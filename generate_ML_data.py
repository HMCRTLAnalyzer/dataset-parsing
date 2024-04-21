#!/usr/bin/python3

import argparse,os
import pandas as pd
import os.path as osp
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt

PLOT_ORIG = False
DELAY_THRESHOLD = 300
AREA_THRESHOLD = 40

def filter_module(group):
    """ Function to help filter out modules that don't have both an area and a
        delay recipe present in the df.
    """
    return all(item in group['recipe_type'].values for item in ['delay', 'area'])

def remove_basename(filepath):
    return osp.dirname(filepath)


def main():
    df = pd.read_csv('synthesis_data_nangate45nm_newarea.csv')

    df.area = df.area.apply(pd.to_numeric, errors='coerce')
    df.delay = df.delay.apply(pd.to_numeric, errors='coerce')

    # drop any rows that have area or delay equal to 0
    df = df[df['area'] != 0]
    df = df[df['delay'] != 0]

    # drop empties
    df.dropna()
    
    #plot area based on if recipe is delay or area 
    if PLOT_ORIG:
        fig, ax1 = plt.subplots()
        colors = ["blue" if i == 'delay' else "orange" for i in df.recipe_type]
        line2 = ax1.scatter(df.design, df.area, color=colors)
        ax1.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_title("Area for each Design based on Recipe Type")
        ax1.legend(['Recipe = Delay', 'Recipe = Area'])

        fig, ax2 = plt.subplots()
        colors = ["blue" if i == 'delay' else "orange" for i in df.recipe_type]
        line2 = ax2.scatter(df.design, df.delay, color=colors)
        ax2.set_xticks(ax1.get_xticks(), ax1.get_xticklabels(), rotation=45, ha='right')
        ax2.set_title("Delay for each Design based on Recipe Type")
        ax2.legend(['Recipe = Delay', 'Recipe = Area'])

    df_filtered = df.groupby('module').filter(filter_module)
    df_filtered.to_csv('filtered.csv')


    delay_df = df_filtered[df_filtered['recipe_type'] == 'delay']
    area_df = df_filtered[df_filtered['recipe_type'] == 'area']

    merged_df = pd.merge(delay_df, area_df, on=['design', 'module'], suffixes=('_delayoptimized', '_areaoptimized'))

    '''
    these next operations basically say: 
    in percent, how much better is the delay recipe compared to the area recipe in delay?
    in percent, how much better is the area recipe compared to the delay recipe in area?
    '''

    merged_df['percent_diff_area']  = (merged_df['area_delayoptimized'] / merged_df['area_areaoptimized'] - 1) * 100
    merged_df['percent_diff_delay'] = (merged_df['delay_areaoptimized'] / merged_df['delay_delayoptimized'] - 1) * 100

    # select only the relevant columns
    df_diffs = merged_df[['design', 'module', 'percent_diff_delay', 'percent_diff_area', 'module_path_delayoptimized']]

    merged_df.to_csv('merged.csv')
    df_diffs.to_csv('diffs.csv')

    df_diffs = df_diffs.rename(columns={"module_path_delayoptimized": "path_to_rtl"})

    df_diffs = df_diffs.dropna()

    # sort by delay difference
    df_diffs = df_diffs.sort_values('percent_diff_delay')
    df_diffs.to_csv('sorted.csv')

    # read in memory labeled CSV
    df_memlabels = pd.read_csv('training_data_memory_labels.csv')
    df_merged_2 = pd.merge(df_diffs, df_memlabels[['path_to_rtl', 'memory']], on='path_to_rtl', how='inner')
    df_merged_2.to_csv('merged_2.csv')
    


    # add sensitivity
    df_diffs['sensitive'] = (df_diffs['percent_diff_delay'] > DELAY_THRESHOLD) | (abs(df_diffs['percent_diff_area']) > AREA_THRESHOLD)
    df_diffs['sensitive'] = df_diffs['sensitive'].astype(int)
    # df_diffs['memory'] = merged_df_2['memory'].astype(int)
    df_diffs['language'] = 'verilog'

    # df_final = df_diffs[['module', 'path_to_rtl', 'language', 'sensitive', 'memory']]
    df_final = df_diffs[['module', 'path_to_rtl', 'language', 'sensitive', 'percent_diff_area', 'percent_diff_delay']]

    df_memlabels = pd.read_csv('training_data_memory_labels.csv')
    df_final = pd.merge(df_final, df_memlabels[['path_to_rtl', 'memory']], on='path_to_rtl', how='inner')
    # df_merged_2.to_csv('merged_2.csv')

    df_final["path_to_rtl"] = df_final["path_to_rtl"].apply(osp.dirname) # only keep directory of RTL file and not the filename


    total_sensitive = sum(df_final['sensitive'])
    total_modules = df_final.shape[0]
    print(f"Total number of modules {total_modules}")
    print(f"Number of sensitive RTL modules: {total_sensitive}")
    print(f"Percent sensitive: {total_sensitive / total_modules * 100}")
    

    df_final.to_csv('/home/qualcomm_clinic/RTL_dataset/training_data_files_diffs_mem.csv')

    if False:
        plt.rcParams.update({'font.size': 16})
        plt.subplot(121)
        plt.scatter(np.arange(len(df_diffs)), df_diffs['percent_diff_delay'], color='red')
        plt.xlabel('Modules sorted by delay')
        plt.ylabel("Percent difference in delay")
        # df_diffs.plot.hist(x="module", y="percent_diff_delay", ax=ax)
        plt.ylim([-100, 500])
        plt.grid(axis='y')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        
        # ax.axhline(300, color='red')

        plt.subplot(122)
        plt.scatter(np.arange(len(df_diffs)), df_diffs['percent_diff_area'], color='red')
        plt.xlabel('Modules sorted by delay')
        plt.ylabel("Percent difference in area")
        plt.ylim([-100, 100])
        plt.grid(axis='y')
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        
        # ax.axhline(25, color='red')
        plt.show()


if __name__ == "__main__":
    main()