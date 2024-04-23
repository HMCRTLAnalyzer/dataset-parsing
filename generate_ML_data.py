#!/usr/bin/python3

import argparse
import pandas as pd
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_DELAY_THRESHOLD = 300
DEFAULT_AREA_THRESHOLD = 40

def filter_module(group):
    """ Function to help filter out modules that don't have both an area and a
        delay recipe present in the df.
    """
    return all(item in group['recipe_type'].values for item in ['delay', 'area'])

def plot_percent_diffs(parsed_df, result_img_path, delay_threshold, area_threshold):

    plt.rcParams.update({'font.size': 16})

    # delay plot
    plt.subplot(121)
    plt.scatter(np.arange(len(parsed_df)), parsed_df['percent_diff_delay'], color='blue')
    plt.xlabel('Modules sorted by delay')
    plt.ylabel("Percent difference in delay")
    plt.ylim([parsed_df['percent_diff_delay'].min() - 10, max(parsed_df['percent_diff_delay'].max(), delay_threshold) + 10])
    plt.grid(axis='y')
   
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    plt.axhline(delay_threshold, color='red')

    # area plot
    plt.subplot(122)
    plt.scatter(np.arange(len(parsed_df)), parsed_df['percent_diff_area'], color='blue')
    plt.xlabel('Modules sorted by delay')
    plt.ylabel("Percent difference in area")
    plt.ylim([parsed_df['percent_diff_area'].min() - 10, max(parsed_df['percent_diff_area'].max(), area_threshold) + 10])    
    plt.grid(axis='y')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    
    plt.axhline(area_threshold, color='red')
    plt.axhline(-area_threshold, color='red')

    # Set face color of current axes to white (non-transparent)
    plt.gca().set_facecolor('white')

    # Set face color of the current figure to transparent
    plt.gcf().patch.set_facecolor('none')
    
    plt.tight_layout(pad=1.1)
    plt.savefig(f"{result_img_path}.png", format='png', dpi=400, bbox_inches='tight')
    # plt.show()

    return

def parse_synthesis_csv(input_csv, output_csv, delay_threshold, area_threshold, memory_label_csv=None):
    df = pd.read_csv(input_csv)

    df.area = df.area.apply(pd.to_numeric, errors='coerce')
    df.delay = df.delay.apply(pd.to_numeric, errors='coerce')

    # drop any rows that have area or delay equal to 0
    df = df[df['area'] != 0]
    df = df[df['delay'] != 0]

    # drop empties
    df.dropna()
    
    df_filtered = df.groupby('module').filter(filter_module)
    # df_filtered.to_csv('filtered.csv')


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

    # merged_df.to_csv('merged.csv')
    # df_diffs.to_csv('diffs.csv')

    df_diffs = df_diffs.rename(columns={"module_path_delayoptimized": "path_to_rtl"})

    df_diffs = df_diffs.dropna()

    # sort by delay difference
    df_diffs = df_diffs.sort_values('percent_diff_delay')
    # df_diffs.to_csv('sorted.csv')

    # read in memory labeled CSV
    
    

    # add sensitivity
    df_diffs['sensitive'] = (df_diffs['percent_diff_delay'] > delay_threshold) | (abs(df_diffs['percent_diff_area']) > area_threshold)
    df_diffs['sensitive'] = df_diffs['sensitive'].astype(int)
    # df_diffs['memory'] = merged_df_2['memory'].astype(int)
    df_diffs['language'] = 'verilog' 

    # df_final = df_diffs[['module', 'path_to_rtl', 'language', 'sensitive', 'memory']]
    df_final = df_diffs[['module', 'path_to_rtl', 'language', 'sensitive', 'percent_diff_area', 'percent_diff_delay']]

    if memory_label_csv is not None:
        df_memlabels = pd.read_csv(memory_label_csv)
        df_final = pd.merge(df_final, df_memlabels[['path_to_rtl', 'memory']], on='path_to_rtl', how='inner')
    # df_merged_2.to_csv('merged_2.csv')
    else:
        df_final['memory'] = 0

    df_final["path_to_rtl"] = df_final["path_to_rtl"].apply(osp.dirname) # only keep directory of RTL file and not the filename


    total_sensitive = sum(df_final['sensitive'])
    total_modules = df_final.shape[0]
    print(f"Total number of modules {total_modules}")
    print(f"Number of sensitive RTL modules: {total_sensitive}")
    print(f"Percent sensitive: {total_sensitive / total_modules * 100}")
    

    df_final.to_csv(output_csv)

    return df_final


def main():
    default_output = '/home/qualcomm_clinic/RTL_dataset/training_data_files_diffs_mem.csv'
    default_input = 'synthesis_data_nangate45nm_newarea.csv'

    parser = argparse.ArgumentParser(description='Parse synthesis data CSV and generate training data for feature generation.')
    parser.add_argument('input_csv', type=str, default=default_input, help='CSV containing module name, path to module, recipe type, delay, and area.')
    parser.add_argument('output_csv', type=str, default=default_output, help='Output CSV file path')
    parser.add_argument('-ml', '--memory_labels', type=str, default=None, help='CSV containing memory labels corresponding to the input CSV.')
    parser.add_argument('-dt', '--delay_threshold', type=float, default=DEFAULT_DELAY_THRESHOLD, help='Percent difference between delay result in delay and area recipe to be considered sensitive.')
    parser.add_argument('-at', '--area_threshold', type=float, default=DEFAULT_AREA_THRESHOLD, help='Percent difference between area result in delay and area recipe to be considered sensitive.')
    parser.add_argument('-p', '--plot', type=str, default=None, help='Filename with no extension to save resulting percent difference plots')
    args = parser.parse_args()

    
    df_final = parse_synthesis_csv(args.input_csv, args.output_csv, args.delay_threshold, args.area_threshold, memory_label_csv=args.memory_labels)
    if args.plot is not None:
        plot_percent_diffs(df_final, args.plot, args.delay_threshold, args.area_threshold)
    return


if __name__ == "__main__":
    main()