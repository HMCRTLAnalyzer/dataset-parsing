#!/usr/bin/python3

import os
import csv
import re
import argparse
import datetime

def is_comment_or_empty(line):
    """Check if a line is a comment or empty."""
    line = line.strip()
    return line.startswith('//') or line.startswith('/*') or line.endswith('*/') or not line

def find_verilog_modules(directory):
    module_pattern = re.compile(r'\bmodule\s+(\w+)\b')
    verilog_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.v') or file.endswith('.sv'):
                verilog_files.append(os.path.join(root, file))

    modules_info = []
    for file in verilog_files:
        with open(file, 'r') as f:
            content = f.readlines()
            inside_comment = False # initialize inside_comment flag
            for line in content:
                if inside_comment:
                    if '*/' in line:
                        inside_comment = False
                    continue  # don't process lines inside multi-line comments
                if is_comment_or_empty(line):
                    # update inside_comment flag for multi-line comments
                    if '/*' in line: # check if this occurs in the middle of the line
                        inside_comment = True
                    continue  # skip comments and empty lines

                modules = module_pattern.findall(line)
                language = 'verilog' if file.endswith('.v') else 'sverilog'
                for module in modules:
                    modules_info.append({'module': module, 'path_to_rtl': os.path.dirname(file), 'language': language})
    
    return modules_info

def write_to_csv(modules_info, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['module', 'path_to_rtl', 'language']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for module_info in modules_info:
            writer.writerow(module_info)

def main():
    parser = argparse.ArgumentParser(description='Generate CSV containing modules, their directory, and the language they are written in.')
    parser.add_argument('directory', type=str, help='Directory containing Verilog source files')
    parser.add_argument('-o', '--output', type=str, default=f'modules_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.csv', help='Output CSV file path')
    args = parser.parse_args()

    modules_info = find_verilog_modules(args.directory)
    write_to_csv(modules_info, args.output)

    print(f"CSV file generated successfully at {args.output}.")

if __name__ == "__main__":
    main()
