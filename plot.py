#! /usr/bin/env python3

"""
Plot average throughput vs number of CPU threads per job.

Command examples:

+ Read 3 csv files and store OUTPUT.pdf and OUTPUT.png:
python3 patatrack-scipts/plot.py scan/reduced_hlt_{ecal,hcal,pixel}_w7900.csv --title Labels --labels ECAL HCAL Pixel -o OUTPUT
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
import os
import argparse

def make_unique_label(new_label, label_list):
    if new_label not in label_list:
        return new_label

    pattern = re.compile(rf"^{re.escape(new_label)}(\d+)?$")
    existing_indices = []

    for label in label_list:
        match = pattern.match(label)
        if match:
            # Extract the index if it exists
            suffix = match.group(1)
            if suffix:
                index = int(suffix)
                existing_indices.append(index)

    # Find the next available index (starting from 2)
    next_index = 2
    while next_index in existing_indices:
        next_index += 1

    return f"{new_label}{next_index}"
    
# Create the parser
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

# Optional arguments
parser.add_argument('-t', '--title', type=str, default="",
                    help="Title of the legend")
parser.add_argument('-o', '--output', type=str, default="throughput_vs_threads",
                    help="Base filename for output files (PNG/PDF)")
parser.add_argument('-x', '--x-axis', default='CPU threads per job',
                    help='Horizontal axis label.')
parser.add_argument('--labels', nargs='+', default=None,
                    help="Labels to show in the legend instead of the CSV file names")

# Positional arguments (CSV files)
parser.add_argument('files', nargs='+', help="CSV files to process")

# Parse arguments
args = parser.parse_args()

# Access the values
title = args.title
filename = args.output
files = args.files
labels = args.labels
if labels is not None:
    assert len(files) == len(labels), "The number of labels must match the number of input CSV files. Each label corresponds to one file, following the order they are provided."

# Dictionary to store per-file datasets
datasets = {}

for file in files:
    # Read CSV and clean column names
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    
    # Keep only relevant columns (ignore "jobs")
    df = df[["CPU threads per job", "average throughput (ev/s)"]]
    
    # Group by CPU threads per job and compute mean & std
    grouped = (
        df.groupby("CPU threads per job")["average throughput (ev/s)"]
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values("CPU threads per job")
    )
    
    # Create a nicer label: remove extension and replace underscores
    label = os.path.basename(file) if labels is None else labels[files.index(file)]
    if label.endswith(".csv"):
        label = label[:-4]
    label = make_unique_label(label.replace("_", " "), datasets.keys())
    datasets[label] = grouped

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for label, df in datasets.items():
    #df["std"] = df["std"].fillna(0)  # In case some groups have a single entry
    color = ax.plot(df["CPU threads per job"], df["mean"], '--', linewidth=1.5)[0].get_color()
    ax.errorbar(df["CPU threads per job"], df["mean"], yerr=df["std"],
                label=label, marker='o', markersize=8, capsize=5, capthick=2,
                ls='none', color=color)

ax.set_xlabel(args.x_axis)
ax.set_ylabel("Average throughput (ev/s)")
ax.set_title("Average throughput vs number of CPU threads per job")
if title:
  ax.legend(title=title, title_fontsize='13', fontsize='11')
else:
  ax.legend()

transparency = dict(alpha=0.7)
ax.grid(True, axis='x', **transparency)
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.grid(True, which='major', axis='both', **transparency)
ax.set_ylim(bottom=0)

# Make the axes (plot area) white
ax.set_facecolor('white')

# Make only the figure background (outside axes) transparent
fig.patch.set_facecolor('none')  # fully transparent
fig.patch.set_alpha(0)

fig.tight_layout()

# Save as PNG and PDF with transparent canvas background
fig.savefig(f"{filename}.png", dpi=600)
fig.savefig(f"{filename}.pdf")
