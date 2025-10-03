#! /usr/bin/env python3

"""
Plot average throughput vs number of CPU threads per job.

Command examples:

+ Read 3 csv files and store OUTPUT.pdf and OUTPUT.png:
python3 patatrack-scipts/plot.py scan/reduced_hlt_{ecal,hcal,pixel}_w7900.csv --title Labels --csv_labels ECAL HCAL Pixel -o OUTPUT
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys
import os
import argparse

# Create the parser
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

# Optional arguments
parser.add_argument('-t', '--title', type=str, default="",
                    help="Title of the legend")
parser.add_argument('-o', '--output', type=str, default="throughput_vs_threads",
                    help="Base filename for output files (PNG/PDF)")
parser.add_argument('--csv_labels', nargs='+', default=None, required=False,
                    help="CSV labels to process (one-to-one mapping with the CSV files)")

# Positional arguments (CSV files)
parser.add_argument('csv_files', nargs='+', help="CSV files to process")

# Parse arguments
args = parser.parse_args()

# Access the values
title = args.title
filename = args.output
csv_files = args.csv_files
csv_labels = args.csv_labels
if csv_labels is not None:
    assert len(csv_files) == len(csv_labels)

# Dictionary to store per-file datasets
datasets = {}

for file in csv_files:
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
    label = os.path.basename(file) if csv_labels is None else csv_labels[csv_files.index(file)]
    if label.endswith(".csv"):
        label = label[:-4]
    label = label.replace("_", " ")
    
    datasets[label] = grouped

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for label, df in datasets.items():
    #df["std"] = df["std"].fillna(0)  # In case some groups have a single entry
    ax.errorbar(df["CPU threads per job"], df["mean"], yerr=df["std"],
                label=label, marker='o', capsize=5, ecolor='black')

ax.set_xlabel("CPU threads per job")
ax.set_ylabel("Average throughput (ev/s)")
ax.set_title("Average throughput vs number of CPU threads per job")
if title:
  ax.legend(title=title, title_fontsize='13', fontsize='11')
else:
  ax.legend()
ax.grid(True, axis='x')
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.grid(True, which='major', axis='x')
ax.grid(True, which='major', axis='y')
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
