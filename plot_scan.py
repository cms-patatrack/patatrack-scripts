#! /usr/bin/env python3

import sys
import os.path
import argparse

parser = argparse.ArgumentParser(
  description = 'Plot the data points in one or more CSV files produced by the "scan" script.',
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('files',
  metavar = 'FILE',
  nargs = '+',
  type = argparse.FileType('r'),
  help = 'data files to plot')

parser.add_argument('-o', '--output',
  metavar = 'FILE',
  default = 'plot.png',
  type = argparse.FileType('wb', 0),
  help = 'save the plot to FILE')

parser.add_argument('-z', '--zoom',
  metavar = 'FILE',
  default = argparse.SUPPRESS,
  nargs = '?',
  const = 'zoom.png',
  type = argparse.FileType('wb', 0),
  help = 'produce a zoomed-in version of the plot and save it to FILE, or zoom.png if FILE is omitted (default: do not produce a zoomed plot)')

parser.add_argument('-n', '--normalise',
  action = 'store_true',
  default = False,
  help = 'plot the average throughput per job instead of the total throughput across all jobs')

parser.add_argument('-x', '--x-axis',
  choices = ['CPU threads per job', 'CPU threads', 'EDM streams per job', 'EDM streams'],
  default = 'EDM streams',
  help = 'plot vs the number of CPU threads or EDM streams, overall or per job')

parser.add_argument('-l', '--labels',
                      dest = 'labels',
                      nargs = '+',
                      default = None,
                      help = 'list of labels to display in the legend. The default is None, which means extract the legend names from the filenames.')

args = parser.parse_args()

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import seaborn as sns

# plot content options
options = {
  'normalise':  False,              # True: plot the average throughput per job, False: plot the total throughput
  'x axis':    'EDM streams',       # 'CPU threads per job', 'CPU threads', 'EDM streams per job', 'EDM streams'
}

# workaround for seaborn 0.9.0
def fix_plot_range(plot, zoom = False):
  data = plot.data[plot._x_var]
  xmin = min(data)
  xmax = max(data)
  step = (xmax - xmin) * 0.05
  plot.set(xlim=(xmin - step, xmax + step))
  if not zoom:
    plot.set(ylim=(0, None))


sns.set(style={                     # based on 'whitegrid'
  'axes.axisbelow': True,
  'axes.edgecolor': '.15',          # .8
  'axes.facecolor': 'white',
  'axes.grid': True,
  'axes.labelcolor': '.15',
  'axes.linewidth': 1,
  'figure.facecolor': 'white',
  'font.family': ['sans-serif'],
  'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
  'grid.color': '.8',
  'grid.linestyle': '-',
  'image.cmap': 'rocket',
  'legend.frameon': False,
  'legend.numpoints': 1,
  'legend.scatterpoints': 1,
  'lines.solid_capstyle': 'round',
  'text.color': '.15',
  'xtick.color': '.15',
  'xtick.direction': 'out',
  'xtick.major.size': 0,
  'xtick.minor.size': 0,
  'ytick.color': '.15',
  'ytick.direction': 'out',
  'ytick.major.size': 0,
  'ytick.minor.size': 0,
})

sns.set_palette([
  (0.,  0.,  1.),                   # ROOT kBlue
  (1.,  0.,  0.),                   # ROOT kRed
  (0.,  0.,  0.),                   # ROOT kBlack
  (1.,  0.4, 0.),                   # ROOT kOrange +7
  (0.8, 0.2, 0.8),                  # ROOT kMagenta -3
], 5)

data = []

for label_idx,file in enumerate(args.files):
  # expected file format:
  #   jobs, overlap, CPU threads per job, EDM streams per job, GPUs per jobs, number of events, average throughput (ev/s), uncertainty (ev/s)
  #   2, 0.994863, 6, 6, 1, 4000, 3591.314398, 1.665309
  #   ...
  values = pd.read_csv(file).rename(columns=lambda x: x.strip())

  if args.labels:
    if len(args.labels) != len(args.files):
      print("Labels mismatch with input filenames {}".format(args.labels))
      sys.exit(1)
    values.insert(0, 'name', args.labels[label_idx], True)
    data.append(values)
  # if the data does not have a name, build it from the file name
  if not 'name' in values:
    name = os.path.basename(file.name)
    if '.' in name:
      i = name.rindex('.')
      name = name[:i]
    values.insert(0, 'name', [ name ] * len(values), True)
    data.append(values)

df = pd.concat(data, ignore_index = True)
del data

# normalise to the number of jobs
if args.normalise:
  df['average throughput (ev/s)'] /= df['jobs']
  df['uncertainty (ev/s)']        /= df['jobs']

# compute the total numer of CPU threads and EDM streams
df['CPU threads'] =  df['CPU threads per job'] * df['jobs']
df['EDM streams'] =  df['EDM streams per job'] * df['jobs']

sides = 1
markers = [(sides * ((n//3)%3 + 4), (n % 3), 0) for n,_ in enumerate(df['name'].unique())]
print(markers)

plot = sns.lmplot(
  data = df,
  x = args.x_axis,
  y = 'average throughput (ev/s)',
  fit_reg = True,                   # estimate and plot a regression model
  order = 4,                        # polynomial fit
  hue = 'name',                     # different categories
  height = 5.4,                     # plot height in inches, at 100 dpi
  aspect = 16./9.,                  # plot aspect ratio
  legend = True,
  legend_out = True,                # show the legend to the right of the plot
  truncate = False,
  markers = markers,
  scatter_kws={"s": 80},
  ci = 95.,
  )

# zoomed-in version of the plot
if 'zoom' in args:
  fix_plot_range(plot, zoom = True)
  plot.savefig(args.zoom)

# full Y axis
fix_plot_range(plot)
plot.savefig(args.output)
