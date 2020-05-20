#! /usr/bin/env python

import sys
import os.path

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

for filename in sys.argv[1:]:
  # expected file format:
  #   jobs, overlap, CPU threads per job, EDM streams per job, GPUs per jobs, number of events, average throughput (ev/s), uncertainty (ev/s)
  #   2, 0.994863, 6, 6, 1, 4000, 3591.314398, 1.665309
  #   ...
  values = pd.read_csv(filename).rename(columns=lambda x: x.strip())

  # if the data does not have a name, build it from the file name
  if not 'name' in values:
    name = os.path.basename(filename)
    if '.' in name:
      i = name.rindex('.')
      name = name[:i]
    values.insert(0, 'name', [ name ] * len(values), True)
    data.append(values)

df = pd.concat(data, ignore_index = True)
del data

# normalise to the number of jobs
if options['normalise']:
  df['average throughput (ev/s)'] /= df['jobs']
  df['uncertainty (ev/s)']        /= df['jobs']

# compute the total numer of CPU threads and EDM streams
df['CPU threads'] =  df['CPU threads per job'] * df['jobs']
df['EDM streams'] =  df['EDM streams per job'] * df['jobs']

plot = sns.lmplot(
  data = df,
  x = options['x axis'],
  y = 'average throughput (ev/s)',
  fit_reg = True,                   # estimate and plot a regression model
  order = 4,                        # polynomial fit
  hue = 'name',                     # different categories
  height = 5.4,                     # plot height in inches, at 100 dpi
  aspect = 16./9.,                  # plot aspect ratio
  legend = True,
  legend_out = True,                # show the legend to the right of the plot
  truncate = False,
  ci = 95.,
  )

# zoomed-in version of the plot
fix_plot_range(plot, zoom = True)   # workaround for seaborn 0.9.0
plot.savefig('zoom.png')

# full Y axis
fix_plot_range(plot)                # workaround for seaborn 0.9.0
plot.savefig('plot.png')
