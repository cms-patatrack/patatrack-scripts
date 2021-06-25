#! /usr/bin/env python3

import sys
import os
import subprocess
import re
import collections


class GPUInfo(object):
  def __init__(self, device = None, model = None):
    self.device = device
    self.model  = model


# cache results across calls
__cache = None


# return a mapping between devices and GPUInfo objects
def get_gpu_info(cache = True):
  global __cache
  if cache and __cache:
    return __cache

  gpus = collections.OrderedDict()

  visible = None
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    if os.environ['CUDA_VISIBLE_DEVICES'] == '':
      visible = []
    else:
      visible = [int(device) for device in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

  devices = subprocess.Popen(['cudaComputeCapabilities', ], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
  for line in devices.splitlines():
    matches = re.match(r' *([0-9]+) +([0-9]+\.[0-9]) +(.*)', line)
    if matches:
      device = int(matches.group(1))
      if visible:
        device = visible[device]
      model  = matches.group(3).strip()
      gpus[device] = GPUInfo(device, model)

  if cache:
    __cache = gpus

  return gpus


if __name__ == "__main__":
  gpus = get_gpu_info()
  if gpus:
    print('%d visible NVIDIA GPUs:' % len(gpus))
    for gpu in list(gpus.values()):
      print('  %d: %s' % (gpu.device, gpu.model))
  else:
    print('No visible NVIDIA GPUs')
