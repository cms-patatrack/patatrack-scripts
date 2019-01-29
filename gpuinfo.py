#! /usr/bin/env python

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
    visible = [int(device) for device in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

  devices = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
  for line in devices.splitlines():
    matches = re.match(r'GPU ([0-9]+): (.*)', line)
    if matches:
      device = int(matches.group(1))
      if visible and not device in visible:
        continue
      model  = matches.group(2).strip()
      gpus[device] = GPUInfo(device, model)

  if cache:
    __cache = gpus

  return gpus


if __name__ == "__main__":
  gpus = get_gpu_info()
  print '%d visible NVIDIA GPUs:' % len(gpus)
  for gpu in gpus.values():
    print '  %d: %s' % (gpu.device, gpu.model)

