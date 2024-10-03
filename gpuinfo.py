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
__cache_amd = None
__cache_nvidia = None


# return a mapping between devices and GPUInfo objects
def get_gpu_info_amd(cache = True):
  global __cache_amd
  if cache and __cache_amd:
    return __cache_amd

  gpus = collections.OrderedDict()

  devices = subprocess.Popen(['rocmComputeCapabilities', ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).communicate()[0]
  for line in devices.splitlines():
    matches = re.match(r' *([0-9]+) +(gfx[0-9]+) +(.*)', line)
    if matches:
      device = int(matches.group(1))
      model  = matches.group(3).strip()
      gpus[device] = GPUInfo(device, model)

  if cache:
    __cache_amd = gpus

  return gpus


def get_gpu_info_nvidia(cache = True):
  global __cache_nvidia
  if cache and __cache_nvidia:
    return __cache_nvidia

  gpus = collections.OrderedDict()

  devices = subprocess.Popen(['cudaComputeCapabilities', ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).communicate()[0]
  for line in devices.splitlines():
    matches = re.match(r' *([0-9]+) +([0-9]+\.[0-9]) +(.*)', line)
    if matches:
      device = int(matches.group(1))
      model  = matches.group(3).strip()
      gpus[device] = GPUInfo(device, model)

  if cache:
    __cache_nvidia = gpus

  return gpus


if __name__ == "__main__":
  # NVIDIA GPUs
  gpus = get_gpu_info_nvidia()
  if gpus:
    print('%d visible NVIDIA CUDA GPUs:' % len(gpus))
    for gpu in gpus.values():
      print('  %d: %s' % (gpu.device, gpu.model))
  else:
    print('No visible NVIDIA CUDA GPUs')
  print()
  
  # AMD GPUs
  gpus = get_gpu_info_amd()
  if gpus:
    print('%d visible AMD ROCm GPUs:' % len(gpus))
    for gpu in gpus.values():
      print('  %d: %s' % (gpu.device, gpu.model))
  else:
    print('No visible AMD ROCm GPUs')
  print()
