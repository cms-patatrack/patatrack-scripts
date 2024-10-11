#! /usr/bin/env python3

import sys
import os
import subprocess
import re
import collections


class GPUInfo(object):
  def __init__(self, device = None, model = None, arch = None):
    self.device = device
    self.model  = model
    self.arch   = arch


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
    matches = re.match(r' *([0-9]+) +(gfx[0-9a-f]+(:(sramecc|xnack)[+-])*) +(.*)', line)
    if matches:
      device = int(matches.group(1))
      model  = matches.group(5).strip()
      arch   = matches.group(2).strip()
      gpus[device] = GPUInfo(device, model, arch)

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
      arch   = matches.group(2).strip()
      # remap 7.5 to sm_75
      arch   = 'sm_' + arch.replace('.', '')
      gpus[device] = GPUInfo(device, model, arch)

  if cache:
    __cache_nvidia = gpus

  return gpus


if __name__ == "__main__":
  # NVIDIA GPUs
  gpus = get_gpu_info_nvidia()
  if gpus:
    print('%d visible NVIDIA CUDA GPUs:' % len(gpus))
    for gpu in gpus.values():
      print('  %d: %s (%s)' % (gpu.device, gpu.model, gpu.arch))
  else:
    print('No visible NVIDIA CUDA GPUs')
  print()
  
  # AMD GPUs
  gpus = get_gpu_info_amd()
  if gpus:
    print('%d visible AMD ROCm GPUs:' % len(gpus))
    for gpu in gpus.values():
      print('  %d: %s (%s)' % (gpu.device, gpu.model, gpu.arch))
  else:
    print('No visible AMD ROCm GPUs')
  print()
