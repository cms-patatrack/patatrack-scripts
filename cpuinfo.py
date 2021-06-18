#! /usr/bin/env python3

import sys
import subprocess
import re
import collections


class CPUInfo(object):
  def __init__(self, socket = None, model = None):
    self.socket = socket
    self.model  = model
    self.cores  = {}
    self.hardware_threads    = []
    self.physical_processors = []

  def add_core(self, core, thread):
    if core in self.cores:
      self.cores[core].append(thread)
    else:
      self.cores[core] = [ thread ]

  def finalise(self):
    for core in self.cores.values():
      self.physical_processors.append(core[0])
      self.hardware_threads.extend(core)
    self.physical_processors.sort()
    self.hardware_threads.sort()


# cache results across calls
__cache = None


# return a mapping between sockets and CPUInfo objects
def get_cpu_info(cache = True):
  global __cache
  if cache and __cache:
    return __cache

  cpus = collections.OrderedDict()

  model = 'Unknown CPU'
  description = subprocess.Popen(['lscpu', ], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
  for line in description.splitlines():
    if 'Model name:' in line:
      model = line.split(':')[1].strip()
      break

  devices = subprocess.Popen(['lscpu', '-b', '-p=SOCKET,NODE,CORE,CPU'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
  for line in devices.splitlines():
    if '#' in line:
      continue

    sock, numa, core, proc = line.split(',')
    sock = int(sock) if sock else 0
    numa = int(numa) if numa else sock      # currently unused
    core = int(core) if core else 0
    proc = int(proc) if proc else 0

    if not sock in cpus:
      cpus[sock] = CPUInfo(sock, model)
    cpus[sock].add_core(core, proc)

  for cpu in cpus.values():
    cpu.finalise()

  if cache:
    __cache = cpus

  return cpus


if __name__ == "__main__":
  cpus = get_cpu_info()
  print '%d CPUs:' % len(cpus)
  for cpu in cpus.values():
    print '  %d: %s (%d cores, %d threads)' % (cpu.socket, cpu.model, len(cpu.physical_processors), len(cpu.hardware_threads))
    print '      cores: %s' % ', '.join(map(str, cpu.physical_processors))
    print '      HT\'s:  %s' % ', '.join(map(str, cpu.hardware_threads))
