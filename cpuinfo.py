#! /usr/bin/env python

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

  try:
    cpuinfo = open('/proc/cpuinfo')
  except:
    (type, value, traceback) = sys.exc_info()
    sys.stderr.write('error: %s\n' % value)

  cpus = collections.OrderedDict()

  proc = None
  core = None
  sock = None
  model = ''
  for line in cpuinfo:

    if re.match('processor\s*:', line):
      proc = int(re.match('processor\s*: (\d+)', line).group(1))
    elif re.match('core id\s*:', line):
      core = int(re.match('core id\s*: (\d+)', line).group(1))
    elif re.match('physical id\s*:', line):
      sock = int(re.match('physical id\s*: (\d+)', line).group(1))
    elif re.match('model name\s*:',  line):
      model = re.match('model name\s*: (.*)', line).group(1).strip()

    elif line.strip() == '':
      if not sock in cpus:
        cpus[sock] = CPUInfo(sock, model)
      cpus[sock].add_core(core, proc)

      proc = None
      core = None
      sock = None
      model = ''

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
