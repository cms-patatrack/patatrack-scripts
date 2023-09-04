#! /usr/bin/env python3

import sys
import os
import copy
import glob
import imp
import itertools
import math
import psutil
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime

# silence NumPy warnings about denormals
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from scipy import stats
warnings.filterwarnings("default", category=UserWarning)

# check that CMSSW_BASE is set
if not 'CMSSW_BASE' in os.environ:
    raise RuntimeError('Please load the CMSSW environment with "cmsenv"')

import FWCore.ParameterSet.Config as cms

from cpuinfo import *
from gpuinfo import *
from threaded import threaded

cpus = get_cpu_info()
gpus = get_gpu_info()

epoch = datetime.now()

# skip at least this many events at the beginning of a job when measuring the throughput
skipevents = 300

def is_iterable(item):
  try:
    iter(item)
    return True
  except TypeError:
    return False


@threaded
def singleCmsRun(filename, workdir, executable = 'cmsRun', logdir = None, keep = [], verbose = False, cpus = None, numa_cpu = None, numa_mem = None, gpus = None, *args):
  command = (executable, filename) + args

  # optionally set CPU affinity
  if cpus is not None:
    command = ('taskset', '-c', cpus) + command

  # optionally set NUMA affinity
  if numa_cpu is not None or numa_mem is not None:
    numa_cmd = ('numactl', )
    numa_cpu_opt = '-N'
    numa_mem_opt = '-m'
    # run on the CPUs of the NUMA nodes "numa_cpu";
    # NUMA nodes may consist of multiple CPUs
    if numa_cpu is not None:
      if is_iterable(numa_cpu) and numa_cpu:
        numa_cmd += (numa_cpu_opt, ','.join(numa_cpu))
      else:
        numa_cmd += (numa_cpu_opt, str(numa_cpu))
    # allocate memory only from the NUMA nodes "numa_mem";
    # allocations will fail when there is not enough memory available on these nodes
    if numa_mem is not None:
      if is_iterable(numa_mem) and numa_mem:
        numa_cmd += (numa_mem_opt, ','.join(numa_mem))
      else:
        numa_cmd += (numa_mem_opt, str(numa_mem))
    command = numa_cmd + command

  # compose the command line for the verbose option
  cmdline = ' '.join(command) + ' &'

  # optionally set GPU affinity
  environment = os.environ.copy()
  if gpus is not None:
    environment['CUDA_VISIBLE_DEVICES'] = gpus
    cmdline = 'CUDA_VISIBLE_DEVICES=' + gpus + ' ' + cmdline

  if verbose:
    print(cmdline)
    sys.stdout.flush()

  # run a job, redirecting standard output and error to files
  lognames = ['stdout', 'stderr']
  logfiles = tuple('%s/%s' % (workdir, name) for name in lognames)
  stdout = open(logfiles[0], 'w')
  stderr = open(logfiles[1], 'w')
  start = datetime.now()
  job = subprocess.Popen(command, cwd = workdir, env = environment, stdout = stdout, stderr = stderr)

  proc = psutil.Process(job.pid)
  raw_data = []
  while True:
    try:
      with proc.oneshot():
        timet = datetime.now()
        mem = proc.memory_full_info()
        raw_data.append(((timet - start).total_seconds(), mem.vms, mem.rss, mem.pss))  # time, vsz, rss, pss
    except psutil.NoSuchProcess:
      break
    try:
      job.communicate()
      time.sleep(1)
      break
    except subprocess.TimeoutExpired:
      pass
  job.communicate()
  stdout.close()
  stderr.close()
  types = np.dtype([('time', 'float'), ('vsz', 'int'), ('rss', 'int'), ('pss','int')])
  monitoring_data = np.array(raw_data, types)

  # if requested, move the logs and any additional artifacts to the log directory
  if logdir:
    # expand any glob patterns in the keep list as-if inside the working directoy
    names = [ name.removeprefix(workdir + '/') for name in itertools.chain(*(glob.glob(workdir + '/' + pattern) for pattern in keep)) ]
    for name in names + lognames:
      source = workdir + '/' + name
      target = '%s/pid%06d/%s' % (logdir, job.pid, name)
      os.makedirs(os.path.dirname(target), exist_ok = True)
      shutil.move(source, target)
    logfiles = tuple('%s/pid%06d/%s' % (logdir, job.pid, name) for name in lognames)

  stderr = open(logfiles[1], 'r')

  if (job.returncode < 0):
    print("The underlying %s job was killed by signal %d" % (executable, -job.returncode))
    print()
    print("The last lines of the error log are:")
    print("".join(stderr.readlines()[-10:]))
    print()
    print("See %s and %s for the full logs" % logfiles)
    sys.stdout.flush()
    stderr.close()
    return None

  elif (job.returncode > 0):
    print("The underlying %s job failed with return code %d" % (executable, job.returncode))
    print()
    print("The last lines of the error log are:")
    print("".join(stderr.readlines()[-10:]))
    print()
    print("See %s and %s for the full logs" % logfiles)
    sys.stdout.flush()
    stderr.close()
    return None

  if verbose:
    print("The underlying %s job completed successfully" % executable)
    sys.stdout.flush()

  # analyse the output
  date_format  = '%d-%b-%Y %H:%M:%S.%f'
  # expected format
  #     100, 18-Mar-2020 12:16:39.172836 CET
  begin_pattern = re.compile(r'%MSG-. ThroughputService:  *AfterModEndJob')
  line_pattern  = re.compile(r' *(\d+), (\d+-...-\d\d\d\d \d\d:\d\d:\d\d.\d\d\d\d\d\d) .*')

  events = []
  times  = []
  matching = False
  for line in stderr:
    # look for the begin marker
    if not matching:
      if begin_pattern.match(line):
        matching = True
      continue

    matches = line_pattern.match(line)
    # check for the end of the events list
    if not matches:
      break

    # read the matching lines
    event = int(matches.group(1))
    timet = datetime.strptime(matches.group(2), date_format)
    events.append(event)
    times.append((timet - epoch).total_seconds())

  stderr.close()
  return (tuple(events), tuple(times), monitoring_data)


def parseProcess(filename):
  # parse the given configuration file and return the `process` object it define
  # the import logic is taken from edmConfigDump
  try:
    handle = open(filename, 'r')
  except:
    print("Failed to open %s: %s" % (filename, sys.exc_info()[1]))
    sys.exit(1)

  # make the behaviour consistent with 'cmsRun file.py'
  sys.path.append(os.getcwd())
  try:
    pycfg = imp.load_source('pycfg', filename, handle)
    process = pycfg.process
  except:
    print("Failed to parse %s: %s" % (filename, sys.exc_info()[1]))
    sys.exit(1)

  handle.close()
  return process


def multiCmsRun(
    process,                        # the cms.Process object to run
    data = None,                    # a file-like object for storing performance measurements
    header = True,                  # write a header before the measurements
    warmup = True,                  # whether to run an extra warm-up job
    tmpdir = None,                  # temporary directory, or None to use a system dependent default temporary directory (default: None)
    logdir = None,                  # a relative or absolute path where to store individual jobs' log files, or None
    keep = [],                      # additional output files to be kept
    verbose = False,                # whether to print extra messages
    plumbing = False,               # print output in a machine-readable format
    events = -1,                    # number of events to process (default: unlimited)
    repeats = 1,                    # number of times to repeat each job (default: 1)
    jobs = 1,                       # number of jobs to run in parallel (default: 1)
    threads = 1,                    # number of CPU threads per job (default: 1)
    streams = 1,                    # number of EDM streams per job (default: 1)
    gpus_per_job = 1,               # number of GPUs per job (default: 1)
    allow_hyperthreading = True,    # whether to use extra CPU cores from HyperThreading
    set_numa_affinity = False,      # FIXME - run each job in a single NUMA node
    set_cpu_affinity = False,       # whether to set CPU affinity
    set_gpu_affinity = False,       # whether yo set GPU affinity
    executable = 'cmsRun',          # executable to run, usually cmsRun
    *args):                         # additional arguments passed to the executable

  # set the number of streams and threads
  process.options.numberOfThreads = cms.untracked.uint32( threads )
  process.options.numberOfStreams = cms.untracked.uint32( streams )

  # set the number of events to process
  process.maxEvents.input = cms.untracked.int32( events )

  # print a message every 100 events
  if not 'ThroughputService' in process.__dict__:
    process.ThroughputService = cms.Service('ThroughputService',
      enableDQM = cms.untracked.bool(False),
    )
  process.ThroughputService.printEventSummary = cms.untracked.bool(True)
  process.ThroughputService.eventResolution = cms.untracked.uint32(100)
  if events > -1:
    process.ThroughputService.eventRange = cms.untracked.uint32(events)

  if not 'MessageLogger' in process.__dict__:
    process.load('FWCore.MessageService.MessageLogger_cfi')
  process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000),
    reportEvery = cms.untracked.int32(1)
  )

  # per-job DAQ output directory
  daqdir = None
  if 'EvFDaqDirector' in process.__dict__:
    daqdir = '%s/run%d' % (process.EvFDaqDirector.baseDir.value(), process.EvFDaqDirector.runNumber.value())

  # make sure the explicit temporary directory exists
  if tmpdir is not None:
      os.makedirs(tmpdir, exist_ok = True)
      tmpdir = os.path.realpath(tmpdir)

  # make a full dump of the configuration, to make changes to the number of threads, streams, etc.
  workdir = tempfile.TemporaryDirectory(prefix = 'multirun', dir = tmpdir)
  config = open(os.path.join(workdir.name, 'process.py'), 'w')
  config.write(process.dumpPython())
  config.close()

  numa_cpu_nodes = [ None ] * jobs
  numa_mem_nodes = [ None ] * jobs
  if set_numa_affinity:
    # FIXME - minimal implementation to test HBM vs DDR memory on Intel Xeon Pro systems
    nodes = sum(len(cpu.nodes) for cpu in cpus.values())
    numa_cpu_nodes = [ job % nodes for job in range(jobs) ]
    numa_mem_nodes = [ job % nodes for job in range(jobs) ]             # use only DDR5
    #numa_mem_nodes = [ job % nodes + nodes for job in range(jobs) ]     # use only HBM

  cpu_assignment = [ None ] * jobs
  if set_cpu_affinity:
    # build the list of CPUs for each job:
    #   - build a list of all "processors", grouped by sockets, cores and hardware threads, e.g.
    #     [ 0,2,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25,27 ]
    #   - split the list by the number of jobs; if the number of jobs is a multiple of the number of sockets
    #     the jobs should automatically be split on socket boundaries
    #   - otherwise some jobs may span multiple sockets, e.g.
    #     [ 0,2,4,6 ], [ 8,10,12,14 ], [ 16,18,20,22 ], [ 24,26,1,3 ], [ 5,7,9,11 ], [ 13,15,17,19 ], [ 21,23,25,27 ]
    # TODO: set the processor assignment as an argument, to support arbitrary splitting
    if allow_hyperthreading:
      cpu_list = list(itertools.chain(*(list(map(str, cpu.hardware_threads)) for cpu in cpus.values())))
    else:
      cpu_list = list(itertools.chain(*(list(map(str, cpu.physical_processors)) for cpu in cpus.values())))

    # if all the jobs fit within individual sockets, assing jobs to sockets in a round-robin
    if len(cpu_list) // len(cpus) // threads * len(cpus) >= jobs:
      cpu_assignment = [ list() for i in range(jobs) ]
      if allow_hyperthreading:
        available_cpus = [ copy.copy(cpu.hardware_threads) for cpu in cpus.values() ]
      else:
        available_cpus = [ copy.copy(cpu.physical_processors) for cpu in cpus.values() ]
      for job in range(jobs):
        socket = job % len(cpus)
        cpu_assignment[job] = ','.join(map(str, available_cpus[socket][0:threads]))
        del available_cpus[socket][0:threads]

    # otherwise, split the list by the number of jobs, and possibly overcommit
    else:
      if len(cpu_list) >= jobs * threads:
        # split the list by the number of jobs
        index = [ i * threads for i in range(jobs+1) ]
      else:
        # fill all cpus and overcommit
        index = [ i * len(cpu_list) // jobs for i in range(jobs+1) ]

      cpu_assignment = [ ','.join(cpu_list[index[i]:index[i+1]]) for i in range(jobs) ]

  gpu_assignment = [ None ] * jobs
  if set_gpu_affinity:
    # build the list of GPUs for each job:
    #   - if the number of GPUs per job is greater than or equal to the number of GPUs in the system,
    #     run each job on all GPUs
    #   - otherwise, assign GPUs to jobs in a round-robin fashon
    # TODO: set the GPU assignment as an argument, to support arbitrary splitting
    if gpus_per_job >= len(gpus):
      gpu_assignment = [ ','.join(map(str, list(gpus.keys()))) for i in range(jobs) ]
    else:
      gpu_repeated   = list(map(str, itertools.islice(itertools.cycle(list(gpus.keys())), jobs * gpus_per_job)))
      gpu_assignment = [ ','.join(gpu_repeated[i*gpus_per_job:(i+1)*gpus_per_job]) for i in range(jobs) ]

  if warmup:
    print('Warming up')
    sys.stdout.flush()
    # recreate logs' directory
    if logdir is not None:
      thislogdir = logdir + '/warmup'
      shutil.rmtree(thislogdir, True)
      os.makedirs(thislogdir)
    else:
      thislogdir = None
    # create work directories and work threads
    job_threads = [ None ] * jobs
    for job in range(jobs):
      jobdir = os.path.join(workdir.name, "warmup_part%02d" % job)
      os.mkdir(jobdir)
      if daqdir is not None:
        if daqdir.startswith('/'):
          os.makedirs(daqdir, exists_ok = True)
        else:
          os.makedirs(os.path.join(jobdir, daqdir))
      job_threads[job] = singleCmsRun(config.name, jobdir, executable = executable, logdir = thislogdir, keep = [], verbose = verbose, cpus = cpu_assignment[job], gpus = gpu_assignment[job], numa_cpu = numa_cpu_nodes[job], numa_mem = numa_mem_nodes[job], *args)

    # start all threads
    for thread in job_threads:
      thread.start()

    # join all threads
    if verbose:
      print("wait")
      sys.stdout.flush()
    for thread in job_threads:
      thread.join()

    # delete all temporary directories
    for job in range(jobs):
      jobdir = os.path.join(workdir.name, "warmup_part%02d" % job)
      shutil.rmtree(jobdir)
    print()
    sys.stdout.flush()

  if repeats > 1:
    n_times = '%d times' % repeats
  elif repeats == 1:
    n_times = 'once'
  else:
    n_times = 'indefinitely'

  if events >= 0:
    n_events = str(events)
  else:
    n_events = 'all'

  print('Running %s over %s events with %d jobs, each with %d threads, %d streams and %d GPUs' % (n_times, n_events, jobs, threads, streams, gpus_per_job))
  sys.stdout.flush()

  # store the values to compute the average throughput over the repetitions
  failed = [ False ] * repeats
  if repeats > 1 and not plumbing:
    throughputs = [ None ] * repeats
    overlaps    = [ None ] * repeats

  # store performance points for later analysis
  if data and header:
    data.write('%s, %s, %s, %s, %s, %s, %s, %s\n' % ('jobs', 'overlap', 'CPU threads per job', 'EDM streams per job', 'GPUs per jobs', 'number of events', 'average throughput (ev/s)', 'uncertainty (ev/s)'))

  iterations = range(repeats) if repeats > 0 else itertools.count()
  for repeat in iterations:
    # run the jobs reading the output to extract the event throughput
    events      = [ None ] * jobs
    times       = [ None ] * jobs
    fits        = [ None ] * jobs
    monit       = [ None ] * jobs
    job_threads = [ None ] * jobs
    # recreate logs' directory
    if logdir is not None:
      thislogdir = logdir + '/step%04d' % repeat
      shutil.rmtree(thislogdir, True)
      os.makedirs(thislogdir)
    else:
      thislogdir = None
    # create work directories and work threads
    for job in range(jobs):
      jobdir = os.path.join(workdir.name, "step%02d_part%02d" % (repeat, job))
      os.mkdir(jobdir)
      if daqdir is not None:
        if daqdir.startswith('/'):
          os.makedirs(daqdir, exists_ok = True)
        else:
          os.makedirs(os.path.join(jobdir, daqdir))
      job_threads[job] = singleCmsRun(config.name, jobdir, executable = executable, logdir = thislogdir, keep = keep, verbose = verbose, cpus = cpu_assignment[job], gpus = gpu_assignment[job], numa_cpu = numa_cpu_nodes[job], numa_mem = numa_mem_nodes[job], *args)

    # start all threads
    for thread in job_threads:
      thread.start()

    # join all threads
    if verbose:
      time.sleep(0.5)
      print("wait")
      sys.stdout.flush()
    failed_jobs = [ False ] * jobs
    consistent_events = defaultdict(int)
    for job, thread in enumerate(job_threads):
      # implicitly wait for the thread to complete
      result = thread.result.get()
      if result is None:
        failed_jobs[job] = True
        continue
      (e, t, m) = result
      if not e or not t:
        failed_jobs[job] = True
        continue
      # skip the entries before skipevents
      ne = tuple(e[i] for i in range(len(e)) if e[i] >= skipevents)
      nt = tuple(t[i] for i in range(len(e)) if e[i] >= skipevents)
      e = ne
      t = nt
      consistent_events[e] += 1
      events[job] = np.array(e)
      times[job]  = np.array(t)
      fits[job]   = stats.linregress(times[job], events[job])
      monit[job]  = m

    # if any jobs failed, skip the whole measurement
    if any(failed_jobs):
      print('%d %s failed, this measurement will be ignored' % (sum(failed_jobs), 'jobs' if sum(failed_jobs) > 1 else 'job'))
      sys.stdout.flush()
      failed[repeat] = True
      continue

    # if all jobs were successful, delete the temporary directories
    for job in range(jobs):
      jobdir = os.path.join(workdir.name, "step%02d_part%02d" % (repeat, job))
      shutil.rmtree(jobdir)

    reference_events = np.array(sorted(consistent_events, key = consistent_events.get, reverse = True)[0])

    # check for jobs with inconsistent events
    inconsistent = False
    for job in range(jobs):
      if (len(events[job]) != len(reference_events)) or any(events[job] != reference_events):
        print('Inconsistent measurement points for job %d' % job)
        sys.stdout.flush()
        inconsistent = True

    # delete data from inconsistent jobs
    if inconsistent:
      print('Inconsistent results detected, this measurement will be ignored')
      sys.stdout.flush()
      failed[repeat] = True
      continue

    # measure the average throughput
    used_events = reference_events[-1] - reference_events[0]
    throughput  = sum(fit.slope for fit in fits)
    error       = math.sqrt(sum(fit.stderr * fit.stderr for fit in fits))
    if jobs > 1:
      # if running more than on job in parallel, estimate and print the overlap among them
      overlap = (min(t[-1] for t in times) - max(t[0] for t in times)) / sum(t[-1] - t[0] for t in times) * len(times)
      if overlap < 0.:
        overlap = 0.
      # machine- or human-readable formatting
      formatting = '%8.1f\t%8.1f\t%d\t%0.1f%%' if plumbing else '%8.1f \u00b1 %5.1f ev/s (%d events, %0.1f%% overlap)'
      print(formatting % (throughput, error, used_events, overlap * 100.))
    else:
      overlap = 1.
      # machine- or human-readable formatting
      formatting = '%8.1f\t%8.1f\t%d' if plumbing else '%8.1f \u00b1 %5.1f ev/s (%d events)'
      print(formatting % (throughput, error, used_events))
    sys.stdout.flush()

    # store the values to compute the average throughput over the repetitions
    if repeats > 1 and not plumbing:
      throughputs[repeat] = throughput
      overlaps[repeat]    = overlap

    # store performance points for later analysis
    if data:
      data.write('%d, %f, %d, %d, %d, %d, %f, %f\n' % (jobs, overlap, threads, streams, gpus_per_job, used_events, throughput, error))

    # do something with the monitoring data
    if thislogdir is not None:
      monit_file = open(thislogdir + '/monit.py', 'w')
      monit_file.write("import numpy as np\n\n")
      monit_file.write("monit = ")
      monit_file.write(repr(monit).replace('array', '\n  np.array'))
      monit_file.write("\n")
      monit_file.close()

  # compute the average throughput over the repetitions
  if repeats > 1 and not plumbing:
    # filter out the failed or inconsistent jobs
    throughputs = [ throughputs[i] for i in range(repeats) if not failed[i] ]
    overlaps    = [ overlaps[i]    for i in range(repeats) if not failed[i] ]
    # filter out the jobs with an overlap lower than 90%
    values      = [ throughputs[i] for i in range(len(throughputs)) if overlaps[i] >= 0.90 ]
    n = len(values)
    if n > 1:
      value = np.average(values)
      error = np.std(values, ddof=1)
    elif n > 0:
      # only a single valid job with an overlap > 90%, use its result
      value = values[0]
      error = float('nan')
    else:
      # no valid jobs with an overlap > 90%, use the "best" one
      value = throughputs[overlaps.index(max(overlaps))]
      error = float('nan')
    print(' --------------------')
    if n == repeats:
      print('%8.1f \u00b1 %5.1f ev/s' % (value, error))
    elif n > 1:
      print('%8.1f \u00b1 %5.1f ev/s (based on %d measurements)' % (value, error, n))
    elif n > 0:
      print('%8.1f ev/s (based on a single measurement)' % (value, ))
    else:
      print('%8.1f ev/s (single measurement with the highest overlap)' % (value, ))

  if not plumbing:
    print()
    sys.stdout.flush()

  # delete the temporary work dir
  workdir.cleanup()


def info():
  print('%d CPUs:' % len(cpus))
  for cpu in cpus.values():
    print('  %d: %s (%d cores, %d threads)' % (cpu.socket, cpu.model, len(cpu.physical_processors), len(cpu.hardware_threads)))
  print()

  print('%d visible NVIDIA GPUs:' % len(gpus))
  for gpu in gpus.values():
    print('  %d: %s' % (gpu.device, gpu.model))
  print()
  sys.stdout.flush()


if __name__ == "__main__":
  options = {
    'verbose'             : False,
    'plumbing'            : False,
    'warmup'              : True,
    'events'              : 4200,
    'repeats'             : 4,
    'jobs'                : 2,
    'threads'             :16,          # per job
    'streams'             : 8,          # per job
    'gpus_per_job'        : 2,          # per job
    'allow_hyperthreading': True,
    'set_numa_affinity'   : False,
    'set_cpu_affinity'    : True,
    'set_gpu_affinity'    : True,
  }

  # TODO parse arguments and options from the command line

  if options['verbose']:
    info()

  if len(sys.argv) > 1:
    process = parseProcess(sys.argv[1])
    multiCmsRun(process, **options)

