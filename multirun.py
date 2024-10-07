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
from slot import Slot
from threaded import threaded

cpus = get_cpu_info()
gpus_nv  = get_gpu_info_nvidia()
gpus_amd = get_gpu_info_amd()

# configure how to merge different files
# 'inputs' can be
#   - 'stdin'   to concatenate all inputs and pass them as standard input (NOT IMPLEMENTED), e.g.
#                   cat in1 in2 in3 ... | command ...
#
#   - 'arg'     to pass all inputs as arguments, e.g.
#                   command in1 in2 in3 ...
#
#   - 'option'  to pass all inputs as arguments after a single option, e.g.
#                   command -i in1 in2 in3 ...
#
#   - 'multi'   to pass all inputs as arguments to individual options, e.g.
#                   command -i in1 -i in2 -i in3 ...
#
# For 'option' and 'multi' the option is given in the "inputs_option" field.
#
# 'output' can be
#   - 'stdout'  to write the combined output to the standard output of the command, e.g.
#                   command ... > output
#
#   - 'arg'     to write the combined output to the (last) argument to the command, e.g.
#                   command ... output
#
#   - 'option'  to write the combined output to the argument to the given option, e.g.
#                   command ... -o output
#
# For 'option' the option is given in the "output_option" field.

auto_merge_map = {
  'resources.json': {
    'cmd': 'mergeResourcesJson.py',
    'args': [],
    'inputs': 'args',
    'inputs_options': None,
    'output': 'stdout',
    'output_options': None,
  }
}

def runMergeCommand(tag, workdir, inputs, output, verbose):
  if not tag in auto_merge_map:
    return

  entry = auto_merge_map[tag]
  cmd = entry['cmd']
  args = entry['args']
  ins = entry['inputs']
  out = entry['output']

  exec = shutil.which(cmd)
  if exec:
    command = [ exec ]
  else:
    raise RuntimeError(f'cannot find command {cmd} .')

  if args:
    command.extend(args)

  stdin = None
  if ins == 'stdin':
    # FIXME does it make sense to pass multiple inputs as stdin ?
    raise NotImplementedError(f'auto merge command {tag} uses input through stdin, which is not supported')
    sys.exit(1)
  elif ins == 'args':
    command.extend(inputs)
  elif ins == 'option':
    opt = entry['inputs_options']
    command.append(opt)
    command.extend(inputs)
  elif ins == 'multi':
    opt = entry['inputs_options']
    for i in inputs:
      command.extend((opt, i))
  else:
    raise NotImplementedError(f'auto merge command for {tag} uses an unknown input schema {ins}.')

  stdout = None
  if out == 'stdout':
    stdout = open(output, 'w')
  elif out == 'arg':
    command.append(output)
  elif out == 'option':
    opt = entry['output_options']
    command.append(opt)
    command.append(output)
  else:
    raise NotImplementedError(f'auto merge command for {tag} uses an unknown output schema {out}.')

  cmdline = ' '.join(command)
  if verbose:
    sys.stdout.write(cmdline + '\n')
    sys.stdout.flush()
  pipe = subprocess.run(command, stdin = stdin, stdout = stdout, stderr = subprocess.PIPE)
  if stdout is not None:
    stdout.close()
  if pipe.returncode != 0:
    raise RuntimeError(f'Exit code {pipe.returncode} while running "' + cmdline + '"\n\n' + pipe.stderr.decode(sys.stdout.encoding))


@threaded
def singleCmsRun(filename, workdir, logdir = None, keep = [], autodelete = [], autodelete_delay = 60., verbose = False, slot = None, executable = 'cmsRun', *args):
  if slot is None:
      slot = Slot()

  # if the slot requires a custom number of events, create a copy of the input file and update it accordingly
  if slot.events is not None:
    # create a new configuration file
    oldfilename = filename
    filename = workdir + '/process.py'
    with open(oldfilename, 'r') as oldfile, open(filename, 'w') as newfile:
      # copy the original content to the new configuration file
      oldfile.seek(0)
      newfile.write(oldfile.read())
      # update the number of events in the temporary file
      newfile.write(f'\n# update the number of events to process\nprocess.maxEvents.input = cms.untracked.int32({slot.events})\n')
      if slot.events > -1:
        newfile.write(f'process.ThroughputService.eventRange = cms.untracked.uint32({slot.events})\n')

  # command to execute
  command = [ executable, filename ] + list(args)
  # shell environment
  environment = os.environ.copy()
  # command line for the verbose option
  cmdline = ' '.join(command) + ' &'

  # optionally set NUMA affinity, CPU affinity, and GPU selection
  prefix, environ = slot.get_execution_parameters()
  # update the command to execute
  if prefix:
      command = prefix + command
  # update the shell environment for the command
  if environ:
      environment.update(environ)
  # update the command line for the verbose option
  if prefix or environ:
      cmdline = slot.get_command_line_prefix() + cmdline

  if verbose:
    #print('Running "' + ' '.join((executable, filename) + args) + '"', slot.describe())
    print(cmdline)
    sys.stdout.flush()

  # run a job, redirecting standard output and error to files
  lognames = ['stdout', 'stderr']
  logfiles = tuple('%s/%s' % (workdir, name) for name in lognames)
  stdout = open(logfiles[0], 'w')
  stderr = open(logfiles[1], 'w')

  # collect the monitoring information about the subprocess
  buffer_type = np.dtype([('time', 'datetime64[ms]'), ('vsz', 'int'), ('rss', 'int'), ('pss','int')])
  buffer_data = []

  # start the subprocess
  timestamp = datetime.now()
  autostamp = timestamp
  buffer_data.append((timestamp, 0, 0, 0))  # time, vsize, rss, pss
  job = subprocess.Popen(command, cwd = workdir, env = environment, stdout = stdout, stderr = stderr)
  proc = psutil.Process(job.pid)

  while job.poll() is None:
    # sleep for 1 second
    time.sleep(1.)
    # flush the subprocess stdin, stdout and stderr
    try:
        job.communicate(timeout=0.)
    except subprocess.TimeoutExpired:
        pass
    # measure the subprocess memory usage
    try:
      with proc.oneshot():
        timestamp = datetime.now()
        mem = proc.memory_full_info()
        buffer_data.append((timestamp, mem.vms, mem.rss, mem.pss))  # time, vsize, rss, pss
    except psutil.NoSuchProcess:
      break
    # if requested, autodelete the files in the working directory
    if autodelete:
      stamp = datetime.now()
      if (stamp - autostamp).total_seconds() > autodelete_delay:
        for pattern in autodelete:
          for f in glob.glob(workdir + '/' + pattern):
            os.remove(f)
        autostamp = stamp

  # flush the subprocess stdin, stdout and stderr
  job.communicate()
  stdout.close()
  stderr.close()
  monitoring_data = np.array(buffer_data, buffer_type)

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
    event_time = datetime.strptime(matches.group(2), date_format)
    events.append(event)
    times.append(event_time)

  stderr.close()
  # FIXME write events, times to a python file in the job directory
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
    resolution = 100,               # sample the number of processed events with the given resolution (default: 100)
    skipevents = 300,               # skip the firts EVENTS in each job, rounded to the next multiple of the event resulution (default: 300)
    repeats = 1,                    # number of times to repeat each job (default: 1)
    wait = 0.,                      # number of seconds to wait between repetitions (default: 0)
    jobs = 1,                       # number of jobs to run in parallel (default: 1)
    threads = 1,                    # number of CPU threads per job (default: 1)
    streams = 1,                    # number of EDM streams per job (default: 1)
    gpus_per_job = 1,               # number of GPUs per job (default: 1)
    allow_hyperthreading = True,    # whether to use extra CPU cores from HyperThreading
    set_numa_affinity = False,      # FIXME - run each job in a single NUMA node
    set_cpu_affinity = False,       # whether to set CPU affinity
    set_gpu_affinity = False,       # whether to set GPU affinity
    slots = [],                     # explit job execution environment
    automerge = True,               # automatically merge supported output across all jobs
    autodelete = [],                # automatically delete files matching the given patterns while running the jobs (default: do not autodelete)
    autodelete_delay = 60.,         # check for files to autodelete with this interval (default: 60s)
    executable = 'cmsRun',          # executable to run, usually cmsRun
    *args):                         # additional arguments passed to the executable

  # set the number of streams and threads
  process.options.numberOfThreads = cms.untracked.uint32(threads)
  process.options.numberOfStreams = cms.untracked.uint32(streams)

  # set the number of events to process
  process.maxEvents.input = cms.untracked.int32(events)

  # print a message every "resolution" events
  if not 'ThroughputService' in process.__dict__:
    process.ThroughputService = cms.Service('ThroughputService',
      enableDQM = cms.untracked.bool(False),
    )
  process.ThroughputService.printEventSummary = cms.untracked.bool(True)
  process.ThroughputService.eventResolution = cms.untracked.uint32(resolution)
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

  if slots:
    # explicit description of the job slots
    slots = list(itertools.islice(itertools.cycle(slots), jobs))

  else:
    # try to build jb slots based on various heuristics
    numa_cpu_nodes = [ None ] * jobs
    numa_mem_nodes = [ None ] * jobs
    cpu_assignment = [ None ] * jobs
    gpu_assignment_nvidia = [ None ] * jobs
    gpu_assignment_amd    = [ None ] * jobs

    if set_numa_affinity:
      # FIXME - minimal implementation to test HBM vs DDR memory on Intel Xeon Pro systems
      nodes = sum(len(cpu.nodes) for cpu in cpus.values())
      numa_cpu_nodes = [ str(job % nodes) for job in range(jobs) ]
      numa_mem_nodes = [ str(job % nodes) for job in range(jobs) ]             # use only DDR5
      #numa_mem_nodes = [ str(job % nodes + nodes) for job in range(jobs) ]     # use only HBM

    if set_cpu_affinity:
      # build the list of CPUs for each job:
      #   - build a list of all "processors", grouped by sockets, cores and hardware threads, e.g.
      #     [ 0,2,4,6,8,10,12,14,16,18,20,22,24,26,1,3,5,7,9,11,13,15,17,19,21,23,25,27 ]
      #   - split the list by the number of jobs; if the number of jobs is a multiple of the number of sockets
      #     the jobs should automatically be split on socket boundaries
      #   - otherwise some jobs may span multiple sockets, e.g.
      #     [ 0,2,4,6 ], [ 8,10,12,14 ], [ 16,18,20,22 ], [ 24,26,1,3 ], [ 5,7,9,11 ], [ 13,15,17,19 ], [ 21,23,25,27 ]
      if allow_hyperthreading:
        cpu_list = list(itertools.chain(*(list(map(str, cpu.hardware_threads)) for cpu in cpus.values())))
      else:
        cpu_list = list(itertools.chain(*(list(map(str, cpu.physical_processors)) for cpu in cpus.values())))

      # if all the jobs fit within individual sockets, assing jobs to sockets in a round-robin
      if len(cpu_list) // len(cpus) // threads * len(cpus) >= jobs:
        cpu_assignment = [ '' for i in range(jobs) ]
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

    if set_gpu_affinity:
      # build the list of GPUs for each job:
      #   - if the number of GPUs per job is greater than or equal to the number of GPUs in the system,
      #     run each job on all GPUs
      #   - otherwise, assign GPUs to jobs in a round-robin fashon
      if gpus_per_job >= len(gpus_nv):
        gpu_assignment_nvidia = [ ','.join(map(str, list(gpus_nv.keys()))) for i in range(jobs) ]
      else:
        gpu_repeated = list(map(str, itertools.islice(itertools.cycle(list(gpus_nv.keys())), jobs * gpus_per_job)))
        gpu_assignment_nvidia = [ ','.join(gpu_repeated[i*gpus_per_job:(i+1)*gpus_per_job]) for i in range(jobs) ]

    # define the execution environments
    slots = [ Slot(numa_cpu = numa_cpu_nodes[job], numa_mem = numa_mem_nodes[job], cpus = cpu_assignment[job], nvidia_gpus = gpu_assignment_nvidia[job], amd_gpus = gpu_assignment_amd[job]) for job in range(jobs) ]

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
      job_threads[job] = singleCmsRun(
        config.name,
        workdir = jobdir,
        logdir = thislogdir,
        keep = [],
        autodelete = autodelete,
        autodelete_delay = autodelete_delay,
        verbose = verbose,
        slot = slots[job],
        executable = executable,
        *args)

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

  print('Running %s over %s events with %d jobs, each with %d threads, %d streams, and %d GPUs' % (n_times, n_events, jobs, threads, streams, gpus_per_job))
  sys.stdout.flush()

  # store the values to compute the average throughput over the repetitions
  failed = [ False ] * repeats
  if repeats > 1 and not plumbing:
    throughputs         = [ None ] * repeats
    overlaps            = [ None ] * repeats
    overlap_throughputs = [ None ] * repeats
    overlap_ranges      = [ None ] * repeats

  # store performance points for later analysis
  if data and header:
    data.write('jobs, overlap, CPU threads per job, EDM streams per job, GPUs per job, jobs start timestamp, jobs stop timestamp, minimum number of events, maximum number of events, average throughput (ev/s), average uncertainty (ev/s), overlap start timestamp, overlap stop timestamp, overlap events, overlap throughput (ev/s), overlap uncertainty (ev/s)\n')

  iterations = range(repeats) if repeats > 0 else itertools.count()
  for repeat in iterations:
    # wait the required number of seconds between the warmup and the measurements and between each repetition
    if warmup or repeat > 0:
      time.sleep(wait)

    # run the jobs reading the output to extract the event throughput
    events       = [ None ] * jobs
    times        = [ None ] * jobs
    fits         = [ None ] * jobs
    overlap_fits = [ None ] * jobs
    overlap_size = [ None ] * jobs
    monit        = [ None ] * jobs
    job_threads  = [ None ] * jobs
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
      job_threads[job] = singleCmsRun(
        config.name,
        workdir = jobdir,
        logdir = thislogdir,
        keep = keep,
        autodelete = autodelete,
        autodelete_delay = autodelete_delay,
        verbose = verbose,
        slot = slots[job],
        executable = executable,
        *args)

    # start all threads
    for thread in job_threads:
      thread.start()

    # join all threads
    if verbose:
      time.sleep(0.5)
      print("wait")
      sys.stdout.flush()
    failed_jobs = [ False ] * jobs
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
      # convert to seconds since the POSIX epoch
      nt = tuple(t[i].timestamp() for i in range(len(e)) if e[i] >= skipevents)
      e = ne
      t = nt
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

    # auto-merge supported outputs
    if thislogdir and automerge:
      for tag in keep:
        if tag in auto_merge_map:
          inputs = glob.glob(f'{thislogdir}/pid*/{tag}')
          output = f'{thislogdir}/{tag}'
          runMergeCommand(tag, workdir, inputs, output, verbose)

    # if all jobs were successful, delete the temporary directories
    for job in range(jobs):
      jobdir = os.path.join(workdir.name, "step%02d_part%02d" % (repeat, job))
      shutil.rmtree(jobdir)

    # find the overlapping ranges
    jobs_start = min(times[job][0] for job in range(jobs))
    jobs_stop  = max(times[job][-1] for job in range(jobs))
    if jobs > 1:
      overlap_start = max(times[job][0] for job in range(jobs))
      overlap_stop  = min(times[job][-1] for job in range(jobs))
      # if overlap_start is >= overlap_stop, there is no overlap
      if overlap_start >= overlap_stop:
        overlap_fits = None
        overlap_size = None
      else:
        for job in range(jobs):
          start_index = times[job].searchsorted(overlap_start, 'left')
          stop_index  = times[job].searchsorted(overlap_stop, 'right')
          e = events[job][start_index:stop_index]
          t = times[job][start_index:stop_index]
          overlap_fits[job] = stats.linregress(t, e)
          overlap_size[job] = e[-1] - e[0]
    else:
      overlap_start = jobs_start
      overlap_stop  = jobs_stop
      overlap_fits  = fits
      overlap_size  = [ events[0][-1] - events[0][0] ]

    # measure the average throughput
    min_events  = min(events[job][-1] - events[job][0] for job in range(jobs))
    max_events  = max(events[job][-1] - events[job][0] for job in range(jobs))
    throughput  = sum(fit.slope for fit in fits)
    error       = math.sqrt(sum(fit.stderr * fit.stderr for fit in fits))
    if overlap_fits is None:
        overlap_events     = 0
        overlap_throughput = 0
        overlap_error      = 0
    else:
        overlap_events     = min(overlap_size)
        overlap_throughput = sum(fit.slope for fit in overlap_fits)
        overlap_error      = math.sqrt(sum(fit.stderr * fit.stderr for fit in overlap_fits))
    if jobs > 1:
      # if running more than on job in parallel, estimate and print the overlap among them
      overlap = (min(t[-1] for t in times) - max(t[0] for t in times)) / sum(t[-1] - t[0] for t in times) * len(times)
      if overlap < 0.:
        overlap = 0.
      if plumbing:
        # machine- or human-readable formatting
        print(', %8.1f\t%8.1f\t%d\t%d\t%0.1f%%\t%8.1f\t%8.1f\t%d' % (throughput, error, min_events, max_events, overlap * 100., overlap_throughput, overlap_error, overlap_events))
      else:
        # human-readable formatting
        if min_events == max_events:
            print('%8.1f \u00b1 %5.1f ev/s (%d events, %0.1f%% overlap)' % (throughput, error, min_events, overlap * 100.), end='')
        else:
            print('%8.1f \u00b1 %5.1f ev/s (%d-%d events, %0.1f%% overlap)' % (throughput, error, min_events, max_events, overlap * 100.), end='')
        if overlap_events > 0:
          print(', %8.1f \u00b1 %5.1f ev/s (\u2a7e %d events, overlap-only)' % (overlap_throughput, overlap_error, overlap_events))
        else:
          print()
    else:
      # with a single job the overlap does not make sense
      overlap = 1.
      overlap_events = min_events
      overlap_throughput = throughput
      overlap_error = error
      # machine- or human-readable formatting
      formatting = '%8.1f\t%8.1f\t%d' if plumbing else '%8.1f \u00b1 %5.1f ev/s (%d events)'
      print(formatting % (throughput, error, min_events))
    sys.stdout.flush()

    # store the values to compute the average throughput over the repetitions
    if repeats > 1 and not plumbing:
      throughputs[repeat]         = throughput
      overlaps[repeat]            = overlap
      overlap_throughputs[repeat] = overlap_throughput
      overlap_ranges[repeat]      = overlap_events

    # store performance points for later analysis
    if data:
      data.write(f'{jobs}, {overlap:0.4f}, {threads}, {streams}, {gpus_per_job}, {jobs_start:.3f}, {jobs_stop:.3f}, {min_events}, {max_events}, {throughput}, {error}, {overlap_start:.3f}, {overlap_stop:.3f}, {overlap_events}, {overlap_throughput}, {overlap_error}\n')

    # do something with the monitoring data
    if thislogdir is not None:
      monit_file = open(thislogdir + '/monit.py', 'w')
      monit_file.write("import numpy as np\n\n")
      monit_file.write("monit = ")
      monit_file.write(repr(monit).replace('array', '\n  np.array'))
      monit_file.write("\n")
      monit_file.close()

  # auto-merge supported outputs
  if logdir and automerge:
    for tag in keep:
      if tag in auto_merge_map:
        inputs = glob.glob(f'{logdir}/step*/{tag}')
        output = f'{logdir}/{tag}'
        runMergeCommand(tag, workdir, inputs, output, verbose)

  # compute the average throughput over the repetitions
  if repeats > 1 and not plumbing:
    # filter out the failed or inconsistent jobs
    throughputs         = [ throughputs[i] for i in range(repeats) if not failed[i] ]
    overlaps            = [ overlaps[i]    for i in range(repeats) if not failed[i] ]
    overlap_throughputs = [ overlap_throughputs[i] for i in range(repeats) if not failed[i] ]
    overlap_ranges      = [ overlap_ranges[i] for i in range(repeats) if not failed[i] ]
    # filter out the jobs with an overlap lower than 90%
    values              = [ throughputs[i] for i in range(len(throughputs)) if overlaps[i] >= 0.90 ]
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
    # overlap-only values
    overlap_value = np.average(overlap_throughputs)
    overlap_error = np.std(overlap_throughputs, ddof=1)
    overlap_range = min(overlap_ranges)
    # print the summary
    print(' --------------------')
    if n == repeats:
      print('%8.1f \u00b1 %5.1f ev/s' % (value, error), end='')
    elif n > 1:
      print('%8.1f \u00b1 %5.1f ev/s (based on %d measurements)' % (value, error, n), end='')
    elif n > 0:
      print('%8.1f ev/s (based on a single measurement)' % (value, ), end='')
    else:
      print('%8.1f ev/s (single measurement with the highest overlap)' % (value, ), end='')
    # print the overlap-only measurements only if at least one repetition had some overlap
    if overlap_range > 0:
      print(', %8.1f \u00b1 %5.1f ev/s (\u2a7e %d events, overlap-only)' % (overlap_value, overlap_error, overlap_range))

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

  if gpus_nv:
    print('%d visible NVIDIA CUDA GPUs:' % len(gpus_nv))
    for gpu in gpus_nv.values():
      print('  %d: %s' % (gpu.device, gpu.model))
  else:
    print('No visible NVIDIA CUDA GPUs')
  print()

  if gpus_amd:
    print('%d visible AMD ROCm GPUs:' % len(gpus_amd))
    for gpu in gpus_amd.values():
      print('  %d: %s' % (gpu.device, gpu.model))
  else:
    print('No visible AMD ROCm GPUs')
  print()

  sys.stdout.flush()


if __name__ == "__main__":
  # parse the command line options
  from options import OptionParser
  parser = OptionParser()
  opts = parser.parse(sys.argv[1:])
  options = {
    'verbose'             : opts.verbose,
    'plumbing'            : opts.plumbing,
    'warmup'              : opts.warmup,
    'events'              : opts.events,
    'resolution'          : opts.event_resolution,
    'skipevents'          : opts.event_skip,
    'repeats'             : opts.repeats,
    'jobs'                : opts.jobs,
    'threads'             : opts.threads,
    'streams'             : opts.streams,
    'gpus_per_job'        : opts.gpus_per_job,
    'allow_hyperthreading': opts.allow_hyperthreading,
    'set_numa_affinity'   : opts.numa_affinity,
    'set_cpu_affinity'    : opts.cpu_affinity,
    'set_gpu_affinity'    : opts.gpu_affinity,
    'slots'               : opts.slots,
    'executable'          : opts.executable,
    'logdir'              : opts.logdir if opts.logdir else None,
    'tmpdir'              : opts.tmpdir,
    'keep'                : opts.keep,
  }

  if options['verbose']:
    info()

  process = parseProcess(opts.config)
  multiCmsRun(process, **options)
