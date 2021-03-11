#! /usr/bin/env python

import sys
import os
import copy
import imp
import itertools
import math
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
import numpy as np
from scipy import stats

# FIXME check that CMSSW_BASE is set
import FWCore.ParameterSet.Config as cms

# set the output encoding to UTF-8 for pipes and redirects
from set_output_encoding import *
set_output_encoding(encoding='utf-8', force=True)

from cpuinfo import *
from gpuinfo import *
from threaded import threaded

cpus = get_cpu_info()
gpus = get_gpu_info()

epoch = datetime.now()

@threaded
def singleCmsRun(filename, workdir, logdir = None, keep = [], verbose = False, cpus = None, gpus = None, *args):
  # optionally set CPU affinity
  command = ('cmsRun', filename) + args
  if cpus is not None:
    command = ('taskset', '-c', cpus) + command
  cmdline = ' '.join(command)

  # optionally set GPU affinity
  environment = os.environ.copy()
  if gpus is not None:
    environment['CUDA_VISIBLE_DEVICES'] = gpus
    cmdline = 'CUDA_VISIBLE_DEVICES=' + gpus + ' ' + cmdline

  if verbose:
    print cmdline
    sys.stdout.flush()

  # run a cmsRun job, redirecting standard output and error to files
  lognames = ('stdout', 'stderr')
  logfiles = tuple('%s/%s' % (workdir, name) for name in  ('stdout', 'stderr'))
  stdout = open(logfiles[0], 'w')
  stderr = open(logfiles[1], 'w')
  job = subprocess.Popen(command, cwd = workdir, env = environment, stdout = stdout, stderr = stderr)
  job.communicate()
  stdout.close()
  stderr.close()

  # if requested, move the logs and any additional artifacts to the log directory
  if logdir:
    for name in list(keep) + list(lognames):
      if os.path.isfile(workdir + '/' + name):
        shutil.move(workdir + '/' + name, '%s/cmsRun%06d.%s' % (logdir, job.pid, name))
    logfiles = tuple('%s/cmsRun%06d.%s' % (logdir, job.pid, name) for name in  lognames)

  stderr = open(logfiles[1], 'r')

  if (job.returncode < 0):
    print "The underlying cmsRun job was killed by signal %d" % -job.returncode
    print
    print "The last lines of the error log are:"
    print "".join(stderr.readlines()[-10:])
    print
    print "See %s and %s for the full logs" % logfiles
    sys.stdout.flush()
    stderr.close()
    return None

  elif (job.returncode > 0):
    print "The underlying cmsRun job failed with return code %d" % job.returncode
    print
    print "The last lines of the error log are:"
    print "".join(stderr.readlines()[-10:])
    print
    print "See %s and %s for the full logs" % logfiles
    sys.stdout.flush()
    stderr.close()
    return None

  if verbose:
    print "The underlying cmsRun job completed successfully"
    sys.stdout.flush()

  # analyse the output
  date_format  = '%d-%b-%Y %H:%M:%S.%f'
  # expected format
  #     100, 18-Mar-2020 12:16:39.172836 CET
  begin_pattern = re.compile(r'%MSG-i ThroughputService:  *AfterModEndJob')
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
    time  = datetime.strptime(matches.group(2), date_format)
    events.append(event)
    times.append((time - epoch).total_seconds())

  stderr.close()
  return (tuple(events), tuple(times))


def parseProcess(filename):
  # parse the given configuration file and return the `process` object it define
  # the import logic is taken from edmConfigDump
  try:
    handle = open(filename, 'r')
  except:
    print "Failed to open %s: %s" % (filename, sys.exc_info()[1])
    sys.exit(1)

  # make the behaviour consistent with 'cmsRun file.py'
  sys.path.append(os.getcwd())
  try:
    pycfg = imp.load_source('pycfg', filename, handle)
    process = pycfg.process
  except:
    print "Failed to parse %s: %s" % (filename, sys.exc_info()[1])
    sys.exit(1)

  handle.close()
  return process


def multiCmsRun(
    process,                        # the cms.Process object to run
    data = None,                    # a file-like object for storing performance measurements
    header = True,                  # write a header before the measurements
    warmup = True,                  # whether to run an extra warm-up job
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
    set_cpu_affinity = False,       # whether to set CPU affinity
    set_gpu_affinity = False,       # whether yo set GPU affinity
    *args):                         # additional arguments passed to cmsRun
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

  # make a full dump of the configuration, to make changes to the number of threads, streams, etc.
  workdir = tempfile.mkdtemp(prefix = 'cmsRun')
  config = open(os.path.join(workdir, 'process.py'), 'w')
  config.write(process.dumpPython())
  config.close()

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
      cpu_list = list(itertools.chain(*(map(str, cpu.hardware_threads) for cpu in cpus.values())))
    else:
      cpu_list = list(itertools.chain(*(map(str, cpu.physical_processors) for cpu in cpus.values())))

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
      gpu_assignment = [ ','.join(map(str, gpus.keys())) for i in range(jobs) ]
    else:
      gpu_repeated   = map(str, itertools.islice(itertools.cycle(gpus.keys()), jobs * gpus_per_job))
      gpu_assignment = [ ','.join(gpu_repeated[i*gpus_per_job:(i+1)*gpus_per_job]) for i in range(jobs) ]

  if warmup:
    # warm up to cache the binaries, data and conditions
    jobdir = os.path.join(workdir, "warmup")
    os.mkdir(jobdir)
    # recreate logs' directory
    if logdir is not None:
      thislogdir = logdir + '/warmup'
      shutil.rmtree(thislogdir, True)
      os.makedirs(thislogdir)
    else:
      thislogdir = None
    print 'Warming up'
    sys.stdout.flush()
    thread = singleCmsRun(config.name, jobdir, thislogdir, [], verbose, cpu_assignment[0], gpu_assignment[0], *args)
    thread.start()
    thread.join()
    shutil.rmtree(jobdir)
    print

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

  print 'Running %s over %s events with %d jobs, each with %d threads, %d streams and %d GPUs' % (n_times, n_events, jobs, threads, streams, gpus_per_job)
  sys.stdout.flush()

  # store the values to compute the average throughput over the repetitions
  failed = [ False ] * repeats
  if repeats > 1 and not plumbing:
    throughputs = [ None ] * repeats
    overlaps    = [ None ] * repeats

  # store performance points for later analysis
  if data and header:
    data.write('%s, %s, %s, %s, %s, %s, %s, %s\n' % ('jobs', 'overlap', 'CPU threads per job', 'EDM streams per job', 'GPUs per jobs', 'number of events', 'average throughput (ev/s)', 'uncertainty (ev/s)'))

  iterations = xrange(repeats) if repeats > 0 else itertools.count()
  for repeat in iterations:
    # run the jobs reading the output to extract the event throughput
    events      = [ None ] * jobs
    times       = [ None ] * jobs
    fits        = [ None ] * jobs
    job_threads = [ None ] * jobs
    # recreate logs' directory
    if logdir is not None:
      thislogdir = logdir + '/step%04d' % repeat
      shutil.rmtree(thislogdir, True)
      os.makedirs(thislogdir)
    else:
      thislogdir = None
    # create work threads
    for job in range(jobs):
      jobdir = os.path.join(workdir, "step%02d_part%02d" % (repeat, job))
      os.mkdir(jobdir)
      job_threads[job] = singleCmsRun(config.name, jobdir, thislogdir, keep, verbose, cpu_assignment[job], gpu_assignment[job], *args)

    # start all threads
    for thread in job_threads:
      thread.start()

    # join all threads
    failed_jobs = [ False ] * jobs
    consistent_events = defaultdict(int)
    for job, thread in enumerate(job_threads):
      # implicitly wait for the thread to complete
      result = thread.result.get()
      if result is None or not(all(result)):
        failed_jobs[job] = True
        continue
      (e, t) = result
      consistent_events[tuple(e)] += 1
      events[job] = np.array(e)
      times[job]  = np.array(t)
      fits[job]   = stats.linregress(times[job], events[job])

    # if any jobs failed, skip the whole measurement
    if any(failed_jobs):
      print '%d %s failed, this measurement will be ignored' % (sum(failed_jobs), 'jobs' if sum(failed_jobs) > 1 else 'job')
      sys.stdout.flush()
      failed[repeat] = True
      continue

    # if all jobs were successful, delete the temporary directories
    for job in range(jobs):
      jobdir = os.path.join(workdir, "step%02d_part%02d" % (repeat, job))
      shutil.rmtree(jobdir)

    reference_events = np.array(sorted(consistent_events, key = consistent_events.get, reverse = True)[0])

    # check for jobs with inconsistent events
    inconsistent = [ False ] * jobs
    for job in range(jobs):
      if (len(events[job]) != len(reference_events)) or any(events[job] != reference_events):
        print 'Inconsistent measurement points for job %d, will be skipped' % job
        sys.stdout.flush()
        inconsistent[job] = True

    # delete data from inconsistent jobs
    for job in range(jobs-1, -1, -1):
      if inconsistent[job]:
        del times[job]
        del fits[job]
        del inconsistent[job]
        jobs -= 1

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
      formatting = '%8.1f\t%8.1f\t%d\t%0.1f%%' if plumbing else u'%8.1f \u00b1 %5.1f ev/s (%d events, %0.1f%% overlap)'
      print formatting % (throughput, error, used_events, overlap * 100.)
    else:
      overlap = 1.
      # machine- or human-readable formatting
      formatting = '%8.1f\t%8.1f\t%d' if plumbing else u'%8.1f \u00b1 %5.1f ev/s (%d events)'
      print formatting % (throughput, error, used_events)
    sys.stdout.flush()

    # store the values to compute the average throughput over the repetitions
    if repeats > 1 and not plumbing:
      throughputs[repeat] = throughput
      overlaps[repeat]    = overlap

    # store performance points for later analysis
    if data:
      data.write('%d, %f, %d, %d, %d, %d, %f, %f\n' % (jobs, overlap, threads, streams, gpus_per_job, used_events, throughput, error))


  # compute the average throughput over the repetitions
  if repeats > 1 and not plumbing:
    # filter out the jobs with an overlap lower than 95%
    values = [ throughputs[i] for i in range(repeats) if overlaps[i] >= 0.95 ]
    n = len(values)
    if n > 0:
      value = np.average(values)
      error = np.std(values, ddof=1)
    else:
      # no jobs with an overlap > 95%, use the "best" one
      value = throughputs[overlaps.index(max(overlaps))]
      error = float('nan')
    print ' --------------------'
    if n == repeats:
      formatting = u'%8.1f \u00b1 %5.1f ev/s'
      print formatting % (value, error)
    elif n > 0:
      formatting = u'%8.1f \u00b1 %5.1f ev/s (based on %d measurements)'
      print formatting % (value, error, n)
    else:
      formatting = u'%8.1f (single measurement with the highest overlap)'
      print formatting % (value, )

  if not plumbing:
    print
    sys.stdout.flush()

  # delete the temporary work dir
  shutil.rmtree(workdir)


def info():
  print '%d CPUs:' % len(cpus)
  for cpu in cpus.values():
    print '  %d: %s (%d cores, %d threads)' % (cpu.socket, cpu.model, len(cpu.physical_processors), len(cpu.hardware_threads))
  print

  print '%d visible NVIDIA GPUs:' % len(gpus)
  for gpu in gpus.values():
    print '  %d: %s' % (gpu.device, gpu.model)
  print
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
    'set_cpu_affinity'    : True,
    'set_gpu_affinity'    : True,
  }

  # TODO parse arguments and options from the command line

  if options['verbose']:
    info()

  if len(sys.argv) > 1:
    process = parseProcess(sys.argv[1])
    multiCmsRun(process, **options)

