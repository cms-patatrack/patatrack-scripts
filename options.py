import argparse
from slot import Slot

class OptionParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class = argparse.RawDescriptionHelpFormatter,
            description = """
Run multiple cmsRun jobs in parallel with a configurable number of threads, streams and gpus per job.
""",
            epilog = """
JOB SLOTS

The execution environment for each job (NUMA nodes for the cpus, NUMA nodes for the memory, individual cpus, NVIDIA and AMD GPUs) can be given explicitly with the '--slot SLOT' option.
This options disables the automatic job assignment to CPUs and GPUs, and makes the program ignore the options '--numa-affinity', '--cpu-affinity' and '--gpu-affinity'.
Each '--slot' option describes the execution environment for a single job. If theare more jobs (see the --jobs option) than slots, they are reused in a round-robin fashion until all jobs are allocated.
The format of SLOT is a colon-separated list of fields, where each field has the format 'keyword=value'.
The possible fields, their formats and descriptions are:
   [events|e]=EVENTS        where EVENTS is a positive integer, or -1 to run over all events in the input dataset, and overrides the --events options for this slot;
   [numa|n]=NODES           where NODES indicates the NUMA nodes of the CPUs to be used by the job;
   [mem|m]=NODES            where NODES indicates the NUMA nodes of the memory to be used by the job;
   [cpu|c]=CPUS             where CPUS indicates the individual CPUs to be used by the job;
   [gpu-nvidia|nv]=GPUS     where GPUS indicates the NVIDIA GPUs to be used by the job;
   [gpu-amd|amd]=GPUS       where GPUS indicates the AMD GPUs to be used by the job.

All fields are optional, but at least one field must be given. Each field should be specified at most once.

NODES should be a comma-separated list of integers or integer ranges, representing the NUMA nodes in the system.
If not specified, or if an empty list is used, no restrictions on the NUMA nodes are applied.

CPUS should be a comma-separated list of integers or integer ranges, representing the individual CPUs in the system.
If not specified, or if an empty list is used, no restrictions on the CPUs are applied.

GPUS should be a comma-separated list of integers, integer ranges, or GPU UUIDs representing the NVIDIA or AMD GPUs in the system.
If not specified, no restrictions on the GPUs are applied.
If an empty list is used, all GPUs are disabled and no GPUs are used by the job.
""")

        self.parser.add_argument('config',
            type = str,
            metavar = 'config.py',
            help = 'cmsRun configuration file to execute')

        self.parser.add_argument('-v', '--verbose',
            dest = 'verbose',
            action = 'store_true',
            default = False,
            help = 'enable verbose mode [default: False]')

        self.parser.add_argument('-E', '--executable',
            dest = 'executable',
            action = 'store',
            type = str,
            default = 'cmsRun',
            help = 'specify what executable to run [default: cmsRun]')

        self.parser.add_argument('-e', '--events',
            dest = 'events',
            action = 'store',
            type = int,
            default = 10300,
            help = 'number of events per cmsRun job [default: 10300]')
        self.parser.add_argument('--event-resolution',
            dest = 'event_resolution',
            metavar = 'EVENTS',
            action = 'store',
            type = int,
            default = 100,
            help = 'sample the number of processed events with the given resolution')
        self.parser.add_argument('--event-skip',
            dest = 'event_skip',
            metavar = 'EVENTS',
            action = 'store',
            type = int,
            default = 300,
            help = 'skip the firts EVENTS in each job, rounded to the next multiple of the event resulution')

        self.parser.add_argument('-j', '--jobs',
            dest = 'jobs',
            action = 'store',
            type = int,
            default = 2,
            help = 'number of concurrent cmsRun jobs per measurement [default: 2]')

        self.parser.add_argument('-r', '--repeats',
            dest = 'repeats',
            action = 'store',
            type = int,
            default = 3,
            help = 'repeat each measurement N times [default: 3]')

        self.parser.add_argument('--wait',
            dest = 'wait',
            action = 'store',
            type = float,
            default = 0.,
            help = 'wait this many seconds between measurements [default: 0]')

        self.parser.add_argument('-t', '--threads',
            dest = 'threads',
            action = 'store',
            type = int,
            default = None,
            help = 'number of threads used in each cmsRun job [default: None -> set automatically to use the whole machine]')

        self.parser.add_argument('-s', '--streams',
            dest = 'streams',
            action = 'store',
            type = int,
            default = None,
            help = 'number of streams used in each cmsRun job [default: None -> set automatically to use the whole machine]')

        self.parser.add_argument('-g', '--gpus-per-job',
            dest = 'gpus_per_job',
            action = 'store',
            type = int,
            default = 1,
            help = 'number of GPUs used in each cmsRun job [default: 1]')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--run-io-benchmark',
            dest = 'run_io_benchmark',
            action= 'store_true',
            default = True,
            help = 'measure the I/O benchmarks before the other measurements [default: True]')
        group.add_argument('--no-run-io-benchmark',
            dest = 'run_io_benchmark',
            action= 'store_false',
            help = 'skip the I/O measurement')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--warmup',
            dest = 'warmup',
            action = 'store_true',
            default = True,
            help = 'do a warmup run before the measurements [default: True]')
        group.add_argument('--no-warmup',
            dest = 'warmup',
            action = 'store_false',
            help = 'skip the warmup run')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('-p', '--plumbing',
            dest = 'plumbing',
            action = 'store_true',
            default = False,
            help = 'enable plumbing output [default: False]')
        group.add_argument('--no-plumbing',
            dest = 'plumbing',
            action = 'store_false',
            help = 'disable plumbing output')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--allow-hyperthreading',
            dest = 'allow_hyperthreading',
            action = 'store_true',
            default = True,
            help = 'allow HyperThreading/Simultaneous multithreading (used only if cpu_affinity = True) [default: True]')
        group.add_argument('--no-hyperthreading',
            dest = 'allow_hyperthreading',
            action = 'store_false',
            help = 'do not allow HyperThreading/Simultaneous multithreading (used only if cpu_affinity = True)')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('-n', '--numa-affinity',
            dest = 'numa_affinity',
            action = 'store_true',
            default = False,
            help = 'enable NUMA affinity [default: False]')
        group.add_argument('--no-numa-affinity',
            dest = 'numa_affinity',
            action = 'store_false',
            help = 'disable NUMA affinity')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--cpu-affinity',
            dest = 'cpu_affinity',
            action = 'store_true',
            default = True,
            help = 'enable CPU affinity [default: True]')
        group.add_argument('--no-cpu-affinity',
            dest = 'cpu_affinity',
            action = 'store_false',
            help = 'disable CPU affinity')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--gpu-affinity',
            dest = 'gpu_affinity',
            action = 'store_true',
            default = True,
            help = 'enable GPU affinity [default: True]')
        group.add_argument('--no-gpu-affinity',
            dest = 'gpu_affinity',
            action = 'store_false',
            help = 'disable GPU affinity')

        self.parser.add_argument('-S', '--slot',
            dest = 'slots',
            metavar = 'SLOT',
            action = 'append',
            type = Slot.parse,
            default = [],
            help = 'ignores --numa-affinity, --cpu-affinity, --gpu-affinity, and define explicitly the execution environment for a job slot (see below)')

        self.parser.add_argument('--csv',
            dest = 'csv',
            metavar = 'FILE',
            action = 'store',
            default = None,
            help = 'write a summary of the measurements to a CSV file [default: None]')
        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--csv-header',
            dest = 'csvheader',
            action = 'store_true',
            default = True,
            help = 'write a header at the top of the CSV file [default: True]')
        group.add_argument('--no-csv-header',
            dest = 'csvheader',
            action = 'store_false',
            help = 'do not write a header at the top of the CSV file [default: True]')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('-l', '--logdir',
            dest = 'logdir',
            action = 'store',
            default = 'logs',
            help = 'path to output directory for log files (if empty, logs are not stored) [default: "logs"]')
        group.add_argument('--no-logdir',
            dest = 'logdir',
            action = 'store_const',
            const = '',
            help = 'do not store log files (equivalent to "--logdir \'\'")')

        self.parser.add_argument('-k', '--keep',
            dest = 'keep',
            nargs = '+',
            default = ['resources.json'],
            help = 'list of additional output files to be kept in logdir, along with the logs [default: "resources.json"]')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--auto-merge',
            dest = 'automerge',
            action = 'store_true',
            default = True,
            help = 'automatically merge supported file types, if the corresponding merger is available [default: True]')
        group.add_argument('--no-auto-merge',
            dest = 'automerge',
            action = 'store_false',
            help = 'do automatically merge supported file types, even if the corresponding merger is available')

        self.parser.add_argument('--tmpdir',
            dest = 'tmpdir',
            action = 'store',
            default = None,
            help = 'path to temporary directory used at runtime [default: None, to use a system-dependent default temporary directory]')
        self.parser.add_argument('--auto-delete',
            dest = 'autodelete',
            metavar = 'PATTERN',
            nargs = '+',
            default = [],
            help = 'automatically delete files matching the given patterns while running the jobs [default: do not delete any files]')
        self.parser.add_argument('--auto-delete-delay',
            dest = 'autodelete_delay',
            metavar = 'DELAY',
            action = 'store',
            type = float,
            default = 60.,
            help = 'check for files to autodelete with this interval [default: 60s]')


    def parse(self, args):
        # parse the command line options
        options, unknown = self.parser.parse_known_args(args)

        if len(unknown) > 0:
            raise RuntimeError('unsupported command-line arguments: ' + str(unknown))

        # if explicit job slots have been defined, disable the '--numa-affinity', '--cpu-affinity' and '--gpu-affinity' options
        if options.slots:
            options.numa_affinity = False
            options.cpu_affinity = False
            options.gpu_affinity = False

        return options


if __name__ == "__main__":
  import sys
  parser = OptionParser()
  opts = parser.parse(sys.argv[1:])
  for key,val in opts.__dict__.items():
    print(f'{key} = {val}')
