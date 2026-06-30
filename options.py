import sys
import os
import argparse
from slot import Slot

# default --logdir template, used when logs are enabled without an explicit directory (no --logdir,
# or a bare --logdir). It expands to a per-run name like "logs_<config>_<J>j_<T>t_<S>s[_<gpus>][_<mps>]"
# (the last two drop out when they do not apply); multirun.expand_logdir does the expansion, and
# logdir_placeholders below documents the names. Pass --no-logdir (or --logdir '') to disable logs.
default_logdir_template = 'logs_%config_%jj_%tt_%ss_%gpus_%mps'

def configStem(config):
    # derive a short configuration name, e.g. "gpu_config.py" -> "gpu", "step3.py" -> "step3"
    stem = os.path.basename(config)
    if stem.endswith('.py'):
        stem = stem[:-3]
    if stem.endswith('_config'):
        stem = stem[:-len('_config')]
    return stem

# --logdir template placeholders: the single source of truth for both the "LOGDIR TEMPLATE" --help
# section and multirun.expand_logdir. Each entry is (name, description, render), where render(params)
# builds the placeholder's value from a run-parameter dict (keys: config, jobs, threads, streams,
# gpus_per_job, gpu_tag, nvidia_mps_tag). %gpus and %mps are optional: an empty value drops the
# placeholder (and its adjacent separator) from the directory name.
logdir_placeholders = [
    ('config', 'the configuration name (basename without ".py" or a trailing "_config", e.g. "gpu")',
               lambda p: configStem(p['config'])),
    ('j',      'the number of jobs',                                       lambda p: str(p['jobs'])),
    ('t',      'the number of threads per job',                            lambda p: str(p['threads'])),
    ('s',      'the number of streams per job',                            lambda p: str(p['streams'])),
    ('gpj',    'the number of GPUs per job (the -g/--gpus-per-job value)', lambda p: str(p['gpus_per_job'])),
    ('gpus',   'the --gpus selection tag ("allGPUs", or e.g. "gpu01"); dropped when no GPU is used',
               lambda p: p['gpu_tag']),
    ('mps',    'the --nvidia-mps tag ("noMPS", or e.g. "MPS50" / "MPS25-33"); dropped unless an NVIDIA GPU is used',
               lambda p: p['nvidia_mps_tag']),
]

def logdir_template_help():
    # build the "LOGDIR TEMPLATE" section appended to the --help epilog, from logdir_placeholders
    width = max(len(name) for name, _, _ in logdir_placeholders) + 1   # +1 for the leading '%'
    lines = [
        'LOGDIR TEMPLATE',
        '',
        'The --logdir value is a template expanded once per run into the output directory name.',
        'The default template is "%s".' % default_logdir_template,
        'The following placeholders are replaced:',
        '',
    ]
    lines += [ '   %-*s  %s' % (width, '%' + name, desc) for name, desc, _ in logdir_placeholders ]
    lines += [
        '',
        'A placeholder may be followed by literal text, e.g. "%jj" expands to "<jobs>j".',
        'The %gpus and %mps placeholders are optional: when they do not apply (no GPU in use, or no',
        'NVIDIA GPU for %mps) they expand to nothing and their adjacent separator is removed, so the',
        'directory name has no dangling separator. Pass --no-logdir (or --logdir "") to disable logs.',
    ]
    return '\n'.join(lines)

# named option presets, applied as defaults by "--preset NAME" (see OptionParser.parse).
# To add a new preset, add an entry here: 'description' is shown in --help, and 'options' maps
# option dests to the default values to pre-load. The special entry input_xml='auto' is only
# applied when the user did not pass --input-collections (the two are mutually exclusive).
presets = {
    'hltRun3': {
        'description': 'standard Run3 HLT timing setup',
        'options': dict(
            events = 10300,
            event_skip = 300,
            event_resolution = 100,
            wait = 0.,
            jobs = 8,
            threads = 32,
            streams = 24,
            output_log = True,
            logdir = default_logdir_template,
            input_collections = 'rawDataCollector',
        ),
    },
    'hltPhase2': {
        'description': 'standard Phase-2 HLT timing setup',
        'options': dict(
            events = 1000,
            event_skip = 100,
            event_resolution = 25,
            wait = 30.,
            jobs = 16,
            threads = 16,
            streams = 16,
            output_log = True,
            logdir = default_logdir_template,
            input_xml = 'auto',
        ),
    },
}

def _config_like_hint(value):
    # if a value that should be numeric looks like a configuration file, the variadic option has
    # most likely swallowed the positional config; point the user at the "--" separator
    if value.endswith('.py') or '/' in value:
        return ' (this looks like a configuration file: place the config files after "--", ' \
               'e.g. "--setup j=J,t=T,s=S -- config.py")'
    return ''

def parse_setup(value):
    # parse an explicit "j=J,t=T,s=S" preset into a (jobs, threads, streams) tuple. The fields may be
    # given in any order, using the keys j/t/s (or the long aliases jobs/threads/streams), and any
    # subset may be provided: an omitted field is left unset (None) and auto-derived downstream,
    # exactly like omitting the matching -j/-t/-s option.
    parts = [ p.strip() for p in value.split(',') if p.strip() ]
    if not parts:
        raise argparse.ArgumentTypeError('an empty setup is not allowed; use "j=J,t=T,s=S"')
    aliases = { 'j': 'jobs', 'jobs': 'jobs', 't': 'threads', 'threads': 'threads',
                's': 'streams', 'streams': 'streams' }
    result = { 'jobs': None, 'threads': None, 'streams': None }
    seen = set()
    for part in parts:
        if '=' not in part:
            raise argparse.ArgumentTypeError(
                'a setup must use the "j=J,t=T,s=S" form, not "%s"' % value
                + _config_like_hint(value))
        key, _, val = part.partition('=')
        field = aliases.get(key.strip().lower())
        if field is None:
            raise argparse.ArgumentTypeError(
                'unknown setup field "%s" in "%s": use j/t/s (or jobs/threads/streams)' % (key.strip(), value))
        if field in seen:
            raise argparse.ArgumentTypeError('the setup field "%s" is set twice in "%s"' % (field, value))
        seen.add(field)
        try:
            result[field] = int(val)
        except ValueError:
            raise argparse.ArgumentTypeError('the setup values must be integers, as in "j=J,t=T,s=S"' + _config_like_hint(value))
    return (result['jobs'], result['threads'], result['streams'])

def parse_nvidia_mps(value):
    # parse the --nvidia-mps percentage as an integer in the range 1-100; the bare "--nvidia-mps"
    # (const=-1, auto-split) bypasses this parser.
    try:
        pct = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('--nvidia-mps expects an integer percentage' + _config_like_hint(value))
    if pct < 1 or pct > 100:
        raise argparse.ArgumentTypeError('--nvidia-mps percentage must be in the range 1-100, got %s' % value)
    return pct

def printCommonArgs(opts):
    print('Common options for multiCmsRun:')
    for key, value in opts.items():
        if key == 'slots':
            print(f'  --{key}')
            for idx, val in enumerate(value):
                print(f'      slot {idx}: {val}')
        else:
            print(f'  --{key}: {value}')
    print()
    sys.stdout.flush()

class OptionParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class = argparse.RawDescriptionHelpFormatter,
            description = """
Run multiple cmsRun jobs in parallel with a configurable number of threads, streams and gpus per job.
""",
            epilog = """
JOB SLOTS

The execution environment for each job (NUMA nodes for the cpus, NUMA nodes for the memory, individual cpus, NVIDIA and AMD GPUs, NVIDIA MPS active thread percentage) can be given explicitly with the '--slot SLOT' option.
This options disables the automatic job assignment to CPUs and GPUs, and makes the program ignore the options '--numa-affinity', '--cpu-affinity' and '--gpu-affinity'.
Each '--slot' option describes the execution environment for a single job. If theare more jobs (see the --jobs option) than slots, they are reused in a round-robin fashion until all jobs are allocated.
The format of SLOT is a colon-separated list of fields, where each field has the format 'keyword=value'.
The possible fields, their formats and descriptions are:
   [events|e]=EVENTS        where EVENTS is a positive integer, or -1 to run over all events in the input dataset, and overrides the --events options for this slot;
   [numa|n]=NODES           where NODES indicates the NUMA nodes of the CPUs to be used by the job;
   [mem|m]=NODES            where NODES indicates the NUMA nodes of the memory to be used by the job;
   [cpu|c]=CPUS             where CPUS indicates the individual CPUs to be used by the job;
   [gpu-nvidia|nv]=GPUS     where GPUS indicates the NVIDIA GPUs to be used by the job;
   nvidia-mps=PERCENT       where PERCENT indicates the NVIDIA MPS active thread percentage to be used by the job;
   [gpu-amd|amd]=GPUS       where GPUS indicates the AMD GPUs to be used by the job.

All fields are optional, but at least one field must be given. Each field should be specified at most once.

NODES should be a comma-separated list of integers or integer ranges, representing the NUMA nodes in the system.
If not specified, or if an empty list is used, no restrictions on the NUMA nodes are applied.

CPUS should be a comma-separated list of integers or integer ranges, representing the individual CPUs in the system.
If not specified, or if an empty list is used, no restrictions on the CPUs are applied.

GPUS should be a comma-separated list of integers, integer ranges, or GPU UUIDs representing the NVIDIA or AMD GPUs in the system.
If not specified, no restrictions on the GPUs are applied.
If an empty list is used, all GPUs are disabled and no GPUs are used by the job.

PERCENT should be a non-negative integer, the NVIDIA MPS active thread percentage to be applied to the job.
It overrides the '--nvidia-mps' option for this slot, and implies it: the NVIDIA MPS control daemon is started even if '--nvidia-mps' was not given.
Slots that do not specify it follow '--nvidia-mps' as usual, or use no NVIDIA MPS if it was not given.

""" + logdir_template_help())

        self.parser.add_argument('configs',
            type = str,
            nargs = '+',
            metavar = 'config.py',
            help = 'one or more cmsRun configuration files to execute. When more than one is given, each is benchmarked in turn. Use "--" to separate them from a preceding "--keep" list.')

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

        self.parser.add_argument('--preset',
            dest = 'preset',
            metavar = 'NAME',
            choices = sorted(presets),
            default = None,
            help = 'apply a named preset of default options before parsing the rest of the command '
                   'line, so any preset value can still be overridden by passing that option explicitly '
                   '(e.g. "--preset hltRun3 --no-input-benchmark" or "--preset hltRun3 --setup j=32,t=8,s=8"). '
                   'Available presets: ' + '; '.join('%s = %s' % (n, presets[n]['description']) for n in sorted(presets)) +
                   ' [default: none]')

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
            help = 'skip the firts EVENTS in each job, rounded to the next multiple of the event resulution [default: 300]')

        self.parser.add_argument('-j', '--jobs',
            dest = 'jobs',
            metavar = 'JOBS',
            action = 'store',
            type = int,
            default = 2,
            help = 'number of concurrent cmsRun jobs per measurement [default: 2]')

        self.parser.add_argument('-r', '--repeats',
            dest = 'repeats',
            metavar = 'N',
            action = 'store',
            type = int,
            default = 3,
            help = 'repeat each measurement N times, or indefinitely if 0 is given [default: 3]')

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

        self.parser.add_argument('--gpus',
            dest = 'gpus',
            metavar = 'LIST',
            action = 'store',
            type = str,
            default = 'all',
            help = 'restrict the benchmark to these GPUs, or "all" for no restriction [default: all]. '
                   'Either a plain comma-separated list of indices, ranges or NVIDIA UUIDs ("0,1", "0-2", '
                   '"GPU-<uuid>"), applied to whichever vendor is present (CUDA_VISIBLE_DEVICES for NVIDIA, '
                   'HIP_VISIBLE_DEVICES for AMD), a per-vendor form for mixed nodes ("nvidia=0,1:amd=0"), or '
                   '"none" to disable all GPUs. The automatic GPU affinity still distributes one GPU per job, '
                   'but only across the selected GPUs. Also used to tag the output directory name.')

        self.parser.add_argument('--setup',
            dest = 'setup',
            metavar = 'j=J,t=T,s=S',
            nargs = '+',
            type = parse_setup,
            default = [],
            help = 'one or more "j=J,t=T,s=S" jobs/threads/streams presets to run for every '
                   'configuration, e.g. --setup j=16,t=16,s=16 j=32,t=8,s=8. Fields may be given in '
                   'any order and any omitted field is auto-derived. Overrides -j/-t/-s and runs the '
                   'full configurations x setups matrix [default: none]')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--nvidia-mps',
            dest = 'nvidia_mps',
            metavar = 'PERCENT',
            nargs = '?',
            type = parse_nvidia_mps,
            const = -1,
            default = None,
            help = 'use NVIDIA MPS: start the NVIDIA MPS control daemon if it is not already running, and '
                   'set the active thread percentage of each job to PERCENT, an integer in the range 1-100 '
                   '(inclusive). If PERCENT is omitted it defaults to ceil(100 / (jobs per GPU)). If the '
                   'daemon was started here it is stopped at the end, otherwise it is left running. '
                   '[default: NVIDIA MPS not used]')
        group.add_argument('--no-nvidia-mps',
            dest = 'no_nvidia_mps',
            action = 'store_true',
            default = False,
            help = 'require NVIDIA MPS to be off: do not start the NVIDIA MPS control daemon, and exit with '
                   'a clear error if one is already running (so a pre-existing daemon cannot silently affect '
                   'the benchmark). [default: NVIDIA MPS not used, but a pre-existing daemon is left alone]')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--input-benchmark', '--run-io-benchmark',
            dest = 'input_benchmark',
            metavar = 'N',
            action= 'store',
            type = int,
            nargs = '?',
            const = -1,
            default = -1,
            help = 'measure the input-only throughput N times before performing any other measurements [default: as many repetitions as given by --repeats]. The legacy option name "--run-io-benchmark N" is deprecated.')
        group.add_argument('--no-input-benchmark', '--no-run-io-benchmark',
            dest = 'input_benchmark',
            action= 'store_false',
            help = 'do not run the input-only throughput measurements (equivalent to "--input-benchmark 0"). The legacy option name "--no-run-io-benchmark" is deprecated.')
        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--input-collections',
            dest = 'input_collections',
            metavar = 'BRANCH[,BRANCH,...]',
            action = 'store',
            type = str,
            default = 'rawDataCollector',
            help = 'comma-separated list of input collections to read for the input-only throughput measurements [default: rawDataCollector]')
        group.add_argument('--input-xml',
            dest = 'input_xml',
            metavar = 'FILE',
            action = 'store',
            type = str,
            default = None,
            help = 'read the list of input collections to read for the input-only throughput measurements from a framework job report XML file. Use the special value "auto" to generate the job report automatically by running a short job over the configuration.')
        self.parser.add_argument('--input-xml-events',
            dest = 'input_xml_events',
            metavar = 'N',
            action = 'store',
            type = int,
            default = 10,
            help = 'number of events to process when auto-generating the input job report ("--input-xml auto") [default: 10]')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('-R', '--reference-benchmark',
            dest = 'reference_benchmark',
            action= 'store_true',
            default = False,
            help = 'benchmark the same configuration using the reference CMSSW release [default: False]')
        group.add_argument('--no-reference-benchmark',
            dest = 'reference_benchmark',
            action= 'store_false',
            help = 'do not benchmark the reference release')

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
            nargs = '?',
            const = default_logdir_template,
            default = default_logdir_template,
            help = 'where to store the log files, given as a template expanded per run (see the '
                   '"LOGDIR TEMPLATE" section below for the placeholders). By default (no --logdir, or '
                   'a bare --logdir with no value) an automatically named directory is used per run; '
                   'pass a value to use it as the directory. Pass --no-logdir or --logdir "" to '
                   'disable logs. [default: automatic]')
        group.add_argument('--no-logdir',
            dest = 'logdir',
            action = 'store_const',
            const = '',
            help = 'disable logs, even when a default or preset would enable them (same as --logdir "")')

        group = self.parser.add_mutually_exclusive_group()
        group.add_argument('--output-log',
            dest = 'output_log',
            action = 'store_true',
            default = False,
            help = 'also save each configuration/setup console output to "<logdir>/output.log" [default: False]')
        group.add_argument('--no-output-log',
            dest = 'output_log',
            action = 'store_false',
            help = 'do not save the console output to a log file [default]')

        self.parser.add_argument('-k', '--keep',
            dest='keep',
            nargs='+',
            default=None,
            metavar='FILE',
            help= 'list of additional output files to be kept in logdir, along with the logs [default: the JSON file written by the FastTimerService of the configuration, if any]. For example, the argument "-k resources.json DQM.root --" keeps resources.json and DQM.root. Note: the dashes "--" avoid the parser to consume unintended arguments afterwards.'
        )

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

        group = self.parser.add_argument_group('debug options')
        group.add_argument('--debug-affinity',
            dest = 'debug_affinity',
            action = 'store_true',
            default = False,
            help = 'Print the jobs CPU and GPU affinity and constraints [default: False].')
        group.add_argument('--debug-cpu-usage',
            dest = 'debug_cpu_usage',
            action = 'store_true',
            default = False,
            help = 'Profile the CPU usage of this script itself [default: False]. Requires the "yappi" module to be installed.')
        group.add_argument('--debug-logs',
            dest = 'debug_logs',
            action = 'store_true',
            default = False,
            help = 'Print full logs on job failure [default: False].')

        group = self.parser.add_argument_group('monitoring options')
        monitor_levels = ['none', 'basic', 'full']
        group.add_argument('--monitor-host',
            dest = 'host_memory_monitoring',
            choices = monitor_levels,
            default = 'basic',
            help = 'per-process host memory monitoring detail: none, basic (VSS+RSS), or full (+PSS, ~10%% CPU per job) [default: basic]')
        group.add_argument('--monitor-gpu',
            dest = 'gpu_monitoring',
            choices = monitor_levels,
            default = 'basic',
            help = 'unified CPU+GPU resource monitoring detail: none, basic (GPU utilization+memory), or full (+power+temperature). Samples the in-use NVIDIA (AMD) GPUs via nvidia-smi (amd-smi) on the same cadence as the aggregate host memory; auto-disabled if no supported GPU is present [default: basic]')


    def parse(self, args):
        # if "--preset NAME" is requested, raise the relevant defaults to that preset. These are
        # applied as argparse defaults, so any option the user passes explicitly still takes
        # precedence over the preset.
        pre, _ = self.parser.parse_known_args(args)
        if pre.preset:
            preset = dict(presets[pre.preset]['options'])
            # input_xml='auto' is only applied if the user did not pick explicit input collections
            # (which are mutually exclusive with --input-xml)
            if 'input_xml' in preset and any(a == '--input-collections' or a.startswith('--input-collections=') for a in args):
                del preset['input_xml']
            self.parser.set_defaults(**preset)

        # parse the command line options
        options, unknown = self.parser.parse_known_args(args)

        if len(unknown) > 0:
            raise RuntimeError('unsupported command-line arguments: ' + str(unknown))

        # if explicit job slots have been defined, disable the '--numa-affinity', '--cpu-affinity' and '--gpu-affinity' options
        if options.slots:
            options.numa_affinity = False
            options.cpu_affinity = False
            options.gpu_affinity = False

        # --no-nvidia-mps overrides any "nvidia-mps=" field in --slot: drop those percentages, so that
        # neither the per-job CUDA_MPS_ACTIVE_THREAD_PERCENTAGE nor the %mps tag claims one
        if options.no_nvidia_mps and any(slot.nvidia_mps is not None for slot in options.slots):
            print('Warning: --no-nvidia-mps was given, but a --slot "nvidia-mps=" field requests NVIDIA MPS; '
                  'not starting the NVIDIA MPS control daemon (the slot setting is ignored).')
            sys.stdout.flush()
            for slot in options.slots:
                slot.nvidia_mps = None

        # check if profiling is supported
        if options.debug_cpu_usage:
          try:
            import yappi
            yappi.set_clock_type("cpu")
          except:
            print("The yappi package is not present, CPU usage profiling will be disabled.")
            options.debug_cpu_usage = False

        # adjust the number of repetitions for the input-only benchmark
        if options.input_benchmark == -1:
            options.input_benchmark = options.repeats if options.repeats != 0 else 3

        return options


if __name__ == "__main__":
  import sys
  parser = OptionParser()
  opts = parser.parse(sys.argv[1:])
  for key,val in opts.__dict__.items():
    print(f'{key} = {val}')
