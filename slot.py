#! /usr/bin/env python3

import collections.abc
import itertools
import re
import sys

# Describe the resources that should be used by a job:
#   - NUMA compute nodes, as understood by `numactl -N ...`
#   - NUMA memory nodes, as understood by `numactl -m ...`
#   - CPUs, as understood by `numactl -C ...`
#   - NVIDIA GPUs, as understood by `CUDA_VISIBLE_DEVICES=...`
#   - AMD GPUs, as understood by `HIP_VISIBLE_DEVICES=...`

# Note: on a mixed system, setting CUDA_VISIBLE_DEVICES affects also the selection of AMD GPUs.

class Slot:
    # a single integer
    integer_format = '-?[0-9]+'
    integer_template = re.compile('^' + integer_format + '$')

    # a comma separated list of integers or integer ranges
    nodes_range = '[0-9]+(-[0-9]+)?'
    nodes_format = f'({nodes_range})(,{nodes_range})*'
    nodes_template = re.compile('^' + nodes_format + '$')

    # a comma-separated list of integers, integer ranges, or GPU UUIDs
    gpus_range = '([0-9]+(-[0-9]+)?)|(GPU-[-0-9a-f]+)'
    gpus_format = f'({gpus_range})(,{gpus_range})*'
    gpus_template = re.compile('^' + gpus_format + '$')

    # [events|e]=EVENTS       where EVENTS is a positive integer, or -1 to run over all events in the input dataset, and overrides the --events options for this slot
    slot_format_events = '(events|e)=' + integer_format
    # [numa|n]=NODES          where NODES is a comma-separated list of integer or integer ranges, representing the NUMA nodes of the CPUs to be used by the job
    slot_format_numa = '(numa|n)=' + nodes_format
    # [mem|m]=NODES           where NODES is a comma-separated list of integer or integer ranges, representing the NUMA nodes of the memory to be used by the job
    slot_format_mem = '(mem|m)=' + nodes_format
    # [cpu|c]=CPUS            where CPUS is a comma-separated list of integer or integer ranges, representing the CPUs to be used by the job
    slot_format_cpu = '(cpus|cpu|c)=' + nodes_format
    # [gpu-nvidia|nv]=GPUS    where GPUS is a comma-separated list of integer, integer ranges, or GPU UUIDs representing the NVIDIA GPUs to be used by the job
    slot_format_gpu_nvidia = '(gpu-nvidia|nv)=(' + gpus_format + ')?'
    # [gpu-amd|amd]=GPUS      where GPUS is a comma-separated list of integer, integer ranges, or GPU UUIDs representing the AMD GPUs to be used by the job
    slot_format_gpu_amd = '(gpu-amd|amd)=(' + gpus_format + ')?'
    # any of the fields above
    slot_format_field = f'({slot_format_events}|{slot_format_numa}|{slot_format_mem}|{slot_format_cpu}|{slot_format_gpu_nvidia}|{slot_format_gpu_amd})'
    # a colon-separated list of fields
    slot_format = f'{slot_format_field}(:{slot_format_field})*'
    slot_template = re.compile('^' + slot_format + '$')


    # expand the range A-B into A,..,..,B
    @staticmethod
    def parse_int_range(arg):
        msg = 'The argument is expected to be a string containing an integer or an integer range.'

        if not isinstance(arg, str):
            raise TypeError(msg)

        if arg.count('-') == 0:
            return [int(arg)]
        elif arg.count('-') > 1:
            raise ValueError(msg)

        a,b = map(int,arg.split('-'))
        return list(range(a, b+1))


    # parse an argument as an integer
    @staticmethod
    def parse_integer(arg):
        msg = 'The argument is expected to be None, or a string containing an integer.'

        if not isinstance(arg, (type(None), str)):
            raise TypeError(msg)

        # None or an empty string indicate no restrictions
        if not arg:
            return None

        # a single integer
        if not Slot.integer_template.match(arg):
            raise ValueError(msg)

        return int(arg)


    # parse an argument as an integer, or list of integers
    @staticmethod
    def parse_value(arg):
        msg = 'The argument is expected to be None, or a string containing a comma-separated list of integers or integer ranges.'

        if not isinstance(arg, (type(None), str)):
            raise TypeError(msg)

        # None or an empty string indicate no restrictions
        if not arg:
            return None

        # a comma separated list of integers or integer ranges
        if not Slot.nodes_template.match(arg):
            raise ValueError(msg)

        return list(itertools.chain.from_iterable(Slot.parse_int_range(a) for a in arg.split(',')))


    # parse an argument as a GPU descriptor: either an integer or "GPU-" followed by a UUID, or a comma-separated list of them
    @staticmethod
    def parse_gpu_descriptor(arg):
        msg = 'The argument is expected to be None, or a string containing a comma-separated list of integers, intege ranges or GPU UUIDs.'

        if not isinstance(arg, (type(None), str)):
            raise TypeError(msg)

        # None represents no restrictions on what GPUs to use
        if arg is None:
            return None

        # an empty string indicates not running on any GPUs
        if not arg:
            return []

        # a comma-separated list of integers, integer ranges, or GPU UUIDs
        if not Slot.gpus_template.match(arg):
            raise ValueError(msg)

        v = []
        for a in arg.split(','):
            if a.startswith('GPU-'):
                v.append(a)
            else:
                v.extend(map(str, Slot.parse_int_range(a)))
        return v


    def __init__(self, events = None, numa_cpu = None, numa_mem = None, cpus = None, nvidia_gpus = None, amd_gpus = None):
        self.events = Slot.parse_integer(events)
        self.numa_cpu = Slot.parse_value(numa_cpu)
        self.numa_mem = Slot.parse_value(numa_mem)
        self.cpus = Slot.parse_value(cpus)
        self.nvidia_gpus = Slot.parse_gpu_descriptor(nvidia_gpus)
        self.amd_gpus = Slot.parse_gpu_descriptor(amd_gpus)

    def __str__(self):
        return ', '.join([f'{k}={v}' for k,v in vars(self).items() if v is not None])
    
    # return "value" if "field=value" is given in arg, or None if field is not in arg
    @staticmethod
    def parse_field(arg, field):
        value = None
        for a in arg.split(':'):
            f, v = a.split('=')
            if f in field:
                if value is None:
                    value = v
                else:
                    raise ValueError(f'Duplicate field "{field[0]}"')
        return value

    @staticmethod
    def parse(arg):
        # syntax:
        # arg should be a colon-separated list of fields, each field with the format keywork=value.
        # The possible fields are
        #   [events|e]=EVENTS       where EVENTS is a positive integer, or -1 to run over all events in the input dataset, and overrides the --events options for this slot
        #   [numa|n]=NODES          where NODES is a comma-separated list of integer or integer ranges, representing the NUMA nodes of the CPUs to be used by the job
        #   [mem|m]=NODES           where NODES is a comma-separated list of integer or integer ranges, representing the NUMA nodes of the memory to be used by the job
        #   [cpus|cpu|c]=CPUS       where CPUS is a comma-separated list of integer or integer ranges, representing the CPUs to be used by the job
        #   [gpu-nvidia|nv]=GPUS    where GPUS is a comma-separated list of integer, integer ranges, or GPU UUIDs representing the NVIDIA GPUs to be used by the job
        #   [gpu-amd|amd]=GPUS      where GPUS is a comma-separated list of integer, integer ranges, or GPU UUIDs representing the AMD GPUs to be used by the job
        if not isinstance(arg, str):
            raise TypeError('The argument should be a string')

        if not Slot.slot_template.match(arg):
            raise ValueError('The argument does not match the slot syntax')

        events = Slot.parse_field(arg, ('events', 'e'))
        numa_cpu = Slot.parse_field(arg, ('numa', 'n'))
        numa_mem = Slot.parse_field(arg, ('mem', 'm'))
        cpus = Slot.parse_field(arg, ('cpus', 'cpu', 'c'))
        nvidia_gpus = Slot.parse_field(arg, ('gpu-nvidia', 'nv'))
        amd_gpus = Slot.parse_field(arg, ('gpu-amd', 'amd'))

        return Slot(events, numa_cpu, numa_mem, cpus, nvidia_gpus, amd_gpus)


    # return the command prefix and environment for the execution environment described by the slot
    def get_execution_parameters(self):
        command = []
        environ = {}
        if self.numa_cpu is not None or self.numa_mem is not None or self.cpus is not None:
            command = ['numactl']
            if self.numa_cpu is not None:
                command += [ '-N', ','.join(map(str, self.numa_cpu)) ]
            if self.numa_mem is not None:
                command += [ '-m', ','.join(map(str, self.numa_mem)) ]
            if self.cpus is not None:
                command += [ '-C', ','.join(map(str, self.cpus)) ]

        if self.nvidia_gpus is not None:
            environ['CUDA_VISIBLE_DEVICES'] = ','.join(self.nvidia_gpus)

        if self.amd_gpus is not None:
            environ['HIP_VISIBLE_DEVICES'] = ','.join(self.amd_gpus)

        return command,environ


    # return the equivalent command line to be executed at a shell prompt
    def get_command_line_prefix(self):
        command,environ = self.get_execution_parameters()
        prefix = ' '.join(list(map('='.join, environ.items())) + command)
        if prefix:
            prefix += ' '
        return prefix


    # return the description of the execution environment described by the slot
    def describe(self):
        desc = []
        if self.numa_cpu is not None:
            if len(self.numa_cpu) == 1:
                desc.append('with the NUMA compute node ' + str(self.numa_cpu[0]))
            else:
                desc.append('with the NUMA compute nodes ' + ','.join(map(str, self.numa_cpu)))

        if self.numa_mem is not None:
            if len(self.numa_mem) == 1:
                desc.append('with the NUMA memory node ' + str(self.numa_mem[0]))
            else:
                desc.append('with the NUMA memory nodes ' + ','.join(map(str, self.numa_mem)))

        if self.cpus is not None:
            if len(self.cpus) == 1:
                desc.append('on the CPU ' + str(self.cpus[0]))
            else:
                desc.append('on the CPUs ' + ','.join(map(str, self.cpus)))

        if self.nvidia_gpus is None:
            desc.append('with any available NVIDIA GPUs')
        elif not self.nvidia_gpus:
            desc.append('without any NVIDIA GPUs')
        elif len(self.nvidia_gpus) == 1:
            desc.append('with the NVIDIA GPU ' + self.nvidia_gpus[0])
        else:
            desc.append('with the NVIDIA GPUs ' + ','.join(self.nvidia_gpus))

        if self.amd_gpus is None:
            desc.append('with any available AMD GPUs')
        elif not self.amd_gpus:
            desc.append('without any AMD GPUs')
        elif len(self.amd_gpus) == 1:
            desc.append('with the AMD GPU ' + self.amd_gpus[0])
        else:
            desc.append('with the AMD GPUs ' + ','.join(self.amd_gpus))

        if self.events is None:
            pass
        elif self.events < 0:
            desc.append('over all events')
        else:
            desc.append(f'over {self.events} events')

        return ', '.join(desc)


# tests
if __name__ == "__main__":
    cmd = 'cmsRun config.py'

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            slot = Slot.parse(arg)
            command,environ = slot.get_execution_parameters()
            print('Running', cmd, slot.describe())
            print(slot.get_command_line_prefix() + cmd)
            print()

    else:
        slot = Slot()
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()

        slot = Slot(events = '-1')
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()

        slot = Slot(numa_cpu='0', numa_mem='8', cpus='1,2,3')
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()

        slot = Slot(numa_cpu='1', numa_mem=None, cpus='8-12')
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()

        slot = Slot(nvidia_gpus='0-1')
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()

        slot = Slot(nvidia_gpus='GPU-3f724da0-76aa-3f79-f0a2-cad8acc97e38', amd_gpus='GPU-c6afa01f760b6075')
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()

        slot = Slot.parse('numa=0:nv=GPU-9107ffaa-3302-0e2a-fdbe-00c02f49913d')
        command,environ = slot.get_execution_parameters()
        print('Running', cmd, slot.describe())
        print(slot.get_command_line_prefix() + cmd)
        print()
