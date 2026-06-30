#! /usr/bin/env python3

def get_read_branch_names(xml_path):
    """
    Open a framework job report XML file and return the list of branch names
    that have bean read from the input files.

    If the ReadBranches element looks like

        <ReadBranches>
        <Branch Name="FEDRawDataCollection_rawDataCollector__LHC." ReadCount="2926"/>
        </ReadBranches>

    this function would return [ 'FEDRawDataCollection_rawDataCollector__LHC' ].

    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    read_branches = root.find("ReadBranches")
    if read_branches is None:
        return []

    branch_names = []
    for branch in read_branches.findall("Branch"):
        name = branch.get("Name")
        if name:
            branch_names.append(name.rstrip("."))  # remove trailing dot(s)

    return branch_names


def loadModuleFromFile(name, filename):
    """
    Import and return a python module from an arbitrary file path (used to load a CMSSW
    configuration by path, like "cmsRun file.py"). The import logic is taken from edmConfigDump.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_job_report(config, events = 10, executable = 'cmsRun', environ = None):
    """
    Run a short job over <events> events to produce a framework job report
    (jobReport.xml) describing the input collections that are read, and return
    the path to the generated XML file.

    This reproduces the manual step performed by the legacy hltTiming.sh wrapper:
    copy the configuration, force a small number of events, and run it with the
    "-j" option to capture the ReadBranches of the input data.
    """
    import atexit
    import os
    import shutil
    import subprocess
    import tempfile

    workdir = tempfile.mkdtemp(prefix = 'jobreport')
    # the returned XML lives in this directory and is read by the caller, so it cannot be removed
    # before returning; defer its cleanup to process exit instead of leaking the directory
    atexit.register(shutil.rmtree, workdir, ignore_errors = True)
    tmpcfg = os.path.join(workdir, 'jobreport_cfg.py')
    xml_path = os.path.join(workdir, 'jobReport.xml')

    # copy the configuration and force a small number of events
    with open(config, 'r') as src, open(tmpcfg, 'w') as dst:
        dst.write(src.read())
        dst.write('\n# limit the number of events while generating the input job report\n')
        dst.write('process.maxEvents.input = cms.untracked.int32(%d)\n' % events)

    command = [ executable, '-j', xml_path, tmpcfg ]
    env = environ.copy() if environ else os.environ.copy()
    print('Generating the input job report by running "%s"' % ' '.join(command))
    result = subprocess.run(command, cwd = workdir, env = env,
                            stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True)
    if result.returncode != 0 or not os.path.exists(xml_path):
        raise RuntimeError('failed to generate the input job report by running "%s":\n%s'
                           % (' '.join(command), result.stdout))
    return xml_path


def resolve_input_collections(config, input_collections, input_xml, input_xml_events = 10, executable = 'cmsRun'):
    """
    Determine the list of input collections for the input-only throughput benchmark: either the
    comma-separated --input-collections, or the ReadBranches of a framework job report given by
    --input-xml FILE (or generated automatically when --input-xml is "auto").
    """
    if input_xml:
        if input_xml == 'auto':
            xml_path = generate_job_report(config, events = input_xml_events, executable = executable)
        else:
            xml_path = input_xml
        return get_read_branch_names(xml_path)
    return input_collections.split(',')


def make_io_process(process, collections):
    """
    Build a trimmed-down copy of `process` that only prefetches the given input collections, for
    benchmarking the input reading in isolation.
    """
    import copy
    import FWCore.ParameterSet.Config as cms

    io_process = copy.deepcopy(process)
    io_process.prefetch = cms.EDAnalyzer("GenericConsumer",
        eventProducts = cms.untracked.vstring(collections)
    )
    io_process.path = cms.Path(io_process.prefetch)
    io_process.schedule = cms.Schedule(io_process.path)
    if 'PrescaleService' in io_process.__dict__:
        del io_process.PrescaleService
    return io_process


if __name__ == "__main__":
    print('\n'.join(get_read_branch_names('test/report.xml')))
