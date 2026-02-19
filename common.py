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


if __name__ == "__main__":
    print('\n'.join(get_read_branch_names('test/report.xml')))
