import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint

import MachineConfigs as machine_configs
import InternalConfig as iConfig
import RunPassTests as rPT
import Helpers as helpers


class PassTestsError(Exception):
    pass
        
        
def main():
    
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    # Add argument for testing directory for render pass tests
    parser.add_argument('-td', '--tests_directory', action='store', help='Specify the testing directory containing the test render graphs.')
    
    # Add argument for specifing build configuration for the test
    parser.add_argument('-bc', '--build_configuration', action='store', help='Build configuration for test. ReleaseD3D12 by default')
    
    # Add argument for choosing if we upload to the source server or not
    parser.add_argument('-l', '--local_only', action='store_true', help='Do not upload generated references to server')
    
    # Add argument for only using a specified graph file in the directory
    parser.add_argument('-gf', '--graph_file', action='store', help='Specify graph file to use within the tests directory')
    
    # Add argument for only using a specified graph within the directory
    parser.add_argument('-gn', '--graph_name', action='store', help='Specify graph name to use within the tests directory')
    
    #
    parser.add_argument('-rsf', '--reference_sub_folder', action='store', help='Optional sub folder name within references directory');
    
    # Parse the Arguments.
    args = parser.parse_args()
    
    if args.build_configuration:
        target_configuration = args.build_configuration
    else:
        target_configuration = iConfig.TestConfig["DefaultConfiguration"]
    
    # This assumes the user always runs the script in the /Tests directory
    root_dir = machine_configs.default_main_dir
    branch_name = helpers.get_git_branch_name(root_dir);
    executable_filepath = helpers.get_executable_directory(target_configuration, '', False);
    executable_filepath = os.path.join(os.path.join(root_dir, executable_filepath), iConfig.viewer_executable)

    references_dir = os.path.join('TestsResults', branch_name)
    helpers.directory_clean_or_make(references_dir)
    references_dir = os.path.join(references_dir, target_configuration);
    helpers.directory_clean_or_make(references_dir)
    
    # Display this before building so user has time to respond during build if unintended
    if not args.tests_directory:
        print('No path specified. Will generate reference images for all passes.')
    
    # Build the falcor solution. Run build target on render pass project.
    helpers.build_solution(root_dir, os.path.join(root_dir, 'Falcor.sln'), target_configuration, False)
    
    if args.tests_directory:
        rPT.run_graph_pass_test(executable_filepath, args.tests_directory, args.graph_file, args.graph_name, references_dir)
    else:
        for subdir, dirs, files in os.walk(root_dir):
            if subdir.lower().endswith(iConfig.TestConfig['LocalTestingDir']):
                rPT.run_graph_pass_test(executable_filepath, subdir, args.graph_file, args.graph_name, references_dir)
    
    # copy top level reference directory to netapp 
    if (not args.local_only):
        if args.reference_sub_folder:
            target_dir = os.path.join(machine_configs.machine_reference_directory, args.reference_sub_folder)
            target_dir = os.path.join(target_dir, branch_name)
        else:
            target_dir = os.path.join(machine_configs.machine_reference_directory, branch_name)
        
        target_dir = os.path.join(target_dir, target_configuration)
        helpers.directory_clean_or_make(target_dir)
        helpers.directory_copy(references_dir, target_dir)

        
if __name__ == '__main__':
    main()
