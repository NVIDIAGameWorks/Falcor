import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint
import getpass
from time import sleep

import xml.etree.ElementTree as ET
import MachineConfigs as machine_configs
import TestConfig as config
import RunPassTests as rPT
import Helpers as helpers
import sys
import WriteTestResultsToHTML as writeTestResultsToHTML
from TeamCityCommon import connect
import StartBuildTest as remoteTests
import GetBuildStatus as getBuildStatus

xml_file_path = './build.xml'

class PassTestsError(Exception):
    pass

def get_generate_references_build_type_ids():
    build_types = getBuildStatus.get_build_types();
    dataString = str(build_types.read().decode())
    xmldata = ET.fromstring(dataString)
    build_type_ids = []
    
    for node in xmldata.iter():
        for buildType in node.findall('buildType'):
            buildTypeId = str(buildType.get('id'))
            if buildTypeId.find('Generate') != -1:
                build_type_ids.append(buildTypeId)
    
    return build_type_ids

def generate_references_remote(branch_name, git_path, tests_directory):
    username = input('Enter username for teamcity.nvidia.com: ')
    getPassPrompt = 'Enter teamcity password for user ' + username + ':'
    password = getpass.getpass(prompt=getPassPrompt )
    
    connect(username, password)
    
    buildTypeIds = get_generate_references_build_type_ids()
    
    for buildTypeId in buildTypeIds:
        remoteTests.start_build(username, xml_file_path, branch_name, git_path, tests_directory, buildTypeId)
    
    sleep(4)
    getBuildStatus.wait_for_running_builds(buildTypeIds)

def main():
    
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    # Add argument for specifing build configuration for the test
    parser.add_argument('--build_configuration', action='store', help='Build configuration for test. ReleaseD3D12 by default')
    
    # Add argument for specifying to only build local instead of dispatching to teamcity
    parser.add_argument('--local', action='store_true', help='Generate local references')
    
    # Add argument for specifying to only build local instead of dispatching to teamcity
    parser.add_argument('--rebuild', action='store_true', help='Force solution rebuild')

    # Add argument for specifying  dispatching to teamcity instead of only building locally
    parser.add_argument('--remote', action='store_true', help='Generate references on all teamcity test machines')
    
    # Add argument to specify to upload resources to the data server
    parser.add_argument('--upload', action='store_true', help='Upload the references to netapp, if tests are local')
            
    # Add argument for branch name. If this is not specified the script will look in the .git directory
    parser.add_argument('--branch_name', action='store', help='Name of the current checkout branch')
    
     # Subfolder within the references directory. Set with the build machine name by the test servers
    parser.add_argument('--machine_name', action='store', help='Optional sub folder name within references directory');
    
    # Add argument for specifying to only generate references that are missing from the directory
    parser.add_argument('--generate_missing', action='store_true', help='Only generate missing reference images.');
    
    # Parse the Arguments.
    args = parser.parse_args()
    
    if not args.remote and not args.local:
        raise(PassTestsError("Please specify 'local' or 'remote' within the script arguments."))
    
    if args.build_configuration:
        target_configuration = args.build_configuration
    else:
        target_configuration = config.defaultConfig
    
    # This assumes the user always runs the script in the /Tests directory
    if args.branch_name:
        branch_name = args.branch_name
    else:
        branch_name = helpers.get_git_branch_name(root_dir);
    
    references_dir = os.path.join('TestsResults', branch_name)
    if not args.generate_missing:
        helpers.directory_clean_or_make(references_dir)
    else:
        helpers.directory_make(references_dir)
    
    references_dir = os.path.join(references_dir, target_configuration);
    if not args.generate_missing:
        helpers.directory_clean_or_make(references_dir)
    else:
        helpers.directory_make(references_dir)
    
    if args.upload and args.generate_missing:
        print('Copying available references from remote netapp server')
        if args.machine_name:
            target_dir = os.path.join(machine_configs.machine_reference_directory, args.machine_name)
            target_dir = os.path.join(target_dir, branch_name)
        else:
            target_dir = os.path.join(machine_configs.machine_reference_directory, branch_name)
        
        target_dir = os.path.join(target_dir, target_configuration)
        if os.path.isdir(target_dir):
            helpers.directory_copy(target_dir, references_dir)
    
    if not args.tests_directory:
            print('No path specified. Will generate reference images for all passes.')
    
    errors = {}
    
    path_to_bin = os.path.join( root_dir, 'Bin')
    if os.path.isdir(path_to_bin):
        helpers.directory_clean(path_to_bin)
    
    if not args.local_only:
        if args.git_url:
            git_path = args.git_url
        else:
            git_path = helpers.get_git_url(root_dir)
        
        generate_references_remote(branch_name, git_path, args.tests_directory)
    else:
        executable_filepath = helpers.get_executable_directory(target_configuration, '', False);
        executable_filepath = os.path.join(os.path.join(root_dir, executable_filepath), iConfig.mogwai_executable)
        
        if not arg.no_rebuild:
            helpers.deletePackmanRepo()
        
        # Build the falcor solution. Run build target on render pass project.
        helpers.build_solution(root_dir, os.path.join(root_dir, 'Falcor.sln'), target_configuration, not args.no_rebuild)
        test_counter = 0;
        if args.tests_directory:
            errors = rPT.run_graph_pass_test(executable_filepath, args.tests_directory, args.graph_file, references_dir, args.generate_missing)
        else:
            for subdir, dirs, files in os.walk(root_dir):
                ignoreThisDir = False
                for ignoreDir in iConfig.IgnoreDirectories[target_configuration]:
                    ignore_abs_path = os.path.abspath(os.path.join(root_dir, str(ignoreDir)))
                    current_abs_path = os.path.abspath(subdir)
                    if (os.path.commonpath([ignore_abs_path]) == os.path.commonpath([ignore_abs_path, current_abs_path])):
                        ignoreThisDir = True
                        break;
                if ignoreThisDir:
                    continue
                
                if subdir.lower().endswith(iConfig.TestConfig['LocalTestingDir']):
                    new_errors =  rPT.run_graph_pass_test(executable_filepath, subdir, args.graph_file, references_dir, args.generate_missing)
                    for error_key in new_errors.keys():
                        errors[error_key] = new_errors[error_key]
        
        # copy top level reference directory to netapp 
        if (args.upload):
            print('Uploading references to remote netapp server')
            if args.machine_name:
                target_dir = os.path.join(machine_configs.machine_reference_directory, args.machine_name)
                target_dir = os.path.join(target_dir, branch_name)
            else:
                target_dir = os.path.join(machine_configs.machine_reference_directory, branch_name)
            
            all_results_data = []
            
            target_dir = os.path.join(target_dir, target_configuration)
            helpers.directory_make(target_dir)
            helpers.directory_copy(references_dir, target_dir)
            
            for subdir, dirs, files in os.walk(target_dir):
                for file in files:
                    all_results_data.append(os.path.abspath(os.path.join(subdir, file)))
            
            # output html file with references 
            html_file_content = writeTestResultsToHTML.write_generate_references_to_html(target_dir, all_results_data, errors)
            helpers.directory_make(machine_configs.machine_relative_checkin_local_results_directory)
            html_file_path = os.path.join(target_dir, "GenerateReferences_Results.html")
            html_file = open(html_file_path, 'w')
            html_file.write(html_file_content)
            html_file.close()
    
    print ('Please confirm that your output images are correct')
    
    #open file browser to generate references
    target_dir = os.path.join(machine_configs.machine_reference_directory, branch_name)
    print (target_dir)
    helpers.open_file_dir(target_dir)
   
if __name__ == '__main__':
    main()
