import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint

import Helpers as helpers
import WriteTestResultsToHTML as write_test_results_to_html
import MachineConfigs as machine_configs
import RunTestsSet as rTS


class TestsCollectionError(Exception):
    pass

# Read the json data, and read and verify the tests collections source.
def read_and_verify_tests_collections_source(json_filename):
    try:
        # Open and parse json file
        json_file = open(json_filename)
        json_data = json.load(json_file)
    except (IOError, OSError, json.decoder.JSONDecodeError) as e:
        raise TestsCollectionError(e.args)

    # Check for a Tests Collection Name.
    if 'Tests Collections Name' not in json_data:
        raise TestsCollectionError('Error - "Tests Collections Name" is not defined in ' + json_filename)

    # Check for a Tests Collection Name.
    if 'Tests Collections' not in json_data:
        raise TestsCollectionError('Error - "Tests Collections" is not defined in ' + json_filename)

    # Check for a non-zero Tests Collection dictionary.
    if len(json_data['Tests Collections'].keys()) == 0:
        raise TestsCollectionError('Error - "Tests Collections" dictionary is not of non-zero size in ' + json_filename)

    # Verify that all of the tests collections are correctly written.
    for key in json_data['Tests Collections']:
        verify_tests_collection(key, json_data['Tests Collections'][key])

    return json_data

# Verify each tests collection.
def verify_tests_collection(tests_name, tests_data):

    # Check for a Repository Target.
    if 'Repository Target' not in tests_data:
        raise TestsCollectionError('Error - "Repository Target" is not defined in ' + tests_name)

    # Check for a Branch Target.
    if 'Source Branch Target' not in tests_data:
        raise TestsCollectionError('Error - "Branch Target" is not defined in ' + tests_name)

    # Check for a Branch Target.
    if 'Compare Branch Target' not in tests_data:
        raise TestsCollectionError('Error - "Branch Target" is not defined in ' + tests_name)

    # Check for a Tests Array.
    if 'Tests' not in tests_data:
        raise TestsCollectionError('Error - "Tests" is not defined in ' + tests_name)

    # Check for a non-zero Tests Collection dictionary.
    if len(tests_data['Tests']) == 0:
        raise TestsCollectionError('Error - "Tests" array is not of non-zero length in ' + tests_name)

    # Verify each of the tests specification.
    for index, current_test_specification in enumerate(tests_data['Tests']):
        if 'Tests Set' not in current_test_specification:
            raise TestsCollectionError('Error - "Tests Set" is not defined in entry ' + str(index) + ' in ' + tests_name)

    return 0


def verify_all_tests_collection_ran_successfully(tests_collections_results):
    verify_tests_collections = {}
    verify_tests_collections['Success'] = True
    verify_tests_collections['Error Messages'] = {}

    for tests_collection_name in tests_collections_results:
        if 'Error' in tests_collections_results[tests_collection_name]:
            verify_tests_collections['Success'] = False

    return verify_tests_collections


# Run all of the Tests Collections.
def run_tests_collections(json_data):
    tests_collection_results = {}
    tests_collection_results = json_data['Tests Collections']

    # Run each test collection.
    for name in tests_collection_results:

        tests_collection_results[name]['Tests Sets Results'] = []

        current_tests_collection_results = tests_collection_results[name]['Tests Sets Results']

        # Run each Test Set.
        for index, current_tests_set in enumerate(tests_collection_results[name]['Tests']):
            # The Clone Directory is the Destination Target + The Branch Target + the Test Collection Name + the Build Configuration Name.
            # For the momemnt, do not add multiple solutions to the same Test Collection, because that will create overlapping clone targets.
            clone_directory = machine_configs.destination_target
            clone_directory = clone_directory +  tests_collection_results[name]['Source Branch Target']
            clone_directory = os.path.join(clone_directory, name)
            clone_directory = os.path.join(clone_directory, os.path.splitext(os.path.basename(current_tests_set['Tests Set']))[0])

            # Clear the directory.
            if helpers.directory_clean_or_make(clone_directory) == None:
                tests_collection_results[name]['Error'] = "Could not clean or make directory - please try manually removing the directory : " + clone_directory
                continue

            # Try to clone the repository
            try:
                cloneRepo.clone(tests_collection_results[name]['Repository Target'], tests_collection_results[name]['Source Branch Target'], clone_directory)
            except (cloneRepo.CloneRepoCloneError, cloneRepo.CloneRepoCleanOrMakeError) as clone_repo_error:
                tests_collection_results[name]['Error'] = "Could not clone the repository. Please try manually removing the directory it is to be cloned into, and verifying the target and branch. " + clone_directory

            # Get the Results and Reference Directory.
            common_directory_path = os.path.join(tests_collection_results[name]['Source Branch Target'], name)
            results_directory = os.path.join('TestsResults', common_directory_path)
            reference_directory = os.path.join(machine_configs.machine_reference_directory, machine_configs.machine_name, tests_collection_results[name]['Compare Branch Target'])

            # Run the Tests Set.
            test_results = rTS.run_tests_set(clone_directory, False, tests_collection_results[name]['Tests Configs Target'] + current_tests_set['Tests Set'], results_directory, reference_directory, True)

            current_tests_collection_results.append(test_results);

    return tests_collection_results


def check_tests_collections_results(tests_collection_results):
    for name in tests_collection_results:
        for result in tests_collection_results[name]['Tests Sets Results']:
            # Get the Tests Set Result.
            rTS.get_tests_set_results(result)


def write_tests_collection_html(tests_collection_results):

    html_outputs = []
    for current_test_collection_name in tests_collection_results:
        for current_test_set_result in tests_collection_results[current_test_collection_name]['Tests Sets Results']:
            tests_set_html_result = write_test_results_to_html.write_test_set_results_to_html(current_test_set_result)

            # Output the file to disk.
            html_output_file_path = os.path.join(current_test_set_result['Results Directory'], helpers.build_html_filename(current_test_set_result))
            html_file = open(html_output_file_path, 'w')
            html_file.write(tests_set_html_result)
            html_file.close()

            current_html_output = {}
            current_html_output['Test Collection Name'] = current_test_collection_name
            current_html_output['Tests Set Filename'] = current_test_set_result['Name']
            current_html_output['HTML File'] = html_output_file_path
            current_html_output['Machine'] = machine_configs.machine_name

            html_outputs.append(current_html_output)

    return html_outputs


def prepare_and_dispatch_email(success, html_outputs):
    date_and_time = date.today().strftime("%m-%d-%y")

    if success:
        subject = "[SUCCESS] "
    else:
        subject = "[FAILED] "

    subject += 'Falcor Automated Tests - ' + machine_configs.machine_name + ' : ' + date_and_time
    attachments = [x['HTML File'] for x in html_outputs]
    
    helpers.dispatch_email(subject, attachments) 

def all_tests_succeeded(tests_collection_results):
    success = True
    for collection_name in tests_collection_results:
        for result in tests_collection_results[collection_name]['Tests Sets Results']:
            success &= result['Success']

    return success

def copy_results_to_target(html_outputs):
    target_base = machine_configs.machine_results_summary_target
    dateStr = date.today().strftime("%m-%d-%y")
    today_dir = os.path.join(target_base, dateStr)    

    #If there isn't already a folder for today's results, make one 
    if not os.path.isdir(today_dir):
        os.mkdir(today_dir)
    
    #copy and rename any output html files    
    for output in html_outputs:
        out_path = output['HTML File']
        out_file = os.path.basename(out_path)
        shutil.copy(out_path, today_dir)
        print('Copied ' + out_path + ' to ' + today_dir)
        copied_path = os.path.join(today_dir, out_file)
        new_name = output['Machine'] + '_' + out_file
        new_path = os.path.join(today_dir, new_name)
        shutil.move(copied_path, new_path)
        print('Renamed ' + copied_path + ' to ' + new_path)

def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-tc', '--tests_collection', action='store', help='Specify the Test Collection.')

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-ne', '--no_email', action='store_true', help='Whether or not to email.')
    
    #whether or not to copy results to network for daily email
    parser.add_argument('-nc', '--no_copy', action='store_true', help='Whether or not to copy html results to directory scanned for daily email')

    # Parse the Arguments.
    args = parser.parse_args()

    try:
        json_data = read_and_verify_tests_collections_source(args.tests_collection)
    except TestsCollectionError as tests_collection_error:
        print(tests_collection_error.args)
        print('Failed to verify Tests Collections source!')
        return

    tests_collection_results = run_tests_collections(json_data)
    check_tests_collections_results(tests_collection_results)
    html_outputs = write_tests_collection_html(tests_collection_results)

    if not args.no_email:
        prepare_and_dispatch_email(all_tests_succeeded(tests_collection_results), html_outputs)
    
    if not args.no_copy:
        copy_results_to_target(html_outputs)

if __name__ == '__main__':
    main()
