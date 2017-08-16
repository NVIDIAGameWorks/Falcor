import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint

import RunTestsSet as rTS
import CloneRepo as cloneRepo
import Helpers as helpers
import WriteTestResultsToHTML as write_test_results_to_html
import MachineConfigs as machine_configs


class TestsCollectionError(Exception):
    pass

# Check whether the json object has the specified attribute.
def json_object_has_attribute(json_data, attribute_name):
    
    if attribute_name in json_data:
        return True
    else:
        return False


# Read the json data, and read and verify the tests collections source.
def read_and_verify_tests_collections_source(json_filename):

    json_data = None
    
    try:
        # Try and open the json file.
        with open(json_filename) as json_file:

            # Try and parse the data from the json file.
            try:
                json_data = json.load(json_file)

                # Check for a Tests Collection Name.
                if not json_object_has_attribute(json_data, "Tests Collections Name"):
                    raise TestsCollectionError('Error - "Tests Collections Name" is not defined in ' + json_filename)

                # Check for a Tests Collection Name.
                if not json_object_has_attribute(json_data, "Tests Collections"):
                    raise TestsCollectionError('Error - "Tests Collections" is not defined in ' + json_filename)

                # Check for a non-zero Tests Collection dictionary.
                if len(json_data["Tests Collections"].keys()) == 0:
                    raise TestsCollectionError('Error - "Tests Collections" dictionary is not of non-zero size in ' + json_filename)

                # Verify that all of the tests collections are correctly written. 
                for key in json_data["Tests Collections"]:
                    verify_tests_collection(key, json_data["Tests Collections"][key])


                return json_data

            # Exception Handling.
            except ValueError:
                print "Error parsing Tests Collection file : " + json_filename
                return None

    # Exception Handling.
    except (IOError, OSError) as json_open_error:
        print 'Error opening Tests Collection json file : ' + json_filename
        return None



# Verify each tests collection.
def verify_tests_collection(tests_name, tests_data):

    # Check for a Repository Target.
    if not json_object_has_attribute(tests_data, "Repository Target"):
        raise TestsCollectionError('Error - "Repository Target" is not defined in ' + tests_name)
        

    # Check for a Repository Folder.
    if not json_object_has_attribute(tests_data, "Repository Folder"):
        raise TestsCollectionError('Error - "Repository Folder" is not defined in ' + tests_name)
        

    # Check for a Branch Target.
    if not json_object_has_attribute(tests_data, "Branch Target"):
        raise TestsCollectionError('Error - "Branch Target" is not defined in ' + tests_name)
        


    # Check for a Repository Target.
    if not json_object_has_attribute(tests_data, "Destination Target"):
        raise TestsCollectionError('Error - "Destination Target" is not defined in ' + tests_name)
        

    # Check for a Repository Target.
    if not json_object_has_attribute(tests_data, "Reference Target"):
        raise TestsCollectionError('Error - "Reference Target" is not defined in ' + tests_name)
        

    # Check for a Tests Array.
    if not json_object_has_attribute(tests_data, "Tests"):
        raise TestsCollectionError('Error - "Tests" is not defined in ' + tests_name)
        

    # Check for a non-zero Tests Collection dictionary.
    if len(tests_data["Tests"]) == 0:
        raise TestsCollectionError('Error - "Tests" array is not of non-zero length in ' + tests_name)

    # Verify each of the tests specification.
    for index, current_test_specification in enumerate(tests_data["Tests"]):

        if not json_object_has_attribute(current_test_specification, "Tests Set"):
            raise TestsCollectionError('Error - "Tests Set" is not defined in entry ' + str(index) + ' in ' + tests_name)

    return 0



def verify_all_tests_collection_ran_successfully(tests_collections_results):




    return True


# Run all of the Tests Collections.
def run_tests_collections(json_data):

    tests_collections_run_results = {}
    tests_collections_run_results = json_data["Tests Collections"]
    
    # Run each test collection.
    for current_tests_collection_name in tests_collections_run_results:
        
        tests_collections_run_results[current_tests_collection_name]['Tests Sets Results'] = {}

        current_tests_collection_results = tests_collections_run_results[current_tests_collection_name]['Tests Sets Results']

        # Run each Test Set.
        for index, current_tests_set in enumerate(json_data["Tests Collections"][current_tests_collection_name]['Tests']):

            # The Clone Directory is the Destination Target + The Branch Target + the Test Collection Name + the Build Configuration Name.
            # For the momemnt, do not add multiple solutions to the same Test Collection, because that will create overlapping clone targets.
            clone_directory = json_data["Tests Collections"][current_tests_collection_name]["Destination Target"] 
            clone_directory = clone_directory +  json_data["Tests Collections"][current_tests_collection_name]["Branch Target"]
            clone_directory = clone_directory + '\\' +  current_tests_collection_name + '\\'
            clone_directory = clone_directory + '\\' + os.path.splitext(os.path.basename(current_tests_set["Tests Set"]))[0] + '\\'

            # Clear the directory.
            # helpers.directory_clean_or_make(clone_directory)

            # Clone the Repositroy to the Clone Directory.
            # cloneRepo.clone(json_data["Tests Collections"][current_tests_collection_name]["Repository Target"], json_data["Tests Collections"][current_tests_collection_name]["Branch Target"], clone_directory)

            # Get the Results and Reference Directory.
            common_directory_path = json_data["Tests Collections"][current_tests_collection_name]["Branch Target"] + "\\" + current_tests_collection_name + '\\'
            results_directory = 'TestsResults' + '\\' + common_directory_path
            reference_directory = json_data["Tests Collections"][current_tests_collection_name]['Reference Target'] + '\\'  + machine_configs.machine_name + '\\' + common_directory_path

            # Run the Tests Set.
            results = rTS.run_tests_set(clone_directory, False, current_tests_set["Tests Set"], results_directory, reference_directory)
            

            #   Get the Tests Groups Results.   
            rTS.verify_tests_groups_expected_output(results['Tests Groups'])

            #   
            current_tests_collection_results[index] = results;

    # 
    return tests_collections_run_results


def check_tests_collections_results(tests_collections_run_results):

    for current_tests_collection_name in tests_collections_run_results:

        for current_tests_set_index in tests_collections_run_results[current_tests_collection_name]['Tests Sets Results']:
            
            # Get the Tests Set Result.
            current_tests_set_result = tests_collections_run_results[current_tests_collection_name]['Tests Sets Results'][current_tests_set_index]
                        
            rTS.get_tests_set_results(current_tests_set_result)



def write_tests_collection_html(tests_collections_run_results):

    html_outputs = []
    for current_test_collection_name in tests_collections_run_results:
            for current_test_set_index in tests_collections_run_results[current_test_collection_name]['Tests Sets Results']:
                current_test_set_result = tests_collections_run_results[current_test_collection_name]['Tests Sets Results'][current_test_set_index]
                tests_set_html_result = write_test_results_to_html.write_test_set_results_to_html(current_test_set_result)

                # Output the file to disk.
                html_file_output = current_test_set_result['Tests Set Results Directory'] + '\\' + "TestResults_" + current_test_set_result['Tests Set Filename']  + ".html" 
                html_file = open(html_file_output, 'w')
                html_file.write(tests_set_html_result)
                html_file.close()

                current_html_output = {}
                current_html_output['Test Collection Name'] = current_test_collection_name
                current_html_output['Tests Set Filename'] = current_test_set_result['Tests Set Filename'] 
                current_html_output['HTML File'] = html_file_output
                current_html_output['Machine'] = machine_configs.machine_name

                html_outputs.append(current_html_output)

    return html_outputs

def dispatch_email(html_outputs):
    date_and_time = date.today().strftime("%m-%d-%y")
    subject = ' Falcor Automated Tests - ' + machine_configs.machine_name + ' : ' + date_and_time
    dispatcher = 'NvrGfxTest@nvidia.com'
    recipients = str(open(machine_configs.email_file, 'r').read())
    subprocess.call(['blat.exe', '-install', 'mail.nvidia.com', dispatcher])
    command = ['blat.exe', '-to', recipients, '-subject', subject, '-body', "   "]
    for html_output in html_outputs:
        command.append('-attach')
        command.append(html_output['HTML File'])
    subprocess.call(command)



    
def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-tc', '--tests_collection', action='store', help='Specify the Test Collection.')

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-ne', '--no_email', action='store_true', help='Whether or not to email.')

    # Parse the Arguments.
    args = parser.parse_args()

    # 
    try: 
        json_data = read_and_verify_tests_collections_source(args.tests_collection)
    except TestsCollectionError as tests_collection_error:
        print (tests_collection_error.args)

    #   
    if json_data is None:
        print 'Falied to Verify Tests Collections Source!'
        return None

    # 
    tests_collections_run_results = run_tests_collections(json_data)
    check_tests_collections_results(tests_collections_run_results)
    html_outputs = write_tests_collection_html(tests_collections_run_results)
    dispatch_email(html_outputs)
    

if __name__ == '__main__':
    main()
