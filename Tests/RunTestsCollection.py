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

        if not json_object_has_attribute(current_test_specification, "Solution Target"):
            raise TestsCollectionError('Error - "Solution Target" is not defined in entry ' + str(index) + ' in ' + tests_name)

        if not json_object_has_attribute(current_test_specification, "Configuration Target"):
            raise TestsCollectionError('Error - "Configuration Target" is not defined in entry ' + str(index) + ' in ' + tests_name)
            

        if not json_object_has_attribute(current_test_specification, "Tests Set"):
            raise TestsCollectionError('Error - "Tests Set" is not defined in entry ' + str(index) + ' in ' + tests_name)

    return 0





# Run all of the Tests Collections.
def run_tests_collections(json_data):


    for current_tests_collection_name in json_data["Tests Collections"]:

        for current_tests_set in json_data["Tests Collections"][current_tests_collection_name]['Tests']:
            
    
            # The Clone Directory is the Destination Target + The Branch Target + the Test Collection Name + the Build Configuration Name.
            # For the momemnt, do not add multiple solutions to the same Test Collection, because that will create overlapping clone targets.
            clone_directory = json_data["Tests Collections"][current_tests_collection_name]["Destination Target"] 
            clone_directory = clone_directory +  json_data["Tests Collections"][current_tests_collection_name]["Branch Target"] 
            clone_directory = clone_directory + '\\' +  current_tests_collection_name + '\\'
            clone_directory = clone_directory + '\\' + current_tests_set["Configuration Target"]

            # Clear the directory.
            helpers.directory_clean_or_make(clone_directory)

            # Clone the Repositroy to the Clone Directory.
            # cloneRepo.clone(json_data["Tests Collections"][current_tests_collection_name]["Repository Target"], json_data["Tests Collections"][current_tests_collection_name]["Branch Target"], clone_directory)

            #   
            # print 'TestsCollectionsAndSets\\' + current_tests_set["Tests Set"]


            # 
            results_directory = "Results\\" + current_tests_collection_name + '\\' + json_data["Tests Collections"][current_tests_collection_name]["Branch Target"]
            results = rTS.run_tests_set_local(clone_directory + '\\' + current_tests_set['Solution Target'], current_tests_set["Configuration Target"], True, 'TestsCollectionsAndSets\\' + current_tests_set["Tests Set"], results_directory)

            # 
            has_expected_test_set_outputs = rTS.check_tests_set_results_expected_output(results['Test Runs Results'])

            # 
            pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(results)

            break

        break



def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-tc', '--tests_collection', action='store', help='Specify the Test Collection.')

    # Parse the Arguments.
    args = parser.parse_args()

    #   
    json_data = read_and_verify_tests_collections_source(args.tests_collection)

    #   
    if json_data is None:

        print 'Falied to Verify Tests Collections Source!'

        return None


    run_tests_collections(json_data)



if __name__ == '__main__':
    main()
