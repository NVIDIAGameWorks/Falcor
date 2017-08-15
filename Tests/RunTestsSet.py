
import subprocess
import argparse
import os
from datetime import date
import time
import shutil
import stat
import sys
import json
import pprint

import Configs as configs
import Helpers as helpers



class TestsSetError(Exception):
    pass

# Get the Executable Directory.
def get_executable_directory(configuration):
    if configuration.lower() == 'released3d12' or configuration.lower() == 'releasevk' :
        return "Bin\\x64\\Release\\" 
    else:
        return "Bin\\x64\\Debug\\"

# Get the Results Directory.
def get_results_directory(configuration, test_name):
    return 'Results\\' + configuration + '\\' + test_name + '\\'



# Build the Solution.
def build_solution(relative_solution_filepath, configuration):

    try:
        # Build the Batch Args.
        batch_args = [configs.gBuildSolutionScript, "rebuild", relative_solution_filepath, configuration.lower()]

        # Build Solution.
        if subprocess.call(batch_args) == 0:
            return 0

        else:
            raise TestsSetError("Error buidling solution : " + relative_solution_filepath + " with configuration : " + configuration.lower())

    except subprocess.CalledProcessError as subprocess_error:
        raise TestsSetError("Error buidling solution : " + relative_solution_filepath + " with configuration : " + configuration.lower())




def run_test_run(executable_filepath, current_arguments, outputfileprefx, output_directory):

    try:
        # Start the process and record the time.
        process = subprocess.Popen(executable_filepath  + ' ' + current_arguments + ' -outputfileprefix ' + outputfileprefx + ' -outputdirectory ' + output_directory)
        start_time = time.time()

        run_results = [True, ""]

        # Wait for the process to finish.
        while process.returncode is None:
            process.poll()
            current_time = time.time()

            difference_time = current_time - start_time

            # If the process has taken too long, kill it.
            if difference_time > configs.gDefaultKillTime:

                print "Kill Process"

                process.kill()
                              
                run_results = [False, "Process ran for too long, had to kill i. Please verify that the program finishes within its hang time, and that it does not crash"]

                # Break.
                break

        return run_results
    
    except (NameError, IOError, OSError) as e:
        raise TestsSetError('Error when trying to run ' + executable_filepath + ' ' + current_arguments + ' ' + 'with outputfileprefix ' + outputfileprefx + ' and outputdirectory ' + output_directory)


# Run the tests locally.
def run_tests_set_local(solution_filepath, configuration, nobuild, json_filepath):

    tests_set_result = {}    
    tests_set_result['Tests Set Error Status'] = False
    tests_set_result['Tests Set Error Message'] = ""
    tests_set_result["Test Runs Results"] = None
    #   
    if not nobuild:
        try:

            build_solution(solution_filepath, configuration)

        except TestsSetError as tests_set_error:
            tests_set_result['Tests Set Error Status'] = True
            tests_set_result['Tests Set Error Message'] = tests_set_error.args 
            return tests_set_result
    #
    json_data = None
    
    try:
        # Try and open the json file.
        with open(json_filepath) as jsonfile:

            # Try and parse the data from the json file.
            try:
                json_data = json.load(jsonfile)

                # Test Runs Results.    
                test_runs_results = {}

                # Absolute path.
                absolutepath = os.path.abspath(os.path.dirname(solution_filepath))
                print json_data
                #   
                for current_test_name in json_data['Tests']:

                    test_runs_results[current_test_name] = {}
                    
                    # Copy over the test data itself so that we can use it from now on.
                    test_runs_results[current_test_name]['Test'] = json_data['Tests'][current_test_name]
                    test_runs_results[current_test_name]['Results'] = {}

                    # Get the executable directory.
                    executable_directory = absolutepath + '\\' + get_executable_directory(configuration)
                    
                    # Get the results directory.
                    results_directory = get_results_directory(configuration, current_test_name) 
                    test_runs_results[current_test_name]['Results Directory'] = results_directory

                    # Create the directory, or clean it.
                    helpers.directory_clean_or_make(results_directory)

                    #   Check if the test is enabled.
                    if test_runs_results[current_test_name]['Test']['Enabled'] == "True":

                        # Initialize all the results.
                        test_runs_results[current_test_name]['Results']["Run Results"] = {}                    
                        test_runs_results[current_test_name]['Results']['Results Directory'] = results_directory
                        test_runs_results[current_test_name]['Results']['Results Error Status'] = {}
                        test_runs_results[current_test_name]['Results']['Results Error Message'] = {}  

                        # Run each test.
                        for index, current_test_args in enumerate(test_runs_results[current_test_name]['Test']['Project Tests Args']) :
                            
                            # Initialize the error status.
                            test_runs_results[current_test_name]['Results']['Results Error Status'][index] = False  
                            test_runs_results[current_test_name]['Results']['Results Error Message'][index] = False  


                            # Try running the test.
                            try:

                                current_test_run_result = run_test_run(executable_directory + test_runs_results[current_test_name]['Test']['Project Name'] + '.exe', current_test_args, current_test_name + str(index), results_directory)                                
                                test_runs_results[current_test_name]['Results']["Run Results"][index] = current_test_run_result 

                            except TestsSetError as tests_set_error:

                                # Check if an error occured.
                                test_runs_results[current_test_name]['Results']['Results Error Status'][index] = True  
                                test_runs_results[current_test_name]['Results']['Results Error Message'][index] = tests_set_error.args  


                tests_set_result["Test Runs Results"] = test_runs_results
                return tests_set_result

            # Exception Handling.
            except ValueError as e:
                tests_set_result['Tests Set Error Status'] = True
                tests_set_result['Tests Set Error Message'] = e.args 
                return tests_set_result


    # Exception Handling.
    except (IOError, OSError) as e:
        tests_set_result['Tests Set Error Status'] = True
        tests_set_result['Tests Set Error Message'] = e.args 
        return tests_set_result



def check_tests_set_results_expected_output(test_runs_results):

    has_expected_output = True

    for current_test_name in test_runs_results:

        if test_runs_results[current_test_name]['Test']['Enabled'] != "True":
            continue

        # For each of the runs, check the errors.
        for index, current_project_run in enumerate(test_runs_results[current_test_name]['Test']['Project Tests Args']):
            
            expected_output_file = test_runs_results[current_test_name]['Results']['Results Directory'] + current_test_name + str(index) + '.json'

            #   Check if the expected file was created.
            if not os.path.isfile(expected_output_file) :

                test_runs_results[current_test_name]['Results']['Results Error Status'][index] = True  
                test_runs_results[current_test_name]['Results']['Results Error Message'][index] = 'Could not find the expected json output file : ' + expected_output_file + ' . Please verify that the program ran correctly.'

                expected_output = False

                continue
    
    return has_expected_output 


#   Check the Tests Set Results, and create the output.
def check_tests_set_results(test_set_results):


    # Check which ones managed to generate an output.    
    has_expected_output = check_tests_set_results_expected_output(test_set_results['Test Runs Results'])
    
    test_runs_results = test_set_results['Test Runs Results']

    # Check the json results for each one.
    for current_test_name in test_runs_results:
        
        #   
        if test_runs_results[current_test_name]['Test']['Enabled'] != "True":
            continue

        # Check that there was not no other error, and open the file if there wasn't.
        for index, current_project_run in enumerate(test_runs_results[current_test_name]['Test']['Project Tests Args']):

            expected_output_file = test_runs_results[current_test_name]['Results Directory'] + current_test_name + str(index) + '.json'

            if test_runs_results[current_test_name]['Results']['Results Error Status'][index] != True:

                check_json_results(current_test_name, test_runs_results[current_test_name], expected_output_file)



#   Check the json results.
def check_json_results(current_test_name, current_test_result, test_output_file):

    



    return    



def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which solution.
    parser.add_argument('-slnfp', '--solutionfilepath', action='store', help='Specify the solution filepath.')

    # Add the Argument for which configuration.
    parser.add_argument('-cfg', '--configuration', action='store', help='Specify the configuration.')

    # Add the Argument for which configuration.
    parser.add_argument('-nb', '--nobuild', action='store_true', help='Specify whether or not to build the solution.')

    # Add the Argument for which Tests Set to run.
    parser.add_argument('-ts', '--testsset', action='store', help='Specify the Tests Set file.')

    # Add the Argument for which reference directory to run against.
    parser.add_argument('-ref', '--referencedirectory', action='store', help='Specify the Tests Set file.')

    # Parse the Arguments.
    args = parser.parse_args()

    #
    tests_set_results = run_tests_set_local(args.solutionfilepath, args.configuration, args.nobuild, args.testsset)

    if tests_set_results['Tests Set Error Status'] is True:

        print tests_set_results['Tests Set Error Message']
    
    else:
        check_tests_set_results(tests_set_results)

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(tests_set_results['Test Runs Results'])


if __name__ == '__main__':
    main()
