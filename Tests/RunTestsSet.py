
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

import MachineConfigs as machine_configs
import Helpers as helpers
import WriteTestResultsToHTML as write_test_results_to_html



class TestsSetError(Exception):
    pass

# Get the Executable Directory.
def get_executable_directory(configuration):
    if configuration.lower() == 'released3d12' or configuration.lower() == 'releasevk' :
        return "Bin\\x64\\Release\\" 
    else:
        return "Bin\\x64\\Debug\\"


# Build the Solution.
def build_solution(relative_solution_filepath, configuration):

    try:
        # Build the Batch Args.
        batch_args = [machine_configs.machine_build_script, "rebuild", relative_solution_filepath, configuration.lower()]

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
            if difference_time > machine_configs.machine_process_default_kill_time:

                print "Kill Process"

                process.kill()
                              
                run_results = [False, "Process ran for too long, had to kill it. Please verify that the program finishes within its hang time, and that it does not crash"]

                # Break.
                break

        return run_results
    
    except (NameError, IOError, OSError) as e:
        print e.args
        raise TestsSetError('Error when trying to run ' + executable_filepath + ' ' + current_arguments + ' ' + 'with outputfileprefix ' + outputfileprefx + ' and outputdirectory ' + output_directory)


# Run the tests set..
def run_tests_set(main_directory, nobuild, json_filepath, results_directory, reference_directory):

    print main_directory
    print results_directory
    print reference_directory

    tests_set_run_result = {}    
    tests_set_run_result['Tests Set Error Status'] = False
    tests_set_run_result['Tests Set Error Message'] = ""
    #
    json_data = None
    
    try:
        # Try and open the json file.
        with open(json_filepath) as jsonfile:
        
            # Try and parse the data from the json file.
            try:
                json_data = json.load(jsonfile)

                tests_set_run_result["Tests Groups"] = None
                tests_set_run_result["Tests Set Filename"] = os.path.splitext(os.path.basename(json_filepath))[0]
                tests_set_run_result["Tests Set Directory"] = os.path.dirname(json_filepath)
                tests_set_run_result['Tests Groups'] = json_data['Tests Groups']
                tests_set_run_result['Solution Target'] = json_data['Solution Target']
                tests_set_run_result['Configuration Target'] = json_data['Configuration Target']
                tests_set_run_result["Tests Set Results Directory"] = results_directory + '\\' + tests_set_run_result["Tests Set Filename"] + '\\' 
                tests_set_run_result["Tests Set Reference Directory"] = reference_directory + '\\' + tests_set_run_result["Tests Set Filename"] + '\\'

                #   
                if not nobuild:

                    try:
                        # Try and Build the Solution.
                        build_solution(main_directory + json_data['Solution Target'], tests_set_run_result['Configuration Target'])
        
                    except TestsSetError as tests_set_error:
                        tests_set_run_result['Tests Set Error Status'] = True
                        tests_set_run_result['Tests Set Error Message'] = tests_set_error.args 
                        return tests_set_run_result


                # Absolute path.
                absolutepath = os.path.abspath(os.path.dirname(main_directory))

                #   
                for current_tests_group_name in tests_set_run_result['Tests Groups']:
                    current_tests_group = tests_set_run_result['Tests Groups'][current_tests_group_name]
                    current_tests_group['Results'] = {}
     
                    # Get the executable directory.
                    executable_directory = absolutepath + '\\' + get_executable_directory(tests_set_run_result['Configuration Target'])
                    # Get the results directory.
                    current_results_directory = tests_set_run_result["Tests Set Results Directory"] + '\\' + current_tests_group_name + '\\'


                    print current_results_directory
                    # Create the directory, or clean it.
                    helpers.directory_clean_or_make(current_results_directory)

                    #   Check if the test is enabled.
                    if current_tests_group['Enabled'] == "True":

                        # Initialize all the results.
                        current_tests_group['Results']["Run Results"] = {}
                        current_tests_group['Results']['Results Directory'] = current_results_directory
                        current_tests_group['Results']['Results Error Status'] = {}
                        current_tests_group['Results']['Results Error Message'] = {}  
                        current_tests_group['Results']['Results Expected Filename'] = {}

                        # Run each test.
                        for index, current_test_args in enumerate(current_tests_group['Project Tests Args']) :
                       
                            # Initialize the error status.
                            current_tests_group['Results']['Results Error Status'][index] = False  
                            current_tests_group['Results']['Results Error Message'][index] = False  

                            # Initialize the expected filename
                            current_tests_group['Results']['Results Expected Filename'][index] = current_tests_group_name + str(index) + '.json'

                            # Try running the test.
                            try:  

                                executable_file = executable_directory + current_tests_group['Project Name'] + '.exe'    
                                current_test_run_result = run_test_run(executable_file, current_test_args, current_tests_group_name + str(index), current_results_directory)                                
                                current_tests_group['Results']["Run Results"][index] = current_test_run_result

                                if current_test_run_result[0] != True :                                    
                                    current_tests_group['Results']['Results Error Status'][index] = True  
                                    current_tests_group['Results Error Message'][index] = current_test_run_result[1]  


                            # Check if an error occured.
                            except (TestsSetError, IOError, OSError) as tests_set_error:
                                current_tests_group['Results'][index] = True  
                                current_tests_group['Results'][index] = tests_set_error.args  


                return tests_set_run_result

            # Exception Handling.
            except ValueError as e:
                tests_set_run_result['Tests Set Error Status'] = True
                tests_set_run_result['Tests Set Error Message'] = e.args 
                return tests_set_run_result


    # Exception Handling.
    except (IOError, OSError) as e:
        tests_set_run_result['Tests Set Error Status'] = True
        tests_set_run_result['Tests Set Error Message'] = e.args 
        return tests_set_run_result



def verify_tests_groups_expected_output(test_groups):

    for current_tests_group_name in test_groups:

        if test_groups[current_tests_group_name]['Enabled'] != "True":
            continue

        # For each of the runs, check the errors.
        for index, current_project_run in enumerate(test_groups[current_tests_group_name]['Project Tests Args']):
            
            expected_output_file = test_groups[current_tests_group_name]['Results']['Results Directory'] + current_tests_group_name + str(index) + '.json'

            #   Check if the expected file was created.
            if not os.path.isfile(expected_output_file) :

                test_groups[current_tests_group_name]['Results']['Results Error Status'][index] = True  
                test_groups[current_tests_group_name]['Results']['Results Error Message'][index] = 'Could not find the expected json output file : ' + expected_output_file + ' . Please verify that the program ran correctly.'

                continue
    


#   Check the Tests Set Results, and create the output.
def get_tests_set_results(tests_set_run_results):

    # 
    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(tests_set_run_results)

    
    # Check which ones managed to generate an output.    
    tests_groups = tests_set_run_results['Tests Groups']
    verify_tests_groups_expected_output(tests_groups)

    # Check the json results for each one that is enabled.
    for current_tests_group_name in tests_groups:
        if tests_groups[current_tests_group_name]['Enabled'] == "True":
            analyze_tests_group(tests_set_run_results, current_tests_group_name)
            

#   Check the json results for a single test.
def analyze_tests_group(tests_set_run_results, current_test_group_name):

    current_test_group_result = tests_set_run_results['Tests Groups'][current_test_group_name] 
    current_test_group_result['Results']['Performance Checks'] = {}
    current_test_group_result['Results']['Memory Checks'] = {}
    current_test_group_result['Results']['Screen Capture Checks'] = {}

    for index, current_test_args in enumerate(current_test_group_result['Project Tests Args']):
        

        #
        if current_test_group_result['Results']['Results Error Status'][index] != True:

            # Try and parse the data from the json file.
            try:

                current_test_reference_directory = tests_set_run_results['Tests Set Reference Directory'] + '\\'  + '\\' + current_test_group_name + '\\'
                current_test_result_directory = current_test_group_result['Results']['Results Directory']

                result_json_filepath = current_test_result_directory + current_test_group_result['Results']['Results Expected Filename'][index]


                # Try and open the json file.
                with open(result_json_filepath) as result_json_file:

                    result_json_data = json.load(result_json_file)
                            
                    # Analyze the performance checks.
                    performance_checks = analyze_performance_checks(result_json_data)
                    current_test_group_result['Results']['Performance Checks'][index] = performance_checks

                    # Analyze the memory checks.
                    memory_checks = analyze_memory_checks(result_json_data)
                    current_test_group_result['Results']['Memory Checks'][index] = memory_checks

                    # Analyze the screen captures.
                    screen_capture_checks = analyze_screen_captures(result_json_data, current_test_result_directory, current_test_reference_directory)
                    current_test_group_result['Results']['Screen Capture Checks'][index] = screen_capture_checks
                        
                # Exception Handling.
            except (IOError, OSError, ValueError) as e:
                current_test_group_result['Results']['Results Error Status'][index] = True  
                current_test_group_result['Results']['Results Error Message'][index] = 'Could not open the expected json output file : ' + result_json_filepath + ' . Please verify that the program ran correctly.'


#   Analzye the Performance Checks.
def analyze_performance_checks(result_json_data):
        
        return []

#   Analzye the Memory Checks.
def analyze_memory_checks(result_json_data):
        
        return []

#
def analyze_screen_captures(result_json_data, current_test_result_directory, current_test_reference_directory):
        

        screen_captures_results = {}

        for index, frame_screen_captures in enumerate(result_json_data['Frame Screen Captures']):

            # Get the test result image.
            test_result_image_filename = current_test_result_directory + frame_screen_captures['Filename']

            # Get the reference image.
            test_reference_image_filename = current_test_reference_directory + frame_screen_captures['Filename']
            
            # Create the test compare imaage.
            test_comapre_image_filepath = current_test_result_directory + os.path.splitext(frame_screen_captures['Filename'])[0] + '_Compare.png'

            # 
            image_compare_command = ['magick', 'compare', '-metric', 'MSE', '-compose', 'Src', '-highlight-color', 'White', '-lowlight-color', 'Black', test_result_image_filename, test_reference_image_filename, test_comapre_image_filepath]
            image_compare_process = subprocess.Popen(image_compare_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            image_compare_result = image_compare_process.communicate()[0]
            image_compare_return_code = image_compare_process.returncode


            if image_compare_return_code == 0:
                space_index = image_compare_result.find(' ')
                image_compare_result_value = image_compare_result[:space_index]
            else:
                image_compare_result_value = image_compare_result

            # Keep the Return Code and the Result.
            screen_captures_results[index] = [image_compare_return_code, image_compare_result_value]


        return screen_captures_results




def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which solution.
    parser.add_argument('-md', '--main_directory', action='store', help='Specify the path to the directory with the solution.')

    # Add the Argument for which configuration.
    parser.add_argument('-nb', '--no_build', action='store_true', help='Specify whether or not to build the solution.')

    # Add the Argument for which Tests Set to run.
    parser.add_argument('-ts', '--tests_set', action='store', help='Specify the Tests Set file.')

    # Add the Argument for which reference directory to run against.
    parser.add_argument('-ref', '--reference_directory', action='store', help='Specify the Tests Set file.')

    # Parse the Arguments.
    args = parser.parse_args()
    
    main_results_directory = machine_configs.machine_relative_checkin_local_results_directory
    main_reference_directory = machine_configs.machine_default_checkin_reference_directory

    #   Run the Test Set.
    tests_set_results = run_tests_set(args.main_directory, args.no_build, args.tests_set, main_results_directory, main_reference_directory)

    # Build the Tests Results.
    get_tests_set_results(tests_set_results)
    
    # Write the Tests Results to HTML.
    tests_set_html_result = write_test_results_to_html.write_test_set_results_to_html(tests_set_results)
    
    # Output the file to disk.
    html_file_output = machine_configs.machine_relative_checkin_local_results_directory + '\\' + "TestResults_" + os.path.splitext(os.path.basename(args.tests_set))[0]  + ".html" 
    html_file = open(html_file_output, 'w')
    html_file.write(tests_set_html_result)
    html_file.close()

    # Open it up.
    os.system("start " + html_file_output)


if __name__ == '__main__':
    main()
