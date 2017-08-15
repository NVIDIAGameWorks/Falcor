
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


class TestsSetOpenError(Exception):
    pass

class TestsSetParseError(Exception):
    pass

class TestsSetBuildSolutionError(Exception):
    pass

class TestsSetRunTestsError(Exception):
    pass


class TestsSetError(Exception):
    pass



def getExecutableDirectoryForConfiguration(configuration):
    if configuration.lower() == 'released3d12' or configuration.lower() == 'releasevk' :
        return "Bin\\x64\\Release\\" 
    else:
        return "Bin\\x64\\Debug\\"






# Parse the Specified Tests Set
def runTestsSet(directorypath, solutionfilename, configuration, jsonfilepath, nobuild):

    # Prep the Tests Set - Build the Solution.
    if not nobuild:
        if prepTestsSet(directorypath, solutionfilename, configuration, jsonfilepath) != 0:
            return None

    try:
        # Try and open the json file.
        with open(jsonfilepath) as jsonfile:

            # Try and parse the data from the json file.
            try:
                jsondata = json.load(jsonfile)

                # pp = pprint.PrettyPrinter(indent=4)
                # pp.pprint(jsondata)

                # Get the absolute path.
                absolutepath = os.path.abspath(directorypath + 'Bin\\x64\\Release\\')
                
                # 
                testsRuns = {}                

                # Iterate over the Tests.
                for currentTest in jsondata['Tests']:
                    
                    # Check if the test is enabled.
                    if(currentTest["Enabled"] != "True"):
                        continue

                    # Output Directory.
                    outputdirectory = 'Results\\' + configuration + '\\' + currentTest['Project Name'] + '\\'

                    # Create the output directory.
                    helpers.directory_clean_or_make(outputdirectory)

                    testsRuns[currentTest['Project Name']] = []

                    # Iterate over the runs.
                    for index, currentRunArgs in enumerate(currentTest["Project Tests Args"]):
                        
                        # Result Filename.
                        outputfileprefix = currentTest['Project Name'] + '_' + str(index)                        

                        # Start the process and record the time.
                        process = subprocess.Popen(absolutepath + '\\' + currentTest['Project Name'] + ".exe" + ' ' + currentRunArgs + ' -outputfileprefix ' + outputfileprefix + ' -outputdirectory ' + outputdirectory)
                        startTime = time.time()

                        testStatus = {}
                        ranSuccessfully = True
                        # Wait for the process to finish.
                        while process.returncode == None:
                            process.poll()
                            currentTime = time.time()

                            differenceTime = currentTime - startTime

                            # If the process has taken too long, kill it.
                            if differenceTime > configs.gDefaultKillTime:
                                print "Kill Process"
                                process.kill()
                              
                                ranSuccessfully = False

                                # Break.
                                break 
                        # 
                        testStatus["Completed"] = ranSuccessfully
                        testStatus["Args"] = currentRunArgs

                        testsRuns[currentTest['Project Name']].append(testStatus)
                
                return testsRuns

            # Exception Handling.
            except ValueError:
                raise TestsSetParseError("Error parsing Tests Set file : " + jsonfilepath)
                return None

    # Exception Handling.
    except (IOError, OSError) as e:
        raise TestsSetOpenError("Error opening Tests Set file : " + jsonfilepath)
        return None



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



# Run the tests locally.
def run_tests_set_local(relative_solution_filepath, configuration, nobuild, json_filepath, reference_target):
    
    #   
    if not nobuild:
        build_solution(relative_solution_filepath, configuration)


    json_data = None
    
    try:
        # Try and open the json file.
        with open(json_filepath) as jsonfile:

            # Try and parse the data from the json file.
            try:
                json_data = json.load(jsonfile)

                # Test Runs Results.    
                test_runs_results = {}

                # Iterate over the Tests.
                for current_test in json_data['Tests']:
                    
                    # Check if the test is enabled.
                    if(current_test["Enabled"] != "True"):
                        continue

                    # Output Directory.
                    output_directory = 'Results\\' + configuration + '\\' + current_test['Test Name'] + '\\'

                    #   
                    helpers.directory_clean_or_make(output_directory);


            # Exception Handling.
            except ValueError:
                raise TestsSetError("Error parsing Tests Set file : " + json_filepath)

    # Exception Handling.
    except (IOError, OSError) as e:
        raise TestsSetError("Error opening Tests Set file : " + json_filepath)



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
    run_tests_set_local(args.solutionfilepath, args.configuration, args.nobuild, args.testsset, args.ref)


if __name__ == '__main__':
    main()
