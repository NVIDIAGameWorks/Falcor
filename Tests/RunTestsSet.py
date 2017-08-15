
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

# Try and Build the Specified Solution with the specified configuration/
def buildSolution(solutionfilepath, configuration):

    try:

        # Build the Batch Args.
        batchArgs = [configs.gBuildSolutionScript, "rebuild", solutionfilepath, configuration.lower()]

        # Build Solution.
        if subprocess.call(batchArgs) == 0:
            return 0
        else:
            raise TestsSetBuildSolutionError("Error buidling solution : " + solutionfilepath + " with configuration : " + configuration.lower())
            return None

    except subprocess.CalledProcessError as subprocessError:
        raise TestsSetBuildSolutionError("Error buidling solution : " + solutionfilepath + " with configuration : " + configuration.lower())
        return None



# Prep for running the Tests Set by building the solution.
def prepTestsSet(directorypath, solutionfilename, configuration, jsonfilepath):

    try:
        # Get the absolute path.
        absolutepath = os.path.abspath(directorypath + solutionfilename)
    
        # Try and Build the Solution.
        if buildSolution(absolutepath, configuration) == 0:
            # Return success.
            return 0
        else:
            return None

        # Exception Handling.
    except (TestsSetBuildSolutionError) as e:
            print e.args
            return None






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




def run_tests_set_local(relative_solution_filepath, configuration, tests_set, nobuild):
        



def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which directory.
    parser.add_argument('-d', '--directory', action='store', help='Specify the directory the solution file is in.')

    # Add the Argument for which solution.
    parser.add_argument('-sln', '--solution', action='store', help='Specify the solution file.')

    # Add the Argument for which configuration.
    parser.add_argument('-cfg', '--configuration', action='store', help='Specify the configuration.')

    # Add the Argument for which configuration.
    parser.add_argument('-nb', '--nobuild', action='store_true', help='Specify whether or not to build the solution.')

    # Add the Argument for which Tests Set to run.
    parser.add_argument('-ts', '--testsSet', action='store', help='Specify the Tests Set file.')

    # Parse the Arguments.
    args = parser.parse_args()

    # Parse the Test Collection.
    return runTestsSet(args.directory, args.solution, args.configuration, args.testsSet, args.nobuild)

if __name__ == '__main__':
    main()
