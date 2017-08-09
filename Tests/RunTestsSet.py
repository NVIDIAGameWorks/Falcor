
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
    except (TestsSetBuildSolutionError) as buildError:
            print buildError.args
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
                
                # Iterate over the Tests.
                for currentTest in jsondata['Tests']:
                    
                    # Check if the test is enabled.
                    if(currentTest["Enabled"] != "True"):
                        continue

                    # 
                    for currentArg in currentTest["Project Tests Args"]:

                        print absolutepath + '\\' + currentTest['Project Name'] + ".exe" + ' ' + currentArg
                        process = subprocess.Popen(absolutepath + '\\' + currentTest['Project Name'] + ".exe" + ' ' + currentArg)
                        startTime = time.time()

                        while process.returncode == None:
                            process.poll()
                            currentTime = time.time()
                                
                            differenceTime = currentTime - startTime
                            print differenceTime
                            if differenceTime > configs.gDefaultKillTime:
                                print "Kill Process"
                                process.kill()
                                return 0

                        break

                    break
            


            # Exception Handling.
            except ValueError:
                raise TestsSetParseError("Error parsing Tests Set file : " + jsonfilepath)
                return None

    # Exception Handling.
    except (IOError, OSError) as e:
        raise TestsSetOpenError("Error opening Tests Set file : " + jsonfilepath)
        return None



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
    parser.add_argument('-ts', '--testsSet', action='store', help='Specify the Tests Set filepath.')

    # Parse the Arguments.
    args = parser.parse_args()

    # Parse the Test Collection.
    return runTestsSet(args.directory, args.solution, args.configuration, args.testsSet, args.nobuild)


if __name__ == '__main__':
    main()
