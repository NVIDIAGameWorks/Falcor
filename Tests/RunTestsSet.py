
import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint


gBuildSolutionScript = "BuildSolution.bat"

class TestsSetOpenError(Exception):
    pass

class TestsSetParseError(Exception):
    pass

class TestsSetBuildSolutionError(Exception):
    pass


# Try and Build the Specified Solution with the specified configuration/
def buildSolution(solutionfile, configuration):

    # Build the Batch Args.
    batchArgs = [gBuildSolutionScript, "rebuild", solutionfile, configuration.lower()]

    # Build Solution.
    if subprocess.call(batchArgs) == 0:
        return 0
    else:
        return None


# Parse the Specified Tests Set
def runTestsSet(directory, solutionfile, configuration, jsonfilename):

    try:
        # Try and open the json file.
        with open(jsonfilename) as jsonfile:

            # Try and parse the data from the json file.
            try:
                jsondata = json.load(jsonfile)

                # Try and Build the Solution.
                if buildSolution(directory + solutionfile, configuration) != 0:
                    raise TestsSetBuildSolutionError("Error buidling solution : " + directory + solutionfile + " with configuration : " + configuration)

                # Return success.
                return 0

            # Exception Handling.
            except ValueError:
                TestsSetParseError("Error parsing Tests Set file : " + jsonfilename)
                return None

    # Exception Handling.
    except (IOError, OSError) as e:
        raise TestsSetOpenError("Error opening Tests Set file : " + jsonfilename)
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
    return runTestsSet(args.directory, args.solution, args.configuration, args.testsSet)


if __name__ == '__main__':
    main()
