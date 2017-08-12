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


# Default Clone Repositories.
gDefaultCloneRepository = 'https://github.com/NVIDIAGameworks/Falcor.git'
gDefaultCloneBranch = 'master'
gDefaultCloneDestination = 'C:\\Falcor\\'


# Parse the Test Collection.
def runTestCollection(json_filename="TestsCollectionsAndSets\\TestsCollection.json"):
    try:
        # Try and open the json file.
        with open(json_filename) as json_file:
            
            # Try and parse the data from the json file.
            try:
                json_data = json.load(json_file)
    
            # Exception Handling.
            except ValueError:
                print "Error parsing Tests Set file : " + json_filename
                return None

            # pp = pprint.PrettyPrinter(indent=4)
            # pp.pprint(json_data)

            # Check if the Tests Name is defined.
            if not json_data['Tests Name']:
                print 'Error "Tests Name" not defined in json file : ' + json_filename
                return None

            # Check if the Tests Array is defined.
            if not json_data['Tests Name']:
                print 'Error "Tests" not defined in json file : ' + json_filename
                return None

            repositoryTarget = gDefaultCloneRepository
            branchTarget = gDefaultCloneBranch
            destinationTarget = gDefaultCloneDestination 

            # Check if the Repository Target is defined.
            if json_data['Repository Target']:
                if json_data['Repository Target'] != "": 
                    repositoryTarget = json_data['Repository Target']

            # Check if the Branch Target is defined.
            if json_data['Branch Target']:
                if json_data['Branch Target'] != "":
                    branchTarget = json_data['Branch Target']

            # Check if the Destination Target is defined.
            if json_data['Destination Target']:
                if json_data['Destination Target'] != "":
                    destinationTarget = json_data['Destination Target']


            # Initialize the Test Results.
            testResults = []

            # Run all the Test Set.
            for currentTestsSet in json_data["Tests"]:

                # Check if a solution target is defined.
                if currentTestsSet['Solution Target']:
                    if currentTestsSet['Solution Target'] == "":
                        continue
                else:
                    continue

                # Check if a configuration target is defined.
                if currentTestsSet['Configuration Target']:
                    if currentTestsSet['Configuration Target'] == "":
                        continue
                else:
                    continue

                # Check if a configuration target is defined.
                if currentTestsSet['Tests Set']:
                    if currentTestsSet['Tests Set'] == "":
                        continue
                else:
                    continue


                try:

                    
                    solutionTarget = currentTestsSet['Solution Target']
                    configurationTarget = currentTestsSet['Configuration Target']
                    testsSet = currentTestsSet['Tests Set']
                    destinationBranchConfigurationTarget = destinationTarget + branchTarget + '\\' + configurationTarget + '\\'                      


                    # Check if we can clone a repository.
                    try:
                        if cloneRepo.clone(repositoryTarget, branchTarget, destinationBranchConfigurationTarget) != 0:
                            return None
                
                    # Exception Handling.
                    except (cloneRepo.CloneRepoCleanOrMakeError, cloneRepo.CloneRepoCloneError) as cloneRepoError:
                        print cloneRepoError.args
                        return None


                    # Run the Tests Set and get the results.
                    currentTestResult = rTS.runTestsSet(destinationBranchConfigurationTarget, solutionTarget, configurationTarget, testsSet)

                    # 
                    if currentTestResult == None
                        continue

                    # Add the Test Result to the Test Results.
                    testResults.append(currentTestResult)

                # Exception Handling.
                except (rTS.TestsSetOpenError, rTS.TestsSetParseError) as runTestsSetError:
                    print runTestsSetError.args
                    continue


        return 0

    # Exception Handling.
    except (IOError, OSError) as jsonopenerror:
        print 'Error opening Tests Collection json file : ' + json_filename
        return None



def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-testsCollection', nargs='?', action='store', help='Specify the Test Collection', default='TestsCollectionsAndSets\\TestsCollection.json')

    # Add the Arguments for do not build and for show summary, and whether to run it locally.
    parser.add_argument("-nb", action='store_true', help='Whether or not to build the solutions again.')
    parser.add_argument("-ss", "--showsummary", action='store_true', help='Whether or not to display the summary at the end.')
    parser.add_argument("-lc", "--local", action='store_true', help='Whether or not to run the tests locally.')

    # Parse the Arguments.
    args = parser.parse_args()

    # Parse the Test Collection.
    return runTestCollection(args.testsCollection)




if __name__ == '__main__':
    main()
