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
gDefaultCloneRepository = 'https://github.com/NVIDIAGameworks/Falcor.git';
gDefaultCloneBranch = 'master';
gDefaultCloneDestination = 'C:\\Falcor\\'


# Parse the Test Collection.
def runTestCollection(json_filename="TestsCollection.json"):
    
    try:
        # Try and open the json file.
        with open(json_filename) as json_file:
            
            # Try and parse the data from the json file.
            try:
                json_data = json.load(json_file)
    
            # Exception Handling.
            except ValueError:
                print "Error parsing Tests Set file : " + json_filename
                return -1;


            # Check if the Tests Name is defined.
            if not json_data['Tests Name']:
                print 'Error "Tests Name" not defined in json file : ' + json_filename
                return -1

            # Check if the Tests Array is defined.
            if not json_data['Tests Name']:
                print 'Error "Tests" not defined in json file : ' + json_filename
                return -1 

            repositoryTarget = gDefaultCloneRepository
            branchTarget = gDefaultCloneBranch
            destinationTarget = gDefaultCloneDestination 

            # Check if the Repository Target is defined.
            if json_data['Repository Target']:
                repositoryTarget = json_data['Repository Target']

            # Check if the Branch Target is defined.
            if json_data['Branch Target']:
                branchTarget = json_data['Branch Target']

            # Check if the Destination Target is defined.
            if json_data['Destination Target']:
                destinationTarget = json_data['Destination Target']



            if cloneRepo.clone(repositoryTarget, branchTarget, destinationTarget) != 0
                return -1


            # Initialize the Test Results.
            testResults = []

            # Run the Test Set.
            for currentTestsSet in (json_data["Tests"]):

                # Run the Test and get the results.
                currentTestResult = rTS.runTestsSet(currentTestsSet)
                
                # Add the Test Result to the Test Results.
                testResults.append(currentTestResult)

        return 0

    # Exception Handling.
    except OSError, info:
        print 'Error opening Tests Collection json file : ' + json_file
        return -1;



def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-testsCollection', nargs='?', action='store', help='Specify the Test Collection', default='TestsCollectionsAndSets/TestsCollection.json')

    # Add the Arguments for do not build and for show summary.
    parser.add_argument("-nb", action='store_true', help='Whether or not to build the solutions again.')
    parser.add_argument("-ss", "--showsummary", action='store_true', help='Whether or not to display the summary at the end.')
    parser.add_argument("-lc", "--local", action='store_true', help='Whether or not to run the tests locally.')

    # Parse the Arguments.
    args = parser.parse_args()

    # Parse the Test Collection.
    return runTestCollection(args.testsCollection)




if __name__ == '__main__':
    main()
