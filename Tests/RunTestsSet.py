
import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint

# Parse the Specified Tests Set
def runTestsSet(json_filename):
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
                                                            
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(json_data)

    # Exception Handling.
    except IOError, info:
        print "Error opening Tests Set file : " + json_filename
        return -1;




def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Arguments for the Repository and the Branch.
    parser.add_argument('-testsSet', nargs='?', action='store', help='Specify the Tests Set', default="TestsSet.json")

    # Parse the Arguments.
    args = parser.parse_args()

    # Parse the Test Collection.
    return runTestsSet(args.testsSet)




if __name__ == '__main__':
    main()
