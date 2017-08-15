import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys
import json
import pprint

import RunTestsCollection as rTC
import Helpers as helpers
import RunTestsSet as rTS






def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-tc', '--tests_collection', action='store', help='Specify the Test Collection.')

    # Parse the Arguments.
    args = parser.parse_args()

    #   
    json_data = rTC.read_and_verify_tests_collections_source(args.testsCollection)

    #   
    if json_data is None:

        print 'Falied to Verify Tests Collections Source!'

        return None


    rTC.run_tests_collections(json_data)


if __name__ == '__main__':
    main()
