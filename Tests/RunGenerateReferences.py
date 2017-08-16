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
import RunTestsSet as rTS
import MachineConfigs as machine_configs
import Helpers as helpers



def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-tc', '--tests_collection', action='store', help='Specify the Test Collection.')

    # Parse the Arguments.
    args = parser.parse_args()

    #   
    json_data = rTC.read_and_verify_tests_collections_source(args.tests_collection)

    #   
    if json_data is None:

        print 'Falied to Verify Tests Collections Source!'

        return None

    # 
    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(json_data)

    #
    tests_collections_results = rTC.run_tests_collections(json_data)
    

    #   
    for current_test_collections in json_data['Tests Collections']:

        # 
        destination_reference_directory = json_data['Tests Collections'][current_test_collections]['Generate Reference Target'] + '\\' + machine_configs.machine_name + '\\' + json_data['Tests Collections'][current_test_collections]['Branch Target'] + '\\' + current_test_collections + '\\'
        
        #
        helpers.directory_clean_or_make(destination_reference_directory)

        #
        helpers.directory_copy('TestsResults\\' + json_data['Tests Collections'][current_test_collections]['Branch Target'] + '\\' + current_test_collections + '\\', destination_reference_directory)


#
if __name__ == '__main__':
    main()
