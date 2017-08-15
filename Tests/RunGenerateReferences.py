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
    # tests_collections_results = rTC.run_tests_collections(json_data)

    # verify_all_ran_successfully = rTC.verify_tests_collections_results(json_data)

    helpers.directory_clean_or_make(json_data['Generate Reference Target'] + '\\' + machine_configs.machine_name + '\\')

    # 
    pp = pprint.PrettyPrinter(indent=4)

    # pp.pprint(tests_collections_results)



if __name__ == '__main__':
    main()
