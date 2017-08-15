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
import RunTestsCollection as rTC
import Configs as configs
import Helpers as helpers




def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Argument for which Test Collection to use.
    parser.add_argument('-testsCollection', nargs='?', action='store', help='Specify the Test Collection', default='TestsCollectionsAndSets\\TestsCollection.json')

    # Add the Arguments for do not build and for show summary, and whether to run it locally.
    parser.add_argument("-nb", action='store_true', help='Whether or not to build the solutions again.')




if __name__ == '__main__':
    main()
