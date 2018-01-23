import subprocess
import os
import shutil
import stat
import pprint
from distutils.dir_util import copy_tree

# CLean the directory if it exists, or make it if it does not.
def directory_clean_or_make(destination):

    # Check if the Directory exists, and make it if it does not.
    if not os.path.isdir(destination):
        try:
            os.makedirs(destination)
            return 0

        except OSError:
            print("Error trying to Create Directory : " + destination)
            return None

    else:
        try:
            # Clean the Directory.
            shutil.rmtree(destination)
            os.makedirs(destination)
            # Return 0 to indicate didn't except 
            return 0 

        # Exception Handling.
        except subprocess.CalledProcessError:
            print("Error trying to clean Directory : " + destination)
            # Return failure.
            return None

def directory_copy(fromDirectory, toDirectory):
    copy_tree(fromDirectory, toDirectory)

def build_html_filename(tests_set):
    if tests_set['Success'] is True:
        header = "[SUCCESS]"
    else:
        header = "[FAILED]"

    return header + tests_set['Name'] + "_Results.html"

