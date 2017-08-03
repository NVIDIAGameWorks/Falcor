import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import sys

# Helpers.
import Helpers as helpers

# Default Clone Repositories.
gDefaultCloneRepository = 'https://github.com/NVIDIAGameworks/Falcor.git';
gDefaultCloneBranch = 'master';

# Clone the Repository with the specified Arguments.
def clone(repository=gDefaultCloneRepository, branch=gDefaultCloneBranch, destination=os.getcwd()):

   # Create the Destination Directory.
    if helpers.directoryCleanOrMake(destination) != 0 :
        return -1

    # Clone the Specified Repository and Branch.
    try: 
        cloneReturnCode = subprocess.call(['git', 'clone', repository, destination, '-b', branch])

        if cloneReturnCode != 0 :
            print 'Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : ' + destination + ' '

        return cloneReturnCode 
            
    except subprocess.CalledProcessError:
        print 'Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : ' + destination + ' ' 
        return -1



# Clone a GitHub Repository.
def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Arguments for the Repository and the Branch.
    parser.add_argument('-repository', nargs='?', action='store', help='Specify the Repository', default = gDefaultCloneRepository)
    parser.add_argument('-branch', nargs='?', action='store', help='Specify the Branch', default = gDefaultCloneBranch)

    # Add the Arguments for the directory.
    parser.add_argument('-destination', nargs='?', action='store', help='Specify the Destination', default = os.getcwd())

    # Parse the Arguments.
    args = parser.parse_args()

    # Clone the repository.
    return clone(args.repository, args.branch, args.destination)


if __name__ == '__main__':
    main()
