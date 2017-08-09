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

# Default Clone Branch,
gDefaultCloneBranch = 'master';


# Error if we failed to clean or make the correct directory.
class CloneRepoCleanOrMakeError(Exception):
    pass

# Error if we failed to clone the repository.
class CloneRepoCloneError(Exception):
    pass


# Clone the Repository with the specified Arguments.
def clone(repository=gDefaultCloneRepository, branch=gDefaultCloneBranch, destination=os.getcwd()):

   # Create the Destination Directory.
    if helpers.directoryCleanOrMake(destination) != 0 :
        raise CloneRepoCleanOrMakeError("Failed To Clean or Make Directory")

    # Clone the Specified Repository and Branch.
    try: 
        cloneReturnCode = subprocess.call(['git', 'clone', repository, destination, '-b', branch])
        
        # Raise an exception if the subprocess did not run correctly.
        if cloneReturnCode != 0 :
            raise CloneRepoCloneError('Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : ' + destination + ' ')

        return cloneReturnCode 
            
    # Exception Handling.
    except subprocess.CalledProcessError:

        # Raise an exception if the subprocess crashed.
        raise CloneRepoCloneError('Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : '  + destination + ' ')
        
        return None



# Clone a GitHub Repository.
def main():

    # Argument Parser.
    parser = argparse.ArgumentParser()

    # Add the Arguments for the Repository.
    parser.add_argument('-repository', nargs='?', action='store', help='Specify the Repository', default = gDefaultCloneRepository)

    # Add the Arguments for the Branch.
    parser.add_argument('-branch', nargs='?', action='store', help='Specify the Branch', default = gDefaultCloneBranch)

    # Add the Arguments for the directory.
    parser.add_argument('-destination', nargs='?', action='store', help='Specify the Destination', default = os.getcwd())

    # Parse the Arguments.
    args = parser.parse_args()

    # Clone the repository.
    return clone(args.repository, args.branch, args.destination)


if __name__ == '__main__':
    main()
