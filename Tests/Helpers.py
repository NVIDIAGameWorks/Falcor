import subprocess
import os
import shutil
import stat


# CLean the directory if it exists, or make it if it does not.
def directoryCleanOrMake(destination) :
    
    # Check if the Directory exists, and make it if it does not.
    if not os.path.isdir(destination):
        try:
            os.makedirs(destination)
            return 0

        except OSError:
            print "Error trying to Create Directory : " + destination
            return None

    else:
        try:
            # Create the arguments.
            batchArgs = ["RemoveDirectoryTree.bat ", destination]

            # Clean the Directory.
            removeDirectoryReturnCode = subprocess.call(batchArgs)

            # Check if it was success.            
            if removeDirectoryReturnCode != 0:
                print "Error trying to clean Directory : " + destination

            # Return the return code.
            return removeDirectoryReturnCode

        # Exception Handling.
        except subprocess.CalledProcessError:
            
            print "Error trying to clean Directory : " + destination

            # Return failure.
            return None




    