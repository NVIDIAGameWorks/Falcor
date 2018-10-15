import subprocess
import os
import shutil
import stat
import pprint
from distutils.dir_util import copy_tree
import MachineConfigs as machine_configs

def get_executable_directory(configuration, test_set, runAsCollection):
    if runAsCollection:
        exe_dir = os.path.join(test_set, 'Bin')
    else:
        exe_dir = 'Bin'
        
    if os.name == 'nt':
        exe_dir = os.path.join(exe_dir, 'x64')
        if configuration.lower() == 'released3d12' or configuration.lower() == 'releasevk' :
            config = 'Release'
        else:
            config = 'Debug'
        return os.path.join(exe_dir, config)
    else:
        return exe_dir
        
# Error if we failed to clean or make the correct directory.
class CloneRepoCleanOrMakeError(Exception):
    pass

# Error if we failed to clone the repository.
class CloneRepoCloneError(Exception):
    pass

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
            remove_directory_return_code = 0
            if os.name == 'nt':
                # Create the arguments.
                batch_args = ["RemoveDirectoryTree.bat ", destination]
                # Clean the Directory.
                remove_directory_return_code = subprocess.call(batch_args)

                # Check if it was success.
                if remove_directory_return_code != 0:
                    print("Error trying to clean Directory : " + destination)
            else:
                # Clean the Directory.
                shutil.rmtree(destination)
            os.makedirs(destination)
            return remove_directory_return_code
            
        # Exception Handling.
        except subprocess.CalledProcessError:
            print("Error trying to clean Directory : " + destination)
            # Return failure.
            return None

# Clone the Repository with the specified Arguments.
def clone(repository, branch, destination):

   # Create the Destination Directory.
    if directory_clean_or_make(destination) != 0 :
        raise CloneRepoCleanOrMakeError("Failed To Clean or Make Directory")

    # Clone the Specified Repository and Branch.
    try: 
        clone_return_code = subprocess.call(['git', 'clone', repository, destination, '-b', branch])
        
        # Raise an exception if the subprocess did not run correctly.
        if clone_return_code != 0 :
            raise CloneRepoCloneError('Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : ' + destination + ' ')

        return clone_return_code 
            
    # Exception Handling.
    except subprocess.CalledProcessError:
        # Raise an exception if the subprocess crashed.
        raise CloneRepoCloneError('Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : '  + destination + ' ')
        

# get branch name without having to store it in a config or require the user to install pygit
def get_git_branch_name(base_dir):
    try:
        git_file = open(os.path.join(base_dir, '.git/HEAD' ))
        git_file_string = git_file.readline()
    except (IOError, OSError) as e:
        raise PassTestsError(e.args)
        
    print(git_file_string)
    
    if git_file_string.find('ref: ') > -1:
        git_file_string = git_file_string[5 : len(git_file_string)]
        return git_file_string[git_file_string.rfind('/') + 1 : len(git_file_string) - 1]
        
    return machine_configs.default_reference_branch_name

# Error if we failed to build the solution.
class BuildSolutionError(Exception):
    pass
    
def build_solution(cloned_dir, relative_solution_filepath, configuration, rebuild):
    if os.name == 'nt':
        windows_build_script = "BuildSolution.bat"
        try:
            # Build the Batch Args.
            buildType = "build"
            if rebuild:
                buildType = "rebuild"
            batch_args = [windows_build_script, buildType, relative_solution_filepath, configuration.lower()]

            # Build Solution.
            if subprocess.call(batch_args) == 0:
                return 0
            else:
                raise BuildSolutionError("Error building solution : " + relative_solution_filepath + " with configuration : " + configuration.lower())

        except subprocess.CalledProcessError as subprocess_error:
            raise BuildSolutionError("Error building solution : " + relative_solution_filepath + " with configuration : " + configuration.lower())
    else:
        prevDir = os.getcwd()
        #Call Makefile
        os.chdir(cloned_dir)
        subprocess.call(['make', 'PreBuild', '-j8'])
        subprocess.call(['make', 'All', '-j8'])
        os.chdir(prevDir)
            
            
def dispatch_email(subject, attachments):
    dispatcher = 'NvrGfxTest@nvidia.com'
    recipients = str(open(machine_configs.machine_email_recipients, 'r').read())

    if os.name == 'nt':
        subprocess.call(['blat.exe', '-install', 'mail.nvidia.com', dispatcher])
        command = ['blat.exe', '-to', recipients, '-subject', subject, '-body', "   "]
        for attachment in attachments:
            command.append('-attach')
            command.append(attachment)
    else:
        command = ['sendEmail', '-s', 'mail.nvidia.com', '-f', 'nvrgfxtest@nvidia.com', '-t', recipients, '-u', subject, '-m', '    ', '-o', 'tls=no' ]
        command.append('-a')
        for attachment in attachments:
            command.append(attachment)
    subprocess.call(command)    
            
def directory_copy(fromDirectory, toDirectory):
    print('Copying directory ' + fromDirectory + ' to ' + toDirectory)
    
    try:
        copy_tree(fromDirectory, toDirectory)
    except IOError:
                print('Failed to copy reference files to server. Please check local directory.')
                return

def build_html_filename(tests_set, configuration):
    if tests_set['Success'] is True:
        header = "[SUCCESS]"
    else:
        header = "[FAILED]"

    return header + configuration + "_Results.html"

