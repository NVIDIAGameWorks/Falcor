import subprocess
import os
import shutil
import stat
import pprint
from distutils.dir_util import copy_tree
import MachineConfigs as machine_configs

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
    copy_tree(fromDirectory, toDirectory)

def build_html_filename(tests_set):
    if tests_set['Success'] is True:
        header = "[SUCCESS]"
    else:
        header = "[FAILED]"

    return header + tests_set['Name'] + "_Results.html"

