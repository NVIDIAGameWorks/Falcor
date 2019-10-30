import subprocess
import os
import shutil
import stat
import pprint
import time
from distutils.dir_util import copy_tree
import shutil
import TestConfig as testConfig
import MachineConfigs as machine_configs
from pathlib import Path
from urllib.parse import urlparse

def getExeDirectory(configuration):
    exeDir = os.path.join(machine_configs.default_main_dir, 'Bin')

    if os.name == 'nt':
        exeDir = os.path.join(exeDir, 'x64')
        if configuration.lower() == 'released3d12' or configuration.lower() == 'releasevk' :
            config = 'Release'
        else:
            config = 'Debug'
        return os.path.join(exeDir, config)
    else:
        return exeDir

def findExecutable(config, exe):
    exePath = os.path.join(getExeDirectory(config), exe)
    if not os.path.isfile(exePath):
        raise FileNotFoundError("Can't find the exe file `" + exe + "`")
    return exePath

# Error if we failed to clean or make the correct directory.
class CloneRepoCleanOrMakeError(Exception):
    pass

# Error if we failed to clone the repository.
class CloneRepoCloneError(Exception):
    pass

def mkdir(dir):
    dir = Path(dir)
    if not dir.is_dir():
        os.makedirs(str(dir))

def rmdir(dir):
    dir = Path(dir)
    if dir.is_dir():
        shutil.rmtree(dir, ignore_errors=True)
    
# Clean the directory if it exists, or make it if it does not.
def cleanDir(dir):
    rmdir(dir)
    mkdir(dir)

# Clone the Repository with the specified Arguments.
def clone(repository, branch, destination):

   # Create the Destination Directory.
    if cleanDir(destination) != 0 :
        raise CloneRepoCleanOrMakeError("Failed To Clean or Make Directory")

    # Clone the Specified Repository and Branch.
    try: 
        errCode = subprocess.call(['git', 'clone', repository, destination, '-b', branch])
        
        # Raise an exception if the subprocess did not run correctly.
        if errCode != 0 :
            raise CloneRepoCloneError('Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : ' + destination + ' ')

        return errCode 
            
    # Exception Handling.
    except subprocess.CalledProcessError:
        # Raise an exception if the subprocess crashed.
        raise CloneRepoCloneError('Error Cloning Repository : ' + repository + ' Branch : ' + branch + ' Destination : '  + destination + ' ')
        

def openFolderInExplorer(folder):
    if os.path.isdir(folder):
        if os.name == 'nt':
            subprocess.call('explorer.exe ' + folder )
        else:
            subprocess.call('nautilus --browser ' + folder )


class GitError(Exception):
    pass
        
# get branch name without having to store it in a config or require the user to install pygit
def getGitBranchName(baseDir):
    try:
        gitFile = open(os.path.join(baseDir, '.git/HEAD' ))
        gitFileStr = gitFile.readline()
    except (IOError, OSError) as e:
        raise GitError(e.args)
            
    if gitFileStr.find('ref: ') > -1:
        gitFileStr = gitFileStr[5 : len(gitFileStr)]
        return gitFileStr[gitFileStr.rfind('/') + 1 : len(gitFileStr) - 1]
        
    return machine_configs.default_reference_branch_name

# get branch name without having to store it in a config or require the user to install pygit
def getGitUrl(baseDir):
    try:
        gitFile = open(os.path.join(baseDir, '.git/config' ))
        gitFileStr = gitFile.read()
    except (IOError, OSError) as e:
        raise GitError(e.args)
    
    ref = gitFileStr.find('url = ')
    if ref > -1:
        rest = gitFileStr[ref:len(gitFileStr)]
        gitUrl = rest[6 : rest.find('\n')]
        return gitUrl
        
    return machine_configs.default_reference_url

def getVcsRoot(baseDir):
    url = getGitUrl(baseDir)
    url = urlparse(url)
    url = url.netloc.split('.')
    for u in url:
        if u.startswith("git@"): u = u.replace("git@", "")
        if u == "gitlab-master" or u == "github": return u
    print("Error. Unknown VCS root `" + url[0] + "`")
    return url[0].lower()


# Error if we failed to build the solution.
class BuildSolutionError(Exception):
    pass
    
def buildSolution(slnDir, slnFile, config, rebuild):
    if os.name == 'nt':
        winBuildScript = "BuildSolution.bat"
        try:
            # Build the Batch Args.
            buildType = "build"
            if rebuild:
                buildType = "rebuild"
            slnPath = Path(slnDir) / Path(slnFile + ".sln")
            batchArgs = [winBuildScript, buildType, str(slnPath), config.lower()]

            # Build Solution.
            if subprocess.call(batchArgs) == 0:
                return 0
            else:
                raise Exception()
            
        except Exception:
            raise BuildSolutionError("Error building solution : " + str(slnPath) + " with configuration : " + config.lower())
    else:
        prevDir = os.getcwd()
        #Call Makefile
        os.chdir(cloned_dir)
        subprocess.call(['make', 'PreBuild', '-j8', '-k'])
        subprocess.call(['make', 'All', '-j24', '-k','TESTS=\'-D _TEST_\''])
        os.chdir(prevDir)
            
def isSupportedImageExt(file):
    for ext in testConfig.imageExtensions:
        if file.endswith(ext):
            return True
    
    return False

def deletePackmanRepo():
    if os.path.isdir(machine_configs.packman_repo):
        print("Deleting the packman repository")
        rmdir(machine_configs.packman_repo)

def dispatchEmail(subject, attachments):
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
            
def copyDir(fromDirectory, toDirectory):
    print('Copying directory ' + fromDirectory + ' to ' + toDirectory)
    
    try:
        for subdir, dirs, files in os.walk(fromDirectory):
            relPath = subdir[len(fromDirectory) + 1 : len(subdir)]
            to_path = os.path.join(toDirectory, relPath)
            if not os.path.isdir(to_path):
                os.mkdir(to_path)
            for file in files:
                if not os.path.exists(os.path.join(to_path, file)):
                    shutil.copyfile(os.path.join(subdir, file), os.path.join(to_path, file))
    except IOError:
        
        print('Failed to copy reference files to server. Please check local directory.')
        return

def createShortcut(source_path, link_file_path):
    if os.name == 'nt':
        # creates a junction to avoid requiring admin rights because of windows
        subprocess.call(['mklink', link_file_path, source_path], shell=True)
    else:
        os.symlink(source_path, link_file_path)

def buildHtmlFilename(tests_sets, configuration):
    header = "[SUCCESS]"
    for tests_set_key in tests_sets.keys():
        if tests_sets[tests_set_key]['Success'] is False:
            header = "[FAILED]"
            break

    return header + configuration + "_Results.html"

class ProcessFailed(RuntimeError):
    pass

class ProcessTimedoutError(Exception):
    pass

def runProcessAsync(cmdArgs):
    try:
        process = subprocess.Popen(cmdArgs, stderr = subprocess.PIPE, stdout = subprocess.PIPE)
        startTime = time.time()
    
        # Wait for the process to finish.
        while process.returncode is None:
            process.poll()

            now = time.time()
            diffTime = now - startTime
            # If the process has taken too long, kill it.
            if diffTime > machine_configs.machine_process_default_kill_time:
                process.kill()
                raise ProcessTimedoutError("Process ran for too long, had to kill it. Please verify that the program finishes within its hang time, and that it does not crash")
                break

        e = "Process log:\n"
        for string in process.stderr:
            e += str(string.decode())

        if process.returncode == 0:
            return "Process " + cmdArgs[0] + " finished. " + e

        else:
            e = "Process " + cmdArgs[0] + " failed with error " + str(process.returncode) + ". " + e
            raise ProcessFailed(e)


    except(NameError, IOError, OSError) as e:
        print(e.args)
        raise RuntimeError('Error when trying to run "' + cmdArgs + '"')

def buildRefSubFolder(args):
    ref = os.path.join(args.vcs_root, os.path.join(args.machine_name, os.path.join(args.branch_name, args.build_config)))
    return ref

def mirror_folders(source, dst):
    if not os.name == 'nt':
        raise RuntimeError("mirror_folders() is not implemented for this OS")
    robocopy = ["Robocopy.exe", source, dst, "/MIR", "/FFT", "/Z", "/XA:H", "/W:5", "/LOG:robocopy.txt", "/np"]
    try:
        subprocess.check_call(robocopy)
    except(subprocess.CalledProcessError) as e:
        if e.returncode > 7:
            raise RuntimeError("Mirroring folders failed")
