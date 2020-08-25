'''
Module with various helpers functions.
'''

import os
import re
import subprocess
import time
import socket
from urllib.parse import urlparse

class GitError(Exception):
    pass

def get_git_head_branch(path):
    '''
    Return the git HEAD branch name by reading from .git/HEAD file.
    '''
    try:
        head = open(os.path.join(path, '.git/HEAD')).read()
        # HEAD either contains a reference to refs/heads or a sha1
        return re.search(r'(ref: refs\/heads\/)?(.*)$', head).group(2)
    except (IOError, OSError, AttributeError) as e:
        raise GitError(e)

def get_git_remote_origin(path, remote='origin'):
    '''
    Return the git remote origin by reading from .git/config file.
    '''
    try:
        config = open(os.path.join(path, '.git/config')).read()
        return re.search(r'^\[remote \"%s\"\].*\n.*url = (.*)$' % (remote), config, flags=re.MULTILINE).group(1)
    except (IOError, OSError, AttributeError) as e:
        raise GitError(e)

def get_hostname():
    '''
    Return the hostname.
    '''
    return socket.gethostname()

def get_vcs_root(path):
    '''
    Return the git version control system root (gitlab-master or github).
    '''
    url = get_git_remote_origin(path)
    url = urlparse(url)
    url = url.netloc.split('.')
    for u in url:
        if u.startswith("git@"): u = u.replace("git@", "")
        if u == "gitlab-master" or u == "github": return u
    print("Error. Unknown VCS root `" + url[0] + "`")
    return url[0].lower()

def mirror_folders(src_dir, dst_dir):
    '''
    Mirror contents from src_dir to dst_dir.
    '''
    if os.name != 'nt':
        raise RuntimeError('mirror_folders() is not implemented for this OS')

    args = ["Robocopy.exe", str(src_dir), str(dst_dir), "/MIR", "/FFT", "/Z", "/XA:H", "/W:5", "/np"]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    return process.returncode <= 7, output.decode('ascii')
