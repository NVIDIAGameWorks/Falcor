'''
Module for running VS compiler.
'''

import os
import subprocess
import argparse

ACTIONS=['build', 'rebuild', 'clean', 'deploy']

def find_visual_studio_install_path(version):
    '''
    Returns the installation path for a given version of Visual Studio.
    Uses vswhere.exe to query the information.
    '''
    process = subprocess.Popen(['%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe', '-version', '[%d,%d)' % (version, version + 1), '-property', 'installationPath'], shell=True, stdout=subprocess.PIPE)
    path = process.communicate()[0].decode('utf-8').strip()
    if process.returncode != 0 or len(path) == 0:
        raise RuntimeError('Visual Studio (version %d) not found' % (version))
    # vswhere.exe can return the property string twice, so lets just take the first line
    return path.splitlines()[0]

def build_solution(solution, action, config, project=None, visual_studio_version=16):
    '''
    Build a Visual Studio solution file using devenv.com.
    '''
    if not action in ACTIONS:
        raise ValueError('Invalid action. Available actions are: %s' % (", ".join(ACTIONS)))

    vs_path = find_visual_studio_install_path(version=visual_studio_version)
    devenv = os.path.join(vs_path, 'Common7\IDE\devenv.com')
    args = [devenv, str(solution), '/' + action, config]
    if project:
        args += ['/project', str(project)]
    process = subprocess.Popen(args)
    process.wait()
    return process.returncode == 0

def main():
    '''
    Command line interface.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--action', type=str, action='store', help='Build action (build, rebuild, clean)', default='build')
    parser.add_argument('-c', '--config', type=str, action='store', help='Build configuration', default='Release')
    parser.add_argument('-p', '--project', type=str, action='store', help='Project to build')
    parser.add_argument('solution', type=str, action='store', help='Solution file')
    args = parser.parse_args()

    try:
        build_solution(solution=args.solution, project=args.project, action=args.action, config=args.config)
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()
