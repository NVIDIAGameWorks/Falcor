'''
Module for running MSVC compiler using msbuild
'''

import os
import subprocess
import argparse

ACTIONS=['build', 'rebuild', 'clean', 'deploy']

def build_solution(packman_dir, solution, action, config, version='14.29.30133', project=None):
    '''
    Build a Visual Studio solution file using msbuild.exe.
    '''
    if not action in ACTIONS:
        raise ValueError('Invalid action. Available actions are: %s' % (", ".join(ACTIONS)))

    msvc_path = os.path.normpath(os.path.join(packman_dir, 'msvc-msbuild'))

    msbuild = os.path.normpath(os.path.join(msvc_path, 'MSBuild/Current/bin/msbuild.exe'))

    # Trailing separators are required
    tools_path = os.path.normpath(os.path.join(msvc_path, 'VC/Tools/MSVC', version)) + os.sep
    windows_kits_path = os.path.normpath(os.path.join(packman_dir, 'WindowsKits')) + os.sep

    tools_property_arg = '-property:VCToolsInstallDir=' + tools_path
    configuration_arg = '-property:Configuration=' + config
    windows_sdk_arg = '-property:WindowsSDKDir=' + windows_kits_path

    target_arg = '-t:' + action
    if project:
        target_arg = '-t:' + project + ':' + action

    # Run a build process on every core
    thread_count_arg = '-m'

    args = [msbuild , tools_property_arg, thread_count_arg, target_arg, configuration_arg, str(solution)]
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
    parser.add_argument('-m', '--packman', type=str, action='store', help='Path to packman directory')
    parser.add_argument('-p', '--project', type=str, action='store', help='Project to build')
    parser.add_argument('solution', type=str, action='store', help='Solution file')
    args = parser.parse_args()

    try:
        build_solution(packman_dir=args.packman, solution=args.solution, project=args.project, action=args.action, config=args.config)
    except ValueError as e:
        print(e)

if __name__ == '__main__':
    main()
