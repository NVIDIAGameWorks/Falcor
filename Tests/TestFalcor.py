import os
import sys
import shutil
import argparse
import TestConfig as testConfig
import Helpers as helpers
import GraphTests as graphTester
import MachineConfigs as machine_configs
import socket

def run_ults(config):
    try:
        ult_exe = helpers.findExecutable(config, testConfig.ultExe)
        print("Running unit-tests from " + ult_exe)
        ret = helpers.runProcessAsync([ult_exe])
        if len(ret):
            print(ret)
        else:
            print("Unit-tests passed") 
    except BaseException as e:
        print('Error when trying to run unit tests. ' + str(e))

def save_results(args):
    if not os.path.isdir(testConfig.localTestDir):
        print("No results were generated, nothing to upload")
        return

    ref = helpers.buildRefSubFolder(args)
    ref = os.path.join(ref, args.build_id)
    ref = os.path.join(machine_configs.results_cache_directory, ref)
    print("Saving test results to " + ref)
    helpers.rmdir(ref)
    shutil.copytree(testConfig.localTestDir, ref)


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_config', action='store', help='Build configuration for test. ReleaseD3D12 by default', default=testConfig.defaultConfig)
    parser.add_argument('--rebuild', action='store_true', help='Force solution rebuild')
    parser.add_argument('--dont_build', action='store_true', help='Don\'t build the solution. If the solution wasn\'t built, the test will fail' )
    parser.add_argument('--save_results', action='store_true', help='Save the results to a local directory')
    parser.add_argument('--build_id', action='store', help='TeamCity build ID', default="")
    parser.add_argument('--branch_name', action='store', help='Name of the current checkout branch', default=helpers.getGitBranchName(".."))
    parser.add_argument('--machine_name', action='store', help='Optional sub folder name within references directory', default=socket.gethostname());
    parser.add_argument('--clean_packman', action='store_true', help='Delete the packman repository')
    parser.add_argument('--vcs_root', action='store', help='The VCS root folder', default=helpers.getVcsRoot(".."));
    return parser.parse_args()

def main():
    success = True
    try:
        args = prepare_args()
        print("Working on GIT branch '" + args.branch_name + "'")
        
        if(args.clean_packman):
            helpers.deletePackmanRepo()

        if not args.dont_build:
            helpers.buildSolution("..", "falcor", args.build_config, args.rebuild)
        
        run_ults(args.build_config)
        remote_ref_dir = os.path.join(machine_configs.remote_reference_directory, helpers.buildRefSubFolder(args)) 
        local_ref_dir = os.path.join(machine_configs.local_reference_directory, helpers.buildRefSubFolder(args))
        print("Mirroring references")
        helpers.mirror_folders(remote_ref_dir, local_ref_dir)
        print("Running graph tests and comparing against " + local_ref_dir)
        success = graphTester.test_all_graphs("../Source", testConfig.localTestDir, "Testing", args.build_config, local_ref_dir)
        if not success: print("Graph tests failed")

    except Exception as e:
        print("TestFalcor failed. " + str(e))
        success = False

    finally:
        if args.save_results:
            save_results(args)
    if success:
        print("All tests passed")

    if not success: sys.exit(-1)

if __name__ == '__main__':
        main()
