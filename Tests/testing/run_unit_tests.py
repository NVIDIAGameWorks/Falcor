'''
Script for running unit tests.
Most of the work is delegated to FalcorTest.
'''

import sys
import argparse
import subprocess

from build_falcor import build_falcor

from core import Environment, config
from core.termcolor import colored

def run_unit_tests(env, filter_regex):
    '''
    Run unit tests by running FalcorTest.
    The optional filter_regex is used to select specific tests to run.
    '''
    args = [str(env.falcor_test_exe)]
    if filter_regex:
        args += ['--filter', str(filter_regex)]

    p = subprocess.Popen(args)
    try:
        p.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
        print('\n\nProcess killed due to timeout')

    success = p.returncode == 0
    status = colored('PASSED', 'green') if success else colored('FAILED', 'red')
    print(f'\nUnit tests {status}.')

    return success

def main():
    available_configs = ', '.join(config.BUILD_CONFIGS.keys())
    parser = argparse.ArgumentParser(description='Utility for running unit tests.')
    parser.add_argument('-c', '--config', type=str, action='store', help=f'Build configuration: {available_configs}', default=config.DEFAULT_BUILD_CONFIG)
    parser.add_argument('-e', '--environment', type=str, action='store', help='Environment', default=config.DEFAULT_ENVIRONMENT)
    parser.add_argument('-f', '--filter', type=str, action='store', help='Regular expression for filtering tests to run')
    parser.add_argument('--skip-build', action='store_true', help='Skip building project before running tests')
    args = parser.parse_args()

    # Load environment.
    try:
        env = Environment(args.environment, args.config)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Build solution before running tests.
    if not args.skip_build:
        if not build_falcor(env):
            print('Build failed. Not running tests.')
            sys.exit(1)

    # Run tests.
    success = run_unit_tests(env, args.filter)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
