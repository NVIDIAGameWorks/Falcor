'''
Frontend for running unit tests.

This script helps to run the falcor unit tests for different build configurations.
'''

import sys
import argparse
import subprocess

from core import Environment, config
from core.environment import find_most_recent_build_config

def run_unit_tests(env: Environment, args):
    '''
    Run unit tests by running FalcorTest.
    '''
    args = [str(env.falcor_test_exe)] + args

    p = subprocess.Popen(args)
    try:
        p.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
        print('\n\nProcess killed due to timeout')

    return p.returncode == 0

def main():
    default_config = find_most_recent_build_config()

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('-h', '--help', action='store_true', help='Show this help message and exit')
    parser.add_argument('--environment', type=str, action='store', help=f'Environment', default=None)
    parser.add_argument('--config', type=str, action='store', help=f'Build configuration (default: {default_config})', default=default_config)
    parser.add_argument('--list-configs', action='store_true', help='List available build configurations')
    args, passthrough_args = parser.parse_known_args()

    # Try to load environment.
    env = None
    try:
        env = Environment(args.environment, args.config)
    except Exception as e:
        env_error = e

    # Print help.
    if args.help:
        parser.print_help()
        if env:
            print(f"\nAdditional arguments consumed by {config.FALCOR_TEST_EXE}:\n")
            subprocess.call([str(env.falcor_test_exe), '-h'])
        else:
            print(f"\nFailed to load environment: {env_error}")
        sys.exit(0)

    # List build configurations.
    if args.list_configs:
        print('Available build configurations:\n' + '\n'.join(config.BUILD_CONFIGS.keys()))
        sys.exit(0)

    # Abort if environment is missing.
    if env == None:
        print(f"\nFailed to load environment: {env_error}")
        sys.exit(1)

    # Run tests.
    success = run_unit_tests(env, passthrough_args)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
