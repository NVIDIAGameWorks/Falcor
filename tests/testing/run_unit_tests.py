'''
Script for running unit tests.
Most of the work is delegated to FalcorTest.
'''

import sys
import argparse
import subprocess

from core import Environment, config

from build_falcor import build_falcor

def run_unit_tests(env, category, device_type, filter_regex, xml_report, repeat_count):
    '''
    Run unit tests by running FalcorTest.
    The optional filter_regex is used to select specific tests to run.
    '''
    args = [str(env.falcor_test_exe)]
    if category:
        args += ['--category', str(category)]
    if device_type:
        args += ['--device-type', str(device_type)]
    if filter_regex:
        args += ['--filter', str(filter_regex)]
    if xml_report:
        args += ['--xml-report', str(xml_report)]
    if repeat_count:
        args += ['--repeat', str(repeat_count)]

    p = subprocess.Popen(args)
    try:
        p.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
        print('\n\nProcess killed due to timeout')

    return p.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Utility for running unit tests.')
    parser.add_argument('-c', '--config', type=str, action='store', help=f'Build configuration')
    parser.add_argument('-e', '--environment', type=str, action='store', help='Environment', default=config.DEFAULT_ENVIRONMENT)
    parser.add_argument('--category', type=str, action='store', help='Test categories to run (default: all).')
    parser.add_argument('-d', '--device-type', type=str, action='store', help='Graphics device type.')
    parser.add_argument('-f', '--filter', type=str, action='store', help='Regular expression for filtering tests to run')
    parser.add_argument('-x', '--xml-report', type=str, action='store', help='XML report output file')
    parser.add_argument('-r', '--repeat', type=int, action='store', help='Number of times to repeat the test.')
    parser.add_argument('--skip-build', action='store_true', help='Skip building project before running tests')
    parser.add_argument('--list-configs', action='store_true', help='List available build configurations.')
    args = parser.parse_args()

    # List build configurations.
    if args.list_configs:
        print('Available build configurations:\n' + '\n'.join(config.BUILD_CONFIGS.keys()))
        sys.exit(0)

    # Load environment.
    try:
        env = Environment(args.environment, args.config)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Build before running tests.
    if not args.skip_build:
        if not build_falcor(env):
            print('Failed to build')
            sys.exit(1)

    # Run tests.
    success = run_unit_tests(env, args.category, args.device_type, args.filter, args.xml_report, args.repeat)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
