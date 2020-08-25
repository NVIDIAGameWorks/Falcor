'''
Script for running image tests.
'''

import os
import sys
import re
import json
import time
import datetime
import argparse
import subprocess
import shutil
from pathlib import Path
from enum import Enum

from build_falcor import build_falcor

from core import Environment, helpers, config
from core.termcolor import colored


def read_header(script_file):
    '''
    Check if script has a IMAGE_TEST dictionary defined at the top and return it's content.
    '''
    with open(script_file) as f:
        script = f.read()
        # Find IMAGE_TEST={} at the top of the script
        m = re.match(r'IMAGE_TEST\s*=\s*({.*})', script, re.DOTALL)
        if m:
            header = None
            # Match curly braces
            depth = 0
            for i, c in enumerate(m.group(1)):
                if c == '{':
                    depth += 1
                if c == '}':
                    depth -= 1
                    if depth == 0:
                        header = m.group(1)[0:i+1]
                        break
            if depth != 0:
                raise Exception(f'Failed to parse script header in {script_file} (curly braces do not match)')
            # Evaluate dictionary
            if header:
                try:
                    return eval(header)
                except Exception as e:
                    raise Exception(f'Failed to parse script header in {script_file} ({e})')

    return {}

class Test:
    '''
    Represents a single image test.
    '''

    class Result(Enum):
        PASSED = 1
        FAILED = 2
        SKIPPED = 3

    COLORED_RESULT_STRING = {
        Result.PASSED: colored('PASSED', 'green'),
        Result.FAILED: colored('FAILED', 'red'),
        Result.SKIPPED: colored('SKIPPED', 'yellow')
    }

    RESULT_STRING = {
        Result.PASSED: 'PASSED',
        Result.FAILED: 'FAILED',
        Result.SKIPPED: 'SKIPPED'
    }

    def __init__(self, script_file, root_dir):
        self.script_file = script_file

        # Test directory relative to root directory.
        self.test_dir = self.script_file.relative_to(root_dir).with_suffix('')

        # Test name derived from test directory.
        self.name = str(self.test_dir.as_posix())

        # Read script header.
        try:
            self.header = read_header(script_file)
        except Exception as e:
            print(e)
            sys.exit(1)

        # Get skipped tests.
        self.skip_message = self.header.get('skipped', None)
        self.skipped = self.skip_message != None

        # Get tolerance.
        self.tolerance = self.header.get('tolerance', config.TOLERANCE)

    def __repr__(self):
        return f'Test(name={self.name},script_file={self.script_file})'

    def collect_images(self, image_dir):
        '''
        Collect all reference and result images in a directory.
        Returns paths relative to the given directory.
        '''
        files = image_dir.iterdir()
        files = map(lambda f: f.relative_to(image_dir), files)
        files = filter(lambda f: not str(f).endswith(config.ERROR_IMAGE_SUFFIX), files)
        files = filter(lambda f: f.suffix.lower() in config.IMAGE_EXTENSIONS, files)
        return list(files)

    def generate_images(self, output_dir, mogwai_exe):
        '''
        Run Mogwai to generate a set of images and store them in output_dir.
        Returns a tuple containing the result code and a list of messages.
        '''
        # Bail out if test is skipped.
        if self.skipped:
            return Test.Result.SKIPPED, [self.skip_message] if self.skip_message != '' else []

        # Determine full output directory.
        output_dir = output_dir / self.test_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # In order to simplify module imports in test scripts, we run Mogwai with
        # a working directory set to the directory the test script resides in.
        # The working directory also specifies the python search path.
        # Here we determine the working directory and a function to resolve paths
        # to be relative to the working directory.
        cwd = self.script_file.parent
        relative_to_cwd = lambda p: os.path.relpath(p, cwd)

        # Write helper script to run test.
        generate_file = output_dir / 'generate.py'
        with open(generate_file, 'w') as f:
            f.write(f'fc.outputDir = r"{output_dir}"\n')
            f.write(f'm.script(r"{relative_to_cwd(self.script_file)}")\n')

        # Run Mogwai to generate images.
        args = [
            str(mogwai_exe),
            '--script', str(relative_to_cwd(generate_file)),
            '--logfile', str(output_dir / 'log.txt'),
            '--silent'
        ]
        p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            outs, errs = p.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            p.kill()
            return Test.Result.FAILED, ['Process killed due to timeout']

        # Check for success.
        if p.returncode != 0:
            # Generate list of errors from stderr.
            errors = list(map(lambda l: l.rstrip(), errs.decode('utf-8').splitlines()))
            return Test.Result.FAILED, errors + [f'{mogwai_exe} exited with return code {p.returncode}']

        # Bail out if no images have been generated.
        if len(self.collect_images(output_dir)) == 0:
            return Test.Result.FAILED, ['Test did not generate any images.']

        return Test.Result.PASSED, []

    def compare_image(self, ref_file, result_file, error_file, tolerance, image_compare_exe):
        '''
        Compare two images using ImageCompare.
        Returns a tuple containing the a boolean to indicate success and the measured error.
        '''
        args = [str(image_compare_exe), '-m', 'mse', '-t', str(tolerance), str(ref_file), str(result_file)]
        if error_file:
            args += ['-e', str(error_file)]
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = process.communicate()[0]
        error = float(output.strip())
        return process.returncode == 0, error

    def compare_images(self, ref_dir, result_dir, image_compare_exe):
        '''
        Run ImageCompare on a set of images in ref_dir and result_dir.
        Checks if error between reference and result image is within a given tolerance.
        Returns a tuple containing the result code, a list of messages and a list of image reports.
        '''
        # Bail out if test is skipped.
        if self.skipped:
            return Test.Result.SKIPPED, [self.skip_message] if self.skip_message != '' else [], []

        # Determine full directory paths for references and results.
        ref_dir = ref_dir / self.test_dir
        result_dir = result_dir / self.test_dir

        # Make sure directories exist.
        if not ref_dir.exists():
            return Test.Result.FAILED, [f'Reference directory "{ref_dir}" does not exist.'], []
        elif not result_dir.exists():
            return Test.Result.FAILED, [f'Result directory "{result_dir}" does not exist.'], []

        # Collect all reference and result images.
        ref_images = self.collect_images(ref_dir)
        result_images = self.collect_images(result_dir)

        # Bail out if no images have been generated.
        if len(result_images) == 0:
            return Test.Result.FAILED, ['Test did not generate any images.'], []

        result = Test.Result.PASSED
        messages = []
        image_reports = []

        # Compare every result image with the corresponding reference image and report missing references.
        for image in result_images:
            if not image in ref_images:
                result = Test.Result.FAILED
                messages.append(f'Test has generated image "{image}" with no corresponding reference image.')
                continue

            ref_file = ref_dir / image
            result_file = result_dir / image
            error_file = result_dir / (str(image) + config.ERROR_IMAGE_SUFFIX)
            compare_success, compare_error = self.compare_image(ref_file, result_file, error_file, self.tolerance, image_compare_exe)
            if not compare_success:
                result = Test.Result.FAILED
                messages.append(f'Test image "{image}" failed with error {compare_error}.')

            image_reports.append({
                'name': str(image),
                'success': compare_success,
                'error': compare_error,
                'tolerance': self.tolerance
            })

        # Report missing result images for existing reference images.
        for image in ref_images:
            if not image in result_images:
                result = Test.Result.FAILED
                messages.append(f'Test has not generated an image for the corresponding reference image "{image}".')

        return result, messages, image_reports

    def run(self, compare_only, ref_dir, result_dir, mogwai_exe, image_compare_exe):
        '''
        Run the image test.
        First, result images are generated (unless compare_only is True).
        Second, result images are compared against reference images.
        Third, writes a JSON report to the result_dir containing details on the test run.
        Returns a tuple containing the result code and a list of messages.
        '''
        # Setup report.
        report = {
            'name': self.name,
            'ref_dir': str(ref_dir / self.test_dir),
            'images': []
        }

        start_time = time.time()
        result = Test.Result.PASSED
        messages = []

        # Generate results images.
        if not compare_only:
            result, messages = self.generate_images(result_dir, mogwai_exe)

        # Compare to references.
        if result == Test.Result.PASSED:
            result, messages, report['images'] = self.compare_images(ref_dir, result_dir, image_compare_exe)

        # Finish report.
        report['result'] = Test.RESULT_STRING[result]
        report['messages'] = messages
        report['duration'] = time.time() - start_time

        # Write JSON report.
        report_dir = result_dir / self.test_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / 'report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)

        return result, messages

def generate_refs(env, tests, ref_dir):
    '''
    Computes references for a set of tests and stores them into ref_dir.
    '''
    print(f'Reference directory: {ref_dir}')
    print(f'Generating references for {len(tests)} tests')

    # Remove existing references.
    if ref_dir.exists():
        shutil.rmtree(ref_dir, ignore_errors=True)

    success = True

    for test in tests:
        print(f'  {test.name:<60} : ', end='', flush=True)

        start_time = time.time()
        result, messages = test.generate_images(ref_dir, env.mogwai_exe)
        elapsed_time = time.time() - start_time

        if result == Test.Result.FAILED:
            success = False

        # Print result and messages.
        status = Test.COLORED_RESULT_STRING[result]
        print(f'{status} ({elapsed_time:.1f} s)')
        for message in messages:
            print(f'    {message}')

    status = colored('PASSED', 'green') if success else colored('FAILED', 'red')
    print(f'\nGenerating references {status}.')

    return success

def run_tests(env, tests, compare_only, ref_dir, result_dir):
    '''
    Runs a set of tests, stores them into result_dir and compares them to ref_dir.
    '''
    print(f'Result directory: {result_dir}')
    print(f'Reference directory: {ref_dir}')
    print(f'Running {len(tests)} tests')

    success = True
    run_date = datetime.datetime.now()
    run_start_time = time.time()

    for test in tests:
        print(f'  {test.name:<60} : ', end='', flush=True)

        start_time = time.time()
        result, messages = test.run(compare_only, ref_dir, result_dir, env.mogwai_exe, env.image_compare_exe)
        elapsed_time = time.time() - start_time

        if result == Test.Result.FAILED:
            success = False

        # Print result and messages.
        status = Test.COLORED_RESULT_STRING[result]
        print(f'{status} ({elapsed_time:.1f} s)')
        for message in messages:
            print(f'    {message}')

    status = colored('PASSED', 'green') if success else colored('FAILED', 'red')
    print(f'\nImage tests {status}.')

    # Setup report.
    report = {
        'date': run_date.isoformat(),
        'result': 'PASSED' if success else 'FAILED',
        'tests': [t.name for t in tests],
        'duration': time.time() - run_start_time
    }

    # Write JSON report.
    report_file = result_dir / 'report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)

    return success

def list_tests(tests):
    '''
    Print a list of tests.
    '''
    print(f'Found {len(tests)} tests')
    for test in tests:
        print(f'  {test.name}')

def collect_tests(root_dir, filter_regex):
    '''
    Collect a list of all tests found in root_dir that are matching the filter_regex.
    A test script needs to be named test_*.py to be detected.
    '''
    print(root_dir)
    # Find all script files.
    script_files = list(root_dir.glob('**/test_*.py'))

    # Filter using regex.
    if filter_regex != '':
        regex = re.compile(filter_regex or '')
        script_files = list(filter(lambda f: regex.search(str(f)) != None, script_files))

    tests = list(map(lambda f: Test(f, root_dir), script_files))
    return tests

def push_refs(ref_dir, remote_ref_dir):
    '''
    Pushes reference images from ref_dir to remote_ref_dir.
    '''
    print(f'Pushing reference images to {remote_ref_dir} : ', end='', flush=True)
    success, log = helpers.mirror_folders(ref_dir, remote_ref_dir)
    print(colored('OK', 'green') if success else colored('FAILED', 'red'))
    if not success:
        print(log)
    return success

def pull_refs(remote_ref_dir, ref_dir):
    '''
    Pulls reference images from remote_ref_dir to ref_dir.
    '''
    print(f'Pulling reference images from {remote_ref_dir} : ', end='', flush=True)
    success, log = helpers.mirror_folders(remote_ref_dir, ref_dir)
    print(colored('OK', 'green') if success else colored('FAILED', 'red'))
    if not success:
        print(log)
    return success

def main():
    available_configs = ', '.join(config.BUILD_CONFIGS.keys())
    parser = argparse.ArgumentParser(description="Utility for running image tests.")
    parser.add_argument('-c', '--config', type=str, action='store', help=f'Build configuration: {available_configs}', default=config.DEFAULT_BUILD_CONFIG)
    parser.add_argument('-e', '--environment', type=str, action='store', help='Environment', default=config.DEFAULT_ENVIRONMENT)
    parser.add_argument('-l', '--list', action='store_true', help='List available tests')
    parser.add_argument('-f', '--filter', type=str, action='store', help='Regular expression for filtering tests to run')
    parser.add_argument('-b', '--ref-branch', help='Reference branch to compare against (defaults to master branch)', default='master')
    parser.add_argument('--compare-only', action='store_true', help='Compare previous results against references without generating new images')
    parser.add_argument('--gen-refs', action='store_true', help='Generate reference images instead of running tests')
    parser.add_argument('--skip-build', action='store_true', help='Skip building project before running')

    additional_group = parser.add_argument_group('extended arguments ', 'Additional options used for testing pipelines on TeamCity.')
    additional_group.add_argument('--pull-refs', action='store_true', help='Pull reference images from remote before running tests')
    additional_group.add_argument('--push-refs', action='store_true', help='Push reference images to remote after generating them')
    additional_group.add_argument('--build-id', action='store', help='TeamCity build ID', default='unknown')

    args = parser.parse_args()

    # Load environment.
    try:
        env = Environment(args.environment, args.config)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Build solution before running tests.
    if not (args.skip_build or args.list):
        if not build_falcor(env):
            print('Build failed. Not running tests.')
            sys.exit(1)

    # Collect tests to run.
    tests = collect_tests(env.image_tests_dir, args.filter)

    if args.list:
        # List available tests.
        list_tests(tests)
    elif args.gen_refs:
        # Generate references.
        ref_dir = env.resolve_image_dir(env.image_tests_ref_dir, env.branch, args.build_id)
        if not generate_refs(env, tests, ref_dir):
            sys.exit(1)

        # Push references to remote.
        if args.push_refs:
            if not env.image_tests_remote_ref_dir:
                print("Remote reference directory is not configured for this environment.")
                sys.exit(1)
            remote_ref_dir = env.resolve_image_dir(env.image_tests_remote_ref_dir, env.branch, args.build_id)
            if not push_refs(ref_dir, remote_ref_dir):
                sys.exit(1)
    else:
        # Determine result and reference directories.
        result_dir = env.resolve_image_dir(env.image_tests_result_dir, env.branch, args.build_id)
        ref_dir = env.resolve_image_dir(env.image_tests_ref_dir, args.ref_branch, args.build_id)

        # Pull references from remote.
        if args.pull_refs:
            if not env.image_tests_remote_ref_dir:
                print("Remote reference directory is not configured for this environment.")
                sys.exit(1)
            remote_ref_dir = env.resolve_image_dir(env.image_tests_remote_ref_dir, args.ref_branch, args.build_id)
            if not pull_refs(remote_ref_dir, ref_dir):
                sys.exit(1)

        # Give some instructions on how to acquire reference images if not available.
        if not ref_dir.exists():
            print(ref_dir)
            print(colored(f'\n!!! Reference images for "{args.ref_branch}" branch are not available !!!', 'red'))
            print('')
            print(f'You have the following options:')
            print('')
            print(f'  1. Checkout "{args.ref_branch}" branch and generate reference images using:')
            print('')
            print(f'       run_image_tests --gen-refs')
            print('')
            print(f'  2. Use references from a different branch using:')
            print('')
            print(f'       run_image_tests --ref-branch BRANCH')
            print('')
            sys.exit(1)

        # Run tests.
        if not run_tests(env, tests, args.compare_only, ref_dir, result_dir):
            sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
