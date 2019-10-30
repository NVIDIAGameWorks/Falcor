import os
import subprocess
import argparse
import Helpers as helpers
import TestConfig as config
from pathlib import Path

def default_comparison(result_image, reference_image):

    result_image = os.path.abspath(result_image)
    reference_image = os.path.abspath(reference_image)
    results_dir = os.path.dirname(result_image)

    # Create the test compare image.
    test_compare_image_filepath = os.path.join(results_dir, os.path.splitext(os.path.basename(result_image))[0] + '_Compare.png')
    
    # Run ImageMagick
    image_compare_command = ['magick', 'compare', '-metric', 'MSE', '-compose', 'Src', '-highlight-color', 'White', '-lowlight-color', 'Black', result_image, reference_image, test_compare_image_filepath]
    
    if os.name == 'nt':
        image_compare_process = subprocess.Popen(image_compare_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    else:
        #don't need "magick" first  or shell=True if on linux
        image_compare_command.pop(0)            
        image_compare_process = subprocess.Popen(image_compare_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    image_compare_result = image_compare_process.communicate()[0]
    
    # Decode if image compare result is a "binary" string
    try:
        image_compare_result = image_compare_result.decode('ascii')
    except AttributeError:
        pass

    success = True

    # Image compare succeeded
    if image_compare_process.returncode <= 1: # 0: Success, 1: Does not match, 2: File not found, or other error?
        result_str = image_compare_result[(image_compare_result.find('(') + 1):image_compare_result.find(')')]
        success = float(result_str) <= config.tolerance
    else:
        print('[FAILED] No output file produced. ')
        success = False

    if success:
        os.remove(test_compare_image_filepath)

    return success

def get_files_to_compare(subdir, file, basedir, otherbase):
    file = os.path.join(subdir, file)
    rel_path = Path(file).relative_to(basedir)
    return file, os.path.join(otherbase, rel_path)


def verify_that_refs_exist(results_dir, reference_dir):
    success = True
    for subdir, dirs, files in os.walk(results_dir):
        for file in files:
            if not helpers.isSupportedImageExt(file):
                continue

            result_file, ref_file = get_files_to_compare(subdir, file, results_dir, reference_dir)

            if (not os.path.exists(ref_file)):
                print('Error: Output file "' + str(result_file) + '" doesn\'t have a reference')
                success = False

    return success

def compare(results_dir, reference_dir):
    if not os.path.isdir(reference_dir):
        print("Can't find reference folder " + reference_dir)
        return False

    success = verify_that_refs_exist(results_dir, reference_dir);

    # make sure that expected references exist for the run tests
    for subdir, dirs, files in os.walk(reference_dir):
        for file in files:
            if not helpers.isSupportedImageExt(file):
                continue
            
            reference_file, result_file = get_files_to_compare(subdir, file, reference_dir, results_dir)
            if (not os.path.exists(result_file)):
                print('Error: Expecting output file "' + str(result_file) + '", but it is missing')
                success = False
                continue

            status = default_comparison(result_file, reference_file)
            if not status:
                # add early out option?
                print('Error: Test failed on comparison between ' +  result_file + ' and reference ' + reference_file)
                success = False

    return success

def main(): 
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    # Add argument for testing directory for render pass tests
    parser.add_argument('-rf', '--ref', action='store', help='Specify the reference directory', required=True)

    # Add argument for testing directory for render pass tests
    parser.add_argument('-r', '--res', action='store', help='Specify the results directory', required=True)

    # Parse the Arguments.
    args = parser.parse_args()
    results = compare(args.res, args.ref)
    if results['Success']:
        print("Test Passed")
    else:
        print("Test Failed")

if __name__ == '__main__':
    main()
