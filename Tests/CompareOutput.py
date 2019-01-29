import os
import subprocess

import Helpers as helpers
import InternalConfig as iConfig

def default_comparison(result_image, result_image_dir, reference_image, screen_captures_results):

    # Create the test compare image.
    test_compare_image_filepath = os.path.join(result_image_dir, os.path.splitext(os.path.basename(result_image))[0] + '_Compare.png')
    
    # Run ImageMagick
    image_compare_command = ['magick', 'compare', '-metric', 'MSE', '-compose', 'Src', '-highlight-color', 'White', '-lowlight-color', 'Black', result_image, reference_image, test_compare_image_filepath]
    
    print('Comparison Test: Source Image: ' + result_image + '\nReference Image: ' + reference_image + '\nOutput Image: ' + test_compare_image_filepath + '\n')
    
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
    
    # Keep the Return Code and the Result.
    result = {}
    
    # Image compare succeeded
    if image_compare_process.returncode <= 1: # 0: Success, 1: Does not match, 2: File not found, or other error?
        result_str = image_compare_result[:image_compare_result.find(' ')]
        result['Compare Result'] = result_str
        result['Test Passed'] = float(result_str) <= iConfig.TestConfig['Tolerance']
        
        # if result['Test Passed'] == False:
        #     print('[FAILED] Comparision above tolerance. Difference was ' + result_str + ' on image ' + result_image)
        
    # Error
    else:
        print('[FAILED] No output file produced. ')
        result['Compare Result'] = "Error"
        result['Test Passed'] = False
    
    result['Return Code'] = image_compare_process.returncode
    result['Source Filename'] = os.path.abspath(result_image)
    result['Reference Filename'] = reference_image
    result['Comparison Filename'] = os.path.abspath(test_compare_image_filepath)
    
    screen_captures_results['Success'] &= result['Test Passed']
    if not result_image_dir in screen_captures_results.keys():
        screen_captures_results[result_image_dir] = []
    screen_captures_results[result_image_dir].append(result)
    
    return result['Test Passed']


# default image comparison
def default_compare(result_file, result_file_dir, reference_file, data):
    result = default_comparison(result_file, result_file_dir, reference_file, data)
    return result;
    

def compare_all_images(results_dir, reference_dir, comparison_func):
    screen_captures_results = {}
    screen_captures_results['Success'] = True
    
    num_tests = len(iConfig.TestConfig['Frames'])
    
    # make sure that expected references exist for the run tests
    for subdir, dirs, files in os.walk(results_dir):
        for file in files:
            if not helpers.isSupportedImageExt(file):
                # .html or .txt files from the previous should already be deleted here
                print('[FAILED] Unsupported reference file type from: ' + file)
                continue
            
            result_file = os.path.join( subdir, file)
            relative_file_path = result_file[len(results_dir) + 1 : len(result_file)]
            reference_file = os.path.join(reference_dir, relative_file_path)
            
            if (not os.path.exists(reference_file)):
                print('Error: Expecting reference' + str(reference_file) + '. refererence file is missing! \n')

    for subdir, dirs, files in os.walk(reference_dir):
        for file in files:
            if not helpers.isSupportedImageExt(file):
                print('[FAILED] Unsupported reference file type from: ' + file)
                continue
            
            reference_file = os.path.join( subdir, file)
            
            # checks that results and references have same number of images
            # make sure there is a subsequent file in results_dir
            relative_file_path = reference_file[len(reference_dir) + 1 : len(reference_file)]
            results_file = os.path.join(results_dir, relative_file_path)
            if (not os.path.exists(results_file)):
                print('Result file: ' + results_file + ' does not exist')
            status = comparison_func(results_file, os.path.split(results_file)[0], reference_file, screen_captures_results)
            if not status:
                # add early out option?
                print('Error: Test failed on comparison between ' +  results_file + ' and reference ' + reference_file)
        
    
    return screen_captures_results
    
    