import os
import subprocess

import InternalConfig as iConfig

def default_comparison(result_image, result_image_dir, reference_image, screen_captures_results):

    # Create the test compare image.
    test_compare_image_filepath = os.path.join(result_image_dir, os.path.splitext(os.path.basename(result_image))[0] + '_Compare.png')
    
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
    
    # Keep the Return Code and the Result.
    result = {}
    
    # Image compare succeeded
    if image_compare_process.returncode <= 1: # 0: Success, 1: Does not match, 2: File not found, or other error?
        result_str = image_compare_result[:image_compare_result.find(' ')]
        result['Compare Result'] = result_str
        result['Test Passed'] = float(result_str) <= iConfig.TestConfig['Tolerance']
    
        if result['Test Passed'] == False:
            print('[FAILED] Comparision above tolerance. Difference was ' + result_str + ' on image ' + result_image)
        
    # Error
    else:
        print('[FAILED] No output file produced. ')
        result['Compare Result'] = "Error"
        result['Test Passed'] = False
    
    result['Return Code'] = image_compare_process.returncode
    result['Source Filename'] = result_image
    result['Reference Filename'] = reference_image
    
    screen_captures_results['Success'] &= result['Test Passed']
    if not result_image in screen_captures_results.keys():
        screen_captures_results[result_image] = []
    screen_captures_results[result_image].append(result)
    
    return screen_captures_results['Success']


# default image comparison
def default_compare(result_file, result_file_dir, reference_file, data):
    result = default_comparison(result_file, result_file_dir, reference_file, data)
    return result;
    

def compare_all_images(results_dir, reference_dir, comparison_func):
    screen_captures_results = {}
    screen_captures_results['Success'] = True
    
    for subdir, dirs, files in os.walk(results_dir):
        for file in files:
            if not file.endswith('.png'):
                continue
                
            result_file = os.path.join( subdir, file )
            # make sure there is a subsequenct file to reference_dir
            relative_file_path = result_file[len(results_dir) + 1 : len(result_file)]
            reference_file = os.path.join(reference_dir, relative_file_path)
            if (not os.path.exists(reference_file)):
                print('Reference file: ' + reference_file + ' does not exist')
                
            status = comparison_func(result_file, subdir, reference_file, screen_captures_results)
            if not status:
                # add early out option?
                print('Test failed on comparison between ' +  result_file + ' and reference ' + reference_file)
    
    return screen_captures_results
    
    