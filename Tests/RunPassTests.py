import subprocess
import argparse
import os
from datetime import date
import shutil
import stat
import ast
import time
import sys
import json
import pprint

import WriteTestResultsToHTML as writeTestResultsToHTML
import CompareOutput as compareOutput
import InternalConfig as iConfig
import Helpers as helpers
import MachineConfigs as machine_configs


class TestsSetError(Exception):
    pass


def run_test_run(executable_filepath, current_arguments, output_file_base_name, output_directory):
    try:
        # Start the process and record the time.
        cmd_line = executable_filepath  + ' ' + current_arguments + ' -outputfilename ' + output_file_base_name + ' -outputdir ' + output_directory
        process = subprocess.Popen(cmd_line.split())
        start_time = time.time()

        output_results = (True, "")

        # Wait for the process to finish.
        while process.returncode is None:
            process.poll()

            if process.returncode is not None and process.returncode > 1:
                output_results = (False, "Process crashed or encountered error." + str(process.returncode) )
                break

            current_time = time.time()
            difference_time = current_time - start_time

            # If the process has taken too long, kill it.
            if difference_time > machine_configs.machine_process_default_kill_time:
                print("Kill Process")
                process.kill()
                output_results = (False, "Process ran for too long, had to kill it. Please verify that the program finishes within its hang time, and that it does not crash")
                break
        
        return output_results

    except (NameError, IOError, OSError) as e:
        print(e.args)
        raise TestsSetError('Error when trying to run ' + executable_filepath + ' ' + current_arguments + ' ' + 'with outputfilename ' + output_file_base_name + ' and outputdir ' + output_directory)


def run_pass_test(executable_filepath, file_path, graph_name, output_directory):
    print('Running tests for graphs in: ' + file_path)
    print('Output directory set to' + output_directory)

    graph_names = []
    num_image_loader_passes = [] # number of image nodes per graph in file
    scenes_loaded = [] # if there is a graph loaded 
    images_loaded = [] # if there is a graph loaded 
    no_default_scenes = [] # if disableLoadDefaultScene is called in graph
    errors = []
    
    #parse render graph file for graphs. 
    graph_file = open( file_path ).read()
    file_ast = ast.parse(graph_file)
    graph_func_prefix = 'render_graph_'
    
    # grab the name of each create function from syntax tree
    for func in file_ast.body:
        num_img_loaders = 0
        graph_loads_scene = False
        graph_loads_image = False
        no_default_scene = False
        
        if(isinstance(func, ast.FunctionDef)):
            # check all statements in the tree to find calls to create an image loader node.
            for expr in ast.walk(func):
                if isinstance(expr, ast.Call): 
                    # print(ast.dump(expr)) # this is useful for debugging this
                    if isinstance(expr.func, ast.Name):
                        if expr.func.id == "createRenderPass":
                            if expr.args[0].s == "ImageLoader":
                                num_img_loaders = num_img_loaders + 1
                                if len(expr.args) >= 1:
                                    passDictionary = expr.args[1]
                                    if isinstance(passDictionary, ast.Dict):
                                        for key in passDictionary.keys:
                                            if(key.s == 'fileName'):
                                                # note: the ast's 'dictionary' is not the same as a dictionary
                                                graph_loads_image = len(passDictionary.values[0].s) > 0
                    else:
                        # member functions so its a bit differently
                        if isinstance(expr.func, ast.Attribute):
                            if expr.func.attr == "setScene":
                                graph_loads_scene = True
                            if expr.func.attr == 'disableLoadDefaultScene':
                                no_default_scene = True
            
            num_image_loader_passes.append(num_img_loaders)
            scenes_loaded.append(graph_loads_scene)
            images_loaded.append(graph_loads_image)
            no_default_scenes.append(no_default_scene)
            
            # function must fit definition of 'render_graph_' + graph_name
            if(func.name.startswith(graph_func_prefix)):
                graph_names.append(func.name[len(graph_func_prefix) : len(func.name)])
        
    # run a test on each graph in the file.
    index = 0
    for graphName in graph_names:
        if graph_name and graph_name != graphName:
            continue
            
        print('Running tests with \'' + graphName + '\' in ' + file_path)
        start_viewer_args = iConfig.test_arguments + ' -graphFile ' + file_path + ' -graphname ' + graphName + ' '
        
        run_image_tests = num_image_loader_passes[index] > 0
        run_scene_tests = not (no_default_scenes[index] and (not scenes_loaded[index]))
        run_all_scenes = (not scenes_loaded[index]) and run_scene_tests
        run_all_images = not images_loaded[index]
        
        if run_all_scenes:
            num_scenes = iConfig.num_scenes
            num_tests = num_scenes
        else:
            num_scenes = 1
        
        if run_all_images:
            num_images = iConfig.num_images
            num_tests = num_images
        else:
            num_images = 1
        
        for test_index in range(0, num_scenes):
            viewer_args = start_viewer_args
            output_file_base_name = graphName + '_'
            
            if run_scene_tests:
                scene_viewer_args = iConfig.get_next_scene_args(test_index)
                viewer_args = scene_viewer_args + viewer_args
                input_arg = scene_viewer_args.split()[1]
                output_file_base_name = output_file_base_name + os.path.splitext(os.path.basename(input_arg))[0] + '_'
            
            if run_image_tests:
                    image_viewer_args = iConfig.get_next_image_args(test_index)
                    viewer_args = image_viewer_args + viewer_args
                    input_arg = image_viewer_args.split()[1]
                    output_image_file_base_name = output_file_base_name + os.path.splitext(os.path.basename(input_arg))[0] + '_'
                    
                    viewer_args_with_images = viewer_args + image_viewer_args
                    output_results = run_test_run(executable_filepath, viewer_args_with_images, output_image_file_base_name, output_directory)
                    errors.append(output_results);
            else:
                output_results = run_test_run(executable_filepath, viewer_args, output_file_base_name, output_directory)
                errors.append(output_results);
            
        print('\n')
        index = index + 1
    
    return errors

def run_graph_pass_test(executable_filepath, path_to_tests, graph_file_name, graph_name, output_directory):
    renderGraphFiles = []
    errors = {}
    
    print('Attempting to run tests for ' + executable_filepath)
    
    # iterate through directory loading each render graph file
    for subdir, dirs, files in os.walk(path_to_tests):
        for file in files:
            # if the pass has it's own solution or make file, build it here
            if (file.endswith('.sln') or file == 'Makefile'):
                rTS.build_solution(path_to_tests, os.path.join(path_to_tests, file), target_configuration, False)
            if (file.endswith('.py')): # assume all found python files are graph files
                path = subdir + os.sep + file;
                renderGraphFiles.append(path)
    
    for graphFile in renderGraphFiles:
        if (graph_file_name and (not graphFile.endswith(graph_file_name)) ):
            continue;
        
        print('\n')
        graph_output_directory = os.path.join(output_directory, os.path.splitext(os.path.basename(graphFile))[0])
        print(graph_output_directory)
        helpers.directory_clean_or_make(graph_output_directory)
        errors[graphFile] = []
        errors[graphFile] = run_pass_test(executable_filepath, graphFile, graph_name, graph_output_directory)
        
    return errors

def main():
    
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    # Add argument for testing directory for render pass tests
    parser.add_argument('-td', '--tests_directory', action='store', help='Specify the testing directory containing the test render graphs.')
    
    # Add argument for specifing build configuration for the test
    parser.add_argument('-bc', '--build_configuration', action='store', help='Build configuration for test. ReleaseD3D12 by default')
    
    # Add argument for only using a specified graph file in the directory
    parser.add_argument('-gf', '--graph_file', action='store', help='Specify graph file to use within the tests directory')
    
    # Add argument for only using a specified graph within the directory
    parser.add_argument('-gn', '--graph_name', action='store', help='Specify graph name to use within the tests directory')
    
    #Add the argument for only doing comparisons with the last generated references
    parser.add_argument('-cmp', '--compare_only', action='store_true', help='Do not generate local images. Only compare last generated.');
    
    #
    parser.add_argument('-rsf', '--reference_sub_folder', action='store', help='Optional sub folder name within references directory');
    
    # Parse the Arguments.
    args = parser.parse_args()
    
    if args.build_configuration:
        target_configuration = args.build_configuration
    else:
        target_configuration = iConfig.TestConfig["DefaultConfiguration"]
    
    # This assumes the user always runs the script in the /Tests directory
    root_dir = machine_configs.default_main_dir
    branch_name = helpers.get_git_branch_name(root_dir);
    executable_filepath = helpers.get_executable_directory(target_configuration, '', False);
    executable_filepath = os.path.join(os.path.join(root_dir, executable_filepath), iConfig.viewer_executable)
    
    if args.reference_sub_folder:
        results_dir = os.path.join(machine_configs.machine_relative_checkin_local_results_directory, args.reference_sub_folder)
        results_dir = os.path.join(results_dir, branch_name)
    else:
        results_dir = os.path.join(machine_configs.machine_relative_checkin_local_results_directory, branch_name)
    
    results_dir = os.path.join(results_dir, target_configuration)
    
    if not args.compare_only:
        helpers.directory_clean_or_make(results_dir)
    
    if args.reference_sub_folder:
        references_dir = os.path.join(machine_configs.machine_reference_directory, args.reference_sub_folder)
        references_dir = os.path.join(references_dir, branch_name)
    else:
        references_dir = os.path.join(machine_configs.machine_reference_directory, branch_name)
    
    references_dir = os.path.join(references_dir, target_configuration)
    
    # Display this before building so user has time to respond during build if unintended
    if not args.tests_directory:
        print('No path specified. Will run full tests for all passes.')
    
    errors = {}
    
    if not args.compare_only:
        # Build the falcor solution. Run build target on render pass project.
        helpers.build_solution(root_dir, os.path.join(root_dir, 'Falcor.sln'), target_configuration, False)
        
        if args.tests_directory:
            errors = run_graph_pass_test(executable_filepath, args.tests_directory, args.graph_file, args.graph_name, results_dir)
        else:
            for subdir, dirs, files in os.walk(root_dir):
                if subdir.lower().endswith('testing'):
                    new_errors = run_graph_pass_test(executable_filepath, subdir, args.graph_file, args.graph_name, results_dir)
                    for error_key in new_errors.keys():
                        errors[error_key] = new_errors[error_key]
                        
    
    print('Running comparisons for all images: \n')
    
    # compare tests to references 
    results_data = compareOutput.compare_all_images(results_dir, references_dir, compareOutput.default_compare)
    
    # write comparison to html file
    html_file_content = writeTestResultsToHTML.write_test_set_results_to_html(results_data, errors)
    
    html_file_path = os.path.join(machine_configs.machine_relative_checkin_local_results_directory, helpers.build_html_filename(results_data, target_configuration))
    html_file = open(html_file_path, 'w')
    html_file.write(html_file_content)#str(results_data))
    html_file.close()
    
    print('Writing comparison file to ' + html_file_path)
    
    # Open it up.
    if os.name == 'nt':
        os.system("start " + html_file_path)
    else:
        webbrowser.open('file://' + os.path.abspath(html_file_path))
    
    print('Done')

if __name__ == '__main__':
    main()
