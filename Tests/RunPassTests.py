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


def run_pass_test(executable_filepath, file_path, output_directory):
    print('Running tests for graphs in: ' + file_path)
    print('Output directory set to' + output_directory)

    graph_names = []
    num_image_loader_passes = [] # number of image nodes per graph in file
    
    #parse render graph file for graphs. 
    graph_file = open( file_path ).read()
    file_ast = ast.parse(graph_file)
    graph_func_prefix = 'render_graph_'
    
    # grab the name of each create function from syntax tree
    for func in file_ast.body:
        num_img_loaders = 0
        if(isinstance(func, ast.FunctionDef)):
            # check all statements in the tree to find calls to create an image loader node
            for expr in ast.walk(func):
                    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "createRenderPass":
                            if expr.args[0].s == "ImageLoader":
                                num_img_loaders = num_img_loaders + 1
            
            num_image_loader_passes.append(num_img_loaders)
            
            # function must fit definition of 'render_graph_' + graph_name
            if(func.name.startswith(graph_func_prefix)):
                graph_names.append(func.name[len(graph_func_prefix) : len(func.name)])
        
    # run a test on each graph in the file.
    index = 0
    for graph_name in graph_names:
        print('Running tests with \'' + graph_name + '\' in ' + file_path)
        viewer_args = iConfig.test_arguments + ' -graphFile ' + file_path + ' -graphname ' + graph_name
        run_image_tests = num_image_loader_passes[index] > 0
        
        if not run_image_tests:
            print('Note: No image loader nodes in graph. Will only input scenes')
        
        print('Num loaders: ' + str(num_image_loader_passes[index]))
        test_index = 0
        test_viewer_args = iConfig.get_next_arguments(run_image_tests, test_index)
        
        while len(test_viewer_args) > 0:
            viewer_args = test_viewer_args + viewer_args
            input_arg = test_viewer_args.split()[1]
            print('Running test for ' + graph_name + ' with ' + input_arg)
            output_file_base_name = graph_name + '_' + os.path.splitext(os.path.basename(input_arg))[0] + '_'
            run_test_run(executable_filepath, viewer_args, output_file_base_name, output_directory)
            test_index = test_index + 1
            test_viewer_args = iConfig.get_next_arguments(run_image_tests, test_index)
            
        print('\n')
        index = index + 1
    
    return 

def run_graph_pass_test(executable_filepath, path_to_tests, output_directory):
    renderGraphFiles = []
    
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
        print('\n')
        graph_output_directory = os.path.join(output_directory, os.path.splitext(os.path.basename(graphFile))[0])
        print(graph_output_directory)
        helpers.directory_clean_or_make(graph_output_directory)
        run_pass_test(executable_filepath, graphFile, graph_output_directory)
        

def main():
    
    # Argument Parser.
    parser = argparse.ArgumentParser()
    
    # Add argument for testing directory for render pass tests
    parser.add_argument('-td', '--tests_directory', action='store', help='Specify the testing directory containing the test render graphs.')
    
    # Add argument for specifing build configuration for the test
    parser.add_argument('-bc', '--build_configuration', action='store', help='Build configuration for test. ReleaseD3D12 by default')
    
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
    
    results_dir = os.path.join(machine_configs.machine_relative_checkin_local_results_directory, branch_name)
    results_dir = os.path.join(results_dir, target_configuration);
    helpers.directory_clean_or_make(results_dir)
    
    references_dir = os.path.join(machine_configs.machine_reference_directory, branch_name)
    references_dir = os.path.join(references_dir, target_configuration)
    
    # Display this before building so user has time to respond during build if unintended
    print('No path specified. Will run full tests for all passes.')
    
    # Build the falcor solution. Run build target on render pass project.
    helpers.build_solution(root_dir, os.path.join(root_dir, 'Falcor.sln'), target_configuration, False)
    
    if args.tests_directory:
        run_graph_pass_test(executable_filepath, args.tests_directory, results_dir)
    else:
        for subdir, dirs, files in os.walk(root_dir):
            if subdir.lower().endswith('testing'):
                run_graph_pass_test(executable_filepath, subdir, results_dir)
    
    # compare tests to references 
    results_data = compareOutput.compare_all_images(results_dir, references_dir, compareOutput.default_compare)
    
    # write comparison to html file
    html_file_content = writeTestResultsToHTML.write_test_set_results_to_html(results_data)
    
    html_file_path = os.path.join(machine_configs.machine_relative_checkin_local_results_directory, helpers.build_html_filename(tests_set_data))
    html_file = open(html_file_path, 'w')
    html_file.write(html_file_content)
    html_file.close()
    

if __name__ == '__main__':
    main()
