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
import webbrowser

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
        cmd_line = executable_filepath  + ' ' + current_arguments + ' -outputdir ' + output_directory
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


def run_pass_test(executable_filepath, file_path, graph_name, output_directory, generate_missing):
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
        else:
            num_scenes = 1
        
        if run_all_images:
            num_images = iConfig.num_images
        else:
            num_images = 1
        
        for test_index in range(0, max(num_scenes, num_images) ):
            viewer_args = start_viewer_args
            output_file_base_name = graphName + '_'
            
            if run_scene_tests:
                scene_viewer_args = iConfig.get_next_scene_args(test_index)
                viewer_args = scene_viewer_args + viewer_args
                input_arg = scene_viewer_args.split()[1]
                output_file_base_name = output_file_base_name + os.path.splitext(os.path.basename(input_arg))[0] + '_'
            
            if run_image_tests:
                image_viewer_args = iConfig.get_next_image_args(test_index)
                input_arg = image_viewer_args.split()[1]
                output_image_file_base_name = output_file_base_name + os.path.splitext(os.path.basename(input_arg))[0] + '_'
                dont_run = False
                
                viewer_args_with_images = viewer_args + image_viewer_args
                if generate_missing:
                    for subdir, dirs, files in os.walk(output_directory):
                        for file in files:
                            if (file.startswith(output_image_file_base_name)):
                                print('Skipping test : ' + output_image_file_base_name + ' . Output files already exists')
                                dont_run = True
                                break;
                if dont_run:
                    continue;
                output_results = run_test_run(executable_filepath, viewer_args_with_images, output_image_file_base_name, output_directory)
                errors.append(output_results);
            else:
                dont_run = False
                if generate_missing:
                    for subdir, dirs, files in os.walk(output_directory):
                        for file in files:
                            if (file.startswith(output_file_base_name)):
                                print('Skipping test : ' + output_file_base_name + ' . Output files already exists')
                                dont_run = True
                                break;
                if dont_run:
                    continue;
                output_results = run_test_run(executable_filepath, viewer_args, output_file_base_name, output_directory)
                errors.append(output_results);
            
        print('\n')
        index = index + 1
    
    return errors

def run_unit_tests(executable_filepath, unit_test_outputfile_path, regex_filter):
    try:
        # set up command line with name of output file and other parameters passed through
        cmd_line = executable_filepath
        if regex_filter:
            cmd_line = cmd_line + " -test_filter " + regex_filter
        
        process = subprocess.Popen(cmd_line.split(), stderr = subprocess.PIPE)
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
        
        unit_test_out_file = open(unit_test_outputfile_path, 'w')
        # unit_test_out_file.write(process.communicate());
        for string in process.stderr:
            unit_test_out_file.write(str(string.decode()))
        unit_test_out_file.close()
        
        return output_results

    except (NameError, IOError, OSError) as e:
        print(e.args)
        raise TestsSetError('Error when trying to run unit tests')

def run_graph_pass_test(executable_filepath, path_to_tests, graph_file_name, graph_name, output_directory, generate_missing):
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
        if not generate_missing:
            helpers.directory_clean_or_make(graph_output_directory)
        else:
            helpers.directory_make(graph_output_directory)
        errors[graphFile] = []
        errors[graphFile] = run_pass_test(executable_filepath, graphFile, graph_name, graph_output_directory, generate_missing)
        
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
    
    # Subfolder wintin the references directory. Set with the build machine name by the test servers
    parser.add_argument('-rsf', '--reference_sub_folder', action='store', help='Optional sub folder name within references directory');
    
    # Add argument for branch name. If this is not specified the script will look in the .git directory
    parser.add_argument('-bn', '--branch_name', action='store', help='Name of the current checkout branch')
    
    # Add argument for reference branch name. If this is not specified will use same as branch_name
    parser.add_argument('-rb', '--reference_branch_name', action='store', help='Name of the branch in which the references were generated from')
    
    # Argument for repostiory name in front of reference machine name
    parser.add_argument('-repo', '--repository_id', action='store', help='Id name for the checkout repository appended in front of the reference sub folder name')
    
    # Add argument to specify to upload results to the data server cache
    parser.add_argument('-u', '--upload', action='store_true', help='Upload the test results to netapp server')
    
    # Add argument to specify regex filter for unit tests run along side the pass tests.
    parser.add_argument('-regex', '--unit_test_filter', action='store', help='Specify regex filter for unit tests run along side the pass tests')
    
    # Add argument to specify comparing only to local references
    parser .add_argument('-l', '--local_only', action ='store_true')
    
    # Parse the Arguments.
    args = parser.parse_args()
    
    if args.build_configuration:
        target_configuration = args.build_configuration
    else:
        target_configuration = iConfig.TestConfig["DefaultConfiguration"]
    
    # This assumes the user always runs the script in the /Tests directory
    root_dir = machine_configs.default_main_dir
    executables_filepath = helpers.get_executable_directory(target_configuration, '', True);
    executables_filepath = os.path.join(root_dir, executables_filepath)
    executable_filepath = os.path.join(executables_filepath, iConfig.viewer_executable)
    unit_tests_executable_filepath = os.path.join(executables_filepath, iConfig.unit_tests_executable)
    if args.branch_name:
        branch_name = args.branch_name
    else:
        branch_name = helpers.get_git_branch_name(root_dir);
    
    results_dir = machine_configs.machine_relative_checkin_local_results_directory
    if args.reference_sub_folder:
        if args.repository_id:
            results_dir = os.path.join(results_dir, args.repository_id)
        results_dir = os.path.join(results_dir, args.reference_sub_folder)
        results_dir = os.path.join(results_dir, branch_name)
    else:
        results_dir = os.path.join(results_dir, branch_name)
    
    results_dir = os.path.join(results_dir, target_configuration)
    
    if not args.compare_only:
        helpers.directory_clean_or_make(results_dir)
    
    references_dir = machine_configs.machine_reference_directory
    
    if args.repository_id:
        references_dir = os.path.join(references_dir, args.repository_id)
    
    # Display this before building so user has time to respond during build if unintended
    if not args.tests_directory:
        print('No path specified. Will run full tests for all passes.')
    
    path_to_bin = os.path.join( root_dir, 'Bin')
    if os.path.isdir(path_to_bin):
        helpers.directory_clean(path_to_bin)
    
    errors = {}
    
    if not args.compare_only:
        # Build the falcor solution. Run build target on render pass project.
        helpers.build_solution(root_dir, os.path.join(root_dir, 'Falcor.sln'), target_configuration, False)
        if args.tests_directory:
            errors = run_graph_pass_test(executable_filepath, args.tests_directory, args.graph_file, args.graph_name, results_dir, False)
        else:
            
            for subdir, dirs, files in os.walk(root_dir):
                ignoreThisDir = False
                for ignoreDir in iConfig.IgnoreDirectories[target_configuration]:
                    ignore_abs_path = os.path.abspath(os.path.join(root_dir, str(ignoreDir)))
                    current_abs_path = os.path.abspath(subdir)
                    if (os.path.commonpath([ignore_abs_path]) == os.path.commonpath([ignore_abs_path, current_abs_path])):
                        ignoreThisDir = True
                        break;
                if ignoreThisDir:
                    continue
                
                if subdir.lower().endswith('testing'):
                    new_errors = run_graph_pass_test(executable_filepath, subdir, args.graph_file, args.graph_name, results_dir, False)
                    for error_key in new_errors.keys():
                        errors[error_key] = new_errors[error_key]
    
    print('Running comparisons for all images: \n')
    
    all_results_data = {}
    
    if args.reference_branch_name:
        reference_branch_name = args.reference_branch_name
    else:
        reference_branch_name = branch_name
    
    if args.upload:
        # upload the local images to remote cache folder
        target_dir = os.path.join(machine_configs.results_cache_directory, branch_name)
        if args.reference_sub_folder:
            if args.repository_id:
                target_dir = os.path.join(target_dir, args.repository_id)
            target_dir = os.path.join(target_dir, args.reference_sub_folder)
        target_dir = os.path.join(target_dir, target_configuration)
        compare_results_dir = target_dir
        if not args.tests_directory:
            helpers.directory_clean_or_make(target_dir)
        helpers.directory_copy(results_dir, target_dir)
        print('target_dir ' + target_dir + '\n\n')
    else:
        compare_results_dir = results_dir
    
    # compare tests to references 
    for references_subdir, dirs, files in os.walk(references_dir): #for references_subdir in os.listdir(references_dir):
        subdir = os.path.join(references_dir, references_subdir)
        if args.reference_sub_folder and (args.reference_sub_folder != os.path.basename(references_subdir)):
            continue;
        if os.path.isdir(subdir):
            subdir = os.path.join( os.path.join(subdir, reference_branch_name), target_configuration)
            if not os.path.isdir(subdir):
                print('No references for ' + target_configuration + ' on ' + references_subdir)
            else:
                all_results_data[os.path.join(subdir, references_subdir)] = compareOutput.compare_all_images(compare_results_dir, subdir, compareOutput.default_compare)
                if args.reference_sub_folder:
                    break;
    
    
    # write comparison to html file
    unit_test_filename = "UnitTestOutput_" + target_configuration + ".txt"
    html_file_content = writeTestResultsToHTML.write_test_set_results_to_html(all_results_data, errors)
    html_file_name = helpers.build_html_filename(all_results_data, target_configuration)
    artifacts_path = machine_configs.machine_relative_checkin_local_results_directory
    if args.reference_sub_folder:
        if args.repository_id:
            artifacts_path = os.path.join(artifacts_path, args.repository_id)
        artifacts_path = os.path.join(artifacts_path, args.reference_sub_folder)
    artifacts_path = os.path.join(artifacts_path, branch_name)
    helpers.directory_make(machine_configs.machine_relative_checkin_local_results_directory)
    artifacts_path = os.path.join(artifacts_path, target_configuration)
    
    html_file_path = os.path.join(artifacts_path, html_file_name)
    print('Writing comparison file to ' + html_file_path)
    html_file = open(html_file_path, 'w')
    html_file.write(html_file_content)
    html_file.close()
    
    regex = ""
    if args.unit_test_filter:
        regex = str(args.unit_test_filter)
    
    unit_test_outputfile_path = os.path.join(artifacts_path, unit_test_filename)
    run_unit_tests(unit_tests_executable_filepath, unit_test_outputfile_path, regex)
    
    unit_test_out_file = open(unit_test_outputfile_path, 'r')
    file_output_string = unit_test_out_file.read()
    
    if not len(file_output_string):
        print ('[Error] no output from unit tests.')
    else:
        print (file_output_string)
    
    unit_test_out_file.close()
    
    if args.upload:
        helpers.directory_copy(results_dir, target_dir)
        remote_html_file_path = os.path.join(target_dir, html_file_name)
        remote_unit_test_out_file = os.path.join(target_dir, unit_test_filename)
        
        if os.name == 'nt':
            os.system("start " + remote_html_file_path)
            os.system("start " + remote_unit_test_out_file)
        else:
            webbrowser.open('file://' + os.path.abspath(remote_html_file_path))
            webbrowser.open('file://' + os.path.abspath(remote_unit_test_out_file))
    else:
        # Open it up.
        if os.name == 'nt':
            os.system("start " + html_file_path)
            os.system("start " + unit_test_outputfile_path)
        else:
            webbrowser.open('file://' + os.path.abspath(html_file_path))
            webbrowser.open('file://' + os.path.abspath(unit_test_outputfile_path))
    
    print('Done')

if __name__ == '__main__':
    main()
