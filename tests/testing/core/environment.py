'''
Module contining Environment class.
'''

import os
import json
import string
from pathlib import Path

from . import config, helpers

def validate_json(data, schema, full_name=None):
    '''
    Validate a given data object to match the schema.
    '''
    type_names = { dict: 'object', str: 'string', int: 'integer' }
    if not isinstance(data, schema['type']):
        raise TypeError(f'Property "{full_name}" is not of type {type_names[(schema["type"])]}')
    if 'properties' in schema:
        for prop_name, prop_schema in schema['properties'].items():
            name = f'{full_name}.{prop_name}' if full_name else prop_name
            if prop_name in data:
                validate_json(data[prop_name], prop_schema, name)
            elif prop_schema.get('optional', False) == False:
                raise TypeError(f'Property "{name}" does not exist')


def find_most_recent_build_config():
    '''
    Find the build config most recently built by checking all
    possible build directories and finding the most recently
    changed Falcor library.
    '''
    project_dir = Path(__file__).parents[3].resolve()

    best_config_name = None
    best_config_time = None

    for config_name, config_values in config.BUILD_CONFIGS.items():
        build_dir = project_dir / config_values["build_dir"]
        falcor_lib = build_dir / config.FALCOR_LIB
        if not os.path.exists(falcor_lib):
            continue
        stat = os.stat(falcor_lib)
        if not best_config_name or stat.st_mtime >= best_config_time:
            best_config_name = config_name
            best_config_time = stat.st_mtime

    return best_config_name

class Environment:
    '''
    Holds a bunch of variables necessary to run the testing infrastructure.
    '''

    def __init__(self, json_file, build_config):
        '''
        Loads the environment from the JSON file and sets up derived variables.
        '''

        self.project_dir = Path(__file__).parents[3].resolve()

        if json_file == None:
            json_file = self.project_dir / config.DEFAULT_ENVIRONMENT

        # Load JSON config.
        if not Path(json_file).exists():
            raise Exception(f'Environment config file "{json_file}" not found.')

        env = json.load(open(json_file))

        # Validate JSON.
        schema = {
            'type': dict,
            'properties': {
                'name': { 'type': str },
                'image_tests': {
                    'type': dict,
                    'properties': {
                        'result_dir': { 'type': str },
                        'ref_dir': { 'type': str },
                        'remote_ref_dir': { 'type': str, 'optional': True }
                    }
                }
            }
        }

        try:
            validate_json(env, schema)
        except TypeError as e:
            raise Exception(f'Invalid environment config file "{json_file}:" {e}.')

        # Validate build configuration.
        if not build_config in config.BUILD_CONFIGS.keys():
            raise Exception(f'Invalid build configuration "{build_config}".')

        # Setup environment variables.
        self.name = env['name']
        self.build_dir = self.project_dir / config.BUILD_CONFIGS[build_config]['build_dir']
        self.cmake_exe = self.project_dir / config.CMAKE_EXE
        # Ideally this information would be parsed from CMakePresets.json, rather than this roundabout way
        self.cmake_dir = self.build_dir.parents[1].resolve()
        self.cmake_config = self.build_dir.parts[-1]
        self.image_tests_dir = self.project_dir / config.IMAGE_TESTS_DIR
        self.image_tests_result_dir = env['image_tests']['result_dir']
        self.image_tests_ref_dir = env['image_tests']['ref_dir']
        self.image_tests_remote_ref_dir = env['image_tests'].get('remote_ref_dir', None)
        self.python_tests_dir = self.project_dir / config.PYTHON_TESTS_DIR

        self.build_config = build_config
        self.branch = helpers.get_git_head_branch(self.project_dir)

        # Resolve common filenames.
        self.falcor_test_exe = self.build_dir / config.FALCOR_TEST_EXE
        self.mogwai_exe = self.build_dir / config.MOGWAI_EXE
        self.image_compare_exe = self.build_dir / config.IMAGE_COMPARE_EXE
        self.python_exe = self.build_dir / config.PYTHON_EXE

    def resolve_image_dir(self, image_dir, branch, build_id):
        '''
        Resolve an image directory name.
        Substitues ${xxx} placeholders in directory name with values in environment.
        '''
        branch = branch.replace("/", "^").replace("\\", "^")
        template = string.Template(image_dir)
        variables = {
            'project_dir': self.project_dir,
            'build_config': self.build_config,
            'vcs_root': helpers.get_vcs_root(self.project_dir),
            'hostname': helpers.get_hostname(),
            'build_id': build_id,
            'branch': branch
        }
        return Path(template.substitute(variables)).resolve()

