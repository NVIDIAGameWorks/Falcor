'''
Script for building Falcor on a build agent
'''

import sys
import os
import argparse
import msbuild

tests_dir = '../Tests/'

# Enable importing core from Tests/testing
# Ugly, but effective
sys.path.insert(0, os.path.join(tests_dir, 'testing'))

from core import Environment, config

def build_falcor(env, rebuild=False):
    '''
    Builds Falcor. Optionally issues a full rebuild.
    '''
    try:
        action = 'rebuild' if rebuild else 'build'
        toolsPackmanDir = env.project_dir / 'Tools' / '.packman'
        return msbuild.build_solution(toolsPackmanDir, env.solution_file, action, env.build_config)
    except (RuntimeError, ValueError) as e:
        print(e)
        return False

def main():
    available_configs = ', '.join(config.BUILD_CONFIGS.keys())
    parser = argparse.ArgumentParser(description='Utility for building Falcor.')
    parser.add_argument('-c', '--config', type=str, action='store', help=f'Build configuration: {available_configs}', default=config.DEFAULT_BUILD_CONFIG)
    parser.add_argument('-e', '--environment', type=str, action='store', help='Environment', default=config.DEFAULT_ENVIRONMENT)
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild')
    args = parser.parse_args()

    # Enviroment config file paths are relative to the Tests directory
    env_path = os.path.join(tests_dir, args.environment)

    # Load environment.
    try:
        env = Environment(env_path, args.config)
    except Exception as e:
        print(e)
        sys.exit(1)

    # Build solution.
    success = build_falcor(env, args.rebuild)
    if (not success):
        print('Build failed.')

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
