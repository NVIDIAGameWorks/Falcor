'''
Script for building Falcor.
'''

import sys
import argparse

from core import Environment, vsbuild, config

def build_falcor(env, rebuild=False):
    '''
    Builds Falcor. Optionally issues a full rebuild.
    '''
    try:
        action = 'rebuild' if rebuild else 'build'
        return vsbuild.build_solution(env.solution_file, action, env.build_config)
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

    # Load environment.
    try:
        env = Environment(args.environment, args.config)
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
