'''
Script for building Falcor.
'''

import sys
import argparse
import subprocess

from core import Environment, config

def build_falcor(env, rebuild=False):
    '''
    Builds Falcor. Optionally issues a full rebuild.
    '''
    args = [
        str(env.cmake_exr),
        "--build", str(env.cmake_dir),
        "--config", str(env.cmake_config)
    ]
    print('Running: ' + ' '.join(args))
    process = subprocess.Popen(args, stderr=subprocess.STDOUT)
    process.communicate()[0]
    return process.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Utility for building Falcor.')
    parser.add_argument('-c', '--config', type=str, action='store', help=f'Build configuration')
    parser.add_argument('-e', '--environment', type=str, action='store', help='Environment', default=config.DEFAULT_ENVIRONMENT)
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild')
    parser.add_argument('--list-configs', action='store_true', help='List available build configurations.')
    args = parser.parse_args()

    # Load environment.
    try:
        env = Environment(args.environment, args.config)
    except Exception as e:
        print(e)
        sys.exit(1)

    # List build configurations.
    if args.list_configs:
        print('Available build configurations:\n' + '\n'.join(config.BUILD_CONFIGS.keys()))
        sys.exit(0)

    # Build.
    success = build_falcor(env, args.rebuild)
    if (not success):
        print('Build failed.')

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
