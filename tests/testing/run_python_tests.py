"""
Frontend for running python unit tests.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

from core import Environment, config
from core.environment import find_most_recent_build_config


def run_python_tests(env: Environment, xml_report, args):
    """
    Run python tests.
    """
    cmd_args = ["python"]
    if xml_report:
        cmd_args += ["-m", "xmlrunner", "--output-file", str(xml_report)]
    else:
        cmd_args += ["-m", "unittest"]
    cmd_args += ["discover", str(env.python_tests_dir)] + args

    environ = os.environ.copy()
    PYTHONPATH = environ.get("PYTHONPATH", "")
    PATH = environ.get("PATH", "")
    LD_LIBRARY_PATH = environ.get("LD_LIBRARY_PATH", "")
    SEP = ";" if os.name == 'nt' else ":"

    # Extend PYTHONPATH with library directory for loading xmlrunner
    PYTHONPATH = str(Path(__file__).parent / "libs") + SEP + PYTHONPATH
    # Extend PYTHONPATH & PATH & LD_LIBRARY_PATH (linux) with build directory for loading falcor
    PYTHONPATH = str(env.build_dir / "python") + SEP + PYTHONPATH
    PATH = str(env.build_dir) + SEP + PATH
    LD_LIBRARY_PATH = str(env.build_dir) + SEP + LD_LIBRARY_PATH

    environ["PYTHONPATH"] = PYTHONPATH
    environ["PATH"] = PATH
    environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

    p = subprocess.Popen(cmd_args, env=environ)
    try:
        p.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
        print("\n\nProcess killed due to timeout")

    return p.returncode == 0


def main():
    default_config = find_most_recent_build_config()

    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )
    parser.add_argument(
        "--environment",
        type=str,
        action="store",
        help=f"Environment",
        default=None,
    )
    parser.add_argument(
        "--config",
        type=str,
        action="store",
        help=f"Build configuration (default: {default_config})",
        default=default_config,
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available build configurations",
    )
    parser.add_argument(
        "--xml-report", type=str, action="store", help="XML report output file"
    )
    args, passthrough_args = parser.parse_known_args()

    # Try to load environment.
    env = None
    try:
        env = Environment(args.environment, args.config)
    except Exception as e:
        env_error = e

    # Print help.
    if args.help:
        parser.print_help()
        if env:
            print(f"\nAdditional arguments consumed by Python's unit test runner:\n")
            subprocess.call(["python", "-m", "unittest", "-h"])
        else:
            print(f"\nFailed to load environment: {env_error}")
        sys.exit(0)

    # List build configurations.
    if args.list_configs:
        print(
            "Available build configurations:\n" + "\n".join(config.BUILD_CONFIGS.keys())
        )
        sys.exit(0)

    # Abort if environment is missing.
    if env == None:
        print(f"\nFailed to load environment: {env_error}")
        sys.exit(1)

    # Run tests.
    success = run_python_tests(env, args.xml_report, passthrough_args)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
