#!/bin/sh

DIR="$(dirname "$(realpath "$0")")"

exec python "${DIR}/testing/run_python_tests.py" "$@"
