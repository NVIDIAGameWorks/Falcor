#!/bin/sh

DIR="$( dirname -- "$BASH_SOURCE"; )";

if [ -z "${CONDA_PYTHON_EXE}" ]; then
    echo "Python tests require conda environment to run."
    exit 1
fi

python ${DIR}/testing/run_python_tests.py $@
