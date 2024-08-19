#!/bin/sh

export pwd="$(dirname "$(realpath "$0")")"
export project_dir=$pwd/..
export python_dir=$project_dir/tools/.packman/python
export python=$python_dir/bin/python3

if [ ! -f "$python" ]; then
    $project_dir/setup.sh
fi

env LD_LIBRARY_PATH="$python_dir/lib" $python $pwd/testing/run_unit_tests.py $@
