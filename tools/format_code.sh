#!/bin/sh

export pwd=`pwd`
export project_dir=$pwd/..
export python_dir=$project_dir/tools/.packman/python
export python=$python_dir/bin/python3
export clang_format=$project_dir/tools/.packman/clang-format/clang-format.exe

if [ ! -f "$python" ] || [ ! -f "$clang_format" ]; then
    $project_dir/setup.sh
fi

cd $project_dir
env LD_LIBRARY_PATH="$python_dir/lib" $python $pwd/run_clang_format.py --clang-format-executable=$clang_format -i -r Source
cd $pwd
