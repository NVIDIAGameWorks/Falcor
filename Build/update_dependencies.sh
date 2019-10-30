#!/bin/bash +x
export FALCOR_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
$FALCOR_ROOT_DIR/packman/packman pull "$FALCOR_ROOT_DIR/dependencies.xml" --platform linux
status=$? 
if [ "$status" -ne "0" ]; then
 echo "Error $status"
 exit 1
fi
