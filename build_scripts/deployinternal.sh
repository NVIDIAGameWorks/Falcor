#!/bin/sh

# $1 -> Project directory
# $2 -> Binary output directory
# $3 -> Build configuration

ExtDir=$1/external/packman/
OutDir=$2

IsDebug=false
if [ "$3" = "Debug" ]; then
    IsDebug=true
fi
