#!/bin/sh

# This script is fetching all dependencies via packman.

if [ "$OSTYPE" = "msys" ]; then
    echo "Do not use "$0" on Windows, use setup.bat instead."
    exit 1
fi

BASE_DIR=$(dirname "$0")
PACKMAN=${BASE_DIR}/tools/packman/packman
PLATFORM=linux-x86_64

echo "Updating git submodules ..."

if ! [ -x "$(command -v git)" ]; then
    echo "Cannot find git on PATH! Please initialize submodules manually and rerun."
    exit 1
else
    git submodule sync --recursive
    git submodule update --init --recursive
fi

echo "Fetching dependencies ..."

${PACKMAN} pull --platform ${PLATFORM} ${BASE_DIR}/dependencies.xml
if [ $? -ne 0 ]; then
    echo "Failed to fetch dependencies!"
    exit 1
fi

if [ ! -d ${BASE_DIR}/.vscode ]; then
    echo "Setting up VS Code workspace ..."
    cp -rp ${BASE_DIR}/.vscode-default ${BASE_DIR}/.vscode
fi

# HACK: Copy libnvtt.so.30106 to libnvtt.so so we can use it in our build.
# This changes the actual packman package, but for now, this is the easiest solution.
echo "Patching NVTT package ..."
cp -fp ${BASE_DIR}/external/packman/nvtt/libnvtt.so.30106 ${BASE_DIR}/external/packman/nvtt/libnvtt.so

exit 0
