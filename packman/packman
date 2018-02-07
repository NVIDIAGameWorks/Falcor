#!/bin/bash

PM_PACKMAN_VERSION=5.1

# Specify where packman command exists
export PM_INSTALL_PATH=$(dirname ${BASH_SOURCE})

# The packages root may already be configured by the user
if [ -z "$PM_PACKAGES_ROOT" ]; then
    # Set variable permanently using .profile for this user
    echo "export PM_PACKAGES_ROOT=\$HOME/packman-repo" >> ~/.profile
    # Set variable temporarily in this process so that the following execution will work 
	export PM_PACKAGES_ROOT="${HOME}/packman-repo"
fi

# Ensure the packages root path exists:
if [ ! -d "$PM_PACKAGES_ROOT" ]; then
	echo "Creating packman packages repository at $PM_PACKAGES_ROOT"
	mkdir -p "$PM_PACKAGES_ROOT"
fi

# The packman module may be externally configured
if [ -z "$PM_MODULE_EXT" ]; then
	PM_MODULE_DIR="$PM_PACKAGES_ROOT/packman-common/$PM_PACKMAN_VERSION"
	export PM_MODULE="$PM_MODULE_DIR/packman.py"
else
	export PM_MODULE="$PM_MODULE_EXT"
fi

fetch_file_from_s3() 
{
	SOURCE=$1
	SOURCE_URL=http://packman.s3.amazonaws.com/$SOURCE
	TARGET=$2
	echo "Fetching $SOURCE from S3 ..."
	if command -v wget >/dev/null 2>&1; then
		wget --quiet -O$TARGET $SOURCE_URL
	else
		curl -o $TARGET $SOURCE_URL -s -S
	fi		
}

# Ensure the packman package exists:
if [ ! -f "$PM_MODULE" ]; then
	PM_MODULE_PACKAGE="packman-common@$PM_PACKMAN_VERSION.zip"
	TARGET="/tmp/$PM_MODULE_PACKAGE"
	# We always fetch packman from S3:
	fetch_file_from_s3 $PM_MODULE_PACKAGE $TARGET
	if [ "$?" -eq "0" ]; then
		echo "Unpacking ..."
		mkdir -p "$PM_MODULE_DIR"
		unzip -q $TARGET -d "$PM_MODULE_DIR"
		rm $TARGET
	else
		echo "Failure while fetching packman module from S3!"
		exit 1
	fi
fi

# For now assume python is installed on the box and we just need to find it
if command -v python2 >/dev/null 2>&1; then
	export PM_PYTHON=python2
else
	export PM_PYTHON=python
fi

# Ensure 7za package exists:
PM_7za_VERSION=16.02
export PM_7za_PATH="$PM_PACKAGES_ROOT/chk/7za/$PM_7za_VERSION"
if [ ! -f "$PM_7za_PATH" ]; then
    $PM_PYTHON "$PM_MODULE" install 7za $PM_7za_VERSION -r packman:cloudfront
    if [ "$?" -ne 0 ]; then
        echo "Failure while installing required 7za package"
        exit 1
    fi
fi

# Generate temporary file name for environment variables:
PM_VAR_PATH=`mktemp -u -t tmp.XXXXX.$$.pmvars`

$PM_PYTHON -u "$PM_MODULE" $* --var-path="$PM_VAR_PATH"
exit_code=$?
# Export the variables if the file was used and remove the file:
if [ -f "$PM_VAR_PATH" ]; then
	while read -r line
	do
        if [ ${#line} -gt 0 ]; then
    		export "$line"
        fi
	done < "$PM_VAR_PATH"
    rm -f "$PM_VAR_PATH"
fi

# Return the exit code from python
if [ "$exit_code" != 0 ]; then
    exit "$exit_code"
fi
