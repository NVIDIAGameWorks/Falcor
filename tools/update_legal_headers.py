# Script to update legal headers across the code base.

import os
import sys
import re
import subprocess
from datetime import datetime
from glob import glob

# List of file extensions requiring the legal header.
EXTENSIONS = ['.h', '.c', '.cpp', '.slang', '.slangh']

# Public legal header.
PUBLIC_HEADER_TEMPLATE = """
/***************************************************************************
 # Copyright (c) {years}, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
"""

def get_sources(include_dirs, exclude_dirs, extensions):
    sources = []

    # get project root directory.
    project_root = os.path.abspath(os.path.join(os.path.split(sys.argv[0])[0], "../"))

    # add files from included directories
    for d in include_dirs:
        sources += glob(os.path.join(project_root, d), recursive=True)

    # remove files from excluded directories
    for d in exclude_dirs:
        sources = list(filter(lambda p: not p.startswith(os.path.abspath(os.path.join(project_root, d))), sources))

    # only keep files with valid extension
    extension_set = set(extensions)
    sources = list(filter(lambda p: os.path.splitext(p)[1] in extension_set, sources))

    return sources


def get_last_modify_year(path):
    cmd = ["git", "log", "-1", "--follow", "--pretty=%aI", path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    str = result.stdout.decode("utf-8")
    year = int(str[0:4])
    return year


def fix_legal_header(include_dirs, exclude_dirs, extensions, header_template):
    # regular expression matching the first block comment (after whitespace)
    rheader = re.compile(r"^\s*\/\*((\*(?!\/)|[^*])*)\*\/\s*")

    # regular expression matching the beginning of a comment
    rcomment = re.compile(r"^\s*\/[\*\/]")

    # prepare header text
    header_template = header_template.strip() + "\n"

    for p in get_sources(include_dirs, exclude_dirs, extensions):
        print("Processing %s" % (p))

        first_year = DEFAULT_FIRST_YEAR
        last_year = max(DEFAULT_LAST_YEAR, get_last_modify_year(p))
        header = header_template.replace("{years}", "%d-%02d" % (first_year, last_year % 100))

        # read file
        text = open(p).read()

        # remove existing header
        if rheader.match(text):
            text = rheader.sub("", text)

        # add extra newline when text starts with a comment
        if rcomment.match(text):
            text = "\n" + text

        # add new header
        text = header + text

        # write back file
        open(p, "w").write(text)


# fix public headers
fix_legal_header(
    include_dirs=["Source/**"],
    exclude_dirs=[],
    extensions=EXTENSIONS,
    header_template=PUBLIC_HEADER_TEMPLATE
)
