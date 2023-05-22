#!/usr/bin/env python

"""
Python macro tool.

Expands macros in C++ code using the python interpreter.
Example:

/* <<<PYMACRO
print("const int table[] = {")
print("    " + ", ".join(str(i) for i in range(8)))
print("};")
>>> */
const int table[] = {
    0, 1, 2, 3, 4, 5, 6, 7
};
/* <<<PYMACROEND>>> */

"""

import sys
import re
import argparse
from pathlib import Path
from enum import Enum
from io import StringIO


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


HEADER_START_RE = re.compile(r"^\s*\/\*\s*<<<PYMACRO\s*$")
HEADER_END_RE = re.compile(r"^\s*>>>\s*\*\/\s*$")
FOOTER_RE = re.compile(r"^\s*\/\*\s+<<<PYMACROEND>>>\s*\*\/\s*$")


class State(Enum):
    IDLE = 1
    HEADER = 2
    CONTENT = 3


def process_file(path: Path, dry_run: bool):

    state = State.IDLE
    script_lines = []

    lines_in = open(path).readlines()
    lines_out = []

    for line in lines_in:
        if state == State.IDLE:
            lines_out.append(line)
            m = HEADER_START_RE.match(line)
            if m:
                script_lines = []
                state = State.HEADER
        elif state == State.HEADER:
            lines_out.append(line)
            m = HEADER_END_RE.match(line)
            if m:
                state = State.CONTENT
                script = "".join(script_lines)
                c = compile(script, "<string>", "exec")
                with Capturing() as output:
                    eval(c)
                lines_out += [l + "\n" for l in output]
            else:
                script_lines.append(line)
        elif state == State.CONTENT:
            m = FOOTER_RE.match(line)
            if m:
                lines_out.append(line)
                state = State.IDLE

    if lines_out != lines_in:
        if dry_run:
            print("".join(lines_out))
        else:
            open(path, "w").writelines(lines_out)


def run(args):
    for file in args.files:
        process_file(Path(file), dry_run=args.dry_run)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without writing files",
    )
    parser.add_argument("files", metavar="file", nargs="+")

    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
