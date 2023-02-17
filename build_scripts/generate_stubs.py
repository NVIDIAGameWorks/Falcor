# This script is heavily inspired by mitsuba3:
# https://github.com/mitsuba-renderer/mitsuba3/blob/master/resources/generate_stub_files.py

# Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.

"""
Usage: generate_stubs.py {package_dir}
This script generates stub files for Python type information for the `falcor`
module. It writes all the objects (classes, methods, functions, enums, etc.)
it finds to the `package_dir` folder. The stub files contain both the signatures
and the docstrings of the objects.
"""

import os
import sys
import json
import logging
import inspect
import re
import time

# ------------------------------------------------------------------------------

buffer = ""


def w(s):
    global buffer
    buffer += f"{s}\n"


# ------------------------------------------------------------------------------


def process_type_hint(s) -> str:
    sub = s
    type_hints = []
    offset = 0
    while True:
        match = re.search(r"[a-zA-Z]+: ", sub)
        if match is None:
            return s
        i = match.start() + len(match.group())
        match_next = re.search(r"[a-zA-Z]+: ", sub[i:])
        if match_next is None:
            j = sub.index(")")
        else:
            j = i + match_next.start() - 2

        type_hints.append((offset + i, offset + j, sub[i:j]))

        if match_next is None:
            break

        offset = offset + j
        sub = s[offset:]

    offset = 0
    result = ""
    for t in type_hints:
        result += s[offset : t[0] - 2]
        offset = t[1]
        # if the type hint is valid, then add it as well
        if not ("::" in t[2]):
            result += f": {t[2]}"
    result += s[offset:]

    # Check is return type hint is not valid
    if "::" in result[result.index(" -> ") :]:
        result = result[: result.index(" -> ")]

    return result


# ------------------------------------------------------------------------------


def process_properties(name, p, indent=0):
    indent = " " * indent

    if not p is None:
        w(f"{indent}{name} = ...")
        if not p.__doc__ is None:
            doc = p.__doc__.splitlines()
            if len(doc) == 1:
                w(f'{indent}"{doc[0]}"')
            elif len(doc) > 1:
                w(f'{indent}"""')
                for l in doc:
                    w(f"{indent}{l}")
                w(f'{indent}"""')


# ------------------------------------------------------------------------------


def process_enums(name, e, indent=0):
    indent = " " * indent

    if not e is None:
        w(f"{indent}{name} = {int(e)}")

        if not e.__doc__ is None:
            doc = e.__doc__.splitlines()
            w(f'{indent}"""')
            for l in doc:
                if l.startswith(f"  {name}"):
                    w(f"{indent}{l}")
            w(f'{indent}"""')


# ------------------------------------------------------------------------------


def process_class(m, c):
    methods = []
    py_methods = []
    properties = []
    enums = []

    for k in dir(c):
        # Skip private attributes
        if k.startswith("_"):
            continue
        if k.endswith("_"):
            continue

        v = getattr(c, k)
        if type(v).__name__ == "instancemethod":
            methods.append((k, v))
        elif type(v).__name__ == "function" and v.__code__.co_varnames[0] == "self":
            py_methods.append((k, v))
        elif type(v).__name__ == "property":
            properties.append((k, v))
        elif str(v).endswith(k):
            enums.append((k, v))

    base = c.__bases__[0]
    base_module = base.__module__
    base_name = base.__qualname__
    has_base = not (
        base_module == "builtins"
        or base_name == "object"
        or base_name == "pybind11_object"
    )

    base_name = base_module + "." + base_name
    base_name = base_name.replace(m.__name__ + ".", "")

    w(f'class {c.__name__}{"(" + base_name + ")" if has_base else ""}:')
    if c.__doc__ is not None:
        doc = c.__doc__.splitlines()
        if len(doc) > 0:
            if doc[0].strip() == "":
                doc = doc[1:]
            if c.__doc__:
                w(f'    """')
                for l in doc:
                    w(f"    {l}")
                w(f'    """')
                w(f"")

    process_function(m, c.__init__, indent=4)
    process_function(m, c.__call__, indent=4)

    if len(properties) > 0:
        for k, v in properties:
            process_properties(k, v, indent=4)
        w(f"")

    if len(enums) > 0:
        for k, v in sorted(enums, key=lambda item: int(item[1])):
            process_enums(k, v, indent=4)
        w(f"")

    for k, v in methods:
        process_function(m, v, indent=4)

    # for k, v in py_methods:
    #     process_py_function(k, v, indent=4)

    w(f"    ...")
    w("")


# ------------------------------------------------------------------------------


def process_function(m, f, indent=0):
    indent = " " * indent
    if f is None or f.__doc__ is None:
        return

    overloads = []
    for l in f.__doc__.splitlines():
        if ") -> " in l:
            l = process_type_hint(l)
            l = l.replace(m.__name__ + ".", "")
            overloads.append((l, []))
        else:
            if len(overloads) > 0:
                overloads[-1][1].append(l)

    for l, doc in overloads:
        has_doc = len(doc) > 1

        # Overload?
        if l[1] == ".":
            w(f"{indent}@overload")
            w(f"{indent}def {l[3:]}:{'' if has_doc else ' ...'}")
        else:
            w(f"{indent}def {l}:{'' if has_doc else ' ...'}")

        if len(doc) > 1:  # first line is always empty
            w(f'{indent}    """')
            for l in doc[1:]:
                w(f"{indent}    {l}")
            w(f'{indent}    """')
            w(f"{indent}    ...")
            w(f"")

    w(f"")


# ------------------------------------------------------------------------------


def process_py_function(name, obj, indent=0):
    indent = " " * indent
    if obj is None:
        return

    has_doc = obj.__doc__ is not None

    signature = str(inspect.signature(obj))
    signature = signature.replace("'", "")

    # Fix parameters that have enums as default values
    enum_match = re.search(r"\=<", signature)
    while enum_match is not None:
        begin = enum_match.start()
        end = begin + signature[begin:].index(">")

        new_default_value = signature[begin + 2 : end]
        new_default_value = new_default_value[: new_default_value.index(":")]

        signature = signature[: begin + 1] + new_default_value + signature[end + 1 :]
        enum_match = re.search(r"\=<", signature[begin])

    w(f"{indent}def {name}{signature}:{'' if has_doc else ' ...'}")

    if has_doc:
        doc = obj.__doc__.splitlines()
        if len(doc) > 0:  # first line is always empty
            w(f'{indent}    """')
            for l in doc:
                w(f"{indent}    {l.strip()}")
            w(f'{indent}    """')
            w(f"{indent}    ...")
            w(f"")


# ------------------------------------------------------------------------------


def process_builtin_type(type, name):
    w(f"class {name}: ...")
    w(f"")


# ------------------------------------------------------------------------------


def process_module(m):
    global buffer

    module_name = m.__name__
    logging.info(f"Processing module '{module_name}' ...")

    submodules = []
    buffer = ""

    w(f"# This file is auto-generated by {os.path.split(__file__)[1]}")
    w(f"import os")
    w(f"from typing import Any, List, Optional, overload")
    w(f"")

    for k in dir(m):
        v = getattr(m, k)

        if inspect.isclass(v):
            logging.debug(f"Found class '{k}'")
            if hasattr(v, "__module__") and not (v.__module__.startswith(module_name)):
                if v in [bool, int, float]:
                    process_builtin_type(v, k)
                continue
            process_class(m, v)
        elif type(v).__name__ in ["method", "function"]:
            logging.debug(f"Found python function '{k}'")
            # process_py_function(k, v)
        elif type(v).__name__ == "builtin_function_or_method":
            logging.debug(f"Found function '{k}'")
            process_function(m, v)
        elif type(v) in [str, bool, int, float]:
            logging.debug(f"Found property '{k}'")
            if k.startswith("_"):
                continue
            process_properties(k, v)
        # elif type(v).__bases__[0].__name__ == "module" or type(v).__name__ == "module":
        elif type(v).__name__ == "module":
            logging.debug(f"Found module '{k}'")

            w("")
            w(f"from . import {v.__name__[len(module_name) + 1:]}")
            w("")

            submodules.append(v)

    if module_name != "falcor":
        w("")
        w(f"from {'.' * (module_name.count('.') + 2)} import falcor")
        w("")
        pass

    return buffer, submodules


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(
            "One argument expected: the falcor python library directory."
        )
    package_dir = sys.argv[1]

    start = time.time()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Generating python stub files in '{package_dir}'")

    import falcor

    logging.debug("Loading plugins ...")
    plugins = json.load(
        open(os.path.join(package_dir, "../plugins/plugins.json"), "r")
    )
    for plugin in plugins:
        logging.debug(f"Loading plugin '{plugin}'")
        falcor.loadPlugin(plugin)

    # Process modules.
    modules = [falcor]
    processed_modules = set()
    while len(modules) > 0:
        m = modules[0]
        modules = modules[1:]
        if m in processed_modules:
            continue

        buffer, submodules = process_module(m)

        module_dir = os.path.join(package_dir, m.__name__.replace(".", os.path.sep))
        os.makedirs(module_dir, exist_ok=True)
        open(os.path.join(module_dir, "__init__.pyi"), "w").write(buffer)

        modules += submodules
        processed_modules.add(m)

    elapsed = time.time() - start
    logging.debug("Done ({:.2f} ms)".format(elapsed * 1000))
