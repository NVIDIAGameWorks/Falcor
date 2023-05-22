import os
import re
import argparse


def remove_hungarian_notation(filename):
    with open(filename, "r") as f:
        contents = f.read()

    pattern = r"([^a-zA-Z0-9_\"'])([msg]?[p][A-Z][a-zA-Z0-9_]*)"

    def remove_hungarian(match):
        var = match.group(2)
        if var[0] == "p":
            return match.group(1) + var[1].lower() + var[2:]
        if var[0] in "msg":
            return match.group(1) + var[0] + var[2:]
        return match.group(0)

    new_contents = re.sub(pattern, remove_hungarian, contents)

    with open(filename, "w") as f:
        f.write(new_contents)


def process_directory(path):
    # recursively process all files in the directory and its subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            # only process C/C++ source files
            if file.endswith(".cpp") or file.endswith(".h"):
                filename = os.path.join(root, file)
                remove_hungarian_notation(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove Hungarian notation from C/C++ identifiers."
    )
    parser.add_argument(
        "path", metavar="path", type=str, help="the path to the directory to process"
    )

    args = parser.parse_args()
    process_directory(path=args.path)
