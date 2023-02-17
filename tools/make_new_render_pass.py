import sys
import argparse
from pathlib import Path

RENDER_PASSES_DIR = Path(sys.argv[0]).parent.parent / "Source" / "RenderPasses"
EXCLUDE_EXT = []
TEMPLATE_NAME = "RenderPassTemplate"


def create_project(name):
    # Source and destination directories.
    src_dir = RENDER_PASSES_DIR / TEMPLATE_NAME
    dst_dir = RENDER_PASSES_DIR / name

    print(f'Creating render pass library "{name}":')

    # Check that destination does not exist.
    if dst_dir.exists():
        print(f'"{name}" already exists!')
        return False

    # Create destination folder.
    dst_dir.mkdir()

    # Copy project template.
    for src_file in filter(lambda f: not f.suffix in EXCLUDE_EXT, src_dir.iterdir()):
        dst_file = dst_dir / (src_file.name.replace(TEMPLATE_NAME, name))

        print(f"Writing {dst_file}.")

        # Replace all occurrences of 'RenderPassTemplate' with new name.
        content = src_file.read_text()
        content = content.replace(TEMPLATE_NAME, name)
        dst_file.write_text(content)

    # Add new subdirectory to CMakeLists.txt.
    cmake_file = RENDER_PASSES_DIR / "CMakeLists.txt"
    lines = [line for line in open(cmake_file, "r").readlines() if line.strip()]
    lines.append(f"add_subdirectory({name})\n")
    lines.sort(key=str.lower)
    open(cmake_file, "w").writelines(lines)

    return True


def main():
    parser = argparse.ArgumentParser(description="Script to create a new render pass.")
    parser.add_argument("name", help="Render pass name")
    args = parser.parse_args()

    success = create_project(args.name)

    return 0 if success else 1


if __name__ == "__main__":
    main()
