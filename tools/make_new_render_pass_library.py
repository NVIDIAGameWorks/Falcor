import sys
import argparse
from pathlib import Path

RENDER_PASSES_DIR = Path(sys.argv[0]).parent.parent / "Source" / "RenderPasses"
EXCLUDE_EXT = []
TEMPLATE_NAME = 'PassLibraryTemplate'

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

        print(f'Writing {dst_file}.')

        # Replace all occurrences 'PassLibraryTemplate' and 'RenderPassTemplate' with new project name.
        content = src_file.read_text()
        content = content.replace(TEMPLATE_NAME, name)
        content = content.replace('RenderPassTemplate', name)
        dst_file.write_text(content)

    # Add new subdirectory to CMakeLists.txt.
    cmake_file = RENDER_PASSES_DIR / "CMakeLists.txt"
    with cmake_file.open("a") as f:
        f.write(f'add_subdirectory({name})\n')

    return True


def main():
    parser = argparse.ArgumentParser(description='Script to create a new render pass library.')
    parser.add_argument('name', help='Render pass library name')
    args = parser.parse_args()

    success = create_project(args.name)

    return 0 if success else 1

if __name__ == '__main__':
    main()
