import os
import sys


def get_packman_module_dir():
    root_dir = os.environ['PM_PACKAGES_ROOT']
    common_dir = '4.1-common'
    module_dir = os.path.join(root_dir, 'packman', common_dir)
    return module_dir


sys.path.insert(0, get_packman_module_dir())
from packman import pack
from packman import push
from packager import get_package_filename
from packager import create_package_from_file_list
