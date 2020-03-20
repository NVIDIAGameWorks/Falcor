import logging
import zipfile
import tempfile
import sys
import shutil

__author__ = 'hfannar'
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger('install_package')


class TemporaryDirectory:
    def __init__(self):
        self.path = None

    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, type, value, traceback):
        # Remove temporary data created
        shutil.rmtree(self.path)


def install_package(package_src_path, package_dst_path):
    with zipfile.ZipFile(package_src_path, allowZip64=True) as zip_file, TemporaryDirectory() as temp_dir:
        zip_file.extractall(temp_dir)
        # Recursively copy (temp_dir will be automatically cleaned up on exit)
        try:
            # Recursive copy is needed because both package name and version folder could be missing in
            # target directory:
            shutil.copytree(temp_dir, package_dst_path)
        except OSError, exc:
            logger.warning("Directory %s already present, packaged installation aborted" % package_dst_path)
        else:
            logger.info("Package successfully installed to %s" % package_dst_path)


install_package(sys.argv[1], sys.argv[2])
