import sys

from pybind11_stubgen import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise RuntimeError(
            "One argument expected: the falcor python library directory."
        )
    package_dir = sys.argv[1]
    main(["-o", package_dir, "--ignore-invalid=all", "--skip-signature-downgrade", "--no-setup-py", "--root-module-suffix=", "falcor"])
