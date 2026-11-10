# Copyright © 2026 mikepapadim

import os

from mlx import extension
from setuptools import setup

if __name__ == "__main__":
    # mlx.extension.CMakeBuild configures the project with CMAKE_ARGS appended:
    # build the bindings against the interpreter's mlx, not the C++ tests
    os.environ["CMAKE_ARGS"] = (
        os.environ.get("CMAKE_ARGS", "")
        + " -DMLXVS_BUILD_PYTHON=ON -DMLXVS_BUILD_TESTS=OFF"
    )
    setup(
        ext_modules=[extension.CMakeExtension("mlxvs._core")],
        cmdclass={"build_ext": extension.CMakeBuild},
    )
