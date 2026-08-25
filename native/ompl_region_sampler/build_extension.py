#!/usr/bin/env python3
"""Build the ABI-matched OMPL regional sampler for this interpreter."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _ompl_installation() -> tuple[str, Path]:
    version = importlib.metadata.version("ompl")
    specification = importlib.util.find_spec("ompl")
    if specification is None or specification.origin is None:
        raise RuntimeError("OMPL is not importable by this Python interpreter")
    package = Path(specification.origin).resolve().parent
    library = package / "libompl.so"
    if not library.is_file():
        raise RuntimeError(f"OMPL shared library was not found at {library}")
    return version, library


def _cxx11_abi(library: Path) -> int:
    result = subprocess.run(
        ["nm", "-D", str(library)], check=True,
        text=True, capture_output=True,
    )
    return int("StateSpace7getNameB5cxx11Ev" in result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ompl-source", type=Path,
        help="existing official OMPL source tree matching the wheel version",
    )
    parser.add_argument("--jobs", type=int, default=2)
    arguments = parser.parse_args()

    version, library = _ompl_installation()
    dependency_root = REPOSITORY / ".native_deps"
    source = (
        arguments.ompl_source.resolve()
        if arguments.ompl_source is not None
        else dependency_root / f"ompl-{version}"
    )
    if not source.is_dir():
        dependency_root.mkdir(parents=True, exist_ok=True)
        _run([
            "git", "clone", "--depth", "1", "--branch", version,
            "https://github.com/ompl/ompl.git", str(source),
        ])
    if not (source / "external" / "nanobind" / "CMakeLists.txt").is_file():
        _run([
            "git", "-C", str(source), "submodule", "update", "--init",
            "--recursive", "--depth", "1", "external/nanobind",
        ])

    configured_headers = dependency_root / f"ompl-{version}-configured"
    if configured_headers.exists():
        shutil.rmtree(configured_headers)
    _run([
        "cmake", "-S", str(source), "-B", str(configured_headers),
        "-DOMPL_BUILD_PYTHON_BINDINGS=OFF", "-DOMPL_BUILD_VAMP=OFF",
        "-DOMPL_BUILD_DEMOS=OFF", "-DOMPL_BUILD_TESTS=OFF",
    ])
    generated_include = configured_headers / "src"
    build = HERE / "build"
    if build.exists():
        shutil.rmtree(build)
    _run([
        "cmake", "-S", str(HERE), "-B", str(build),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DOMPL_SOURCE_DIR={source}",
        f"-DOMPL_GENERATED_INCLUDE_DIR={generated_include}",
        f"-DOMPL_LIBRARY={library}",
        f"-DOMPL_CXX11_ABI={_cxx11_abi(library)}",
    ])
    _run(["cmake", "--build", str(build), f"-j{max(1, arguments.jobs)}"])
    candidates = sorted(HERE.glob("_ompl_region_sampler*.so"))
    if not candidates:
        raise RuntimeError("build completed but the extension module is missing")
    print(f"Built {candidates[-1]}")


if __name__ == "__main__":
    main()
