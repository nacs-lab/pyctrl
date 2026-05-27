"""Phase 0: the environment is wired up correctly. No engine, no hardware."""

import sys

import pytest

pytestmark = pytest.mark.no_hardware


def test_python_version():
    assert sys.version_info >= (3, 8)


def test_compare_bytes_importable():
    import compare_bytes  # noqa: F401  (provided via pythonpath = tools)

    assert hasattr(compare_bytes, "decode")
    assert hasattr(compare_bytes, "encode")
    assert hasattr(compare_bytes, "diff")


def test_references_present():
    from conftest import matlab_reference_files

    files = matlab_reference_files()
    assert files, "expected matlab_new/lib/test/seq*.json reference files"
