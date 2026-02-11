"""Example tests for Qanta."""

import qanta


def test_import() -> None:
    """Test that qanta can be imported."""
    assert hasattr(qanta, '__version__')


def test_version_is_string() -> None:
    """Test that version is a string."""
    assert isinstance(qanta.__version__, str)
