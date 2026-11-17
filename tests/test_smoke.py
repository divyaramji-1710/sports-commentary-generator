"""Basic checks for the commentary application."""
import ast
from pathlib import Path

import pytest

from app.encoding import format_prompt


def test_page_is_valid_python():
    source = Path("app/main.py").read_text(encoding="utf-8")
    ast.parse(source)


def test_prompt_format():
    assert format_prompt("  Goal for City.  ") == (
        "Event: Goal for City.\nCommentary:\n"
    )


def test_empty_event_is_rejected():
    with pytest.raises(ValueError):
        format_prompt("   ")