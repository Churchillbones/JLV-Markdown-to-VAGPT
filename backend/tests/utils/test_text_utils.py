import pytest
from utils.text_utils import chunk_text_by_paragraphs # Assuming text_utils.py is in backend/utils

def test_chunk_empty_text():
    assert chunk_text_by_paragraphs("") == []
    assert chunk_text_by_paragraphs("   ") == []
    assert chunk_text_by_paragraphs("\n\n\n") == []

def test_chunk_single_paragraph():
    text = "This is a single paragraph."
    assert chunk_text_by_paragraphs(text) == [text]
    text_with_newlines = "This is a single paragraph\nwith internal newlines."
    assert chunk_text_by_paragraphs(text_with_newlines) == [text_with_newlines]

def test_chunk_multiple_paragraphs():
    text = "Paragraph one.\n\nParagraph two.\n\n\nParagraph three."
    expected = ["Paragraph one.", "Paragraph two.", "Paragraph three."]
    assert chunk_text_by_paragraphs(text) == expected

def test_chunk_with_leading_trailing_newlines():
    text = "\n\nParagraph one.\n\nParagraph two.\n\n"
    expected = ["Paragraph one.", "Paragraph two."]
    assert chunk_text_by_paragraphs(text) == expected

def test_chunk_with_mixed_spacing():
    text = "Para 1.\n \nPara 2.\n\n   \n\nPara 3."
    expected = ["Para 1.", "Para 2.", "Para 3."]
    assert chunk_text_by_paragraphs(text) == expected

def test_chunk_real_text_example():
    text = """This is the first paragraph.
It has multiple lines.

This is the second paragraph. It also has
multiple lines.

This is the third paragraph.
"""
    expected = [
        "This is the first paragraph.\nIt has multiple lines.",
        "This is the second paragraph. It also has\nmultiple lines.",
        "This is the third paragraph."
    ]
    assert chunk_text_by_paragraphs(text) == expected
