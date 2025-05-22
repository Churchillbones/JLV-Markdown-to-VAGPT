import unittest
import sys
import os

# Add src to Python path to allow direct import of src.utils.text_utils
# This assumes test_text_utils.py is in src/tests and text_utils.py is in src/utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_utils import chunk_text_by_paragraphs

class TestTextChunking(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(chunk_text_by_paragraphs(""), [])
        self.assertEqual(chunk_text_by_paragraphs("   "), []) # Only whitespace

    def test_single_paragraph(self):
        text = "This is a single paragraph with no breaks."
        self.assertEqual(chunk_text_by_paragraphs(text), [text])
        text_with_newlines = "This is a single paragraph\nwith internal newlines."
        self.assertEqual(chunk_text_by_paragraphs(text_with_newlines), [text_with_newlines])

    def test_standard_paragraph_breaks(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        expected = ["Paragraph one.", "Paragraph two.", "Paragraph three."]
        self.assertEqual(chunk_text_by_paragraphs(text), expected)

    def test_multiple_consecutive_blank_lines(self):
        text = "Para 1.\n\n\nPara 2.\n\n\n\nPara 3."
        expected = ["Para 1.", "Para 2.", "Para 3."]
        self.assertEqual(chunk_text_by_paragraphs(text), expected)
        
        text_with_spaces_in_blank_lines = "Para A.\n \nPara B.\n  \n\nPara C."
        expected_b = ["Para A.", "Para B.", "Para C."]
        self.assertEqual(chunk_text_by_paragraphs(text_with_spaces_in_blank_lines), expected_b)

    def test_leading_trailing_whitespace_and_blank_lines(self):
        text = "\n\n  Leading blank lines and spaces.\n\nMiddle paragraph.\n\nTrailing paragraph with spaces.  \n\n"
        expected = [
            "Leading blank lines and spaces.",
            "Middle paragraph.",
            "Trailing paragraph with spaces."
        ]
        self.assertEqual(chunk_text_by_paragraphs(text), expected)

        text_only_blank_lines = "\n\n\n   \n\n"
        self.assertEqual(chunk_text_by_paragraphs(text_only_blank_lines), [])


    def test_mixed_newlines(self):
        text = "Paragraph one using LF.\n\nParagraph two using CRLF.\r\n\r\nParagraph three using LF again."
        # The regex \n\s*\n should handle both \n and \r\n as \n after strip() and split
        # and considering how re.split works with newlines.
        expected = [
            "Paragraph one using LF.",
            "Paragraph two using CRLF.", # Assuming \r is stripped or handled by regex
            "Paragraph three using LF again."
        ]
        # Normalizing input for consistent testing across platforms if regex doesn't fully cover \r
        normalized_text = text.replace('\r\n', '\n')
        self.assertEqual(chunk_text_by_paragraphs(normalized_text), expected)
        # Test directly as well, as re.split should handle it
        self.assertEqual(chunk_text_by_paragraphs(text), expected)


    def test_chunks_are_stripped(self):
        text = "  Chunk 1 with spaces.  \n\n  Chunk 2 also.  "
        expected = ["Chunk 1 with spaces.", "Chunk 2 also."]
        self.assertEqual(chunk_text_by_paragraphs(text), expected)

if __name__ == '__main__':
    unittest.main()
