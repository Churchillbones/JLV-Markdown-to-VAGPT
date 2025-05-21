import unittest
from unittest.mock import MagicMock, patch
import io
import sys
import os

# Add src to Python path to allow direct import of src.utils.converter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.converter import MarkdownConverter
from PyPDF2 import errors as PyPDF2Errors # For simulating PDF errors

class TestMetadataExtraction(unittest.TestCase):

    @patch('utils.converter.PdfReader') # Patch PdfReader in the context of where it's used
    def test_extract_metadata_per_page(self, mock_pdf_reader_constructor):
        # --- Mocking PyPDF2 ---
        mock_pdf_instance = MagicMock()
        mock_pdf_reader_constructor.return_value = mock_pdf_instance

        # Define page content and expected metadata
        page_contents_and_expected_metadata = [
            {
                "text": "Page 1 content.\nSome other text.\nSigned by: John Doe, MD on Date: January 1, 2023",
                "expected_metadata": "Signed by John Doe, MD on January 1, 2023"
            },
            {
                "text": "Page 2 has only a name.\nMore text.\nPhysician: Dr. Jane Smith",
                "expected_metadata": "Signed by Dr. Jane Smith (date not found)"
            },
            {
                "text": "Page 3 has only a date.\nFinal remarks.\nDate: 12/25/2022",
                "expected_metadata": "Date found: 12/25/2022 (signer not found)"
            },
            {
                "text": "Page 4 has metadata in the middle: Signed by: Mid Doc on Feb 2, 2024. This should not be picked up if heuristic focuses on bottom.",
                "expected_metadata": "Metadata is missing" # Assuming heuristic focuses on bottom
            },
            {
                "text": "Page 5 is empty.", # Test with page_object.extract_text() returning None or empty
                "expected_metadata": "Metadata is missing"
            },
            {
                "text": "Page 6 with complex name and date.\nElectronically Signed by: Dr. Emily R. White-PhD, NP on March 15th, 2023",
                "expected_metadata": "Signed by Dr. Emily R. White-PhD, NP on March 15th, 2023" # Note: "th" might not be captured by current date regex
            },
             {
                "text": "Page 7 with just a simple name.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nAlice Wonderland", # Fallback name
                "expected_metadata": "Signed by Alice Wonderland (date not found)"
            },
            {
                "text": "Page 8 with only a date at the very end.\n\n\n05-05-2025",
                "expected_metadata": "Date found: 05-05-2025 (signer not found)"
            }
        ]

        mock_pages = []
        for i, content_spec in enumerate(page_contents_and_expected_metadata):
            page_mock = MagicMock()
            if content_spec["text"] == "Page 5 is empty.": # Simulate extract_text returning None
                page_mock.extract_text.return_value = None
            else:
                page_mock.extract_text.return_value = content_spec["text"]
            mock_pages.append(page_mock)
        
        mock_pdf_instance.pages = mock_pages

        # --- Call the method ---
        # Simulate binary file content (not strictly needed due to mocking, but good practice)
        dummy_file_content = b"%PDF-1.4 dummy content"
        results = MarkdownConverter.extract_text_and_metadata_per_page(dummy_file_content)

        # --- Assertions ---
        self.assertEqual(len(results), len(page_contents_and_expected_metadata))

        for i, (page_num, page_text, metadata_string) in enumerate(results):
            self.assertEqual(page_num, i + 1)
            expected_page_text = page_contents_and_expected_metadata[i]["text"]
            if expected_page_text == "Page 5 is empty.":
                 self.assertEqual(page_text, "") # Ensure None from extract_text becomes empty string
            else:
                self.assertEqual(page_text, expected_page_text)
            
            # Refine expected metadata for page 6 due to "th" in date
            if i == 5: # Page 6 (0-indexed)
                 self.assertEqual(metadata_string, "Signed by Dr. Emily R. White-PhD, NP on March 15, 2023")
            else:
                self.assertEqual(metadata_string, page_contents_and_expected_metadata[i]["expected_metadata"])


    @patch('utils.converter.PdfReader')
    def test_pdf_read_error(self, mock_pdf_reader_constructor):
        """Test behavior when PdfReader raises PdfReadError."""
        mock_pdf_reader_constructor.side_effect = PyPDF2Errors.PdfReadError("Simulated PDF read error")
        
        dummy_file_content = b"corrupted pdf content"
        results = MarkdownConverter.extract_text_and_metadata_per_page(dummy_file_content)
        
        self.assertEqual(results, [])

    @patch('utils.converter.PdfReader')
    def test_general_exception_during_processing(self, mock_pdf_reader_constructor):
        """Test behavior with a general exception during page processing."""
        mock_pdf_instance = MagicMock()
        mock_pdf_reader_constructor.return_value = mock_pdf_instance
        
        # Simulate an error during text extraction for the first page
        page_mock = MagicMock()
        page_mock.extract_text.side_effect = Exception("Simulated extraction error")
        mock_pdf_instance.pages = [page_mock]
        
        dummy_file_content = b"%PDF-1.4 working pdf"
        results = MarkdownConverter.extract_text_and_metadata_per_page(dummy_file_content)
        
        # Current implementation returns [] on any Exception.
        # If it were to process other pages, this test would change.
        self.assertEqual(results, []) # Or check for partial results if that's the design

    @patch('utils.converter.PdfReader')
    def test_no_pages_pdf(self, mock_pdf_reader_constructor):
        """Test behavior with a PDF that has no pages."""
        mock_pdf_instance = MagicMock()
        mock_pdf_reader_constructor.return_value = mock_pdf_instance
        mock_pdf_instance.pages = [] # No pages

        dummy_file_content = b"%PDF-1.4 empty pdf"
        results = MarkdownConverter.extract_text_and_metadata_per_page(dummy_file_content)
        self.assertEqual(results, [])


if __name__ == '__main__':
    unittest.main()
