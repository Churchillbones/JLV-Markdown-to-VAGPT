import pytest
from unittest.mock import patch, MagicMock, mock_open
from utils.converter import MarkdownConverter
from PyPDF2 import PdfReader, errors as PyPDF2Errors
import io
import subprocess

# Sample DOCX content (simplified for testing)
SAMPLE_DOCX_CONTENT = b"word_doc_content" # Actual DOCX bytes would be more complex
EXPECTED_DOCX_MARKDOWN = "Paragraph 1 text.\n\nParagraph 2 text."

# Sample PDF content for PyPDF2 mocking
SAMPLE_PDF_TEXT_PAGE_1 = "This is page 1 of the PDF. Signature: John Doe Date: 01/01/2023"
SAMPLE_PDF_TEXT_PAGE_2 = "This is page 2. Signed by Jane Smith on 02/15/2023"
EXPECTED_PDF_MARKDOWN_FALLBACK = SAMPLE_PDF_TEXT_PAGE_1 + "\n\n" + SAMPLE_PDF_TEXT_PAGE_2 + "\n\n"

@pytest.fixture
def mock_docx_document():
    # Create a mock Document object
    mock_doc = MagicMock()
    para1 = MagicMock()
    para1.text = "Paragraph 1 text."
    para2 = MagicMock()
    para2.text = "Paragraph 2 text."
    para3 = MagicMock()
    para3.text = "   " # Empty paragraph
    mock_doc.paragraphs = [para1, para2, para3]
    return mock_doc

def test_convert_docx_to_markdown(mocker, mock_docx_document):
    mocker.patch('docx.Document', return_value=mock_docx_document)
    
    markdown = MarkdownConverter.convert_docx_to_markdown(SAMPLE_DOCX_CONTENT)
    assert markdown == EXPECTED_DOCX_MARKDOWN

def test_convert_docx_to_markdown_error(mocker):
    mocker.patch('docx.Document', side_effect=Exception("DOCX parsing error"))
    result = MarkdownConverter.convert_docx_to_markdown(b"bad_docx_content")
    assert "Error converting DOCX to markdown: DOCX parsing error" in result

@pytest.fixture
def mock_pdf_reader_pages():
    # Mock PdfReader.pages
    page1_mock = MagicMock()
    page1_mock.extract_text.return_value = SAMPLE_PDF_TEXT_PAGE_1
    page2_mock = MagicMock()
    page2_mock.extract_text.return_value = SAMPLE_PDF_TEXT_PAGE_2
    return [page1_mock, page2_mock]

def test_extract_text_from_pdf(mocker, mock_pdf_reader_pages):
    mock_pdf = MagicMock()
    mock_pdf.pages = mock_pdf_reader_pages
    mocker.patch('PyPDF2.PdfReader', return_value=mock_pdf)
    
    text = MarkdownConverter.extract_text_from_pdf(b"sample_pdf_bytes")
    assert text == EXPECTED_PDF_MARKDOWN_FALLBACK

def test_extract_text_from_pdf_read_error(mocker):
    mocker.patch('PyPDF2.PdfReader', side_effect=PyPDF2Errors.PdfReadError("Bad PDF"))
    result = MarkdownConverter.extract_text_from_pdf(b"bad_pdf_bytes")
    assert "Error extracting PDF text: Bad PDF" in result

def test_convert_pdf_to_markdown_markitdown_success(mocker):
    mock_subprocess_run = mocker.patch('subprocess.run')
    mock_subprocess_run.return_value = MagicMock(
        returncode=0, stdout="Markitdown successful output", stderr=""
    )
    # Mock open for reading the output file
    mocker.patch('builtins.open', mock_open(read_data="Markitdown successful output"))
    
    # Mock os.path.exists and os.unlink to simulate file operations
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.unlink')

    markdown = MarkdownConverter.convert_pdf_to_markdown(b"sample_pdf_bytes", "test.pdf")
    assert markdown == "Markitdown successful output"
    subprocess.run.assert_called_once() # Check if markitdown was called
    assert "markitdown" in subprocess.run.call_args[0][0]

def test_convert_pdf_to_markdown_markitdown_fail_fallback(mocker, mock_pdf_reader_pages):
    # Mock subprocess.run to simulate markitdown failure
    mock_subprocess_run = mocker.patch('subprocess.run')
    mock_subprocess_run.return_value = MagicMock(
        returncode=1, stdout="Error output", stderr="Markitdown error details"
    )
    
    # Mock the fallback: PdfReader for extract_text_from_pdf
    mock_pdf_fallback = MagicMock()
    mock_pdf_fallback.pages = mock_pdf_reader_pages
    mocker.patch('PyPDF2.PdfReader', return_value=mock_pdf_fallback)

    # Mock os.path.exists and os.unlink
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('os.unlink')

    markdown = MarkdownConverter.convert_pdf_to_markdown(b"sample_pdf_bytes", "test.pdf")
    
    assert "Error during markitdown conversion. Falling back to direct text extraction:" in markdown
    assert EXPECTED_PDF_MARKDOWN_FALLBACK in markdown
    subprocess.run.assert_called_once()

def test_extract_text_and_metadata_per_page(mocker, mock_pdf_reader_pages):
    mock_pdf = MagicMock()
    mock_pdf.pages = mock_pdf_reader_pages
    mocker.patch('PyPDF2.PdfReader', return_value=mock_pdf)

    results = MarkdownConverter.extract_text_and_metadata_per_page(b"sample_pdf_bytes")
    
    assert len(results) == 2
    
    # Page 1
    assert results[0][0] == 1 # Page number
    assert results[0][1] == SAMPLE_PDF_TEXT_PAGE_1
    assert "Signed by John Doe" in results[0][2] # Metadata string
    assert "01/01/2023" in results[0][2]

    # Page 2
    assert results[1][0] == 2 # Page number
    assert results[1][1] == SAMPLE_PDF_TEXT_PAGE_2
    assert "Signed by Jane Smith" in results[1][2]
    assert "02/15/2023" in results[1][2]

def test_extract_text_and_metadata_no_match():
    # Test with text that doesn't match regex patterns
    page_text_no_match = "This page has no standard signature or date format."
    
    mock_page = MagicMock()
    mock_page.extract_text.return_value = page_text_no_match
    
    mock_pdf_reader = MagicMock()
    mock_pdf_reader.pages = [mock_page]
    
    with patch('PyPDF2.PdfReader', return_value=mock_pdf_reader):
        results = MarkdownConverter.extract_text_and_metadata_per_page(b"dummy_pdf_content")
        
    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] == page_text_no_match
    assert results[0][2] == "Metadata is missing"

def test_convert_to_markdown_pdf(mocker):
    mocker.patch.object(MarkdownConverter, 'convert_pdf_to_markdown', return_value="pdf_markdown")
    result = MarkdownConverter.convert_to_markdown(b"pdf_content", "file.pdf")
    assert result == "pdf_markdown"
    MarkdownConverter.convert_pdf_to_markdown.assert_called_once_with(b"pdf_content", "file.pdf")

def test_convert_to_markdown_docx(mocker):
    mocker.patch.object(MarkdownConverter, 'convert_docx_to_markdown', return_value="docx_markdown")
    result = MarkdownConverter.convert_to_markdown(b"docx_content", "file.docx")
    assert result == "docx_markdown"
    MarkdownConverter.convert_docx_to_markdown.assert_called_once_with(b"docx_content")

def test_convert_to_markdown_unsupported():
    result = MarkdownConverter.convert_to_markdown(b"txt_content", "file.txt")
    assert "Unsupported file type: .txt" in result
