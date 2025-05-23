import pytest
from unittest.mock import patch, MagicMock
import io
import json
from datastore import get_document, document_store # To inspect the store after upload

# client fixture is from conftest.py

def test_upload_no_file(client):
    response = client.post('/api/upload')
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == "No file part"

def test_upload_empty_filename(client):
    data = {'file': (io.BytesIO(b"dummy content"), '')}
    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == "No selected file"

@patch('utils.converter.MarkdownConverter.convert_to_markdown')
@patch('utils.converter.MarkdownConverter.extract_text_and_metadata_per_page')
def test_upload_docx_file(mock_extract_metadata, mock_convert_to_markdown, client):
    mock_convert_to_markdown.return_value = "Converted DOCX Markdown."
    # For DOCX, extract_text_and_metadata_per_page is not directly called by the main path,
    # but it's good to have it mocked if any internal logic changes.
    # We expect chunk_text_by_paragraphs to be called on the "Converted DOCX Markdown."
    
    file_content = b"dummy docx content"
    file_name = "test.docx"
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 200
    json_data = response.get_json()
    
    assert 'doc_id' in json_data
    doc_id = json_data['doc_id']
    assert json_data['filename'] == file_name
    assert json_data['markdown_text'] == "Converted DOCX Markdown."
    # Based on "Converted DOCX Markdown.", chunk_text_by_paragraphs will return ["Converted DOCX Markdown."]
    assert json_data['chunks'] == ["Converted DOCX Markdown."] 
    assert json_data['chunk_metadata'] == ["No specific metadata extracted for this chunk."]

    mock_convert_to_markdown.assert_called_once_with(file_content, file_name)
    
    # Verify datastore
    stored_doc = get_document(doc_id)
    assert stored_doc is not None
    assert stored_doc['markdown_text'] == "Converted DOCX Markdown."
    assert stored_doc['chunks'] == ["Converted DOCX Markdown."]
    assert stored_doc['chunk_metadata'] == ["No specific metadata extracted for this chunk."]

@patch('utils.converter.MarkdownConverter.convert_to_markdown')
@patch('utils.converter.MarkdownConverter.extract_text_and_metadata_per_page')
def test_upload_pdf_file(mock_extract_metadata, mock_convert_to_markdown, client):
    # This will be the raw output of convert_to_markdown, often used for display
    mock_convert_to_markdown.return_value = "Full PDF Markdown from markitdown or fallback."
    
    # This is what extract_text_and_metadata_per_page returns for PDF processing
    mock_extract_metadata.return_value = [
        (1, "Page 1 text.", "Metadata for page 1"),
        (2, "Page 2 chunk 1.\n\nPage 2 chunk 2.", "Metadata for page 2")
    ]
    
    file_content = b"dummy pdf content"
    file_name = "test.pdf"
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 200
    json_data = response.get_json()
    
    assert 'doc_id' in json_data
    doc_id = json_data['doc_id']
    assert json_data['filename'] == file_name
    assert json_data['markdown_text'] == "Full PDF Markdown from markitdown or fallback."
    
    # Expected chunks and metadata based on mock_extract_metadata
    expected_chunks = ["Page 1 text.", "Page 2 chunk 1.", "Page 2 chunk 2."]
    expected_metadata = ["Metadata for page 1", "Metadata for page 2", "Metadata for page 2"]
    
    assert json_data['chunks'] == expected_chunks
    assert json_data['chunk_metadata'] == expected_metadata

    mock_convert_to_markdown.assert_called_once_with(file_content, file_name)
    mock_extract_metadata.assert_called_once_with(file_content)
    
    # Verify datastore
    stored_doc = get_document(doc_id)
    assert stored_doc is not None
    assert stored_doc['markdown_text'] == "Full PDF Markdown from markitdown or fallback."
    assert stored_doc['chunks'] == expected_chunks
    assert stored_doc['chunk_metadata'] == expected_metadata

@patch('utils.converter.MarkdownConverter.convert_to_markdown', side_effect=Exception("Conversion failed badly"))
def test_upload_processing_error(mock_convert_to_markdown, client):
    file_content = b"some content"
    file_name = "test.txt" # Using .txt to ensure it hits a path that calls convert_to_markdown
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 500
    json_data = response.get_json()
    assert json_data['error'] == "An unexpected error occurred during file processing."
    assert "Conversion failed badly" in json_data['details']

@patch('utils.converter.MarkdownConverter.convert_to_markdown')
@patch('utils.converter.MarkdownConverter.extract_text_and_metadata_per_page')
def test_upload_pdf_file_empty_page_text(mock_extract_metadata, mock_convert_to_markdown, client):
    mock_convert_to_markdown.return_value = "Full PDF Markdown."
    mock_extract_metadata.return_value = [
        (1, "Page 1 text.", "Metadata for page 1"),
        (2, "", "Metadata for page 2 but no text"), # Empty text for page 2
        (3, "Page 3 text.", "Metadata for page 3")
    ]
    
    file_content = b"dummy pdf content"
    file_name = "test.pdf"
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 200
    json_data = response.get_json()
    doc_id = json_data['doc_id']
    
    expected_chunks = ["Page 1 text.", "Page 3 text."]
    expected_metadata = ["Metadata for page 1", "Metadata for page 3"]
    
    assert json_data['chunks'] == expected_chunks
    assert json_data['chunk_metadata'] == expected_metadata

    stored_doc = get_document(doc_id)
    assert stored_doc['chunks'] == expected_chunks
    assert stored_doc['chunk_metadata'] == expected_metadata

@patch('utils.converter.MarkdownConverter.convert_to_markdown')
@patch('utils.converter.MarkdownConverter.extract_text_and_metadata_per_page')
def test_upload_pdf_file_no_metadata_extracted(mock_extract_metadata, mock_convert_to_markdown, client):
    mock_convert_to_markdown.return_value = "Full PDF Markdown."
    # Simulate case where extract_text_and_metadata_per_page returns empty list
    mock_extract_metadata.return_value = [] 
    
    file_content = b"dummy pdf content"
    file_name = "test.pdf"
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 200
    json_data = response.get_json()
    doc_id = json_data['doc_id']
    
    # Fallback logic should chunk the entire "Full PDF Markdown."
    expected_chunks = ["Full PDF Markdown."]
    expected_metadata = ["No specific metadata extracted for this chunk."]
    
    assert json_data['chunks'] == expected_chunks
    assert json_data['chunk_metadata'] == expected_metadata

    stored_doc = get_document(doc_id)
    assert stored_doc['chunks'] == expected_chunks
    assert stored_doc['chunk_metadata'] == expected_metadata
    mock_extract_metadata.assert_called_once_with(file_content)

@patch('utils.converter.MarkdownConverter.convert_to_markdown')
def test_upload_unsupported_file_type_message_in_markdown(mock_convert_to_markdown, client):
    # Simulate converter returning the "Unsupported file type" message
    unsupported_message = "Unsupported file type: .xyz. Please upload a PDF or DOCX file."
    mock_convert_to_markdown.return_value = unsupported_message
    
    file_content = b"some content"
    file_name = "test.xyz" # An unsupported type
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 200 # Endpoint should still succeed
    json_data = response.get_json()
    doc_id = json_data['doc_id']
    
    assert json_data['markdown_text'] == unsupported_message
    # The fallback chunking logic for error messages in markdown_text
    assert json_data['chunks'] == ["Error in document processing or document is empty."]
    assert json_data['chunk_metadata'] == ["No specific metadata extracted for this chunk."]

    stored_doc = get_document(doc_id)
    assert stored_doc['markdown_text'] == unsupported_message
    assert stored_doc['chunks'] == ["Error in document processing or document is empty."]
    assert stored_doc['chunk_metadata'] == ["No specific metadata extracted for this chunk."]

@patch('utils.converter.MarkdownConverter.convert_to_markdown')
def test_upload_empty_markdown_from_converter(mock_convert_to_markdown, client):
    # Simulate converter returning empty string or only whitespace
    mock_convert_to_markdown.return_value = "   " 
    
    file_content = b"some content"
    file_name = "test.docx"
    data = {'file': (io.BytesIO(file_content), file_name)}

    response = client.post('/api/upload', content_type='multipart/form-data', data=data)
    
    assert response.status_code == 200
    json_data = response.get_json()
    doc_id = json_data['doc_id']
    
    assert json_data['markdown_text'] == "   "
    # Fallback logic for empty/whitespace markdown
    assert json_data['chunks'] == ["Error in document processing or document is empty."]
    assert json_data['chunk_metadata'] == ["No specific metadata extracted for this chunk."]

    stored_doc = get_document(doc_id)
    assert stored_doc['markdown_text'] == "   "
    assert stored_doc['chunks'] == ["Error in document processing or document is empty."]
    assert stored_doc['chunk_metadata'] == ["No specific metadata extracted for this chunk."]
