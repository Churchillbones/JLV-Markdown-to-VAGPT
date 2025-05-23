import pytest
from datastore import (
    add_document,
    get_document,
    document_exists,
    update_document_markdown,
    update_document_chunks,
    update_document_embeddings,
    remove_document,
    clear_store  # Imported for potential direct use in tests, though conftest handles it
)

# conftest.py should handle clearing the store for each test

def test_add_and_get_document():
    doc_id = "test_doc_1"
    assert add_document(doc_id) is True
    assert document_exists(doc_id) is True
    
    doc = get_document(doc_id)
    assert doc is not None
    assert doc["markdown_text"] == ""
    assert doc["chunks"] == []
    assert doc["chunk_metadata"] == []
    assert doc["embeddings"] == []

    assert add_document(doc_id) is False # Cannot add existing document

def test_get_non_existent_document():
    assert get_document("non_existent_doc") is None

def test_document_exists_non_existent():
    assert document_exists("non_existent_doc") is False

def test_update_document_markdown():
    doc_id = "test_doc_md"
    add_document(doc_id)
    
    new_markdown = "This is some markdown text."
    assert update_document_markdown(doc_id, new_markdown) is True
    doc = get_document(doc_id)
    assert doc["markdown_text"] == new_markdown
    
    assert update_document_markdown("non_existent_doc", "text") is False

def test_update_document_chunks():
    doc_id = "test_doc_chunks"
    add_document(doc_id)
    
    new_chunks = ["chunk1", "chunk2"]
    new_metadata = ["meta1", "meta2"]
    assert update_document_chunks(doc_id, new_chunks, new_metadata) is True
    doc = get_document(doc_id)
    assert doc["chunks"] == new_chunks
    assert doc["chunk_metadata"] == new_metadata
    
    assert update_document_chunks("non_existent_doc", [], []) is False

def test_update_document_embeddings():
    doc_id = "test_doc_embeddings"
    add_document(doc_id)
    
    new_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    assert update_document_embeddings(doc_id, new_embeddings) is True
    doc = get_document(doc_id)
    assert doc["embeddings"] == new_embeddings
    
    assert update_document_embeddings("non_existent_doc", []) is False

def test_remove_document():
    doc_id = "test_doc_remove"
    add_document(doc_id)
    assert document_exists(doc_id) is True
    
    assert remove_document(doc_id) is True
    assert document_exists(doc_id) is False
    assert get_document(doc_id) is None
    
    assert remove_document("non_existent_doc") is False

def test_clear_store_indirectly_via_fixture():
    # This test relies on the autouse fixture in conftest.py
    # First, add some data
    add_document("doc1")
    add_document("doc2")
    assert document_exists("doc1")
    assert document_exists("doc2")
    # The fixture should clear the store before the next test runs.
    # To verify clear_store itself (though redundant if fixture works):
    # clear_store()
    # assert not document_exists("doc1")
    # assert not document_exists("doc2")
    # This test doesn't explicitly check clear_store, but relies on it for isolation.
    pass
