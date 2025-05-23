import pytest
import json
from unittest.mock import MagicMock, patch
from datastore import (
    add_document,
    update_document_chunks,
    get_document,
    document_store, # For direct inspection if needed, but get_document is preferred
    clear_store # Though conftest handles this, good for clarity
)

# client fixture is from conftest.py
# mocker fixture is from pytest-mock

MOCK_EMBEDDING_MODEL_NAME = "test-embedding-model"

@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-api-key")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", MOCK_EMBEDDING_MODEL_NAME)


def test_embed_successful(client, mocker):
    doc_id = "test_doc_embed_success"
    chunks = ["This is chunk 1.", "Another chunk here."]
    metadata = ["meta1", "meta2"]
    
    add_document(doc_id)
    update_document_chunks(doc_id, chunks, metadata)

    mock_embedding_vector_1 = [0.1, 0.2, 0.3]
    mock_embedding_vector_2 = [0.4, 0.5, 0.6]

    mock_azure_embedder_instance = MagicMock()
    mock_azure_embedder_instance.get_embedding.side_effect = [
        mock_embedding_vector_1, 
        mock_embedding_vector_2
    ]
    
    mocker.patch('app.AzureEmbedder', return_value=mock_azure_embedder_instance) # Patch in app's context
    
    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['doc_id'] == doc_id
    assert json_data['status'] == "success"
    assert json_data['successful_embeddings_count'] == 2
    assert json_data['total_chunks_processed'] == 2

    stored_doc = get_document(doc_id)
    assert stored_doc is not None
    assert stored_doc['embeddings'] == [mock_embedding_vector_1, mock_embedding_vector_2]
    
    # Verify AzureEmbedder was called with the correct model name from env var
    # The patch path for AzureEmbedder should be where it's looked up (i.e., in 'app.py' or 'backend.app')
    app_module_path = 'app' # Assuming app.py is in backend/ and tests run from backend/
    # If tests run from root, it might be 'backend.app.AzureEmbedder'
    # For this setup, 'app.AzureEmbedder' is common if app.py is in the same dir as tests/ or PYTHONPATH is set up.
    # Let's assume app is where AzureEmbedder is imported.
    # We need to patch 'app.AzureEmbedder' or 'backend.app.AzureEmbedder'
    # Based on the provided app.py, it's imported as 'from utils.embedder import AzureEmbedder'
    # So, the path for patching within app.py context would be 'app.AzureEmbedder' if tests are run from backend folder
    # Or, more robustly, patch where it's defined: 'utils.embedder.AzureEmbedder'
    # However, the prompt implies patching where it's USED in the endpoint, so app.AzureEmbedder
    
    # The actual constructor call is made within the app.py, so we check its call args
    # This requires patching 'utils.embedder.AzureEmbedder' if we want to check constructor,
    # or ensuring the mocked instance from `app.AzureEmbedder` was constructed correctly.
    # The current mock `mocker.patch('app.AzureEmbedder', ...)` replaces the class itself.
    # So, app.AzureEmbedder.call_args will give us how the class was called.
    # For simplicity, we trust the mock was set up. The critical check is get_embedding calls.
    
    mock_azure_embedder_instance.get_embedding.assert_any_call(chunks[0])
    mock_azure_embedder_instance.get_embedding.assert_any_call(chunks[1])


def test_embed_document_not_found(client):
    response = client.post('/api/embed', json={'doc_id': "non_existent_doc"})
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data['error'] == "Document not found"

def test_embed_no_doc_id_in_request(client):
    response = client.post('/api/embed', json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['error'] == "Missing doc_id in request body"

    response_no_json = client.post('/api/embed')
    assert response_no_json.status_code == 400 # Flask default for missing JSON with get_json()
    # The error message might vary slightly depending on Flask version and if request.get_json(silent=True) is used
    # assert "Failed to decode JSON" in response_no_json.get_json()['error'] or "Invalid JSON"

def test_embed_no_chunks_in_document(client):
    doc_id = "test_doc_no_chunks"
    add_document(doc_id)
    update_document_chunks(doc_id, [], []) # Empty chunks

    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['doc_id'] == doc_id
    assert json_data['status'] == "success"
    assert json_data['message'] == "No chunks to embed." # Or "No chunks found..."
    assert json_data['embeddings_count'] == 0

    stored_doc = get_document(doc_id)
    assert stored_doc['embeddings'] == []

def test_embed_azure_embedder_init_fails_value_error(client, mocker):
    doc_id = "test_doc_embed_init_fail_value"
    add_document(doc_id)
    update_document_chunks(doc_id, ["chunk1"], ["meta1"])

    # Patch AzureEmbedder in the context of the 'app' module where it's used
    mocker.patch('app.AzureEmbedder', side_effect=ValueError("Azure config error"))
    
    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 500
    json_data = response.get_json()
    assert "Azure OpenAI configuration error" in json_data['error']

def test_embed_azure_embedder_init_fails_other_error(client, mocker):
    doc_id = "test_doc_embed_init_fail_other"
    add_document(doc_id)
    update_document_chunks(doc_id, ["chunk1"], ["meta1"])

    mocker.patch('app.AzureEmbedder', side_effect=Exception("Some other init error"))
    
    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 500
    json_data = response.get_json()
    assert "Could not initialize embedding service" in json_data['error']


def test_embed_individual_chunk_fails(client, mocker):
    doc_id = "test_doc_chunk_fail"
    chunks = ["chunk success 1", "chunk fail", "chunk success 2", "   ", ""] # Added empty/whitespace chunks
    metadata = ["meta1", "meta2", "meta3", "meta4", "meta5"]
    
    add_document(doc_id)
    update_document_chunks(doc_id, chunks, metadata)

    mock_embedding_vector_1 = [0.1, 0.1, 0.1]
    mock_embedding_vector_3 = [0.3, 0.3, 0.3]

    mock_azure_embedder_instance = MagicMock()
    
    def get_embedding_side_effect(text):
        if text == "chunk success 1":
            return mock_embedding_vector_1
        elif text == "chunk fail":
            raise Exception("Embedding failed for this chunk")
        elif text == "chunk success 2":
            return mock_embedding_vector_3
        # Empty/whitespace chunks should be skipped by the endpoint logic before calling get_embedding
        return None # Should not be called for empty/whitespace

    mock_azure_embedder_instance.get_embedding.side_effect = get_embedding_side_effect
    mocker.patch('app.AzureEmbedder', return_value=mock_azure_embedder_instance)
    
    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['doc_id'] == doc_id
    assert json_data['status'] == "success"
    # Chunks processed = 5 (chunk success 1, chunk fail, chunk success 2, "   ", "")
    # Successful embeddings = 2 (chunk success 1, chunk success 2)
    # Failed embedding = 1 (chunk fail)
    # Skipped = 2 ("   ", "")
    assert json_data['total_chunks_processed'] == 5 
    assert json_data['successful_embeddings_count'] == 2 

    stored_doc = get_document(doc_id)
    assert stored_doc is not None
    # Expected: vector, None (for failure), vector, None (skipped whitespace), None (skipped empty)
    assert stored_doc['embeddings'] == [
        mock_embedding_vector_1, 
        None, 
        mock_embedding_vector_3,
        None, # For "   "
        None  # For ""
    ]

    # Check calls to get_embedding (should only be called for non-empty chunks)
    assert mock_azure_embedder_instance.get_embedding.call_count == 3
    mock_azure_embedder_instance.get_embedding.assert_any_call("chunk success 1")
    mock_azure_embedder_instance.get_embedding.assert_any_call("chunk fail")
    mock_azure_embedder_instance.get_embedding.assert_any_call("chunk success 2")

def test_embed_empty_chunks_list(client, mocker):
    doc_id = "test_doc_empty_chunks_list"
    add_document(doc_id)
    update_document_chunks(doc_id, [], []) # Explicitly empty list

    mock_azure_embedder_constructor = mocker.patch('app.AzureEmbedder')

    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['message'] == "No chunks to embed."
    assert json_data['embeddings_count'] == 0
    
    stored_doc = get_document(doc_id)
    assert stored_doc['embeddings'] == []
    
    # AzureEmbedder should not even be initialized if there are no chunks
    mock_azure_embedder_constructor.assert_not_called()

def test_embed_chunks_key_exists_but_is_none_or_empty_in_datastore(client, mocker):
    # This tests a more direct manipulation of datastore for robustness,
    # though normal upload flow should prevent 'chunks': None.
    doc_id = "test_doc_chunks_none"
    add_document(doc_id) # Creates with "chunks": []
    document_store[doc_id]['chunks'] = None # Manually set to None

    mock_azure_embedder_constructor = mocker.patch('app.AzureEmbedder')
    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 200 # Endpoint handles this as "no chunks"
    json_data = response.get_json()
    assert json_data['message'] == "No chunks found for this document to embed."
    assert json_data['embeddings_count'] == 0
    
    stored_doc = get_document(doc_id)
    assert stored_doc['embeddings'] == [] 
    mock_azure_embedder_constructor.assert_not_called()

    # Case: "chunks" key exists but is an empty list
    clear_store()
    doc_id_2 = "test_doc_chunks_empty_list_direct"
    add_document(doc_id_2) # Creates with "chunks": []
    # update_document_chunks(doc_id_2, [], []) # This is the standard way

    response_2 = client.post('/api/embed', json={'doc_id': doc_id_2})
    assert response_2.status_code == 200
    json_data_2 = response_2.get_json()
    assert json_data_2['message'] == "No chunks to embed." # Or "No chunks found..."
    assert json_data_2['embeddings_count'] == 0
    
    stored_doc_2 = get_document(doc_id_2)
    assert stored_doc_2['embeddings'] == []
    mock_azure_embedder_constructor.assert_not_called() # Should still be not called

def test_embed_with_only_whitespace_or_empty_chunks(client, mocker):
    doc_id = "test_doc_only_empty_chunks"
    chunks = ["   ", "", "  \n  "]
    metadata = ["meta1", "meta2", "meta3"]
    
    add_document(doc_id)
    update_document_chunks(doc_id, chunks, metadata)

    mock_azure_embedder_instance = MagicMock()
    mocker.patch('app.AzureEmbedder', return_value=mock_azure_embedder_instance)
    
    response = client.post('/api/embed', json={'doc_id': doc_id})
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['doc_id'] == doc_id
    assert json_data['status'] == "success"
    assert json_data['total_chunks_processed'] == 3
    assert json_data['successful_embeddings_count'] == 0 # No actual text to embed

    stored_doc = get_document(doc_id)
    assert stored_doc is not None
    assert stored_doc['embeddings'] == [None, None, None] # None for each skipped chunk
    
    # get_embedding should not be called as all chunks are empty/whitespace
    mock_azure_embedder_instance.get_embedding.assert_not_called()
    # AzureEmbedder constructor would still be called once.
    app.AzureEmbedder.assert_called_once()

# Helper to ensure conftest clear_store is working as expected between tests
def test_datastore_is_clear_for_this_test():
    assert len(document_store) == 0

# Test for AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME being used
def test_embed_uses_env_var_for_model_name(client, mocker, monkeypatch):
    doc_id = "test_model_name_usage"
    custom_model_name = "my-custom-embedding-model"
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", custom_model_name)

    add_document(doc_id)
    update_document_chunks(doc_id, ["chunk1"], ["meta1"])

    mock_embedder_instance = MagicMock()
    mock_embedder_instance.get_embedding.return_value = [0.5, 0.5]
    
    # We want to assert that AzureEmbedder (the class) was called with the correct model_name
    mock_azure_embedder_class_constructor = mocker.patch('app.AzureEmbedder', return_value=mock_embedder_instance)
    
    client.post('/api/embed', json={'doc_id': doc_id})

    # Check that the constructor of AzureEmbedder was called with the custom model name
    mock_azure_embedder_class_constructor.assert_called_once_with(model_name=custom_model_name)
    mock_embedder_instance.get_embedding.assert_called_once_with("chunk1")

    # Reset the env var if other tests depend on the default MOCK_EMBEDDING_MODEL_NAME
    # (though pytest-monkeypatch handles unsetting for this test, if it was set before)
    # monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", MOCK_EMBEDDING_MODEL_NAME)
    # This is generally handled by fixture scoping or explicit setup/teardown if needed globally.
    # The autouse fixture `set_env_vars` will reset it for the next test.

    # Final check: make sure the env var is reset by the autouse fixture for subsequent tests
    # This can be verified by having another test that relies on the default MOCK_EMBEDDING_MODEL_NAME
    # as set by the set_env_vars fixture. For example, the first test `test_embed_successful`
    # implicitly relies on this.
    # No direct assertion here, but it's an important aspect of the test suite design.

# This test specifically checks the default model name if the set_env_vars fixture had a different scope
# or if we wanted to be very explicit about the default name without relying on set_env_vars for this specific check.
# However, with set_env_vars as autouse, this test is similar to test_embed_uses_env_var_for_model_name
# but confirms the default set by the fixture.
def test_embed_uses_default_model_name_from_fixture_env(client, mocker):
    doc_id = "test_default_model_name"
    add_document(doc_id)
    update_document_chunks(doc_id, ["chunk default"], ["meta_default"])

    mock_embedder_instance = MagicMock()
    mock_embedder_instance.get_embedding.return_value = [0.6, 0.6]
    
    mock_azure_embedder_class_constructor = mocker.patch('app.AzureEmbedder', return_value=mock_embedder_instance)
    
    client.post('/api/embed', json={'doc_id': doc_id})

    # MOCK_EMBEDDING_MODEL_NAME is set by the autouse set_env_vars fixture
    mock_azure_embedder_class_constructor.assert_called_once_with(model_name=MOCK_EMBEDDING_MODEL_NAME)
    mock_embedder_instance.get_embedding.assert_called_once_with("chunk default")
