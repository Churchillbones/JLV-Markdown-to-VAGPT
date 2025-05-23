import pytest
import json
from unittest.mock import MagicMock, patch, ANY # ANY is useful for some call assertions
from datastore import (
    add_document,
    update_document_chunks,
    get_document,
    document_store,
    clear_store
)

# client fixture is from conftest.py
# mocker fixture is from pytest-mock

MOCK_CHAT_MODEL_NAME = "test-chat-deployment"
MOCK_CHAT_API_VERSION = "2023-05-15" # Or whatever your .env uses

@pytest.fixture(autouse=True)
def set_env_vars_for_chat(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-chat-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-chat-api-key")
    monkeypatch.setenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", MOCK_CHAT_MODEL_NAME)
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", MOCK_CHAT_API_VERSION)


EXPECTED_SYSTEM_PROMPT = "You are a helpful assistant. Answer questions based on the provided context. If the context is empty or not relevant, answer to the best of your ability."
MOCKED_AI_RESPONSE = "Mocked AI response."

def test_chat_successful_with_client_context(client, mocker):
    question = "What is the capital of France?"
    context_chunks = ["Paris is the capital of France.", "France is in Europe."]
    
    mock_chat_completer_instance = MagicMock()
    mock_chat_completer_instance.get_chat_completion.return_value = MOCKED_AI_RESPONSE
    
    # Patch where AzureChatCompleter is instantiated in app.py
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={
        'question': question,
        'context_chunks': context_chunks
    })
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['question'] == question
    assert json_data['answer'] == MOCKED_AI_RESPONSE
    assert json_data['doc_id'] is None # No doc_id was sent

    expected_context_str = "Context:\n---\n" + "\n---\n".join(context_chunks) + "\n---\n"
    expected_user_prompt = f"{expected_context_str}Question: {question}"
    
    app.AzureChatCompleter.assert_called_once_with() # Check constructor was called (without specific args here)
    mock_chat_completer_instance.get_chat_completion.assert_called_once_with(
        expected_user_prompt,
        EXPECTED_SYSTEM_PROMPT
    )

def test_chat_successful_with_doc_id_context(client, mocker):
    doc_id = "chat_doc_id_context"
    question = "What does apple taste like?"
    stored_chunks = ["Apples are sweet.", "Some apples are sour."]
    
    add_document(doc_id)
    update_document_chunks(doc_id, stored_chunks, ["meta1", "meta2"]) # Metadata not used by chat but good to have
    
    mock_chat_completer_instance = MagicMock()
    mock_chat_completer_instance.get_chat_completion.return_value = MOCKED_AI_RESPONSE
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={
        'doc_id': doc_id,
        'question': question
    })
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['doc_id'] == doc_id
    assert json_data['question'] == question
    assert json_data['answer'] == MOCKED_AI_RESPONSE

    expected_context_str = "Context:\n---\n" + "\n---\n".join(stored_chunks) + "\n---\n"
    expected_user_prompt = f"{expected_context_str}Question: {question}"
    
    mock_chat_completer_instance.get_chat_completion.assert_called_once_with(
        expected_user_prompt,
        EXPECTED_SYSTEM_PROMPT
    )

def test_chat_successful_no_context_provided(client, mocker):
    question = "What is 2 + 2?"
    
    mock_chat_completer_instance = MagicMock()
    mock_chat_completer_instance.get_chat_completion.return_value = MOCKED_AI_RESPONSE
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={'question': question})
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['question'] == question
    assert json_data['answer'] == MOCKED_AI_RESPONSE
    assert json_data['doc_id'] is None

    # Expected user prompt will not have the "Context:\n---\n..." part
    expected_user_prompt = f"Question: {question}" 
    
    mock_chat_completer_instance.get_chat_completion.assert_called_once_with(
        expected_user_prompt,
        EXPECTED_SYSTEM_PROMPT
    )

def test_chat_missing_question(client):
    response = client.post('/api/chat', json={}) # Missing question
    assert response.status_code == 400
    assert response.get_json()['error'] == "Missing question in request body"

    response_empty_q = client.post('/api/chat', json={'question': "   "})
    assert response_empty_q.status_code == 400
    assert response_empty_q.get_json()['error'] == "Question cannot be empty"


def test_chat_doc_id_not_found(client):
    # No client_context_chunks provided, so it will try to fetch from doc_id
    response = client.post('/api/chat', json={
        'doc_id': "non_existent_chat_doc",
        'question': "This should fail."
    })
    assert response.status_code == 404
    assert response.get_json()['error'] == "Document not found"

def test_chat_azure_completer_init_fails(client, mocker):
    # Patch the constructor of AzureChatCompleter within app.py context
    mocker.patch('app.AzureChatCompleter.__init__', side_effect=ValueError("Azure config error from test"))
    
    response = client.post('/api/chat', json={'question': "Test question"})
    assert response.status_code == 500
    assert "Azure OpenAI configuration error" in response.get_json()['error']

def test_chat_get_completion_call_fails(client, mocker):
    mock_chat_completer_instance = MagicMock()
    # Simulate a failure during the API call to Azure
    mock_chat_completer_instance.get_chat_completion.side_effect = Exception("LLM API call failed")
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={'question': "Test question"})
    assert response.status_code == 500
    assert "Failed to get chat completion from language model" in response.get_json()['error']
    assert "LLM API call failed" in response.get_json()['details']


def test_chat_empty_client_context_chunks(client, mocker):
    question = "General question with empty client context"
    
    mock_chat_completer_instance = MagicMock()
    mock_chat_completer_instance.get_chat_completion.return_value = MOCKED_AI_RESPONSE
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={
        'question': question,
        'context_chunks': ["", "   "] # Empty and whitespace only chunks
    })
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['answer'] == MOCKED_AI_RESPONSE

    # Context should be empty, so prompt is just the question
    expected_user_prompt = f"Question: {question}"
    mock_chat_completer_instance.get_chat_completion.assert_called_once_with(
        expected_user_prompt,
        EXPECTED_SYSTEM_PROMPT
    )

def test_chat_doc_id_context_chunks_are_empty_or_none(client, mocker):
    doc_id = "chat_doc_empty_chunks"
    question = "Question for doc with no real chunks"
    
    add_document(doc_id)
    update_document_chunks(doc_id, ["", "  \n  "], ["m1", "m2"]) # Store only empty/whitespace chunks
    
    mock_chat_completer_instance = MagicMock()
    mock_chat_completer_instance.get_chat_completion.return_value = MOCKED_AI_RESPONSE
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={
        'doc_id': doc_id,
        'question': question
    })
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['answer'] == MOCKED_AI_RESPONSE

    # Context from doc_id should be empty after filtering
    expected_user_prompt = f"Question: {question}"
    mock_chat_completer_instance.get_chat_completion.assert_called_once_with(
        expected_user_prompt,
        EXPECTED_SYSTEM_PROMPT
    )

def test_chat_doc_id_context_no_chunks_key(client, mocker):
    # Test if doc_data.get('chunks') returns None or key is missing
    doc_id = "chat_doc_no_chunks_key"
    question = "Question for doc with no chunks key"
    
    # Add document, but don't call update_document_chunks, so "chunks" will be its default []
    # To simulate it being None or missing, we'd have to manipulate datastore directly more deeply,
    # but the current datastore.add_document initializes 'chunks': [].
    # The endpoint's `doc_data.get('chunks', [])` handles this gracefully.
    # Let's test the case where it's present but empty (which is covered by previous test as well)
    add_document(doc_id) 
    # document_store[doc_id]['chunks'] = None # This would be a more direct test for None
    
    mock_chat_completer_instance = MagicMock()
    mock_chat_completer_instance.get_chat_completion.return_value = MOCKED_AI_RESPONSE
    mocker.patch('app.AzureChatCompleter', return_value=mock_chat_completer_instance)
    
    response = client.post('/api/chat', json={
        'doc_id': doc_id,
        'question': question
    })
    
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['answer'] == MOCKED_AI_RESPONSE

    expected_user_prompt = f"Question: {question}" # No context expected
    mock_chat_completer_instance.get_chat_completion.assert_called_once_with(
        expected_user_prompt,
        EXPECTED_SYSTEM_PROMPT
    )

# To ensure datastore is clean for each test (handled by conftest.py)
def test_chat_datastore_is_clear():
    assert len(document_store) == 0
