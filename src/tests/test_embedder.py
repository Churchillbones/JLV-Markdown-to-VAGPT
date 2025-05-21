import unittest
import os
from unittest.mock import patch, MagicMock
from openai import APIError # For testing API call failures

# Adjust the import path based on your project structure
# This assumes test_embedder.py is in src/tests and embedder.py is in src/utils
import sys
# Add src to Python path to allow direct import of src.utils.embedder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.embedder import AzureEmbedder

class TestAzureEmbedder(unittest.TestCase):

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://fake-endpoint.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "fake_api_key"
    })
    @patch('openai.AzureOpenAI')
    def test_successful_initialization_and_embedding(self, mock_azure_openai_client_constructor):
        # Mock the client instance and its methods
        mock_client_instance = MagicMock()
        mock_embeddings_create_response = MagicMock()
        mock_embeddings_create_response.data = [MagicMock()]
        mock_embeddings_create_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client_instance.embeddings.create.return_value = mock_embeddings_create_response
        
        # Configure the constructor mock to return our client instance
        mock_azure_openai_client_constructor.return_value = mock_client_instance

        embedder = AzureEmbedder(model_name="test-embedding-model")
        
        # Test initialization
        mock_azure_openai_client_constructor.assert_called_once_with(
            azure_endpoint="https://fake-endpoint.openai.azure.com/",
            api_key="fake_api_key",
            api_version="2023-05-15" 
        )
        self.assertEqual(embedder.model_name, "test-embedding-model")

        # Test get_embedding
        test_text = "Hello, world!"
        embedding = embedder.get_embedding(test_text)

        mock_client_instance.embeddings.create.assert_called_once_with(
            input=[test_text.replace("\n", " ")], # Ensure newline replacement
            model="test-embedding-model"
        )
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "fake_api_key"}, clear=True)
    def test_missing_azure_openai_endpoint(self):
        with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_ENDPOINT environment variable not set."):
            AzureEmbedder()

    @patch.dict(os.environ, {"AZURE_OPENAI_ENDPOINT": "https://fake-endpoint.openai.azure.com/"}, clear=True)
    def test_missing_azure_openai_api_key(self):
        with self.assertRaisesRegex(ValueError, "AZURE_OPENAI_API_KEY environment variable not set."):
            AzureEmbedder()

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://fake-endpoint.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "fake_api_key"
    })
    @patch('openai.AzureOpenAI')
    def test_azure_api_call_failure(self, mock_azure_openai_client_constructor):
        # Mock the client instance
        mock_client_instance = MagicMock()
        # Configure the embeddings.create method to raise an APIError
        mock_client_instance.embeddings.create.side_effect = APIError("Azure API Error", response=MagicMock(), body=None)
        
        # Configure the constructor mock to return our client instance
        mock_azure_openai_client_constructor.return_value = mock_client_instance

        embedder = AzureEmbedder()
        
        with self.assertRaises(APIError):
            embedder.get_embedding("This will fail.")

if __name__ == '__main__':
    unittest.main()
