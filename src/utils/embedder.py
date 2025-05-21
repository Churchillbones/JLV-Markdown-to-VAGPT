import os
from openai import AzureOpenAI

# To use this class, you need to set the following environment variables:
# AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint (e.g., "https://<your-resource-name>.openai.azure.com/")
# AZURE_OPENAI_API_KEY: Your Azure OpenAI API key

class AzureEmbedder:
    """
    A class to get text embeddings using Azure OpenAI.
    """
    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initializes the AzureOpenAI client.

        Args:
            model_name (str): The deployment name for the embedding model.
                              Defaults to "text-embedding-3-large".

        Raises:
            ValueError: If the AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY
                        environment variables are not set.
        """
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set.")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")

        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2023-05-15"  # Or your desired API version
        )
        self.model_name = model_name

    def get_embedding(self, text: str) -> list[float]:
        """
        Gets the embedding for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector for the text.
        """
        text = text.replace("\n", " ") # Azure OpenAI API doesn't like newlines
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding
