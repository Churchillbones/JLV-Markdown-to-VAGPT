# backend/utils/chat.py
import os
from openai import AzureOpenAI
import logging # Using standard logging, assuming app.logger might not be directly available here

# It's better to get a logger instance than rely on a global app.logger if this module is to be self-contained
logger = logging.getLogger(__name__)

class AzureChatCompleter:
    def __init__(self, deployment_name: str = None, api_version: str = None):
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

        if not self.azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set.")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable not set.")
        if not self.deployment_name:
            # Ensure this matches the .env variable name from the prompt if it's different
            chat_deployment_env_var = "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME" 
            if not self.deployment_name: # Check again if it was None and env var is also None
                 raise ValueError(f"{chat_deployment_env_var} environment variable not set for chat model.")
        if not self.api_version:
            raise ValueError("AZURE_OPENAI_API_VERSION environment variable not set.")

        logger.info(f"Initializing AzureChatCompleter with deployment: {self.deployment_name}, version: {self.api_version}, endpoint: {self.azure_endpoint[:20]}...")

        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )

    def get_chat_completion(self, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            logger.debug(f"Requesting chat completion from deployment: {self.deployment_name} with system prompt: '{system_prompt}' and user prompt starting with: '{user_prompt[:100]}...'")
            response = self.client.chat.completions.create(
                model=self.deployment_name, # This should be the deployment name
                messages=messages
            )
            
            if response.choices and len(response.choices) > 0:
                completion_text = response.choices[0].message.content
                logger.debug(f"Received chat completion: '{completion_text[:100]}...'")
                return completion_text
            else:
                logger.warning("Azure chat completion returned no choices or empty response.")
                return "Error: Received no response from language model." # Or raise an error

        except Exception as e:
            # The prompt suggested app.logger.error, but this class should be independent.
            # Logging is done here, and the exception is re-raised to be handled by the calling endpoint.
            logger.error(f"Azure chat completion request failed: {e}", exc_info=True)
            raise
