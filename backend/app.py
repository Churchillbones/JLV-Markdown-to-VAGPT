from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from dotenv import load_dotenv
import logging # Added for logging

# Assuming utils are in backend.utils and datastore in backend.datastore
from utils.converter import MarkdownConverter
from utils.text_utils import chunk_text_by_paragraphs
from utils.embedder import AzureEmbedder
from utils.chat import AzureChatCompleter # Added for chat
from datastore import (
    add_document,
    update_document_markdown,
    update_document_chunks,
    get_document,
    document_exists, # Added document_exists
    update_document_embeddings # Added update_document_embeddings
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Configure basic logging if not already configured
if not app.debug: # Example: only configure if not in debug mode, or use a more robust logging setup
    logging.basicConfig(level=logging.INFO)
    # For production, you might want to configure logging to a file or a logging service

@app.route('/')
def hello():
    return "Flask backend is running!"

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            file_content = file.read()
            file_name = file.filename
            
            doc_id = str(uuid.uuid4())
            add_document(doc_id) # Initialize entry in datastore

            # Convert to Markdown
            markdown_text = MarkdownConverter.convert_to_markdown(file_content, file_name)
            # It's okay if markdown_text contains an error message from conversion,
            # it will be stored and returned. Client can inspect.
            update_document_markdown(doc_id, markdown_text)

            # Chunking with metadata
            chunks_with_metadata_tuples = []
            # Use a default metadata string if none is found for a chunk
            default_metadata = "No specific metadata extracted for this chunk."

            if file_name.lower().endswith('.pdf'):
                pages_data = MarkdownConverter.extract_text_and_metadata_per_page(file_content)
                if pages_data:
                    for _page_num, page_text, page_meta in pages_data:
                        if page_text and page_text.strip():
                            page_specific_chunks = chunk_text_by_paragraphs(page_text)
                            for chunk_text in page_specific_chunks:
                                if chunk_text.strip(): # Ensure chunk itself is not just whitespace
                                    chunks_with_metadata_tuples.append((chunk_text, page_meta if page_meta and page_meta.strip() else default_metadata))
            
            final_chunks = []
            final_chunk_metadata = []

            # Fallback or primary for non-PDFs / if PDF processing yielded no chunks
            if not chunks_with_metadata_tuples:
                if markdown_text.startswith("Error during conversion:") or \
                   markdown_text.startswith("Unexpected error") or \
                   markdown_text.startswith("Unsupported file type:") or \
                   not markdown_text.strip():
                    final_chunks = ["Error in document processing or document is empty."]
                    final_chunk_metadata = [default_metadata]
                else:
                    full_text_chunks = chunk_text_by_paragraphs(markdown_text)
                    if not full_text_chunks: 
                        final_chunks = ["Document content is empty or could not be chunked."]
                        final_chunk_metadata = [default_metadata]
                    else:
                        # This was missing the re-assignment to chunks_with_metadata_tuples
                        # and subsequent extraction, fixed now.
                        for chunk in full_text_chunks:
                            if chunk.strip():
                                chunks_with_metadata_tuples.append((chunk, default_metadata))
                        
                        if not chunks_with_metadata_tuples: # Still possible if all chunks were whitespace
                            final_chunks = ["Document content is empty or could not be chunked."]
                            final_chunk_metadata = [default_metadata]
                        else:
                            final_chunks = [item[0] for item in chunks_with_metadata_tuples]
                            final_chunk_metadata = [item[1] for item in chunks_with_metadata_tuples]
            else: 
                  # This means chunks_with_metadata_tuples was populated (likely from PDF processing)
                final_chunks = [item[0] for item in chunks_with_metadata_tuples]
                final_chunk_metadata = [item[1] for item in chunks_with_metadata_tuples]
            
            # Ensure final_chunks and final_chunk_metadata are not empty if they haven't been populated
            # by any of the above conditions, which can happen if initial chunks_with_metadata_tuples was empty
            # and then the conditions for error/empty markdown were not met, but chunking still resulted in nothing.
            if not final_chunks:
                final_chunks = ["No content could be extracted or chunked."]
                final_chunk_metadata = [default_metadata]


            update_document_chunks(doc_id, final_chunks, final_chunk_metadata)
            
            return jsonify({
                "doc_id": doc_id,
                "filename": file_name,
                "markdown_text": markdown_text, # Full markdown for display
                "chunks": final_chunks, # List of chunk texts
                "chunk_metadata": final_chunk_metadata # List of corresponding metadata
            }), 200

        except Exception as e:
            # Consider logging the exception e here
            # import logging
            # logging.exception("Error processing upload")
            app.logger.error(f"Error processing upload: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred during file processing.", "details": str(e)}), 500
            
    return jsonify({"error": "File type not allowed or other issue"}), 400

@app.route('/api/embed', methods=['POST'])
def embed_document():
    data = request.get_json()
    if not data or 'doc_id' not in data:
        app.logger.warning("Missing doc_id in /api/embed request")
        return jsonify({"error": "Missing doc_id in request body"}), 400

    doc_id = data['doc_id']
    app.logger.info(f"Received embed request for doc_id: {doc_id}")

    if not document_exists(doc_id):
        app.logger.warning(f"Document not found for embed request: {doc_id}")
        return jsonify({"error": "Document not found"}), 404

    doc_data = get_document(doc_id)
    # Note: The prompt example had 'not doc_data.get('chunks')'. 
    # If doc_data is None (which document_exists should prevent if used first), this would error.
    # Assuming document_exists(doc_id) implies doc_data is not None.
    if not doc_data.get('chunks'): # Checks for presence of 'chunks' key and if it's non-empty
        app.logger.info(f"No chunks found for document {doc_id} to embed.")
        # Storing empty embeddings list is fine, or just return this message.
        # Let's ensure embeddings are updated to empty if they were not already.
        update_document_embeddings(doc_id, []) 
        return jsonify({"doc_id": doc_id, "status": "success", "message": "No chunks found for this document to embed.", "embeddings_count": 0}), 200

    chunks = doc_data['chunks']
    if not chunks: # Explicitly handles empty list of chunks
        app.logger.info(f"Chunks list is empty for document {doc_id}.")
        update_document_embeddings(doc_id, [])
        return jsonify({"doc_id": doc_id, "status": "success", "message": "No chunks to embed.", "embeddings_count": 0}), 200

    try:
        embedder = AzureEmbedder(model_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))
    except ValueError as ve:
        app.logger.error(f"AzureEmbedder initialization failed: {ve}", exc_info=True)
        return jsonify({"error": "Azure OpenAI configuration error. Check server logs."}), 500
    except Exception as e_init: 
        app.logger.error(f"AzureEmbedder unforeseen initialization error: {e_init}", exc_info=True)
        return jsonify({"error": "Could not initialize embedding service. Check server logs."}), 500

    all_embeddings = []
    try:
        app.logger.info(f"Starting embedding process for {len(chunks)} chunks in doc_id: {doc_id}")
        for i, chunk_text in enumerate(chunks):
            if not chunk_text or not chunk_text.strip(): # Skip empty or whitespace-only chunks
                all_embeddings.append(None) 
                app.logger.info(f"Skipped embedding for empty chunk {i} in doc_id: {doc_id}")
                continue
            try:
                embedding_vector = embedder.get_embedding(chunk_text)
                all_embeddings.append(embedding_vector)
            except Exception as e_chunk:
                app.logger.error(f"Error embedding chunk {i} for doc_id {doc_id}: {e_chunk}", exc_info=True)
                all_embeddings.append(None) # Add None if a chunk fails
        
        update_document_embeddings(doc_id, all_embeddings)
        successful_embeddings_count = sum(1 for emb in all_embeddings if emb is not None)
        app.logger.info(f"Successfully generated {successful_embeddings_count} embeddings for {len(chunks)} chunks in doc_id: {doc_id}")

        return jsonify({
            "doc_id": doc_id,
            "status": "success",
            "total_chunks_processed": len(chunks),
            "successful_embeddings_count": successful_embeddings_count
        }), 200

    except Exception as e:
        app.logger.error(f"Error during embedding process for doc_id {doc_id}: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during embedding generation.", "details": str(e)}), 500

if __name__ == '__main__':
    # Basic logging configuration for local development
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, port=os.getenv("FLASK_PORT", 5000))


@app.route('/api/chat', methods=['POST'])
def chat_with_document():
    data = request.get_json()
    if not data or 'question' not in data: # doc_id is optional for general chat
        app.logger.warning("Chat: Missing question in request body")
        return jsonify({"error": "Missing question in request body"}), 400

    doc_id = data.get('doc_id') # Optional
    question = data['question']
    client_context_chunks = data.get('context_chunks') # Optional list of strings

    if not question.strip():
        app.logger.warning(f"Chat: Empty question received (doc_id: {doc_id})")
        return jsonify({"error": "Question cannot be empty"}), 400
    
    app.logger.info(f"Chat request received. doc_id: {doc_id}, question: '{question[:50]}...'")

    context_chunks_for_prompt = []

    if client_context_chunks is not None and isinstance(client_context_chunks, list):
        app.logger.info(f"Chat: Using {len(client_context_chunks)} context chunks provided by client for doc_id: {doc_id}.")
        # Filter any empty strings from client-provided chunks
        context_chunks_for_prompt = [chunk for chunk in client_context_chunks if chunk and chunk.strip()]
    elif doc_id:
        if not document_exists(doc_id):
            app.logger.warning(f"Chat: Document not found: {doc_id} when trying to fetch context.")
            return jsonify({"error": "Document not found"}), 404
        
        doc_data = get_document(doc_id)
        if doc_data and doc_data.get('chunks'):
            # Filter out empty or whitespace-only chunks from datastore
            stored_chunks = [chunk for chunk in doc_data.get('chunks', []) if chunk and chunk.strip()]
            if stored_chunks:
                 context_chunks_for_prompt = stored_chunks
                 app.logger.info(f"Chat: Using {len(context_chunks_for_prompt)} chunks from datastore for doc_id: {doc_id}.")
            else:
                app.logger.info(f"Chat: No valid (non-empty) chunks found in datastore for doc_id: {doc_id}.")
        else:
            app.logger.info(f"Chat: No chunks found in datastore for doc_id: {doc_id}.")
    else:
        app.logger.info("Chat: No doc_id provided and no context_chunks from client. Proceeding without specific document context.")

    # Prompt Formatting
    system_prompt = "You are a helpful assistant. Answer questions based on the provided context. If the context is empty or not relevant, answer to the best of your ability."
    
    context_str = ""
    if context_chunks_for_prompt:
        formatted_chunks = "\n---\n".join(context_chunks_for_prompt)
        context_str = f"Context:\n---\n{formatted_chunks}\n---\n"
    
    user_prompt = f"{context_str}Question: {question}"

    app.logger.debug(f"Chat: System Prompt: '{system_prompt}'")
    app.logger.debug(f"Chat: User Prompt (first 200 chars): '{user_prompt[:200]}...'")

    try:
        # AZURE_OPENAI_CHAT_DEPLOYMENT_NAME and AZURE_OPENAI_API_VERSION are used from env vars by default by AzureChatCompleter
        chat_completer = AzureChatCompleter() 
    except ValueError as ve:
        app.logger.error(f"Chat: AzureChatCompleter initialization failed: {ve}", exc_info=True)
        return jsonify({"error": "Azure OpenAI configuration error. Check server logs."}), 500
    except Exception as e_init:
        app.logger.error(f"Chat: AzureChatCompleter unforeseen initialization error: {e_init}", exc_info=True)
        return jsonify({"error": "Could not initialize chat service. Check server logs."}), 500

    try:
        model_response_text = chat_completer.get_chat_completion(user_prompt, system_prompt)
        app.logger.info(f"Chat: Successfully received response from chat model for doc_id: {doc_id}, question: '{question[:50]}...'")
    except Exception as e_chat:
        app.logger.error(f"Chat: get_chat_completion failed: {e_chat}", exc_info=True)
        # The actual error from AzureChatCompleter might be more specific if it catches openai.APIError
        return jsonify({"error": "Failed to get chat completion from language model.", "details": str(e_chat)}), 500

    return jsonify({
        "doc_id": doc_id if doc_id else None, # Return None if doc_id was not part of the request
        "question": question,
        "answer": model_response_text
    }), 200


@app.route('/api/search', methods=['POST'])
def search_document():
    data = request.get_json()
    if not data or 'doc_id' not in data or 'query' not in data:
        app.logger.warning("Search: Missing doc_id or query in request body")
        return jsonify({"error": "Missing doc_id or query in request body"}), 400

    doc_id = data['doc_id']
    query = data['query']

    if not query.strip():
        app.logger.warning(f"Search: Empty query for doc_id {doc_id}")
        return jsonify({"error": "Query cannot be empty"}), 400

    app.logger.info(f"Search request for doc_id: {doc_id}, query: '{query[:50]}...'") # Log first 50 chars of query

    if not document_exists(doc_id):
        app.logger.warning(f"Search: Document not found: {doc_id}")
        return jsonify({"error": "Document not found"}), 404

    doc_data = get_document(doc_id)
    if not doc_data: # Should ideally not happen if document_exists passed
        app.logger.error(f"Search: Document data inconsistent for doc_id {doc_id} after existence check.")
        return jsonify({"error": "Document data is inconsistent."}), 500

    # Retrieve and validate stored data
    stored_chunks = doc_data.get('chunks', [])
    stored_metadata = doc_data.get('chunk_metadata', [])
    stored_embeddings = doc_data.get('embeddings', [])

    if not stored_chunks or not stored_embeddings:
        app.logger.info(f"Search: Document {doc_id} has no chunks or embeddings.")
        return jsonify({"error": "Document has no chunks or embeddings. Please process and embed the document first."}), 400
    
    # Filter out items where embedding was None (failed) or chunk is empty
    # Also ensure metadata list is at least as long as chunks, and handle potential inconsistencies.
    valid_indices = []
    for i, emb in enumerate(stored_embeddings):
        if emb is not None and i < len(stored_chunks) and stored_chunks[i] and stored_chunks[i].strip():
            valid_indices.append(i)
        else:
            app.logger.debug(f"Search: Filtering out chunk {i} for doc_id {doc_id} due to missing embedding or empty chunk content.")


    if not valid_indices:
        app.logger.info(f"Search: No valid embeddings available for search in document {doc_id} after filtering.")
        return jsonify({"error": "No valid embeddings available for search in this document."}), 400
        
    searchable_chunks = [stored_chunks[i] for i in valid_indices]
    # Ensure metadata list is also filtered and handle cases where metadata might be shorter than chunks
    searchable_metadata = []
    for i in valid_indices:
        if i < len(stored_metadata):
            searchable_metadata.append(stored_metadata[i])
        else:
            # This case should ideally not happen if data is consistent from upload
            app.logger.warning(f"Search: Missing metadata for valid chunk index {i} in doc_id {doc_id}. Using default.")
            searchable_metadata.append("Default metadata (missing original)") 
            
    searchable_embeddings_matrix = np.array([stored_embeddings[i] for i in valid_indices])

    # This check is largely redundant due to 'if not valid_indices' but acts as a safeguard.
    if searchable_embeddings_matrix.shape[0] == 0:
        app.logger.info(f"Search: No searchable content found after filtering for doc_id {doc_id}.")
        return jsonify({"error": "No searchable content found after filtering invalid embeddings."}), 400

    try:
        # Use the embedding model name from environment variable, as done in /api/embed
        embedder = AzureEmbedder(model_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"))
    except ValueError as ve:
        app.logger.error(f"Search: AzureEmbedder initialization failed: {ve}", exc_info=True)
        return jsonify({"error": "Azure OpenAI configuration error. Check server logs."}), 500
    except Exception as e_init:
        app.logger.error(f"Search: AzureEmbedder unforeseen initialization error: {e_init}", exc_info=True)
        return jsonify({"error": "Could not initialize embedding service for search. Check server logs."}), 500

    try:
        query_embedding_vector = embedder.get_embedding(query)
        query_embedding = np.array(query_embedding_vector).reshape(1, -1)
    except Exception as e_query_emb:
        app.logger.error(f"Search: Error generating embedding for query '{query}': {e_query_emb}", exc_info=True)
        return jsonify({"error": "Failed to generate embedding for your query."}), 500

    # Calculate cosine similarities
    try:
        similarities = cosine_similarity(query_embedding, searchable_embeddings_matrix)
    except Exception as e_similarity:
        app.logger.error(f"Search: Error calculating cosine similarity for doc_id {doc_id}: {e_similarity}", exc_info=True)
        return jsonify({"error": "Failed to calculate search similarities."}), 500
    
    results_with_scores = []
    if similarities.size > 0:
        for i, score in enumerate(similarities[0]):
            # Ensure index i is valid for searchable_chunks and searchable_metadata
            if i < len(searchable_chunks) and i < len(searchable_metadata):
                results_with_scores.append({
                    "score": float(score), # Ensure score is float for JSON
                    "chunk": searchable_chunks[i],
                    "metadata": searchable_metadata[i]
                })
            else:
                # This indicates an inconsistency if valid_indices was built correctly
                app.logger.error(f"Search: Index out of bounds when compiling results for doc_id {doc_id}. Index {i}, Chunks len {len(searchable_chunks)}")

        # Sort results by similarity score in descending order
        sorted_results = sorted(results_with_scores, key=lambda x: x['score'], reverse=True)
        top_n_results = sorted_results[:5] # Display top 5
        app.logger.info(f"Search for doc_id {doc_id} completed. Found {len(sorted_results)} results, returning top {len(top_n_results)}.")
    else:
        app.logger.info(f"Search for doc_id {doc_id} completed. No similarity scores generated (similarities.size is 0).")
        top_n_results = []

    return jsonify({
        "doc_id": doc_id,
        "query": query,
        "search_results": top_n_results
    }), 200
