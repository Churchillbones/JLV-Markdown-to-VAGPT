# backend/datastore.py

document_store = {} # Global in-memory store

def add_document(doc_id: str):
    if doc_id not in document_store:
        document_store[doc_id] = {
            "markdown_text": "",
            "chunks": [],         # List of text chunks
            "chunk_metadata": [], # List of metadata strings, parallel to chunks
            "embeddings": []      # List of embedding vectors, parallel to chunks
        }
        return True
    return False # Document already exists

def get_document(doc_id: str):
    return document_store.get(doc_id)

def document_exists(doc_id: str) -> bool:
    return doc_id in document_store

def update_document_markdown(doc_id: str, markdown_text: str):
    if document_exists(doc_id):
        document_store[doc_id]["markdown_text"] = markdown_text
        return True
    return False

def update_document_chunks(doc_id: str, chunks: list[str], chunk_metadata: list[str]):
    if document_exists(doc_id):
        document_store[doc_id]["chunks"] = chunks
        document_store[doc_id]["chunk_metadata"] = chunk_metadata
        return True
    return False

def update_document_embeddings(doc_id: str, embeddings: list[list[float]]):
    if document_exists(doc_id):
        document_store[doc_id]["embeddings"] = embeddings
        return True
    return False

# Optional: A function to clear the store or remove a document if needed for testing/MVP management
def clear_store():
    document_store.clear()

def remove_document(doc_id: str):
    if doc_id in document_store:
        del document_store[doc_id]
        return True
    return False
