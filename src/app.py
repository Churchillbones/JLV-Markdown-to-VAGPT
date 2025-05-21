import streamlit as st
import base64
import pyperclip
from components.file_uploader import FileUploader
from utils.converter import MarkdownConverter
from src.utils.embedder import AzureEmbedder # Import AzureEmbedder
from src.utils.text_utils import chunk_text_by_paragraphs # Import for chunking
import subprocess
import sys
import os
import json
import numpy as np # For numerical operations
from sklearn.metrics.pairwise import cosine_similarity # For semantic search

def check_markitdown_installed():
    """Check if markitdown is installed and install if necessary"""
    try:
        subprocess.run(["markitdown", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.warning("Installing markitdown package. Please wait...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "markitdown"])
            st.success("markitdown installed successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to install markitdown: {str(e)}")
            return False

def create_download_link(markdown_text, filename="converted_document.md"):
    """Create a download link for the markdown text"""
    b64 = base64.b64encode(markdown_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download markdown file</a>'
    return href

def main():
    st.set_page_config(
        page_title="PDF to Markdown Converter",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize state for clipboard success message
    if 'show_clipboard_success' not in st.session_state:
        st.session_state.show_clipboard_success = False

    st.sidebar.title("About")
    st.sidebar.info(
        "This app converts PDF documents to Markdown format for use with GPT and other AI chat bots. "
        "Upload your document, click convert, and then copy the markdown to your chat interface."
    )
    
    # Check if markitdown is installed
    check_markitdown_installed()
    
    # Render file uploader
    uploaded_file, convert_clicked, gpt_prompt = FileUploader.render()
    
    # Store the conversion results in session state to preserve across re-renders
    if uploaded_file is not None and convert_clicked:
        # Clear previous chunk embeddings on new file conversion
        st.session_state.text_chunks_with_embeddings = None

        with st.spinner("Converting to Markdown..."):
            file_content = uploaded_file.getvalue()
            markdown_text = MarkdownConverter.convert_to_markdown(file_content, uploaded_file.name)
            
            # Store in session state
            st.session_state.markdown_text = markdown_text
            st.session_state.filename = uploaded_file.name
            st.session_state.gpt_prompt = gpt_prompt

            # --- Chunking and Embedding Generation ---
            if st.session_state.markdown_text:
                st.info("Processing document: Chunking text and generating embeddings...")
                text_chunks = chunk_text_by_paragraphs(st.session_state.markdown_text)
                
                if not text_chunks:
                    st.warning("Document is empty or could not be chunked. No embeddings generated.")
                    st.session_state.text_chunks_with_embeddings = []
                else:
                    try:
                        embedder = AzureEmbedder()
                        chunks_with_embeddings = []
                        st.progress(0.0) # Initialize progress bar
                        for i, chunk in enumerate(text_chunks):
                            try:
                                if chunk.strip(): # Ensure chunk is not just whitespace
                                    embedding = embedder.get_embedding(chunk)
                                    chunks_with_embeddings.append((chunk, embedding))
                                else:
                                    # Optionally, store empty chunks or skip them
                                    # chunks_with_embeddings.append((chunk, None)) 
                                    pass 
                            except Exception as e_chunk:
                                st.warning(f"Could not generate embedding for chunk {i+1}: {e_chunk}")
                            st.progress((i + 1) / len(text_chunks)) # Update progress bar
                        
                        st.session_state.text_chunks_with_embeddings = chunks_with_embeddings
                        if chunks_with_embeddings:
                             st.success(f"Document chunks processed. {len(chunks_with_embeddings)} embeddings generated.")
                        else:
                            st.warning("No embeddings were generated for the document chunks.")
                        st.progress(1.0) # Ensure progress bar completes

                    except ValueError as ve:
                        st.error(f"Azure Credentials Not Set: {ve}. Embeddings for chunks not generated. Search functionality will be limited.")
                        st.session_state.text_chunks_with_embeddings = [] # Indicate checked but unavailable
                    except Exception as e_embed_init:
                        st.error(f"Error initializing embedder: {e_embed_init}. Embeddings for chunks not generated.")
                        st.session_state.text_chunks_with_embeddings = []
            else:
                st.warning("Markdown content is empty. No embeddings generated for chunks.")
                st.session_state.text_chunks_with_embeddings = []
    
    # Display results if available in session state
    if 'markdown_text' in st.session_state:
        markdown_text = st.session_state.markdown_text
        filename = st.session_state.filename
        gpt_prompt = st.session_state.gpt_prompt
            
        # Check for error messages in the markdown text
        if markdown_text.startswith("Error during conversion:") or markdown_text.startswith("Unexpected error"):
            st.warning("Conversion encountered issues. Showing fallback text extraction.")
            
            # Display error details in an expander
            with st.expander("View Error Details"):
                st.code(markdown_text.split("Falling back to direct text extraction:")[0], language="text")
            
            # Extract just the fallback text part if it exists
            if "Falling back to direct text extraction:" in markdown_text:
                markdown_text = markdown_text.split("Falling back to direct text extraction:")[1].strip()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Markdown Output")
            st.text_area("Markdown Content", 
                         value=markdown_text, 
                         height=400,
                         key="markdown_output")
            
            st.markdown(create_download_link(markdown_text, f"{os.path.splitext(filename)[0]}.md"), 
                        unsafe_allow_html=True)
            
            # Add combined copy feature
            combined_text = f"{gpt_prompt}\n\n{markdown_text}"
            st.write("### Combined Prompt and Markdown")
            st.text_area("Ready to paste into GPT", 
                         value=combined_text,
                         height=200,
                         key="combined_output")
            
            # Function to handle clipboard operation
            def copy_to_clipboard():
                pyperclip.copy(combined_text)
                st.session_state.show_clipboard_success = True
            
            # Use Streamlit's built-in button with callback
            copy_btn = st.button("Copy Combined Text to Clipboard", 
                                 on_click=copy_to_clipboard, 
                                 key="copy_btn",
                                 type="primary")
            
            # Display success message when needed
            if st.session_state.show_clipboard_success:
                st.code(f'''
# Text copied to clipboard! Use keyboard shortcut Ctrl+V to paste.
# For convenience, here's the text that was copied:

{combined_text[:100]}... (truncated)
                ''')
                st.session_state.show_clipboard_success = False
            
        with col2:
            st.write("### Preview")
            st.markdown(markdown_text)

        # --- Azure Embedding Section ---
        st.subheader("Generate Embeddings")
        if st.button("Generate Embedding with Azure"):
            if 'markdown_text' in st.session_state and st.session_state.markdown_text:
                try:
                    embedder = AzureEmbedder() # Uses default model
                    embedding_vector = embedder.get_embedding(st.session_state.markdown_text)
                    st.success("Successfully generated embedding.")
                    st.write("Embedding preview (first 5 dimensions):", embedding_vector[:5])
                except ValueError as ve:
                    st.error(f"Configuration Error: {ve}. Please ensure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables are set correctly.")
                except Exception as e:
                    st.error(f"An error occurred during embedding generation: {e}")
            else:
                st.warning("Please convert a document to Markdown first to generate embeddings.")

        # --- Semantic Search Section ---
        if st.session_state.get('text_chunks_with_embeddings') and isinstance(st.session_state.text_chunks_with_embeddings, list) and len(st.session_state.text_chunks_with_embeddings) > 0:
            st.subheader("Search Document Content")
            search_query = st.text_input("Enter your search query:", key="search_query_input")

            # Clear previous search results if query is cleared
            if not search_query:
                st.session_state.search_results = None

            if st.button("Search", key="search_doc_button"):
                if search_query:
                    st.session_state.search_results = None # Clear previous results for this specific search
                    try:
                        embedder = AzureEmbedder()
                    except ValueError as ve:
                        st.error(f"Azure Credentials Not Set: {ve}. Search functionality is unavailable.")
                        st.stop() # Stop execution for this callback if embedder fails
                    except Exception as e_init:
                        st.error(f"Error initializing Azure Embedder: {e_init}. Search is unavailable.")
                        st.stop()
                    
                    try:
                        query_embedding_vector = embedder.get_embedding(search_query)
                        query_embedding = np.array(query_embedding_vector).reshape(1, -1)
                    except Exception as e_query:
                        st.error(f"Error generating embedding for your query: {e_query}")
                        st.stop()

                    # Retrieve chunks and their embeddings
                    chunks_with_embeddings = st.session_state.text_chunks_with_embeddings
                    
                    # Filter out any chunks that might have failed embedding (if any)
                    valid_chunks_with_embeddings = [item for item in chunks_with_embeddings if item[1] is not None and len(item[1]) > 0]

                    if not valid_chunks_with_embeddings:
                        st.warning("No valid chunk embeddings available to search against.")
                        st.stop()

                    chunk_texts = [item[0] for item in valid_chunks_with_embeddings]
                    chunk_embeddings_list = [item[1] for item in valid_chunks_with_embeddings]
                    
                    if not chunk_embeddings_list:
                        st.warning("No chunk embeddings available to search.")
                        st.stop()
                    
                    chunk_embeddings_matrix = np.array(chunk_embeddings_list)

                    # Calculate cosine similarities
                    similarities = cosine_similarity(query_embedding, chunk_embeddings_matrix)
                    
                    results_with_scores = []
                    if similarities.size > 0:
                        for i, score in enumerate(similarities[0]):
                            results_with_scores.append((score, chunk_texts[i]))
                        
                        # Sort results by similarity score in descending order
                        sorted_results = sorted(results_with_scores, key=lambda x: x[0], reverse=True)
                        st.session_state.search_results = sorted_results[:5] # Display top 5
                    else:
                        st.session_state.search_results = []
                else:
                    st.warning("Please enter a search query.")
                    st.session_state.search_results = None # Clear results if search is clicked with empty query

            # Display search results
            if st.session_state.get('search_results') is not None:
                st.write("### Search Results")
                if not st.session_state.search_results:
                    st.write("No relevant chunks found for your query or search criteria not met.")
                else:
                    for score, chunk_text in st.session_state.search_results:
                        st.markdown(f"**Similarity: {score:.4f}**")
                        st.markdown(f"> {chunk_text}") # Using blockquote for results
                        st.markdown("---")
        elif 'markdown_text' in st.session_state : # Only show this if markdown is generated but no embeddings
             if st.session_state.get('text_chunks_with_embeddings') is None or len(st.session_state.get('text_chunks_with_embeddings', [])) == 0:
                st.info("Document embeddings for search are not available. This might be due to missing Azure credentials or an issue during the embedding process. Convert a new document or check logs if issues persist.")


    st.sidebar.title("Tips")
    st.sidebar.markdown("""
    **How to use with GPT:**
    1. Upload your PDF
    2. Enter your prompt for GPT
    3. Convert to Markdown
    4. Use the "Copy Combined Text" button 
    5. Paste directly into a GPT chat
    
    **Troubleshooting:**
    - If conversion fails, the app will fall back to direct text extraction
    - For best results, use PDFs with selectable text rather than scanned documents
    - Very large PDFs may take longer to process
    """)

if __name__ == "__main__":
    main()