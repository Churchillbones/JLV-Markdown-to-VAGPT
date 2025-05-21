import streamlit as st
import base64
import pyperclip
from components.file_uploader import FileUploader
from utils.converter import MarkdownConverter
from src.utils.embedder import AzureEmbedder # Import AzureEmbedder
import subprocess
import sys
import os
import json

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
        with st.spinner("Converting to Markdown..."):
            file_content = uploaded_file.getvalue()
            markdown_text = MarkdownConverter.convert_to_markdown(file_content, uploaded_file.name)
            
            # Store in session state
            st.session_state.markdown_text = markdown_text
            st.session_state.filename = uploaded_file.name
            st.session_state.gpt_prompt = gpt_prompt
    
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