import streamlit as st
from pathlib import Path
import pyperclip

class FileUploader:
    @staticmethod
    def render():
        """
        Render the file uploader component in Streamlit
        
        Returns:
            tuple: (uploaded_file, convert_clicked, gpt_prompt)
                uploaded_file - The uploaded file object
                convert_clicked - Boolean indicating if conversion was requested
                gpt_prompt - The user's GPT prompt to prepend to the markdown
        """
        st.header("PDF to Markdown Converter")
        st.write("Upload a PDF file to convert it to Markdown format for use with GPT chat bots.")
        
        uploaded_file = st.file_uploader("Choose a file", 
                                        type=["pdf", "docx"],
                                        help="Upload a PDF or DOCX file to convert to Markdown")
        
        convert_clicked = False
        gpt_prompt = ""
        
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            
            st.write("### File Details")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")
                
            st.write("### GPT Prompt")
            st.write("Enter the prompt you want to use with GPT:")
            gpt_prompt = st.text_area(
                "Your prompt (e.g., 'Summarize this document for me:')",
                value="""Summarize these non-VA documents. Please provide:

1. Active Medications List
2. Lab Test Results
3. Imaging Results
4. Procedures Performed
5. Clinical Problems/Diagnoses
6. Brief summaries of any:
   - Discharge Summaries
   - Clinic Visits
   - Consults
7. List of Health Systems Used

Please use markdown formatting in your response.""",
                help="This prompt will be combined with the markdown content"
            )
                
            convert_clicked = st.button("Convert to Markdown", key="convert_btn")
            
        return uploaded_file, convert_clicked, gpt_prompt
        
    @staticmethod
    def copy_to_clipboard(prompt, markdown_content):
        """
        Create a button to copy both prompt and markdown to clipboard
        
        Args:
            prompt: The GPT prompt text
            markdown_content: The markdown content
            
        Returns:
            None
        """
        combined_text = f"{prompt}\n\n{markdown_content}"
        
        if st.button("Copy Prompt + Markdown to Clipboard", key="copy_combined"):
            try:
                pyperclip.copy(combined_text)
                st.success("Copied to clipboard! Ready to paste into a GPT chat.")
            except Exception as e:
                st.error(f"Failed to copy to clipboard: {str(e)}")
                st.info("Please manually copy the text from the box below:")
                st.text_area("Combined Content", value=combined_text, height=100)