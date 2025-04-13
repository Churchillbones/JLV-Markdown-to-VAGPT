import tempfile
import os
import subprocess
import sys
import requests
import io
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document

class MarkdownConverter:
    @staticmethod
    def convert_pdf_to_markdown(file_content, file_name):
        """
        Convert PDF content to Markdown using markitdown
        
        Args:
            file_content: The binary content of the PDF file
            file_name: The name of the file
            
        Returns:
            str: Markdown content
        """
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        # Create a temporary output file for the markdown
        output_path = os.path.join(tempfile.gettempdir(), f"{Path(file_name).stem}.md")
        
        try:
            # Call markitdown using subprocess and capture output
            result = subprocess.run(
                ["markitdown", tmp_path, "--output", output_path],
                check=False,  # Don't raise an exception yet
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Log the actual error output
                error_msg = f"Error during conversion: (Exit code {result.returncode})\n"
                if result.stderr:
                    error_msg += f"Error details: {result.stderr}\n"
                if result.stdout:
                    error_msg += f"Output: {result.stdout}"
                
                # Fall back to direct PDF extraction
                fallback_text = MarkdownConverter.extract_text_from_pdf(file_content)
                return f"{error_msg}\n\nFalling back to direct text extraction:\n\n{fallback_text}"
            
            # Read the markdown output
            with open(output_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
                
            return markdown_content
        except Exception as e:
            return f"Unexpected error during conversion: {str(e)}"
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

    @staticmethod
    def extract_text_from_pdf(file_content):
        """
        Extract text directly from PDF (fallback method)
        
        Args:
            file_content: The binary content of the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            pdf = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            return f"Error extracting PDF text: {str(e)}"
    
    @staticmethod
    def convert_docx_to_markdown(file_content):
        """
        Convert DOCX content to Markdown
        
        Args:
            file_content: The binary content of the DOCX file
            
        Returns:
            str: Markdown content
        """
        try:
            doc = Document(io.BytesIO(file_content))
            text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text
        except Exception as e:
            return f"Error converting DOCX to markdown: {str(e)}"
    
    @staticmethod
    def convert_to_markdown(file_content, file_name):
        """
        Convert various file types to Markdown
        
        Args:
            file_content: The binary content of the file
            file_name: The name of the file
            
        Returns:
            str: Markdown content
        """
        file_extension = Path(file_name).suffix.lower()
        
        if file_extension == '.pdf':
            try:
                return MarkdownConverter.convert_pdf_to_markdown(file_content, file_name)
            except:
                # Fallback to direct text extraction if markitdown fails
                return MarkdownConverter.extract_text_from_pdf(file_content)
        elif file_extension == '.docx':
            return MarkdownConverter.convert_docx_to_markdown(file_content)
        else:
            return f"Unsupported file type: {file_extension}. Please upload a PDF or DOCX file."