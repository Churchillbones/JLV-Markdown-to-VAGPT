import tempfile
import os
import subprocess
import logging
import sys
import requests
import io
import re # Import re for regular expressions
from pathlib import Path
from PyPDF2 import PdfReader, errors as PyPDF2Errors # Import errors for specific handling
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
                logging.error(error_msg)
                # Fall back to direct PDF extraction
                fallback_text = MarkdownConverter.extract_text_from_pdf(file_content)
                return f"Error during markitdown conversion. Falling back to direct text extraction:\n\n{fallback_text}"

            # Read the markdown output
            with open(output_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            return markdown_content
        except Exception as e:
            logging.error(f"Unexpected error during conversion: {str(e)}")
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
                page_text = page.extract_text()
                if page_text:  # Ensure text was extracted
                    text += page_text + "\n\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting PDF text: {str(e)}")
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
            logging.error(f"Error converting DOCX to markdown: {str(e)}")
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
            return MarkdownConverter.convert_pdf_to_markdown(file_content, file_name)
        elif file_extension == '.docx':
            return MarkdownConverter.convert_docx_to_markdown(file_content)
        else:
            logging.warning(f"Unsupported file type: {file_extension}")
            return f"Unsupported file type: {file_extension}. Please upload a PDF or DOCX file."

    @staticmethod
    def extract_text_and_metadata_per_page(file_content: bytes) -> list[tuple[int, str, str]]:
        """
        Extracts text and attempts to find heuristic metadata (signer, date) from each page of a PDF.

        Args:
            file_content: The binary content of the PDF file.

        Returns:
            A list of tuples, where each tuple contains:
            (page_number (1-indexed), page_text (str), metadata_string (str)).
            Returns an empty list if the PDF cannot be processed.
        """
        results = []
        try:
            pdf_reader = PdfReader(io.BytesIO(file_content))
            num_pages = len(pdf_reader.pages)

            # Regex patterns (can be refined)
            # Date: Matches MM/DD/YYYY, MM-DD-YYYY, Month DD, YYYY, etc.
            date_pattern = re.compile(
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4})\b',
                re.IGNORECASE
            )
            # Name: Looks for capitalized words after common titles or "Signed by"
            # This is highly heuristic and may need significant refinement.
            name_pattern = re.compile(
                r'(?:Signed by|Electronically Signed by|Physician|Provider|Doctor|DR\.?|MD|NP|DO)\s*:?\s*([A-Z][a-zA-Z\'-]+(?:\s+[A-Z][a-zA-Z\'-]+){0,3}(?:,\s*(?:MD|DO|NP|PA|RN))?)',
                re.IGNORECASE
            )
            # Alternative simple name pattern if the above is too complex or fails
            simple_name_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')


            for page_num in range(num_pages):
                page_object = pdf_reader.pages[page_num]
                page_text = ""
                try:
                    page_text = page_object.extract_text()
                    if page_text is None: # extract_text can return None
                        page_text = ""
                except Exception: # Catch any error during text extraction for a page
                    page_text = "" # Default to empty string

                metadata_string = "Metadata is missing"
                found_name = None
                found_date = None

                if page_text.strip(): # Only process if there's text on the page
                    # Define "bottom of the page" - e.g., last 20 lines
                    lines = page_text.splitlines()
                    bottom_section_lines = lines[-20:] # Take last 20 lines
                    bottom_section_text = "\n".join(bottom_section_lines)

                    date_match = date_pattern.search(bottom_section_text)
                    if date_match:
                        found_date = date_match.group(0)

                    name_match = name_pattern.search(bottom_section_text)
                    if name_match:
                        found_name = name_match.group(1).strip()
                    else: # Fallback to a simpler name pattern if the primary one fails
                        # Search for any capitalized name-like structure in the bottom section
                        # This is very broad, so it's a fallback.
                        simple_name_matches = simple_name_pattern.findall(bottom_section_text)
                        if simple_name_matches:
                             # Heuristic: pick the longest suitable name, or one near a keyword if possible
                             # For now, just picking the first one if it's reasonably long
                            for potential_name in simple_name_matches:
                                if len(potential_name.split()) >= 2: # Require at least two words for a name
                                    found_name = potential_name
                                    break
                            if not found_name and simple_name_matches: # if no two-word name, take first one
                                found_name = simple_name_matches[0]


                # Construct metadata string if both are found
                if found_name and found_date:
                    metadata_string = f"Signed by {found_name} on {found_date}"
                elif found_name:
                    metadata_string = f"Signed by {found_name} (date not found)"
                elif found_date:
                    metadata_string = f"Date found: {found_date} (signer not found)"
                
                results.append((page_num + 1, page_text, metadata_string))

        except PyPDF2Errors.PdfReadError as e: # More specific error for bad PDFs
            logging.error(f"PyPDF2 PdfReadError: {e}")
            return [] # Return empty list for bad PDFs
        except Exception as e: # Catch any other unexpected errors during PDF processing
            logging.error(f"Unexpected error in extract_text_and_metadata_per_page: {e}")
            return [] # Return empty list

        return results