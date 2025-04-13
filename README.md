# PDF to Markdown Converter

A Streamlit application that converts PDF documents to Markdown format for use with GPT and other AI chat bots.

## Features

- PDF and DOCX file upload
- Conversion to markdown format using Microsoft's MarkItDown tool
- Preview of the markdown output
- Copy functionality for easy pasting into a GPT chat
- Download option for the converted markdown file

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install the MarkItDown package:

```bash
pip install markitdown
```

Note: The application will try to automatically install the markitdown package if it's not already installed.

## Usage

1. Start the Streamlit app:

```bash
cd markitdown-streamlit
streamlit run src/app.py
```

2. Upload a PDF or DOCX file using the file uploader
3. Click the "Convert to Markdown" button
4. Copy the generated markdown text from the text area
5. Paste the markdown into your GPT chat interface with a prompt like "Summarize this document for me:"

## Requirements

- Python 3.7+
- Streamlit
- PyPDF2
- python-docx
- markitdown (Microsoft's PDF to Markdown converter)

## How It Works

The application uses Microsoft's MarkItDown tool to convert PDF documents to Markdown format. If the conversion fails, it falls back to basic text extraction from the PDF.

For DOCX files, it extracts text content directly from the document.

## Tips for Best Results

- For best results, use PDF documents with well-structured text
- Very large files may take longer to process
- Complex document formatting (tables, images, etc.) may not convert perfectly to markdown
- After conversion, you might need to make small edits to optimize the markdown for your specific use case

## License

This project uses the MIT License.