import re

def chunk_text_by_paragraphs(text: str) -> list[str]:
    """
    Splits a given text into chunks based on paragraphs.

    Paragraphs are typically separated by one or more blank lines in Markdown.
    This function identifies such separations and splits the text accordingly.
    Empty strings resulting from multiple blank lines are filtered out.

    Args:
        text (str): The input string, expected to be Markdown content or
                    plain text where paragraphs are separated by blank lines.

    Returns:
        list[str]: A list of non-empty strings, where each string is a
                   paragraph (chunk) from the input text.
    """
    if not text or not text.strip():
        return []

    # Normalize newlines (optional, but can help with mixed Windows/Unix newlines)
    # text = text.replace('\r\n', '\n') # Not strictly necessary with re.split on \n\s*\n

    # Split by one or more blank lines (sequences of newlines, possibly with whitespace)
    # text.strip() handles leading/trailing newlines on the entire text
    chunks = re.split(r'\n\s*\n', text.strip())
    
    # Filter out any empty strings that might result from the split
    # (e.g., if there were more than two newlines between paragraphs)
    non_empty_chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    
    return non_empty_chunks

if __name__ == '__main__':
    # Example Usage for direct testing
    sample_text_1 = """This is the first paragraph.
It has multiple lines.

This is the second paragraph.

This is the third.


This is the fourth, after extra blank lines."""
    
    sample_text_2 = "Single paragraph only."
    
    sample_text_3 = ""
    
    sample_text_4 = "\n\n   \n\nFirst paragraph after leading newlines.\n\nSecond one. \n\n"

    print("Sample 1 Chunks:")
    for i, chunk in enumerate(chunk_text_by_paragraphs(sample_text_1)):
        print(f"Chunk {i+1}:\n---\n{chunk}\n---")

    print("\nSample 2 Chunks:")
    for i, chunk in enumerate(chunk_text_by_paragraphs(sample_text_2)):
        print(f"Chunk {i+1}:\n---\n{chunk}\n---")

    print("\nSample 3 Chunks (empty input):")
    print(chunk_text_by_paragraphs(sample_text_3))
    
    print("\nSample 4 Chunks (leading/trailing newlines and spaces):")
    for i, chunk in enumerate(chunk_text_by_paragraphs(sample_text_4)):
        print(f"Chunk {i+1}:\n---\n{chunk}\n---")

    sample_text_5 = "Para1\n\n\nPara2\n\n\n\nPara3"
    print("\nSample 5 Chunks (multiple empty lines between paragraphs):")
    for i, chunk in enumerate(chunk_text_by_paragraphs(sample_text_5)):
        print(f"Chunk {i+1}:\n---\n{chunk}\n---")
