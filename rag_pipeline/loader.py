from llama_parse import LlamaParse

def parse_ncert_to_markdown(file_path: str) -> str:
    """
    Converts the NCERT PDF into clean Markdown using LlamaParse.
    This preserves the layout, tables, and structural headers.
    """
    print(f"Parsing {file_path} to Markdown...")
    parser = LlamaParse(
        result_type="markdown",  # Crucial for structural chunking
        verbose=True,
        language="en"
    )

    # Extract documents
    documents = parser.load_data(file_path)

    # Combine pages into a single markdown string
    full_markdown = "\n\n".join([doc.text for doc in documents])
    return full_markdown
