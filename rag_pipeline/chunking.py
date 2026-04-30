from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

def chunk_ncert_content(markdown_text: str):
    """
    Applies Structural Chunking followed by a Recursive Character Fallback.
    """
    # STEP 1: Structural Splitting
    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Main_Topic"),
        ("###", "Sub_Topic"),
        ('####', "Sub_Sub_Topic"),
        ("#####", "Sub_Sub_Sub_Topic")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the header in the chunk text for LLM context
    )

    structural_chunks = markdown_splitter.split_text(markdown_text)
    print(f"Generated {len(structural_chunks)} structural chunks.")

    # STEP 2: Recursive Fallback
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Characters (approx 200-250 tokens)
        chunk_overlap=150,   # Preserve context across splits
        separators=["\n\n", "\n", ". ", " ", ""] # Hierarchical splitting
    )

    final_chunks = recursive_splitter.split_documents(structural_chunks)

    return final_chunks
