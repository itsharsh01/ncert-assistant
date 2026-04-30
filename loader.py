from langchain_opendataloader_pdf import OpenDataLoaderPDFLoader

loader = OpenDataLoaderPDFLoader(
    file_path="chapter.pdf",
    format="markdown"
)

docs = loader.load()