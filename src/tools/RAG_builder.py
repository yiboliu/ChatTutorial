from src.tools import utils
from src.tools import vectorizer
from src.tools import chunker


def build_RAG(files, weaviate_client):
    vectorizer.initiate_storage(weaviate_client)
    for file in files:
        content = utils.extract_content(file.data)
        chunker.chunk_text(content, 3)
