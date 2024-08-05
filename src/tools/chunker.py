import nltk
from nltk.tokenize import sent_tokenize

from src.tools.vectorizer import add_text_chunk_to_db

nltk.download('punkt')


def chunk_text(text, max_chunk_size):
    sentences = sent_tokenize(text)
    chunks = []

    for sentence in sentences:
        chunks.append(sentence)
        if len(chunks) == max_chunk_size:
            process_chunk(chunks)
            chunks = []
    if chunks:
        process_chunk(chunks)


def process_chunk(chunks: []):
    content = ' '.join(chunks)
    add_text_chunk_to_db(content)
