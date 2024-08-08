from src.tools import utils
import weaviate
import weaviate.classes as wvc
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

class_name = "TextChunk"
nltk.download("punkt")


def build_rag(files, weaviate_client):
    """This function builds up the RAG pipeline. It takes the file and database client as input, initiate the storage
    and extract the contents of each file to store in the vector database.
    Args:
        files: the files that are to be used in the RAG
        weaviate_client: the weaviate client for storing file contents in the vector database.
    """
    initiate_storage(weaviate_client)
    for file in files:
        content = utils.extract_content(file.data)
        chunk_text(content, 3)


def add_text_chunk_to_db(chunk: str):
    """This function stores the chunk of text in the database, so that it can be used for semantic search later on.
    Args:
        chunk: the string value of that text chunk
    """
    weaviate_client = weaviate.connect_to_local(
        host="weaviate", port=8080, grpc_port=50051
    )
    try:
        text_chunk = weaviate_client.collections.get(class_name)
        result = text_chunk.data.insert(
            {
                "content": chunk,
            }
        )
        print(f"content is {chunk}")
        print(f"result is {result}")
    finally:
        weaviate_client.close()


def initiate_storage(weaviate_client):
    """This function initiate the storage of the weaviate database. NOTE: this should only be called once for each user
    session.
    Args:
        weaviate_client: the database client that is used to create the collection.
    """
    try:
        weaviate_client.collections.delete_all()  # clear any existing data so that new data is not polluted.
        weaviate_client.collections.create(
            name=class_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
            properties=[
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=wvc.config.Tokenization.LOWERCASE,
                ),
            ],
        )
    finally:
        weaviate_client.close()


def chunk_text(text: str, max_chunk_size: int):
    """For the text from the entire file, this function tokenizes it by sentences, and then for each sentence, it
    combines the max_chunk_size number of sentences together as a chunk and process it. The remainder of unprocessed
    sentences will be processed as a chunk.
    Args:
        text: the text from the entire file
        max_chunk_size: the max number of sentences to be included in a chunk
    """
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
    """Chunks will be joined together and stored in the database."""
    content = " ".join(chunks)
    add_text_chunk_to_db(content)


def semantic_search(query: str, chunk_num: int) -> str:
    """This functions takes a query string and chunk_num integer as input. The query string is used as the criterion for
    query: convert the string to a vector and find the chunk_num closest chunks, which combined to be the output.
    Args:
        query: the string that user inputs as the query
        chunk_num: the number of chunks to be used in output, meaning how long would be the context information.
    """
    client = weaviate.connect_to_local(host="weaviate", port=8080, grpc_port=50051)
    try:
        collection = client.collections.get(class_name)
        response = collection.query.fetch_objects(include_vector=True)
        vectors = [o.vector["default"] for o in response.objects]

        model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        query_vector = model.encode(query)

        dot_product = np.dot(vectors, query_vector)
        query_norm = np.linalg.norm(query_vector)
        chunk_norms = np.linalg.norm(vectors, axis=1)
        cosine_sim = dot_product / (query_norm * chunk_norms)
        max_indices = np.argsort(cosine_sim)[-chunk_num:]
        most_sim_contents = [
            response.objects[i].properties["content"] for i in max_indices
        ]
        return " ".join(most_sim_contents)

    finally:
        client.close()
