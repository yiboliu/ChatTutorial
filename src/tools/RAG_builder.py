from src.tools import utils
import weaviate
import weaviate.classes as wvc
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

class_name = 'TextChunk'
nltk.download('punkt')


def build_rag(files, weaviate_client):
    initiate_storage(weaviate_client)
    for file in files:
        content = utils.extract_content(file.data)
        chunk_text(content, 3)


def add_text_chunk_to_db(chunk: str):
    weaviate_client = weaviate.connect_to_local(host='weaviate', port=8080, grpc_port=50051)
    try:
        text_chunk = weaviate_client.collections.get(class_name)
        result = text_chunk.data.insert({
            'content': chunk,
        })
        print(f'content is {chunk}')
        print(f'result is {result}')
    finally:
        weaviate_client.close()


def initiate_storage(weaviate_client):
    try:
        weaviate_client.collections.delete_all()
        weaviate_client.collections.create(
            name=class_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
            properties=[
                wvc.config.Property(
                    name='content',
                    data_type=wvc.config.DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=wvc.config.Tokenization.LOWERCASE
                ),
            ]
        )
    finally:
        weaviate_client.close()


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


def semantic_search(query: str, chunk_num: int) -> str:
    client = weaviate.connect_to_local(host='weaviate', port=8080, grpc_port=50051)
    try:
        collection = client.collections.get(class_name)
        response = collection.query.fetch_objects(include_vector=True)
        vectors = [o.vector['default'] for o in response.objects]

        model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        query_vector = model.encode(query)

        dot_product = np.dot(vectors, query_vector)
        query_norm = np.linalg.norm(query_vector)
        chunk_norms = np.linalg.norm(vectors, axis=1)
        cosine_sim = dot_product / (query_norm * chunk_norms)
        max_indices = np.argsort(cosine_sim)[-chunk_num:]
        most_sim_contents = [response.objects[i].properties['content'] for i in max_indices]
        return ' '.join(most_sim_contents)

    finally:
        client.close()