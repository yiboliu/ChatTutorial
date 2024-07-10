import weaviate
from frontend.tools.vectorizer import class_name
from sentence_transformers import SentenceTransformer
import numpy as np


def semantic_search(query: str, chunk_num: int) -> str:
    client = weaviate.connect_to_local()
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
