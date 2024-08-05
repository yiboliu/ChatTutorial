import weaviate
import weaviate.classes as wvc


class_name = 'TextChunk'


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
