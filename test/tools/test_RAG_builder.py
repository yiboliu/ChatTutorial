import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.tools.RAG_builder import (
    build_rag,
    add_text_chunk_to_db,
    initiate_storage,
    chunk_text,
    process_chunk,
    semantic_search,
)


class TestRAGBuilder(unittest.TestCase):

    @patch("src.tools.RAG_builder.initiate_storage")
    @patch("src.tools.RAG_builder.utils.extract_content")
    @patch("src.tools.RAG_builder.chunk_text")
    def test_build_rag(
        self, mock_chunk_text, mock_extract_content, mock_initiate_storage
    ):
        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.data = b"test data"
        mock_extract_content.return_value = "extracted content"

        build_rag([mock_file], mock_client)

        mock_initiate_storage.assert_called_once_with(mock_client)
        mock_extract_content.assert_called_once_with(b"test data")
        mock_chunk_text.assert_called_once_with("extracted content", 3)

    @patch("src.tools.RAG_builder.weaviate.connect_to_local")
    def test_add_text_chunk_to_db(self, mock_connect):
        mock_client = MagicMock()
        mock_connect.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        add_text_chunk_to_db("test chunk")

        mock_connect.assert_called_once_with(
            host="weaviate", port=8080, grpc_port=50051
        )
        mock_client.collections.get.assert_called_once_with("TextChunk")
        mock_collection.data.insert.assert_called_once_with({"content": "test chunk"})
        mock_client.close.assert_called_once()

    def test_initiate_storage(self):
        mock_client = MagicMock()

        initiate_storage(mock_client)

        mock_client.collections.delete_all.assert_called_once()
        mock_client.collections.create.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("src.tools.RAG_builder.sent_tokenize")
    @patch("src.tools.RAG_builder.process_chunk")
    def test_chunk_text(self, mock_process_chunk, mock_sent_tokenize):
        mock_sent_tokenize.return_value = [
            "Sentence 1.",
            "Sentence 2.",
            "Sentence 3.",
            "Sentence 4.",
        ]

        chunk_text("Test text", 2)

        mock_sent_tokenize.assert_called_once_with("Test text")
        self.assertEqual(mock_process_chunk.call_count, 2)
        mock_process_chunk.assert_any_call(["Sentence 1.", "Sentence 2."])
        mock_process_chunk.assert_any_call(["Sentence 3.", "Sentence 4."])

    @patch("src.tools.RAG_builder.add_text_chunk_to_db")
    def test_process_chunk(self, mock_add_text_chunk_to_db):
        process_chunk(["Chunk 1", "Chunk 2"])

        mock_add_text_chunk_to_db.assert_called_once_with("Chunk 1 Chunk 2")

    @patch("src.tools.RAG_builder.weaviate.connect_to_local")
    @patch("src.tools.RAG_builder.SentenceTransformer")
    def test_semantic_search(self, mock_sentence_transformer, mock_connect):
        mock_client = MagicMock()
        mock_connect.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_response = MagicMock()
        mock_response.objects = [
            MagicMock(
                vector={"default": np.array([1, 0, 0])},
                properties={"content": "Content 1"},
            ),
            MagicMock(
                vector={"default": np.array([0, 1, 0])},
                properties={"content": "Content 2"},
            ),
            MagicMock(
                vector={"default": np.array([0, 0, 1])},
                properties={"content": "Content 3"},
            ),
        ]
        mock_collection.query.fetch_objects.return_value = mock_response

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([1, 1, 0])
        mock_sentence_transformer.return_value = mock_model

        result = semantic_search("test query", 2)

        mock_connect.assert_called_once_with(
            host="weaviate", port=8080, grpc_port=50051
        )
        mock_client.collections.get.assert_called_once_with("TextChunk")
        mock_collection.query.fetch_objects.assert_called_once_with(include_vector=True)
        mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        )
        mock_model.encode.assert_called_once_with("test query")
        mock_client.close.assert_called_once()

        self.assertEqual(result, "Content 1 Content 2")


if __name__ == "__main__":
    unittest.main()
