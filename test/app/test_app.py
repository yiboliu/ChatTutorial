import unittest
from unittest.mock import patch, MagicMock
from src.app.app import app, db, File, UserSession
from io import BytesIO
import uuid


class TestApp(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
        self.app = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_index_route(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)

    def test_upload_file(self):
        data = {"file": (BytesIO(b"test file content"), "test.txt")}
        response = self.app.post(
            "/upload", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 302)  # Redirect status code
        with app.app_context():
            file = File.query.filter_by(filename="test.txt").first()
            self.assertIsNotNone(file)
            self.assertEqual(file.data, b"test file content")

    def test_list_files(self):
        with app.app_context():
            file = File(filename="test.txt", data=b"test content")
            db.session.add(file)
            db.session.commit()

        response = self.app.get("/files")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"test.txt", response.data)

    @patch("src.app.app.weaviate.connect_to_local")
    @patch("src.app.app.RAG_builder.build_RAG")
    @patch("src.app.app.tempfile.mkdtemp")
    @patch("src.app.app.uuid.uuid4")
    def test_perform_operation(
        self, mock_uuid4, mock_mkdtemp, mock_build_rag, mock_connect_to_local
    ):
        # Set up mocks
        test_uuid = str(uuid.uuid4())
        mock_uuid4.return_value = test_uuid
        mock_mkdtemp.return_value = "/tmp/test_dir"
        mock_client = MagicMock()
        mock_connect_to_local.return_value = mock_client

        # Create a test file in the database
        with app.app_context():
            file = File(filename="test.txt", data=b"test content")
            db.session.add(file)
            db.session.commit()
            file_id = file.id

        # Perform the operation
        response = self.app.post(
            "/perform_operation", data={"file_ids": [file_id]}
        )

        # Check redirect
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/inference")

        # Check if mocks were called correctly
        mock_connect_to_local.assert_called_once_with(
            host="weaviate", port=8080, grpc_port=50051
        )
        mock_build_rag.assert_called_once()

        # Check if a UserSession was created
        with app.app_context():
            user_session = UserSession.query.filter_by(
                session_id=test_uuid
            ).first()
            self.assertIsNotNone(user_session)
            self.assertEqual(user_session.temp_dir, "/tmp/test_dir")

        # Check if flash message was set
        with self.app.session_transaction() as session:
            flash_messages = session.get("_flashes", [])
            self.assertTrue(
                any(
                    "Operation performed successfully" in message
                    for _, message in flash_messages
                )
            )

        # Check if session variables were set correctly
        with self.app.session_transaction() as session:
            self.assertEqual(session["temp_dir"], "/tmp/test_dir")
            self.assertEqual(session["id"], test_uuid)

    @patch("src.app.app.RAG_builder.semantic_search")
    @patch("src.app.app.OpenAI")
    def test_inference_page(self, mock_openai, mock_semantic_search):
        mock_semantic_search.return_value = "Mock search results"
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Mock LLM response"
        mock_openai.return_value.chat.completions.create.return_value = (
            mock_completion
        )

        with app.app_context():
            user_session = UserSession(
                session_id="test_session", temp_dir="/tmp/test"
            )
            db.session.add(user_session)
            db.session.commit()

        with self.app.session_transaction() as session:
            session["id"] = "test_session"

        response = self.app.post(
            "/inference", data={"query_text": "test query"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Mock search results", response.data)
        self.assertIn(b"Mock LLM response", response.data)

    @patch("src.app.app.shutil.rmtree")
    @patch("src.app.app.os.path.exists")
    def test_cleanup(self, mock_exists, mock_rmtree):
        mock_exists.return_value = (
            True  # Simulate that the temp directory exists
        )

        with app.app_context():
            user_session = UserSession(
                session_id="test_session", temp_dir="/tmp/test"
            )
            db.session.add(user_session)
            db.session.commit()

        with self.app.session_transaction() as session:
            session["id"] = "test_session"

        response = self.app.get("/cleanup")
        self.assertEqual(response.status_code, 302)  # Redirect status code

        mock_rmtree.assert_called_once_with("/tmp/test")

        with app.app_context():
            user_session = UserSession.query.filter_by(
                session_id="test_session"
            ).first()
            self.assertIsNone(user_session)

        # Check if flash message was set
        with self.app.session_transaction() as session:
            flash_messages = session.get("_flashes", [])
            self.assertTrue(
                any(
                    "Temporary data cleaned up" in message
                    for _, message in flash_messages
                )
            )


if __name__ == "__main__":
    unittest.main()
