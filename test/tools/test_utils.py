import unittest
from unittest.mock import patch, MagicMock
import io
import socket
from src.tools.utils import find_available_port, extract_content


class TestUtils(unittest.TestCase):

    @patch('socket.socket')
    def test_find_available_port_success(self, mock_socket):
        mock_socket.return_value.__enter__.return_value.bind.side_effect = [OSError, None]

        port = find_available_port(start_port=8000, max_port=8002)

        self.assertEqual(port, 8001)
        self.assertEqual(mock_socket.call_count, 2)

    @patch('socket.socket')
    def test_find_available_port_no_free_ports(self, mock_socket):
        mock_socket.return_value.__enter__.return_value.bind.side_effect = OSError

        with self.assertRaises(IOError):
            find_available_port(start_port=8000, max_port=8002)

        self.assertEqual(mock_socket.call_count, 2)

    @patch('PyPDF2.PdfReader')
    def test_extract_content(self, mock_pdf_reader):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_pdf_reader.return_value.pages = [mock_page1, mock_page2]

        file_data = b'mock pdf content'
        result = extract_content(file_data)

        self.assertEqual(result, "Page 1 content\nPage 2 content")
        mock_pdf_reader.assert_called_once()
        self.assertEqual(mock_pdf_reader.call_args[0][0].getvalue(), file_data)
        mock_page1.extract_text.assert_called_once()
        mock_page2.extract_text.assert_called_once()

    @patch('PyPDF2.PdfReader')
    def test_extract_content_empty_pdf(self, mock_pdf_reader):
        mock_pdf_reader.return_value.pages = []

        file_data = b'empty pdf'
        result = extract_content(file_data)

        self.assertEqual(result, "")
        mock_pdf_reader.assert_called_once()
        self.assertEqual(mock_pdf_reader.call_args[0][0].getvalue(), file_data)


if __name__ == '__main__':
    unittest.main()