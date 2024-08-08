import io
import socket
import PyPDF2


def find_available_port(start_port=8000, max_port=9000):
    """Find the next available port between start_port and max_port, which defaults to 8000 and 9000."""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise IOError("No free ports")


def extract_content(file_data):
    """This function extracts the text data from file provided. Only pdf files are supported."""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
    text = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text.append(page.extract_text())
    return "\n".join(text)
