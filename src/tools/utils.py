import io
import socket
import PyPDF2


def find_available_port(start_port=8000, max_port=9000):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise IOError("No free ports")


def extract_content(file_data):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
    text = []
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text.append(page.extract_text())
    return '\n'.join(text)

