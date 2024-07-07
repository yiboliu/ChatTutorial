from flask import Flask, request, redirect, url_for, render_template, flash, session
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import os
import tempfile
import shutil

app = Flask(__name__, template_folder='../webpages', static_folder='../static')
app.config['UPLOAD_FOLDER'] = '../../uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../../files.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'  # Needed for flashing messages

db = SQLAlchemy(app)


class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

    def __init__(self, filename, data):
        self.filename = filename
        self.data = data


with app.app_context():
    db.create_all()


def extract_content(file_data):
    # Dummy content extraction for demonstration purposes
    return np.random.rand(512).astype('float32')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        filename = file.filename
        data = file.read()
        new_file = File(filename=filename, data=data)
        db.session.add(new_file)
        db.session.commit()
        flash('File successfully uploaded')
        return redirect(url_for('index'))


@app.route('/files')
def list_files():
    files = File.query.all()
    return render_template('files.html', files=files)


@app.route('/perform_operation', methods=['POST'])
def perform_operation():
    selected_files = request.form.getlist('file_ids')
    if not selected_files:
        flash('No files selected')
        return redirect(url_for('list_files'))

    files = [File.query.get(file_id) for file_id in selected_files]
    temp_dir = tempfile.mkdtemp()
    session['temp_dir'] = temp_dir


    return redirect(url_for('inference_page'))


@app.route('/inference', methods=['GET', 'POST'])
def inference_page():
    if request.method == 'POST':
        flash(f'reading indexes and search')
        # query_vector = np.random.rand(512).astype('float32')  # Dummy query vector
        # temp_dir = session.get('temp_dir')
        # if temp_dir:
        #     index = faiss.read_index(os.path.join(temp_dir, 'vector_index'))
        #     distances, indices = index.inference(np.array([query_vector]), k=5)
        #     results = [indices[0]]  # For demonstration, returning the indices of top results
        #     flash(f'inference results: {results}')
        # else:
        #     flash('No temporary vector database found')

    return render_template('inference.html')


@app.route('/cleanup')
def cleanup():
    temp_dir = session.pop('temp_dir', None)
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        flash('Temporary vector database cleaned up')
    return redirect(url_for('list_files'))


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
