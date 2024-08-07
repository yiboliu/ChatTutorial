import time

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template,
    flash,
    session,
)
from flask_sqlalchemy import SQLAlchemy
import weaviate

import os
from openai import OpenAI
import tempfile
import shutil
import uuid

from src.tools import RAG_builder

app = Flask(__name__, template_folder="../webpages", static_folder="../static")
app.config["UPLOAD_FOLDER"] = "../../uploads"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///../../files.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your_secret_key"  # Needed for flashing messages

db = SQLAlchemy(app)

client_map = {}


class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)

    def __init__(self, filename, data):
        self.filename = filename
        self.data = data


class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), nullable=False, unique=True)
    temp_dir = db.Column(db.String(255), nullable=True)

    def __init__(self, session_id, temp_dir):
        self.session_id = session_id
        self.temp_dir = temp_dir


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.Integer, default=int(time.time()))

    def __init__(self, session_id, role, content):
        self.session_id = session_id
        self.role = role
        self.content = content


with app.app_context():
    db.create_all()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)

    if file:
        filename = file.filename
        data = file.read()
        new_file = File(filename=filename, data=data)
        db.session.add(new_file)
        db.session.commit()
        flash("File successfully uploaded")
        return redirect(url_for("index"))


@app.route("/files")
def list_files():
    files = File.query.all()
    return render_template("files.html", files=files)


def get_weaviate_client(weaviate_host: str):
    return weaviate.Client(weaviate_host)


@app.route("/perform_operation", methods=["POST"])
def perform_operation():
    selected_files = request.form.getlist("file_ids")
    if not selected_files:
        flash("No files selected")
        return redirect(url_for("list_files"))

    files = [File.query.get(file_id) for file_id in selected_files]
    temp_dir = tempfile.mkdtemp()
    session["temp_dir"] = temp_dir

    session_id = str(uuid.uuid4())
    session["id"] = session_id

    user_session = UserSession(session_id=session_id, temp_dir=temp_dir)
    db.session.add(user_session)
    db.session.commit()

    client = weaviate.connect_to_local(
        host="weaviate", port=8080, grpc_port=50051
    )
    client_map[session_id] = client
    RAG_builder.build_rag(files, client)

    flash("Operation performed successfully")
    return redirect(url_for("inference_page"))


@app.route("/inference", methods=["GET", "POST"])
def inference_page():
    session_id = session.get("id")
    if not session_id:
        flash("No active session found")
        return redirect(url_for("list_files"))

    user_session = UserSession.query.filter_by(session_id=session_id).first()
    if not user_session:
        flash("No Weaviate session found")
        return redirect(url_for("list_files"))

    conversation = Conversation.query.filter_by(session_id=session_id).order_by(Conversation.timestamp).all()

    if request.method == "POST":
        query_text = request.form.get("query_text", "")

        # Save user message
        user_message = Conversation(session_id=session_id, role="user", content=query_text)
        db.session.add(user_message)
        db.session.commit()

        search_result = RAG_builder.semantic_search(query_text, 3)

        try:
            base_url = os.environ.get("EXTERNAL_SERVER_URL", "http://localhost:8081")
            openai_client = OpenAI(base_url=f"{base_url}/v1", api_key="sk-no-key-required")

            # Prepare conversation history for the API call
            messages = [{"role": msg.role, "content": msg.content} for msg in conversation[-5:]]  # Last 5 messages
            messages.append({"role": "user", "content": f"{query_text}\nContext: {search_result}"})
            before_time = int(time.time())
            flash(f"user input time: {before_time}")

            completion = openai_client.chat.completions.create(
                model="LLaMA_CPP",
                messages=messages
            )

            ai_response = completion.choices[0].message.content
            flash(f"response created time: {completion.created}")
            flash(f"Llamafile response time consumption: {completion.created - before_time}")

            # Save AI response
            ai_message = Conversation(session_id=session_id, role="assistant", content=ai_response)
            db.session.add(ai_message)
            db.session.commit()

        except Exception as e:
            ai_response = f"Error running LlamaFile: {str(e)}"
            flash(ai_response)

        return redirect(url_for("inference_page"))

    return render_template("inference.html", conversation=conversation)


@app.route("/cleanup")
def cleanup():
    session_id = session.get("id")
    if not session_id:
        flash("No active session found")
        return redirect(url_for("list_files"))
    user_session = UserSession.query.filter_by(session_id=session["id"]).first()

    # If the session exists
    if user_session:
        # Remove the temporary directory if it exists
        if user_session.temp_dir and os.path.exists(user_session.temp_dir):
            shutil.rmtree(user_session.temp_dir)

        # Delete the session details from the database
        db.session.delete(user_session)
        db.session.commit()

        flash("Temporary data cleaned up")
    else:
        flash("No user session found")
    return redirect(url_for("list_files"))


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(host="0.0.0.0", port=5000, debug=True)
