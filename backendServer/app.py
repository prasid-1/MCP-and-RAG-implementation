import datetime
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import chromadb
import uuid # For generating unique IDs for ChromaDB documents

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configuration ---
# Get the upload folder path from environment variables
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '../data/upload')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER) # Create the upload folder if it doesn't exist

# Initialize ChromaDB client
# This will create a local ChromaDB instance in a directory named 'chroma_data'
# For persistent storage, ensure this directory is not deleted.
client = chromadb.PersistentClient(path="./chroma_data")

# Get or create a collection
# A collection is where your documents (with embeddings and metadata) are stored
collection_name = "user_dropped_files"
try:
    collection = client.get_or_create_collection(name=collection_name)
    print(f"ChromaDB collection '{collection_name}' ready.")
except Exception as e:
    print(f"Error initializing ChromaDB collection: {e}")
    # Exit or handle the error appropriately if ChromaDB cannot be initialized

# --- Helper Function for Content Extraction (Basic Example) ---
def extract_content_for_chroma(filepath, filename):
    """
    A basic function to extract content from a file for ChromaDB.
    In a real application, you would use libraries to parse various file types
    (e.g., PyPDF2 for PDFs, Pillow for images, docx for Word docs).
    For this example, we'll just read text files or use a placeholder.
    """
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()

    if file_extension in ['.jpg', '.csv']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Could not read text from {filename}: {e}")
            return f"Content of {filename}. Could not read text directly."
    else:
        # For non-text files, we'll just use a descriptive string.
        # In a real app, you might use OCR for images, or specific parsers.
        return f"Binary file: {filename}. Stored at {filepath}. Content not extracted for embedding."


# --- API Route for File Upload and ChromaDB Integration ---
@app.route('/upload-to-chroma', methods=['POST'])
def upload_to_chroma():
    if 'files' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'message': 'No selected file'}), 400

    processed_files_info = []
    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for file in files:
        if file.filename == '':
            continue

        # Generate a unique filename to prevent overwrites
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

        try:
            # Save the file to the server's specific path
            file.save(filepath)
            print(f"File saved to: {filepath}")

            # Extract content for ChromaDB (or use a placeholder)
            document_content = extract_content_for_chroma(filepath, file.filename)

            # Prepare data for ChromaDB
            doc_id = str(uuid.uuid4()) # Unique ID for each document in ChromaDB
            documents_to_add.append(document_content)
            metadatas_to_add.append({
                "original_filename": file.filename,
                "server_path": filepath,
                "file_size_bytes": file.content_length,
                "mime_type": file.mimetype,
                "uploaded_at": datetime.now().isoformat() # Add a timestamp
            })
            ids_to_add.append(doc_id)

            processed_files_info.append({
                'original_filename': file.filename,
                'server_path': filepath,
                'chroma_doc_id': doc_id
            })

        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            return jsonify({'message': f'Error processing file {file.filename}: {str(e)}'}), 500

    try:
        # Add all processed documents to ChromaDB in a single batch
        if documents_to_add:
            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            print(f"Added {len(documents_to_add)} documents to ChromaDB.")
            # Example: Query ChromaDB to verify (optional, for debugging)
            # results = collection.query(query_texts=["example query"], n_results=2)
            # print("ChromaDB query results (for verification):", results)
        else:
            print("No documents to add to ChromaDB.")

    except Exception as e:
        print(f"Error adding documents to ChromaDB: {e}")
        return jsonify({'message': f'Error adding documents to ChromaDB: {str(e)}'}), 500

    return jsonify({
        'message': 'Files uploaded and processed successfully!',
        'details': processed_files_info
    }), 200

# --- Basic Route for Health Check (Optional) ---
@app.route('/')
def home():
    return "ChromaDB Backend is running!"

if __name__ == '__main__':
    # Ensure the upload folder exists before starting the app
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5000) # Run on port 5000, debug=True for development
