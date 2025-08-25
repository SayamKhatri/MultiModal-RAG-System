from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import fitz  
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction, SentenceTransformerEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import google.generativeai as genai
import base64
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('pdf_images', exist_ok=True)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="multimodal_db")

# Text embedding function
text_embedder = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Image embedding function
image_embedder = OpenCLIPEmbeddingFunction()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_or_create_collections(document_id):
    """Get or create collections for a specific document"""
    text_collection_name = f"pdf_text_collection_{document_id}"
    image_collection_name = f"pdf_image_collection_{document_id}"
    
    # Try to get existing collections
    try:
        text_collection = client.get_collection(
            name=text_collection_name,
            embedding_function=text_embedder
        )
    except:
        # Create new collection if it doesn't exist
        text_collection = client.create_collection(
            name=text_collection_name,
            embedding_function=text_embedder
        )
    
    try:
        image_collection = client.get_collection(
            name=image_collection_name,
            embedding_function=image_embedder,
            data_loader=ImageLoader()
        )
    except:
        # Create new collection if it doesn't exist
        image_collection = client.create_collection(
            name=image_collection_name,
            embedding_function=image_embedder,
            data_loader=ImageLoader()
        )
    
    return text_collection, image_collection

def clear_collections(document_id):
    """Clear existing collections for a document"""
    text_collection_name = f"pdf_text_collection_{document_id}"
    image_collection_name = f"pdf_image_collection_{document_id}"
    
    # Delete existing collections if they exist
    try:
        client.delete_collection(name=text_collection_name)
    except:
        pass
    
    try:
        client.delete_collection(name=image_collection_name)
    except:
        pass

def process_pdf(pdf_path, document_id):
    """Process PDF and store embeddings in ChromaDB"""
    try:
        # Clear existing collections for this document
        clear_collections(document_id)
        
        # Get fresh collections
        text_collection, image_collection = get_or_create_collections(document_id)
        
        # Extract text
        doc = fitz.open(pdf_path)
        text_chunks = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                text_chunks.extend(text.split("\n"))  

        # Extract images
        images = []
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                img_path = f"pdf_images/{document_id}_page{page_num}_{img_index}.png"
                pix.save(img_path)
                images.append(img_path)

        # Store in ChromaDB
        if text_chunks:
            text_collection.add(
                ids=[f"text_{i}" for i in range(len(text_chunks))],
                documents=text_chunks
            )

        if images:
            image_collection.add(
                ids=[f"img_{i}" for i in range(len(images))],
                uris=images
            )

        return {
            'text_count': len(text_chunks),
            'image_count': len(images),
            'pages': len(doc)
        }
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise e

def encode_image_to_base64(image_path):
    """Convert image to base64 for frontend display"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and processing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PDF files are allowed.'}), 400
        
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{document_id}_{filename}")
        
        # Save file
        file.save(pdf_path)
        
        # Process PDF
        stats = process_pdf(pdf_path, document_id)
        
        return jsonify({
            'success': True,
            'document_id': document_id,
            'filename': filename,
            'stats': stats,
            'message': f'PDF processed successfully! Extracted {stats["text_count"]} text chunks and {stats["image_count"]} images from {stats["pages"]} pages.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests"""
    try:
        data = request.get_json()
        query_text = data.get('query', '')
        top_k = data.get('top_k', 3)
        document_id = data.get('document_id')
        
        if not query_text:
            return jsonify({'error': 'Query is required'}), 400
        
        if not document_id:
            return jsonify({'error': 'Document ID is required'}), 400
        
        # Get collections for this document
        text_collection, image_collection = get_or_create_collections(document_id)
        
        # Check if collections have data
        text_count = text_collection.count()
        image_count = image_collection.count()
        
        if text_count == 0 and image_count == 0:
            return jsonify({'error': 'No document data found. Please upload a PDF first.'}), 400
        
        # Initialize Gemini model
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Text retrieval
        retrieved_texts = []
        text_distances = []
        if text_count > 0:
            try:
                text_results = text_collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, text_count),
                    include=["documents", "distances"]
                )
                retrieved_texts = text_results['documents'][0]
                text_distances = text_results['distances'][0]
            except Exception as e:
                print(f"Error in text retrieval: {e}")
                retrieved_texts = []
                text_distances = []

        # Image retrieval
        retrieved_images = []
        image_distances = []
        if image_count > 0:
            try:
                image_results = image_collection.query(
                    query_texts=[query_text],
                    n_results=min(top_k, image_count),
                    include=["uris", "distances"]
                )
                retrieved_images = image_results['uris'][0]
                image_distances = image_results['distances'][0]
            except Exception as e:
                print(f"Error in image retrieval: {e}")
                retrieved_images = []
                image_distances = []

        # Convert images to base64 for frontend
        encoded_images = []
        for i, img_path in enumerate(retrieved_images):
            encoded_img = encode_image_to_base64(img_path)
            if encoded_img:
                encoded_images.append({
                    'path': img_path,
                    'data': encoded_img,
                    'distance': image_distances[i] if i < len(image_distances) else 0.0
                })

        # Prepare text results with distances
        text_results_formatted = []
        for i, (text, distance) in enumerate(zip(retrieved_texts, text_distances)):
            text_results_formatted.append({
                'text': text,
                'distance': distance,
                'id': f"text_{i}"
            })

        # Generate response using Gemini
        prompt_parts = [
            "You are a helpful assistant that answers based on PDF context.",
            f"User query: {query_text}",
        ]
        
        if retrieved_texts:
            prompt_parts.append("Relevant text snippets from the document:\n" + "\n---\n".join(retrieved_texts))
        
        try:
            if retrieved_images:
                prompt_parts.append("Relevant images are attached below.")
                image_parts = [genai.upload_file(img) for img in retrieved_images]
                response = model.generate_content(prompt_parts + image_parts)
            else:
                response = model.generate_content(prompt_parts)
            
            answer = response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            answer = f"I encountered an error while processing your query: {str(e)}"
        
        return jsonify({
            'query': query_text,
            'retrieved_texts': text_results_formatted,
            'retrieved_images': encoded_images,
            'answer': answer,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in query endpoint: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of uploaded documents"""
    try:
        documents = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if filename.endswith('.pdf'):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file_size = os.path.getsize(file_path)
                    documents.append({
                        'filename': filename,
                        'size': file_size,
                        'document_id': filename.split('_')[0]
                    })
        
        return jsonify({'documents': documents})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)
