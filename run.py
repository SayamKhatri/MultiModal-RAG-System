#!/usr/bin/env python3
"""
Startup script for the Multi-Modal RAG System
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import chromadb
        import google.generativeai
        import sentence_transformers
        print("All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if Google API key is set"""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("GOOGLE_API_KEY environment variable is not set")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your_api_key_here'")
        return False
    print("Google API key is configured")
    return True

def clear_and_prepare_directories():
    """Clear all data and prepare fresh directories for new session"""
    uploads_dir = 'uploads'
    pdf_images_dir = 'pdf_images'
    multimodal_db_dir = 'multimodal_db'
    
    # Clear uploads directory
    if os.path.exists(uploads_dir):
        import shutil
        shutil.rmtree(uploads_dir)
    
    # Clear pdf_images directory
    if os.path.exists(pdf_images_dir):
        import shutil
        shutil.rmtree(pdf_images_dir)
    
    # Clear ChromaDB directory
    if os.path.exists(multimodal_db_dir):
        import shutil
        shutil.rmtree(multimodal_db_dir)
    
    # Create fresh directories
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(pdf_images_dir, exist_ok=True)
    
    return True

def main():
    """Main startup function"""
    print("Starting Multi-Modal RAG System...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_dependencies():
        sys.exit(1)
    
    if not check_api_key():
        sys.exit(1)
    
    if not clear_and_prepare_directories():
        sys.exit(1)
    
    print("=" * 50)
    print("All checks passed! Starting fresh web server...")
    port = int(os.getenv('FLASK_PORT', 5001))
    print(f"The application will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app
        port = int(os.getenv('FLASK_PORT', 5001))
        app.run(debug=True, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
