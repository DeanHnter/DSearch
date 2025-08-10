#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
import uuid
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class DocumentSearchHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, doc_indexer, config, *args, **kwargs):
        self.doc_indexer = doc_indexer
        self.config = config
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        if path == '/search':
            self.handle_search(query_params)
        elif path == '/test':
            self.handle_test(query_params)
        elif path == '/health':
            self.handle_health()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/index':
            self.handle_index()
        elif path == '/add':
            self.handle_add()
        else:
            self.send_error(404, "Not Found")
    
    def handle_search(self, query_params):
        try:
            query = query_params.get('q', [''])[0]
            if not query:
                self.send_json_response({"error": "Missing required parameter 'q'"}, 400)
                return
            
            max_results = int(query_params.get('max_results', [self.config['max_results']])[0])
            threshold = float(query_params.get('threshold', [self.config['similarity_threshold']])[0])
            
            results = self.doc_indexer.search(query, max_results, threshold)
            response_data = {
                "query": query,
                "results": len(results),
                "documents": results
            }
            
            if results:
                merged_content = self.doc_indexer.merge_results(results)
                response_data["merged_content"] = merged_content
            
            self.send_json_response(response_data)
            
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_index(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({"error": "Missing request body"}, 400)
                return
            
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            directory_path = data.get('directory')
            if not directory_path:
                self.send_json_response({"error": "Missing required field 'directory'"}, 400)
                return
            
            if not os.path.exists(directory_path):
                self.send_json_response({"error": f"Directory '{directory_path}' does not exist"}, 400)
                return
            
            self.doc_indexer.index_directory(directory_path)
            self.send_json_response({"message": f"Successfully indexed documents in {directory_path}"})
            
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON in request body"}, 400)
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_add(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({"error": "Missing request body"}, 400)
                return
            
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            content = data.get('content')
            if not content:
                self.send_json_response({"error": "Missing required field 'content'"}, 400)
                return
            
            index_path = data.get('index_path', self.config['index_path'])
            self.doc_indexer.add_document(content, index_path)
            self.send_json_response({"message": "Document added and indexed successfully"})
            
        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON in request body"}, 400)
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_test(self, query_params):
        try:
            index_path = query_params.get('index_path', [self.config['index_path']])[0]
            run_simple_test(self.doc_indexer, index_path)
            self.send_json_response({"message": "Test completed successfully"})
        except Exception as e:
            self.send_json_response({"error": str(e)}, 500)
    
    def handle_health(self):
        self.send_json_response({"status": "healthy", "service": "document-search"})
    
    def send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def create_http_handler(doc_indexer, config):
    def handler(*args, **kwargs):
        return DocumentSearchHTTPHandler(doc_indexer, config, *args, **kwargs)
    return handler

def main():
    parser = argparse.ArgumentParser(description="Index and search local documents using ChromaDB")
    parser.add_argument("--index", "-i", type=str, help="Directory path to index documents")
    parser.add_argument("--search", "-s", type=str, help="Search query string")
    parser.add_argument("--add", "-a", type=str, help="Add document content as string to index")
    parser.add_argument("--test", "-t", action="store_true", help="Run a simple test of add and search functionality")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Path to ChromaDB database")
    parser.add_argument("--collection", type=str, default="documents", help="ChromaDB collection name")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results to return")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Minimum similarity threshold")
    parser.add_argument("--index-path", type=str, default="./documents", help="Directory to save added documents")
    parser.add_argument("--server", action="store_true", help="Start HTTP server mode")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument("--host", type=str, default="localhost", help="HTTP server host")
    
    args = parser.parse_args()
    
    if not args.server and not args.index and not args.search and not args.add and not args.test:
        parser.error("Either --server, --index, --search, --add, or --test must be specified")
    
    doc_indexer = DocumentIndexer(args.db_path, args.collection)
    
    if args.server:
        config = {
            'db_path': args.db_path,
            'collection': args.collection,
            'max_results': args.max_results,
            'similarity_threshold': args.similarity_threshold,
            'index_path': args.index_path
        }
        
        handler = create_http_handler(doc_indexer, config)
        server = HTTPServer((args.host, args.port), handler)
        
        print(f"Starting HTTP server on http://{args.host}:{args.port}")
        print("Available endpoints:")
        print(f"  GET  /health - Health check")
        print(f"  GET  /search?q=<query>&max_results=<n>&threshold=<t> - Search documents")
        print(f"  POST /index - Index directory (JSON: {{'directory': 'path'}})")
        print(f"  POST /add - Add document (JSON: {{'content': 'text', 'index_path': 'path'}})")
        print(f"  GET  /test?index_path=<path> - Run test")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
        return
    
    if args.index:
        if not os.path.exists(args.index):
            print(f"Error: Directory '{args.index}' does not exist")
            sys.exit(1)
        print(f"Indexing documents in: {args.index}")
        doc_indexer.index_directory(args.index)
        print("Indexing completed!")
    
    if args.test:
        print("Running simple test...")
        run_simple_test(doc_indexer, args.index_path)
    
    if args.add:
        print(f"Adding document content to index...")
        doc_indexer.add_document(args.add, args.index_path)
        print("Document added and indexed successfully!")
    
    if args.search:
        print(f"Searching for: {args.search}")
        results = doc_indexer.search(args.search, args.max_results, args.similarity_threshold)
        if results:
            merged_content = doc_indexer.merge_results(results)
            print("\n" + "="*80)
            print("MERGED SEARCH RESULTS")
            print("="*80)
            print(merged_content)
        else:
            print("No matching documents found.")

class DocumentIndexer:
    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Connected to ChromaDB at {self.db_path}")
            
            # Preload embedding model by running a dummy query
            print("Preloading embedding model...")
            try:
                self.collection.query(
                    query_texts=["dummy query to initialize embeddings"],
                    n_results=1,
                    include=["documents"]
                )
                print("Embedding model preloaded successfully")
            except Exception as e:
                print(f"Warning: Could not preload embedding model: {e}")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            sys.exit(1)
    
    def index_directory(self, directory_path: str):
        """Recursively index all documents in a directory"""
        documents = []
        metadatas = []
        ids = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if any(file.lower().endswith(ext) for ext in self._get_supported_extensions()):
                    try:
                        content = self._load_document(file_path)
                        if content and len(content.strip()) > 0:
                            documents.append(content)
                            metadatas.append({
                                "file_path": file_path,
                                "filename": file,
                                "size": os.path.getsize(file_path)
                            })
                            ids.append(str(uuid.uuid4()))
                            print(f"Indexed: {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully indexed {len(documents)} documents")
        else:
            print("No documents found to index")
    
    def _load_document(self, file_path: str) -> str:
        """Load and extract text content from a document"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    return f.read()
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                return ""
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def _get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        return ['.txt', '.md', '.yak']
    
    def search(self, query: str, max_results: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for documents similar to the query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            filtered_results = []
            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    similarity = 1 - distance
                    if similarity >= threshold:
                        filtered_results.append({
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity,
                            'distance': distance
                        })
            
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            if filtered_results:
                print(f"Found {len(filtered_results)} matching documents:")
                for i, result in enumerate(filtered_results, 1):
                    print(f"{i}. {result['metadata']['filename']} (similarity: {result['similarity']:.3f})")
            
            return filtered_results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def merge_results(self, results: List[Dict]) -> str:
        """Merge search results into a single content string"""
        if not results:
            return "No results to merge."
        
        merged_content = []
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            document = result['document']
            similarity = result['similarity']
            
            separator = f"\n{'='*60}\n"
            header = f"DOCUMENT {i}: {metadata['filename']}"
            file_info = f"Path: {metadata['file_path']}"
            similarity_info = f"Similarity: {similarity:.3f}"
            content_header = f"Content:"
            
            merged_content.extend([
                separator,
                header,
                file_info,
                similarity_info,
                content_header,
                "-" * 40,
                document.strip(),
                ""
            ])
        
        return "\n".join(merged_content)
    
    def add_document(self, content: str, index_path: str):
        """Add a document from string content, save as .md file, and index it"""
        import datetime
        import hashlib
        
        # Create index directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Generate filename using timestamp and content hash
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        filename = f"doc_{timestamp}_{content_hash}.md"
        file_path = os.path.join(index_path, filename)
        
        # Save content to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Document saved to: {file_path}")
            
            # Index the document immediately
            doc_id = str(uuid.uuid4())
            self.collection.add(
                documents=[content],
                metadatas=[{
                    "file_path": file_path,
                    "filename": filename,
                    "size": len(content),
                    "added_timestamp": timestamp
                }],
                ids=[doc_id]
            )
            print(f"Document indexed with ID: {doc_id}")
            
        except Exception as e:
            print(f"Error saving or indexing document: {e}")
            raise

def run_simple_test(doc_indexer: DocumentIndexer, index_path: str):
    """Run a simple test to verify add and search functionality"""
    import time
    
    print("\n" + "="*60)
    print("RUNNING SIMPLE TEST")
    print("="*60)
    
    # Test documents to add
    test_docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning involves training algorithms to make predictions from data.",
        "ChromaDB is a vector database designed for storing and searching embeddings."
    ]
    
    print("\n1. Adding test documents...")
    for i, doc in enumerate(test_docs, 1):
        print(f"   Adding document {i}: {doc[:50]}...")
        doc_indexer.add_document(doc, index_path)
    
    # Wait a moment for indexing to complete
    time.sleep(1)
    
    print("\n2. Testing search functionality...")
    test_queries = [
        "programming language",
        "machine learning algorithms",
        "vector database"
    ]
    
    for query in test_queries:
        print(f"\n   Searching for: '{query}'")
        results = doc_indexer.search(query, max_results=2, threshold=0.3)
        if results:
            for result in results:
                print(f"   - Found: {result['document'][:60]}... (similarity: {result['similarity']:.3f})")
        else:
            print("   - No results found")
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()