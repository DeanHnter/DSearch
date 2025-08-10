#!/usr/bin/env python3
import argparse, os, sys, uuid, json, re, datetime, hashlib, time
from pathlib import Path
from typing import List, Dict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import chromadb


# ────────────────────────────────────────────────────────────
# document indexer
# ────────────────────────────────────────────────────────────
class DocumentIndexer:
    @staticmethod
    def _abs(p: str) -> str:
        return os.path.abspath(p)

    def __init__(self, db_path: str, collection_name: str, index_path: str = None):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        
        # Store or retrieve the indexed directory path
        self.index_path = self._get_or_set_index_path(index_path)
        
        # warm-up embeddings
        try:
            self.collection.query(query_texts=["ping"], n_results=1)
        except Exception:
            pass

    # -------------------------------------------------- helpers
    def _get_or_set_index_path(self, index_path: str = None) -> str:
        """Get or set the indexed directory path in the collection metadata"""
        collection_metadata = self.collection.get(include=["metadatas"])
        
        # Check if we have any documents with index_path metadata
        if collection_metadata["metadatas"]:
            for metadata in collection_metadata["metadatas"]:
                if "index_path" in metadata:
                    stored_path = metadata["index_path"]
                    if index_path and os.path.abspath(index_path) != os.path.abspath(stored_path):
                        print(f"Warning: Provided index path '{index_path}' differs from stored path '{stored_path}'")
                        print(f"Using stored path: {stored_path}")
                    return stored_path
        
        # No stored path found, use provided or default
        new_path = os.path.abspath(index_path or "./documents")
        return new_path
    
    def _store_index_path_metadata(self, file_path: str, index_path: str):
        """Store the index path in document metadata"""
        return {
            "filename": os.path.basename(file_path),
            "file_path": file_path,
            "size": os.path.getsize(file_path),
            "index_path": index_path
        }
    
    def sync_with_filesystem(self):
        """Sync the database with the filesystem, checking for changes"""
        if not os.path.exists(self.index_path):
            print(f"Warning: Index path '{self.index_path}' does not exist")
            return
            
        print(f"Syncing database with filesystem at: {self.index_path}")
        
        # Get all documents from database
        all_docs = self.collection.get(include=["metadatas"])
        db_files = set()
        
        for metadata in all_docs["metadatas"]:
            if "file_path" in metadata:
                db_files.add(metadata["file_path"])
        
        # Get all files from filesystem
        fs_files = set()
        for root, _, files in os.walk(self.index_path):
            for f in files:
                if f.lower().endswith(self._supported()):
                    fs_files.add(os.path.abspath(os.path.join(root, f)))
        
        # Find files to remove (in DB but not on filesystem)
        to_remove = db_files - fs_files
        if to_remove:
            print(f"Removing {len(to_remove)} files that no longer exist")
            # Remove documents that no longer exist
            for file_path in to_remove:
                self._remove_document_by_path(file_path)
        
        # Find files to add (on filesystem but not in DB)
        to_add = fs_files - db_files
        if to_add:
            print(f"Adding {len(to_add)} new files")
            self._add_files_to_collection(to_add)
        
        # Check for modified files
        for file_path in fs_files & db_files:
            if self._file_modified(file_path):
                print(f"Re-indexing modified file: {file_path}")
                self._remove_document_by_path(file_path)
                self._add_files_to_collection([file_path])
    
    def _remove_document_by_path(self, file_path: str):
        """Remove a document from the collection by file path"""
        all_docs = self.collection.get(include=["metadatas"])
        ids_to_delete = []
        
        for i, metadata in enumerate(all_docs["metadatas"]):
            if metadata.get("file_path") == file_path:
                ids_to_delete.append(all_docs["ids"][i])
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
    
    def _file_modified(self, file_path: str) -> bool:
        """Check if a file has been modified since it was indexed"""
        all_docs = self.collection.get(include=["metadatas"])
        
        for metadata in all_docs["metadatas"]:
            if metadata.get("file_path") == file_path:
                stored_size = metadata.get("size", 0)
                current_size = os.path.getsize(file_path)
                return stored_size != current_size
        
        return True  # If not found in DB, consider it modified
    
    def _add_files_to_collection(self, file_paths: List[str]):
        """Add multiple files to the collection"""
        documents, metadatas, ids = [], [], []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    txt = self._clean_content(fp.read())
                if txt:
                    documents.append(txt)
                    metadatas.append(self._store_index_path_metadata(file_path, self.index_path))
                    ids.append(str(uuid.uuid4()))
            except Exception as e:
                print(f"Skip {file_path}: {e}")
        
        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def _clean_content(self, content: str) -> str:
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        return content.strip()

    def _supported(self):                                                   # noqa
        return ('.txt', '.md', '.yaml', '.yml', '.json', '.py', '.js',
                '.html', '.css', '.yak')

    # -------------------------------------------------- main jobs
    def index_directory(self, directory: str):
        directory = os.path.abspath(directory)
        self.index_path = directory  # Update the stored index path
        
        documents, metadatas, ids = [], [], []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(self._supported()):
                    try:
                        p = os.path.abspath(os.path.join(root, f))
                        with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                            txt = self._clean_content(fp.read())
                        if txt:
                            documents.append(txt)
                            metadatas.append(self._store_index_path_metadata(p, directory))
                            ids.append(str(uuid.uuid4()))
                        print("Indexed", p)
                    except Exception as e:
                        print("Skip", f, e)
        if documents:
            self.collection.add(documents=documents,
                                metadatas=metadatas,
                                ids=ids)
            print("Added", len(documents), "documents")

    def add_document(self, content: str, title: str | None = None):
        # Use the stored index path instead of a passed directory
        os.makedirs(self.index_path, exist_ok=True)

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (re.sub(r'[^A-Za-z0-9 _-]', '', title).replace(' ', '_')
                if title else f"doc_{hashlib.md5(content.encode()).hexdigest()[:8]}")
        filename = f"{name}_{ts}.md"

        path = os.path.abspath(os.path.join(self.index_path, filename))
        text = f"# {title}\n\n{content}" if title else content

        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(text)

        metadata = self._store_index_path_metadata(path, self.index_path)
        metadata["title"] = title or ""
        metadata["size"] = len(text)  # Override with actual content size

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )
        print("Saved & indexed", path)

    def search(self, query: str, n: int, thr: float) -> List[Dict]:
        res = self.collection.query(query_texts=[query], n_results=n,
                                    include=['documents', 'metadatas', 'distances'])
        out = []
        for i, dist in enumerate(res['distances'][0]):
            sim = 1 - dist
            if sim >= thr:
                txt = self._clean_content(res['documents'][0][i])
                if len(txt) > 2000:
                    txt = txt[:2000] + "... [truncated]"
                out.append({"document": txt,
                            "metadata": res['metadatas'][0][i],
                            "similarity": sim})
        out.sort(key=lambda x: x['similarity'], reverse=True)
        return out

    def merge_results(self, results: List[Dict]) -> str:
        if not results:
            return ""
        merged = []
        for i, r in enumerate(results, 1):
            merged.append("\n" + "="*60)
            merged.append(f"DOC {i}: {self._abs(r['metadata']['file_path'])} "
                        f"(sim {r['similarity']:.3f})")
            merged.append("-"*60)
            merged.append(r['document'])
        return "\n".join(merged)

# ────────────────────────────────────────────────────────────
# HTTP layer
# ────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def __init__(self, indexer: DocumentIndexer, config: dict, *a, **kw):
        self.indexer = indexer
        self.config  = config
        super().__init__(*a, **kw)

    # ---------------------------- helpers
    def send_json_response(self, data, status_code: int = 200):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    # ---------------------------- routes
    def do_GET(self):
        p = urlparse(self.path)
        q = parse_qs(p.query)
        if p.path == '/health':
            return self.send_json_response({"status": "healthy"})
        if p.path == '/search':
            return self._search(q)
        if p.path == '/test':
            run_simple_test(self.indexer)
            return self.send_json_response({"message": "test ok"})
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length) if length else b'{}'
        data   = json.loads(body or '{}')
        if self.path == '/index':
            d = data.get('directory')
            if not d:
                return self.send_json_response({"error": "directory missing"}, 400)
            if not os.path.isdir(d):
                return self.send_json_response({"error": "not a directory"}, 400)
            self.indexer.index_directory(d)
            return self.send_json_response({"message": "indexed"})
        if self.path == '/add':
            c = data.get('content')
            if not c:
                return self.send_json_response({"error": "content missing"}, 400)
            self.indexer.add_document(c, data.get('title'))
            return self.send_json_response({"message": "added"})
        self.send_error(404)

    # ---------------------------- actual work
    def _search(self, q):
        query  = q.get('q', [''])[0]
        if not query:
            return self.send_json_response({"error": "q missing"}, 400)
        n      = int(q.get('max_results', [self.config['max_results']])[0])
        thr    = float(q.get('threshold',  [self.config['similarity_threshold']])[0])
        fmt    = q.get('format', ['json'])[0]

        res    = self.indexer.search(query, n, thr)
        if fmt == 'text':                       # plain-text branch
            out = ["Found %d documents" % len(res)]
            for i, r in enumerate(res, 1):
                out.append(f"{i}. {DocumentIndexer._abs(r['metadata']['file_path'])} "
                        f"(sim {r['similarity']:.3f})")
            if res:
                out.append("\n" + "="*80)
                out.append("MERGED RESULT")
                out.append("="*80)
                out.append(self.indexer.merge_results(res))
            txt = "\n".join(out) + "\n"
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            return self.wfile.write(txt.encode('utf-8'))

        # default JSON
        payload = {"query": query,
                   "results": len(res),
                   "documents": res}
        if res:
            payload["merged_content"] = self.indexer.merge_results(res)
        self.send_json_response(payload)

# factory – needed because http.server instantiates handler by itself
def handler_factory(indexer, cfg):
    return lambda *a, **kw: Handler(indexer, cfg, *a, **kw)

# ────────────────────────────────────────────────────────────
# simple test
def run_simple_test(indexer: DocumentIndexer):
    print("Running test …")
    docs = ["Python is a language", "Vector DB", "Machine learning"]
    for d in docs:
        indexer.add_document(d, None)
    time.sleep(1)
    for q in ["language", "vector", "machine"]:
        print("search:", q)
        print(indexer.search(q, 2, 0.3))

# ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default="./db")
    p.add_argument("--collection", default="documents")
    p.add_argument("--index-path", default="./documents")
    p.add_argument("--max-results", type=int, default=5)
    p.add_argument("--similarity-threshold", type=float, default=0.7)
    sub = p.add_subparsers(dest="cmd", required=True)

    server_parser = sub.add_parser("server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8080)
    sub.add_parser("index").add_argument("directory")
    sub.add_parser("search").add_argument("query")
    addp = sub.add_parser("add");  addp.add_argument("content"); addp.add_argument("--title")
    sub.add_parser("test")

    args = p.parse_args()
    idx  = DocumentIndexer(args.db_path, args.collection, args.index_path)

    if args.cmd == "server":
        # Sync database with filesystem on startup
        print("Syncing database with filesystem...")
        idx.sync_with_filesystem()
        print("Sync complete.")
        
        cfg = vars(args)
        httpd = HTTPServer((args.host, args.port), handler_factory(idx, cfg))
        print(f"Serving on http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.cmd == "index":
        idx.index_directory(args.directory)
    elif args.cmd == "search":
        print(idx.search(args.query, args.max_results, args.similarity_threshold))
    elif args.cmd == "add":
        idx.add_document(args.content, args.title)
    elif args.cmd == "test":
        run_simple_test(idx)

if __name__ == "__main__":
    main()