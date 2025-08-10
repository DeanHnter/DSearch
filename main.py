#!/usr/bin/env python3
import argparse, os, sys, uuid, json, re, datetime, hashlib, time
from pathlib import Path
from typing import List, Dict, Set
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import chromadb
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from difflib import SequenceMatcher

# Initialize NLTK components
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception:
    pass

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = set()

def get_synonyms(word: str) -> Set[str]:
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
    except Exception:
        pass
    return synonyms

def fuzzy_match_score(word1: str, word2: str) -> float:
    """Calculate fuzzy matching score between two words"""
    return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()

def expand_query_terms(text: str) -> List[str]:
    """Expand query with synonyms and variations"""
    if not text:
        return []
    
    # Tokenize and get base terms
    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
    
    expanded_terms = []
    
    for token in tokens:
        if len(token) > 2 and token not in stop_words:
            # Add original term
            expanded_terms.append(token)
            
            # Add stemmed version
            stemmed = stemmer.stem(token)
            if stemmed != token:
                expanded_terms.append(stemmed)
            
            # Add lemmatized version
            lemmatized = lemmatizer.lemmatize(token)
            if lemmatized != token:
                expanded_terms.append(lemmatized)
            
            # Add synonyms (limited to avoid noise)
            synonyms = get_synonyms(token)
            for syn in list(synonyms)[:3]:  # Limit to top 3 synonyms
                if len(syn) > 2:
                    expanded_terms.append(syn)
    
    return list(set(expanded_terms))  # Remove duplicates

def preprocess_text(text: str, preserve_original: bool = True) -> str:
    """Enhanced preprocessing with multiple strategies for better search accuracy including partial word matching"""
    if not text:
        return ""
    
    # Store original for case-insensitive matching
    original_text = text
    
    # Convert to lowercase for processing
    text_lower = text.lower()
    
    # Remove special characters but preserve word boundaries
    text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
    
    # Tokenize
    try:
        tokens = word_tokenize(text_clean)
    except Exception:
        tokens = text_clean.split()
    
    # Process tokens with multiple strategies
    processed_tokens = []
    
    # Strategy 1: Keep original tokens (for exact matching)
    if preserve_original:
        for token in word_tokenize(original_text.lower()):
            if token and len(token) > 1:  # Keep single chars for acronyms
                processed_tokens.append(token)
    
    # Strategy 2: Stemmed and lemmatized versions
    for token in tokens:
        if token and len(token) > 1:
            # Skip stop words only for very common ones, keep others for context
            if token not in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                # Add stemmed version
                stemmed = stemmer.stem(token)
                if stemmed and len(stemmed) > 1:
                    processed_tokens.append(stemmed)
                
                # Add lemmatized version
                lemmatized = lemmatizer.lemmatize(token)
                if lemmatized and len(lemmatized) > 1:
                    processed_tokens.append(lemmatized)
                
                # Strategy 2.5: Add word prefixes for partial matching
                if len(token) > 3:
                    for i in range(3, len(token)):
                        prefix = token[:i]
                        processed_tokens.append(f"prefix_{prefix}")
                    
                    # Add suffixes too
                    for i in range(3, len(token)):
                        suffix = token[i:]
                        if len(suffix) >= 3:
                            processed_tokens.append(f"suffix_{suffix}")
    
    # Strategy 3: N-grams for phrase matching
    if len(tokens) > 1:
        # Add bigrams
        for i in range(len(tokens) - 1):
            if tokens[i] not in stop_words and tokens[i+1] not in stop_words:
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                processed_tokens.append(bigram)
    
    # Strategy 4: Enhanced character n-grams for fuzzy and partial matching
    clean_text = re.sub(r'\s+', '', text_lower)
    if len(clean_text) > 2:
        # Add character trigrams
        for i in range(len(clean_text) - 2):
            trigram = clean_text[i:i+3]
            if trigram.isalpha():
                processed_tokens.append(f"#{trigram}#")
        
        # Add character 4-grams for better partial matching
        if len(clean_text) > 3:
            for i in range(len(clean_text) - 3):
                fourgram = clean_text[i:i+4]
                if fourgram.isalpha():
                    processed_tokens.append(f"#{fourgram}#")
    
    # Strategy 5: Word-level character n-grams for each word
    for token in tokens:
        if token and len(token) > 3 and token not in stop_words:
            # Add character trigrams within each word
            for i in range(len(token) - 2):
                char_trigram = token[i:i+3]
                if char_trigram.isalpha():
                    processed_tokens.append(f"@{char_trigram}@")
            
            # Add character 4-grams within each word
            if len(token) > 4:
                for i in range(len(token) - 3):
                    char_fourgram = token[i:i+4]
                    if char_fourgram.isalpha():
                        processed_tokens.append(f"@{char_fourgram}@")
    
    return ' '.join(processed_tokens)

def enhanced_query_preprocessing(query: str) -> str:
    """Enhanced query preprocessing with expansion and multiple matching strategies including partial word support"""
    if not query:
        return ""
    
    # Expand query terms
    expanded_terms = expand_query_terms(query)
    
    # Apply standard preprocessing to expanded terms
    all_processed = []
    
    # Add original query (case-insensitive)
    all_processed.append(query.lower())
    
    # Add expanded and processed terms
    for term in expanded_terms:
        processed = preprocess_text(term, preserve_original=True)
        if processed:
            all_processed.append(processed)
    
    # Add phrase-level processing
    phrase_processed = preprocess_text(query, preserve_original=True)
    if phrase_processed:
        all_processed.append(phrase_processed)
    
    # Special handling for partial words in query
    query_tokens = query.lower().split()
    for token in query_tokens:
        if len(token) > 2:
            # Add prefixes for partial matching
            for i in range(3, len(token) + 1):
                prefix = token[:i]
                all_processed.append(f"prefix_{prefix}")
            
            # Add character n-grams for the query token
            for i in range(len(token) - 2):
                char_trigram = token[i:i+3]
                if char_trigram.isalpha():
                    all_processed.append(f"@{char_trigram}@")
            
            if len(token) > 3:
                for i in range(len(token) - 3):
                    char_fourgram = token[i:i+4]
                    if char_fourgram.isalpha():
                        all_processed.append(f"@{char_fourgram}@")
    
    return ' '.join(all_processed)


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
        """Enhanced content cleaning and preprocessing"""
        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        cleaned = content.strip()
        
        # Create multiple representations for better matching
        representations = []
        
        # 1. Original cleaned content (for exact phrase matching)
        representations.append(cleaned)
        
        # 2. Preprocessed content (for semantic matching)
        preprocessed = preprocess_text(cleaned, preserve_original=True)
        representations.append(preprocessed)
        
        # 3. Case-normalized version
        case_normalized = cleaned.lower()
        representations.append(case_normalized)
        
        # 4. Word-level processed version
        words = re.findall(r'\b\w+\b', cleaned.lower())
        word_processed = ' '.join(words)
        representations.append(word_processed)
        
        # Combine all representations
        return ' '.join(representations)

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
        """Enhanced search with multiple query strategies and robust fallback"""
        if not query.strip():
            return []
        
        all_results = []
        
        # Strategy 1: Try exact query first (most conservative)
        try:
            res1 = self.collection.query(
                query_texts=[query.lower()], 
                n_results=min(n * 2, 50),
                include=['documents', 'metadatas', 'distances']
            )
            results1 = self._process_search_results(res1, query)
            all_results.extend(results1)
            print(f"Strategy 1 (exact): {len(results1)} results")
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Try individual words (robust fallback)
        query_words = [word.strip() for word in query.lower().split() if len(word.strip()) > 2]
        for word in query_words:
            try:
                res2 = self.collection.query(
                    query_texts=[word], 
                    n_results=min(n * 2, 50),
                    include=['documents', 'metadatas', 'distances']
                )
                results2 = self._process_search_results(res2, query)
                all_results.extend(results2)
                print(f"Strategy 2 (word '{word}'): {len(results2)} results")
            except Exception as e:
                print(f"Strategy 2 for '{word}' failed: {e}")
        
        # Strategy 3: Try enhanced preprocessing (but don't let it dominate)
        try:
            enhanced_query = enhanced_query_preprocessing(query)
            res3 = self.collection.query(
                query_texts=[enhanced_query], 
                n_results=min(n * 2, 50),
                include=['documents', 'metadatas', 'distances']
            )
            results3 = self._process_search_results(res3, query)
            all_results.extend(results3)
            print(f"Strategy 3 (enhanced): {len(results3)} results")
        except Exception as e:
            print(f"Strategy 3 failed: {e}")
        
        # Strategy 4: Try partial word matching for incomplete words
        partial_words = [word for word in query_words if len(word) >= 3]
        for partial_word in partial_words:
            # Try with wildcard-like matching using character n-grams
            partial_query_parts = []
            
            # Add the partial word itself
            partial_query_parts.append(partial_word)
            
            # Add character trigrams from the partial word
            for i in range(len(partial_word) - 2):
                trigram = partial_word[i:i+3]
                if trigram.isalpha():
                    partial_query_parts.append(f"@{trigram}@")
            
            # Add prefix matching
            if len(partial_word) >= 3:
                partial_query_parts.append(f"prefix_{partial_word}")
            
            partial_search_query = ' '.join(partial_query_parts)
            
            try:
                res4 = self.collection.query(
                    query_texts=[partial_search_query], 
                    n_results=min(n, 30),
                    include=['documents', 'metadatas', 'distances']
                )
                results4 = self._process_search_results(res4, query)
                all_results.extend(results4)
                print(f"Strategy 4 (partial '{partial_word}'): {len(results4)} results")
            except Exception as e:
                print(f"Strategy 4 for '{partial_word}' failed: {e}")
        
        # Strategy 5: Fallback to simple preprocessing if nothing else works
        if not all_results:
            try:
                simple_processed = preprocess_text(query, preserve_original=True)
                res5 = self.collection.query(
                    query_texts=[simple_processed], 
                    n_results=min(n * 2, 50),
                    include=['documents', 'metadatas', 'distances']
                )
                results5 = self._process_search_results(res5, query)
                all_results.extend(results5)
                print(f"Strategy 5 (simple preprocessing): {len(results5)} results")
            except Exception as e:
                print(f"Strategy 5 failed: {e}")
        
        # Remove duplicates and rerank
        unique_results = self._deduplicate_and_rerank(all_results, query, thr)
        
        # Apply threshold and limit
        filtered_results = [r for r in unique_results if r['similarity'] >= thr]
        final_results = filtered_results[:n]
        
        print(f"Final results: {len(final_results)} (after dedup and threshold {thr})")
        return final_results
    
    def _process_search_results(self, res, original_query: str) -> List[Dict]:
        """Process raw search results into standardized format"""
        results = []
        if not res['distances'] or not res['distances'][0]:
            return results
            
        for i, dist in enumerate(res['distances'][0]):
            sim = 1 - dist
            doc_content = res['documents'][0][i]
            metadata = res['metadatas'][0][i]
            file_path = metadata.get('file_path')
            
            # Get original content for display
            display_content = self._get_display_content(doc_content, file_path)
            
            # Apply additional scoring based on query matching
            enhanced_sim = self._calculate_enhanced_similarity(display_content, original_query, sim)
            
            results.append({
                "document": display_content,
                "metadata": metadata,
                "similarity": enhanced_sim,
                "original_similarity": sim
            })
        
        return results
    
    def _get_display_content(self, doc_content: str, file_path: str) -> str:
        """Get clean content for display"""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    original_content = fp.read()
                # Clean but don't preprocess for display
                original_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', original_content)
                original_content = original_content.replace('\r\n', '\n').replace('\r', '\n').strip()
                content = original_content
            except Exception:
                content = doc_content
        else:
            content = doc_content
        
        # Truncate if too long
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        
        return content
    
    def _calculate_enhanced_similarity(self, content: str, query: str, base_sim: float) -> float:
        """Calculate enhanced similarity score with additional factors, prioritizing individual word matches"""
        enhanced_score = base_sim
        
        # Get query words
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Strong boost for exact phrase matches (case-insensitive)
        if query.lower() in content.lower():
            enhanced_score += 0.2
        
        # Strong boost for individual word matches (this is key for the user's issue)
        if query_words:
            word_overlap = len(query_words.intersection(content_words)) / len(query_words)
            enhanced_score += word_overlap * 0.3  # Increased from 0.05 to 0.3
            
            # Extra boost if ALL query words are found
            if word_overlap == 1.0:
                enhanced_score += 0.1
        
        # Moderate boost for partial word matches
        for query_word in query_words:
            for content_word in content_words:
                if len(query_word) >= 3 and len(content_word) >= 3:
                    # Check if query word is a prefix of content word
                    if content_word.startswith(query_word):
                        enhanced_score += 0.15
                    # Check if query word is contained in content word
                    elif query_word in content_word:
                        enhanced_score += 0.1
                    # Fuzzy matching for typos
                    elif fuzzy_match_score(query_word, content_word) > 0.8:
                        enhanced_score += 0.05
        
        # Boost for word order preservation
        if len(query_words) > 1:
            query_word_list = re.findall(r'\b\w+\b', query.lower())
            content_lower = content.lower()
            
            # Check if words appear in the same order
            last_pos = -1
            order_preserved = True
            for word in query_word_list:
                pos = content_lower.find(word, last_pos + 1)
                if pos == -1:
                    order_preserved = False
                    break
                last_pos = pos
            
            if order_preserved:
                enhanced_score += 0.1
        
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def _deduplicate_and_rerank(self, results: List[Dict], query: str, threshold: float) -> List[Dict]:
        """Remove duplicates and rerank results"""
        # Group by file path to remove duplicates
        file_results = {}
        for result in results:
            file_path = result['metadata'].get('file_path', '')
            if file_path not in file_results or result['similarity'] > file_results[file_path]['similarity']:
                file_results[file_path] = result
        
        # Convert back to list and sort by similarity
        unique_results = list(file_results.values())
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return unique_results

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
        n      = int(q.get('max_results', [self.config.get('max_results', 5)])[0])
        thr    = float(q.get('threshold',  [self.config.get('similarity_threshold', 0.7)])[0])
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
        else:                                   # JSON branch
            return self.send_json_response({"results": res})

def handler_factory(indexer: DocumentIndexer, cfg: dict):
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

if __name__ == '__main__':
    main()