#!/usr/bin/env python3
import argparse, os, sys, uuid, json, re, datetime, hashlib, time
from pathlib import Path
from typing import List, Dict, Set, Tuple
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
from collections import Counter, defaultdict
import math

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

# Add common programming stop words that shouldn't dominate matching
programming_stop_words = {
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
    'how', 'why', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that', 'these',
    'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
    'shall', 'should', 'ought'
}

def normalize_word(word: str) -> Set[str]:
    """Generate all normalized forms of a word (stem, lemma, plural/singular)"""
    if not word or len(word) < 2:
        return {word.lower()}
    
    word_lower = word.lower()
    normalized_forms = {word_lower}
    
    # Add stemmed version
    try:
        stemmed = stemmer.stem(word_lower)
        if stemmed and len(stemmed) >= 2:
            normalized_forms.add(stemmed)
    except:
        pass
    
    # Add lemmatized versions (try different POS tags)
    try:
        # Try as noun
        lemma_noun = lemmatizer.lemmatize(word_lower, pos='n')
        if lemma_noun != word_lower:
            normalized_forms.add(lemma_noun)
        
        # Try as verb
        lemma_verb = lemmatizer.lemmatize(word_lower, pos='v')
        if lemma_verb != word_lower:
            normalized_forms.add(lemma_verb)
        
        # Try as adjective
        lemma_adj = lemmatizer.lemmatize(word_lower, pos='a')
        if lemma_adj != word_lower:
            normalized_forms.add(lemma_adj)
    except:
        pass
    
    # Handle common programming plurals manually
    programming_plurals = {
        'class': {'class', 'classes'},
        'classes': {'class', 'classes'},
        'function': {'function', 'functions'},
        'functions': {'function', 'functions'},
        'method': {'method', 'methods'},
        'methods': {'method', 'methods'},
        'variable': {'variable', 'variables'},
        'variables': {'variable', 'variables'},
        'array': {'array', 'arrays'},
        'arrays': {'array', 'arrays'},
        'object': {'object', 'objects'},
        'objects': {'object', 'objects'},
        'type': {'type', 'types'},
        'types': {'type', 'types'},
        'struct': {'struct', 'structs'},
        'structs': {'struct', 'structs'},
        'interface': {'interface', 'interfaces'},
        'interfaces': {'interface', 'interfaces'},
        'callback': {'callback', 'callbacks'},
        'callbacks': {'callback', 'callbacks'},
        'event': {'event', 'events'},
        'events': {'event', 'events'},
        'property': {'property', 'properties'},
        'properties': {'property', 'properties'},
        'parameter': {'parameter', 'parameters'},
        'parameters': {'parameter', 'parameters'},
        'argument': {'argument', 'arguments'},
        'arguments': {'argument', 'arguments'},
    }
    
    if word_lower in programming_plurals:
        normalized_forms.update(programming_plurals[word_lower])
    
    # Add partial forms for longer words
    if len(word_lower) > 4:
        # Add prefixes
        for i in range(3, len(word_lower)):
            prefix = word_lower[:i]
            normalized_forms.add(f"prefix_{prefix}")
        
        # Add suffixes
        for i in range(1, len(word_lower) - 2):
            suffix = word_lower[i:]
            if len(suffix) >= 3:
                normalized_forms.add(f"suffix_{suffix}")
    
    return normalized_forms

def get_synonyms(word: str) -> Set[str]:
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word.lower() and len(synonym) > 2:
                    synonyms.add(synonym)
    except Exception:
        pass
    return synonyms

def extract_key_concepts(text: str) -> Set[str]:
    """Extract key programming and domain concepts from text with morphological awareness"""
    # Programming concepts and keywords (include both singular and plural)
    programming_concepts = {
        'class', 'classes', 'object', 'objects', 'inheritance', 'polymorphism',
        'function', 'functions', 'method', 'methods', 'variable', 'variables',
        'array', 'arrays', 'list', 'lists', 'dictionary', 'dictionaries',
        'loop', 'loops', 'condition', 'conditions', 'if', 'else', 'while', 'for',
        'struct', 'structs', 'interface', 'interfaces', 'module', 'modules',
        'import', 'export', 'package', 'packages', 'library', 'libraries',
        'type', 'types', 'datatype', 'datatypes', 'string', 'integer', 'boolean',
        'callback', 'callbacks', 'event', 'events', 'handler', 'handlers',
        'pattern', 'patterns', 'algorithm', 'algorithms', 'data', 'structure',
        'pointer', 'pointers', 'reference', 'references', 'memory', 'allocation',
        'constructor', 'destructor', 'getter', 'setter', 'property', 'properties',
        'static', 'dynamic', 'public', 'private', 'protected', 'abstract',
        'virtual', 'override', 'implement', 'implements', 'extend', 'extends',
        'generic', 'generics', 'template', 'templates', 'namespace', 'namespaces',
        'parameter', 'parameters', 'argument', 'arguments', 'return', 'returns',
        'create', 'creating', 'creation', 'define', 'defining', 'definition',
        'declare', 'declaring', 'declaration', 'initialize', 'initializing', 'initialization'
    }
    
    # Extract words from text with morphological normalization
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Find concepts using normalized forms
    found_concepts = set()
    for word in words:
        normalized_forms = normalize_word(word)
        if normalized_forms.intersection(programming_concepts):
            found_concepts.add(word)
            # Also add the normalized forms that match concepts
            found_concepts.update(normalized_forms.intersection(programming_concepts))
    
    # Look for compound concepts and patterns
    text_lower = text.lower()
    concept_patterns = [
        r'\bclass\s+\w+', r'\bstruct\s+\w+', r'\bfunction\s+\w+',
        r'\bdef\s+\w+', r'\binterface\s+\w+', r'\benum\s+\w+',
        r'\btype\s+\w+', r'\bvar\s+\w+', r'\blet\s+\w+', r'\bconst\s+\w+',
        r'\bcreating\s+\w+', r'\bdefining\s+\w+', r'\bdeclaring\s+\w+'
    ]
    
    for pattern in concept_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Extract both words from the pattern
            words_in_match = match.split()
            found_concepts.update(words_in_match)
    
    return found_concepts

def calculate_morphological_similarity(query_words: List[str], content_words: List[str]) -> float:
    """Calculate similarity considering morphological variations"""
    if not query_words:
        return 0.0
    
    # Normalize all words
    query_normalized = set()
    for word in query_words:
        if word not in programming_stop_words:
            query_normalized.update(normalize_word(word))
    
    content_normalized = set()
    for word in content_words:
        if word not in programming_stop_words:
            content_normalized.update(normalize_word(word))
    
    if not query_normalized:
        return 0.0
    
    # Calculate overlap
    intersection = len(query_normalized.intersection(content_normalized))
    union = len(query_normalized.union(content_normalized))
    
    # Jaccard similarity
    jaccard = intersection / union if union > 0 else 0.0
    
    # Coverage (how many query concepts are covered)
    coverage = intersection / len(query_normalized)
    
    # Combine with emphasis on coverage
    return (jaccard * 0.4) + (coverage * 0.6)

def calculate_concept_overlap(query_concepts: Set[str], content_concepts: Set[str]) -> float:
    """Calculate overlap between concept sets with morphological awareness"""
    if not query_concepts:
        return 0.0
    
    # Normalize concepts
    query_normalized = set()
    for concept in query_concepts:
        query_normalized.update(normalize_word(concept))
    
    content_normalized = set()
    for concept in content_concepts:
        content_normalized.update(normalize_word(concept))
    
    intersection = len(query_normalized.intersection(content_normalized))
    
    if intersection == 0:
        return 0.0
    
    # Calculate both Jaccard and coverage
    union = len(query_normalized.union(content_normalized))
    jaccard = intersection / union if union > 0 else 0.0
    coverage = intersection / len(query_normalized)
    
    # Emphasize coverage for better recall
    return (jaccard * 0.3) + (coverage * 0.7)

def extract_topic_keywords(text: str) -> Dict[str, float]:
    """Extract topic-specific keywords with morphological normalization"""
    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Remove stop words and very common programming words
    filtered_words = [w for w in words if w not in programming_stop_words and len(w) > 2]
    
    # Normalize words and count frequencies
    normalized_word_freq = Counter()
    for word in filtered_words:
        normalized_forms = normalize_word(word)
        for form in normalized_forms:
            normalized_word_freq[form] += 1
    
    total_words = sum(normalized_word_freq.values())
    
    # Calculate TF scores with concept boosting
    topic_keywords = {}
    programming_concepts = {
        'class', 'classes', 'object', 'objects', 'inheritance', 'polymorphism',
        'function', 'functions', 'method', 'methods', 'variable', 'variables',
        'array', 'arrays', 'struct', 'structs', 'interface', 'interfaces',
        'type', 'types', 'callback', 'callbacks', 'event', 'events',
        'create', 'creating', 'define', 'defining', 'declare', 'declaring'
    }
    
    for word, freq in normalized_word_freq.items():
        if word.startswith('prefix_') or word.startswith('suffix_'):
            continue  # Skip partial forms for keyword extraction
        
        # Basic TF score
        tf = freq / total_words if total_words > 0 else 0
        
        # Boost important programming concepts
        concept_boost = 3.0 if word in programming_concepts else 1.0
        
        # Penalize very common words
        commonality_penalty = 1.0
        if freq > total_words * 0.1:  # If word appears in >10% of content
            commonality_penalty = 0.5
        
        score = tf * concept_boost * commonality_penalty
        topic_keywords[word] = score
    
    return topic_keywords

def calculate_topic_relevance(query: str, content: str) -> float:
    """Calculate how relevant the content is to the query topic with morphological awareness"""
    query_concepts = extract_key_concepts(query)
    content_concepts = extract_key_concepts(content)
    
    # If no concepts found in query, fall back to keyword matching
    if not query_concepts:
        query_words = [w for w in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query.lower()) 
                      if w not in programming_stop_words]
        query_concepts = set(query_words)
    
    # Calculate concept overlap with morphological awareness
    concept_score = calculate_concept_overlap(query_concepts, content_concepts)
    
    # Extract topic keywords with morphological normalization
    query_keywords = extract_topic_keywords(query)
    content_keywords = extract_topic_keywords(content)
    
    # Calculate keyword relevance with morphological matching
    keyword_score = 0.0
    if query_keywords and content_keywords:
        total_relevance = 0.0
        matched_keywords = 0
        
        for query_word, query_weight in query_keywords.items():
            # Check for exact match
            if query_word in content_keywords:
                content_weight = content_keywords[query_word]
                relevance = min(query_weight, content_weight) * 3.0  # Boost exact matches
                total_relevance += relevance
                matched_keywords += 1
            else:
                # Check for morphological matches
                query_normalized = normalize_word(query_word)
                for content_word, content_weight in content_keywords.items():
                    content_normalized = normalize_word(content_word)
                    if query_normalized.intersection(content_normalized):
                        relevance = min(query_weight, content_weight) * 2.0  # Boost morphological matches
                        total_relevance += relevance
                        matched_keywords += 1
                        break  # Only count first match to avoid double counting
        
        # Normalize by query keyword count
        keyword_score = total_relevance / len(query_keywords) if query_keywords else 0.0
        
        # Boost if many keywords matched
        coverage_bonus = (matched_keywords / len(query_keywords)) * 0.2 if query_keywords else 0.0
        keyword_score += coverage_bonus
    
    # Combine scores with emphasis on concept matching
    final_score = (concept_score * 0.8) + (keyword_score * 0.2)
    
    return min(final_score, 1.0)

def calculate_semantic_distance(query: str, content: str) -> float:
    """Calculate semantic distance between query and content with morphological awareness"""
    # Extract main topics/themes
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Define topic clusters with morphological variations
    topic_clusters = {
        'classes_oop': {'class', 'classes', 'object', 'objects', 'inheritance', 'polymorphism', 
                       'constructor', 'destructor', 'method', 'methods', 'property', 'properties',
                       'static', 'public', 'private', 'protected', 'abstract', 'virtual',
                       'create', 'creating', 'creation', 'define', 'defining', 'definition'},
        'functions': {'function', 'functions', 'callback', 'callbacks', 'procedure', 'procedures',
                     'lambda', 'closure', 'higher-order', 'functional', 'parameter', 'parameters',
                     'argument', 'arguments', 'return', 'returns'},
        'data_structures': {'array', 'arrays', 'list', 'lists', 'dictionary', 'dictionaries',
                           'map', 'maps', 'set', 'sets', 'tree', 'trees', 'graph', 'graphs',
                           'tuple', 'tuples', 'collection', 'collections'},
        'types': {'type', 'types', 'datatype', 'datatypes', 'struct', 'structs', 'interface',
                 'interfaces', 'enum', 'enums', 'generic', 'generics', 'template', 'templates'},
        'control_flow': {'loop', 'loops', 'condition', 'conditions', 'if', 'else', 'while', 'for',
                        'switch', 'case', 'break', 'continue', 'return', 'yield'},
        'memory': {'pointer', 'pointers', 'reference', 'references', 'memory', 'allocation',
                  'garbage', 'collection', 'stack', 'heap', 'buffer', 'address'}
    }
    
    # Find which topics are present in query and content with morphological matching
    query_topics = set()
    content_topics = set()
    
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    content_words = set(re.findall(r'\b\w+\b', content_lower))
    
    for topic, keywords in topic_clusters.items():
        # Check query topics with morphological matching
        query_matched = False
        for query_word in query_words:
            query_normalized = normalize_word(query_word)
            for keyword in keywords:
                keyword_normalized = normalize_word(keyword)
                if query_normalized.intersection(keyword_normalized):
                    query_matched = True
                    break
            if query_matched:
                break
        if query_matched:
            query_topics.add(topic)
        
        # Check content topics with morphological matching
        content_matched = False
        for content_word in content_words:
            content_normalized = normalize_word(content_word)
            for keyword in keywords:
                keyword_normalized = normalize_word(keyword)
                if content_normalized.intersection(keyword_normalized):
                    content_matched = True
                    break
            if content_matched:
                break
        if content_matched:
            content_topics.add(topic)
    
    # Calculate topic overlap
    if not query_topics:
        return 0.5  # Neutral if no clear topics in query
    
    topic_overlap = len(query_topics.intersection(content_topics)) / len(query_topics)
    
    # Penalize topic mismatch heavily
    if topic_overlap == 0.0:
        return 0.1  # Very low score for completely different topics
    
    return topic_overlap

class MorphologicalSimilarity:
    """Advanced similarity calculator with morphological awareness"""
    
    def __init__(self):
        self.weights = {
            'embedding': 0.15,      # Reduced weight for embedding similarity
            'topic_relevance': 0.40, # Increased weight for topic relevance
            'semantic_distance': 0.25, # Semantic topic distance
            'morphological_match': 0.15, # Morphological word matching
            'exact_matches': 0.05   # Reduced weight for exact matches
        }
    
    def calculate_similarity(self, query: str, content: str, embedding_similarity: float) -> Dict[str, float]:
        """Calculate morphologically-aware similarity score"""
        query_lower = query.lower().strip()
        content_lower = content.lower().strip()
        
        if not query_lower or not content_lower:
            return {'final_score': 0.0, 'embedding': embedding_similarity, 'topic_relevance': 0.0,
                   'semantic_distance': 0.0, 'morphological_match': 0.0, 'exact_matches': 0.0}
        
        scores = {}
        
        # 1. Embedding similarity (reduced weight)
        scores['embedding'] = embedding_similarity
        
        # 2. Topic relevance (high weight, morphologically aware)
        scores['topic_relevance'] = calculate_topic_relevance(query, content)
        
        # 3. Semantic distance (topic clustering with morphological matching)
        scores['semantic_distance'] = calculate_semantic_distance(query, content)
        
        # 4. Morphological matching
        scores['morphological_match'] = self._calculate_morphological_match(query_lower, content_lower)
        
        # 5. Exact matches (reduced weight)
        scores['exact_matches'] = self._calculate_exact_matches(query_lower, content_lower)
        
        # Calculate weighted final score
        final_score = sum(scores[component] * self.weights[component] 
                         for component in self.weights.keys())
        
        # Apply topic mismatch penalty
        if scores['topic_relevance'] < 0.1 and scores['semantic_distance'] < 0.2:
            final_score *= 0.2  # Heavy penalty for topic mismatch
        
        # Boost if morphological matching is strong
        if scores['morphological_match'] > 0.7:
            final_score *= 1.2  # Boost for strong morphological matches
        
        scores['final_score'] = min(final_score, 1.0)
        
        return scores
    
    def _calculate_morphological_match(self, query: str, content: str) -> float:
        """Calculate morphological word matching"""
        query_words = [w for w in re.findall(r'\b\w+\b', query) if w not in programming_stop_words]
        content_words = [w for w in re.findall(r'\b\w+\b', content) if w not in programming_stop_words]
        
        return calculate_morphological_similarity(query_words, content_words)
    
    def _calculate_exact_matches(self, query: str, content: str) -> float:
        """Calculate exact phrase and word matches"""
        score = 0.0
        
        # Exact phrase match
        if query in content:
            score += 0.6
        
        # Word overlap (excluding stop words)
        query_words = set(re.findall(r'\b\w+\b', query)) - programming_stop_words
        content_words = set(re.findall(r'\b\w+\b', content)) - programming_stop_words
        
        if query_words:
            word_overlap = len(query_words.intersection(content_words)) / len(query_words)
            score += word_overlap * 0.4
        
        return min(score, 1.0)

def preprocess_text(text: str, preserve_original: bool = True) -> str:
    """Enhanced preprocessing with morphological awareness"""
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
    
    # Process tokens with morphological awareness
    processed_tokens = []
    
    # Keep original tokens for exact matching
    if preserve_original:
        for token in word_tokenize(original_text.lower()):
            if token and len(token) > 1:
                processed_tokens.append(token)
    
    # Add morphologically normalized versions
    for token in tokens:
        if token and len(token) > 2 and token not in programming_stop_words:
            normalized_forms = normalize_word(token)
            processed_tokens.extend(normalized_forms)
    
    return ' '.join(processed_tokens)

def enhanced_query_preprocessing(query: str) -> str:
    """Query preprocessing with morphological expansion"""
    if not query:
        return ""
    
    # Add original query
    all_processed = [query.lower()]
    
    # Add morphologically processed version
    processed = preprocess_text(query, preserve_original=True)
    if processed:
        all_processed.append(processed)
    
    # Add concept-focused expansion with morphological variants
    concepts = extract_key_concepts(query)
    for concept in concepts:
        all_processed.append(concept)
        # Add morphological variants
        normalized_forms = normalize_word(concept)
        all_processed.extend(normalized_forms)
        
        # Add synonyms for key concepts
        synonyms = get_synonyms(concept)
        for syn in list(synonyms)[:2]:  # Limit synonyms
            if len(syn) > 2:
                all_processed.append(syn)
    
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
        
        # Initialize morphological similarity calculator
        self.similarity_calculator = MorphologicalSimilarity()
        
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
        """Morphologically-aware content cleaning"""
        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        cleaned = content.strip()
        
        # Create morphologically-aware representations
        representations = []
        
        # 1. Original cleaned content (for exact matching)
        representations.append(cleaned)
        
        # 2. Morphologically processed content
        preprocessed = preprocess_text(cleaned, preserve_original=True)
        representations.append(preprocessed)
        
        # 3. Extract and emphasize key concepts with morphological variants
        concepts = extract_key_concepts(cleaned)
        if concepts:
            concept_variants = set()
            for concept in concepts:
                concept_variants.add(concept)
                concept_variants.update(normalize_word(concept))
            concept_text = ' '.join(concept_variants)
            representations.append(concept_text)
        
        # Combine representations
        return ' '.join(representations)

    def _supported(self):                                                   # noqa
        return ('.txt', '.md', '.mdx', '.yaml', '.yml', '.json', '.py', '.js',
                '.html', '.css', '.yaka')

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
        """Morphologically-aware search"""
        if not query.strip():
            return []
        
        all_results = []
        
        # Strategy 1: Morphological query search
        try:
            morphological_query = enhanced_query_preprocessing(query)
            res1 = self.collection.query(
                query_texts=[morphological_query], 
                n_results=min(n * 4, 100),  # Get more results for reranking
                include=['documents', 'metadatas', 'distances']
            )
            results1 = self._process_search_results(res1, query)
            all_results.extend(results1)
            print(f"Morphological search: {len(results1)} results")
        except Exception as e:
            print(f"Morphological search failed: {e}")
        
        # Strategy 2: Concept-based search with morphological variants
        query_concepts = extract_key_concepts(query)
        if query_concepts:
            try:
                concept_variants = set()
                for concept in query_concepts:
                    concept_variants.add(concept)
                    concept_variants.update(normalize_word(concept))
                concept_query = ' '.join(concept_variants)
                
                res2 = self.collection.query(
                    query_texts=[concept_query], 
                    n_results=min(n * 3, 75),
                    include=['documents', 'metadatas', 'distances']
                )
                results2 = self._process_search_results(res2, query)
                all_results.extend(results2)
                print(f"Concept search: {len(results2)} results")
            except Exception as e:
                print(f"Concept search failed: {e}")
        
        # Strategy 3: Original query fallback
        try:
            res3 = self.collection.query(
                query_texts=[query.lower()], 
                n_results=min(n * 2, 50),
                include=['documents', 'metadatas', 'distances']
            )
            results3 = self._process_search_results(res3, query)
            all_results.extend(results3)
            print(f"Original search: {len(results3)} results")
        except Exception as e:
            print(f"Original search failed: {e}")
        
        # Morphological reranking and deduplication
        unique_results = self._deduplicate_and_rerank_morphological(all_results, query, thr)
        
        # Apply threshold and limit
        filtered_results = [r for r in unique_results if r['similarity'] >= thr]
        final_results = filtered_results[:n]
        
        print(f"Final results: {len(final_results)} (after morphological reranking and threshold {thr})")
        return final_results
    
    def _process_search_results(self, res, original_query: str) -> List[Dict]:
        """Process raw search results into standardized format"""
        results = []
        if not res['distances'] or not res['distances'][0]:
            return results
            
        for i, dist in enumerate(res['distances'][0]):
            embedding_sim = 1 - dist
            doc_content = res['documents'][0][i]
            metadata = res['metadatas'][0][i]
            file_path = metadata.get('file_path')
            
            # Get limited content with match positions
            display_data = self._get_display_content(doc_content, file_path, original_query)
            
            results.append({
                "document": display_data,
                "metadata": metadata,
                "embedding_similarity": embedding_sim,
                "original_query": original_query
            })
        
        return results
    
    def _deduplicate_and_rerank_morphological(self, results: List[Dict], query: str, threshold: float) -> List[Dict]:
        """Remove duplicates and rerank using morphological similarity"""
        # Group by file path to remove duplicates
        file_results = {}
        for result in results:
            file_path = result['metadata'].get('file_path', '')
            if file_path not in file_results or result['embedding_similarity'] > file_results[file_path]['embedding_similarity']:
                file_results[file_path] = result
        
        # Calculate morphological similarity for each unique result
        morphological_results = []
        for result in file_results.values():
            # Get original content for better similarity calculation
            original_content = self._get_original_content(result['metadata'].get('file_path', ''))
            if not original_content:
                # Fallback to first context if available
                contexts = result['document'].get('contexts', [])
                original_content = contexts[0]['context'] if contexts else ""
            
            # Calculate morphological similarity
            similarity_scores = self.similarity_calculator.calculate_similarity(
                query, original_content, result['embedding_similarity']
            )
            
            # Update result with morphological scores
            result['similarity'] = similarity_scores['final_score']
            result['similarity_breakdown'] = similarity_scores
            
            morphological_results.append(result)
        
        # Sort by morphological similarity
        morphological_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return morphological_results
    
    def _get_original_content(self, file_path: str) -> str:
        """Get original file content for similarity calculation"""
        if not file_path or not os.path.exists(file_path):
            return ""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                content = fp.read()
            # Clean but don't preprocess
            content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
            content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
            return content
        except Exception:
            return ""
    
    def _find_match_positions(self, content: str, query: str) -> List[Dict]:
        """Find line and column positions of query matches with morphological awareness"""
        matches = []
        lines = content.split('\n')
        
        # Focus on meaningful words, not stop words
        query_words = [w for w in re.findall(r'\b\w+\b', query.lower()) 
                      if w not in programming_stop_words and len(w) > 2]
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            # Check for exact phrase match first
            phrase_match = re.search(re.escape(query.lower()), line_lower)
            if phrase_match:
                matches.append({
                    'line': line_num,
                    'column': phrase_match.start() + 1,
                    'match_type': 'phrase',
                    'matched_text': line[phrase_match.start():phrase_match.end()]
                })
            else:
                # Check for morphological word matches
                line_words = re.findall(r'\b\w+\b', line_lower)
                for query_word in query_words:
                    query_normalized = normalize_word(query_word)
                    
                    for line_word in line_words:
                        line_normalized = normalize_word(line_word)
                        
                        # Check for morphological match
                        if query_normalized.intersection(line_normalized):
                            # Find the position of this word in the line
                            word_matches = list(re.finditer(r'\b' + re.escape(line_word) + r'\b', line_lower))
                            for match in word_matches:
                                matches.append({
                                    'line': line_num,
                                    'column': match.start() + 1,
                                    'match_type': 'morphological',
                                    'matched_text': line[match.start():match.end()]
                                })
        
        return matches[:10]  # Limit to first 10 matches
    
    def _get_context_around_matches(self, content: str, matches: List[Dict], context_words: int = 50) -> List[Dict]:
        """Get limited context around each match"""
        if not matches:
            return []
        
        lines = content.split('\n')
        contexts = []
        
        for match in matches:
            line_idx = match['line'] - 1
            if line_idx >= len(lines):
                continue
                
            # Get the line with the match
            match_line = lines[line_idx]
            
            # Extract words around the match position
            words_before = match_line[:match['column'] - 1].split()
            match_word = match['matched_text']
            words_after = match_line[match['column'] - 1 + len(match_word):].split()
            
            # Take context_words/2 before and after
            half_context = context_words // 2
            context_before = ' '.join(words_before[-half_context:]) if words_before else ''
            context_after = ' '.join(words_after[:half_context]) if words_after else ''
            
            # Build context string
            context_parts = []
            if context_before:
                context_parts.append(context_before)
            context_parts.append(f"**{match_word}**")  # Highlight the match
            if context_after:
                context_parts.append(context_after)
            
            context_text = ' '.join(context_parts).strip()
            
            contexts.append({
                'line': match['line'],
                'column': match['column'],
                'context': context_text,
                'match_type': match['match_type']
            })
        
        return contexts
    
    def _get_display_content(self, doc_content: str, file_path: str, query: str = None) -> Dict:
        """Get limited context with line/column info instead of full content"""
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
        
        # If we have a query, find matches and return limited context
        if query:
            matches = self._find_match_positions(content, query)
            contexts = self._get_context_around_matches(content, matches, context_words=100)
            
            return {
                'contexts': contexts,
                'total_matches': len(matches),
                'file_size': len(content)
            }
        else:
            # Fallback: return first 100 words if no query
            words = content.split()
            limited_content = ' '.join(words[:100])
            if len(words) > 100:
                limited_content += "... [truncated]"
            
            return {
                'contexts': [{'line': 1, 'column': 1, 'context': limited_content, 'match_type': 'fallback'}],
                'total_matches': 0,
                'file_size': len(content)
            }

    def merge_results(self, results: List[Dict]) -> str:
        if not results:
            return ""
        merged = []
        for i, r in enumerate(results, 1):
            merged.append("\n" + "="*60)
            merged.append(f"DOC {i}: {self._abs(r['metadata']['file_path'])} "
                        f"(sim {r['similarity']:.3f})")
            
            # Show morphological similarity breakdown
            if 'similarity_breakdown' in r:
                breakdown = r['similarity_breakdown']
                merged.append(f"  Topic Relevance: {breakdown.get('topic_relevance', 0):.3f}, "
                            f"Semantic Distance: {breakdown.get('semantic_distance', 0):.3f}, "
                            f"Morphological: {breakdown.get('morphological_match', 0):.3f}, "
                            f"Embedding: {breakdown.get('embedding', 0):.3f}")
            
            merged.append("-"*60)
            
            # Handle document structure with contexts
            doc_data = r['document']
            if isinstance(doc_data, dict) and 'contexts' in doc_data:
                merged.append(f"Total matches: {doc_data['total_matches']}")
                merged.append(f"File size: {doc_data['file_size']} characters")
                merged.append("")
                
                for j, context in enumerate(doc_data['contexts'], 1):
                    merged.append(f"Match {j} at line {context['line']}, column {context['column']} ({context['match_type']}):")
                    merged.append(f"  {context['context']}")
                    merged.append("")
            else:
                # Fallback for old format
                merged.append(str(doc_data))
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
        results = idx.search(args.query, args.max_results, args.similarity_threshold)
        print(f"Found {len(results)} documents")
        for i, r in enumerate(results, 1):
            print(f"{i}. {DocumentIndexer._abs(r['metadata']['file_path'])} "
                  f"(sim {r['similarity']:.3f})")
            if 'similarity_breakdown' in r:
                breakdown = r['similarity_breakdown']
                print(f"   Topic Relevance: {breakdown.get('topic_relevance', 0):.3f}, "
                      f"Semantic Distance: {breakdown.get('semantic_distance', 0):.3f}, "
                      f"Morphological: {breakdown.get('morphological_match', 0):.3f}")
        if results:
            print("\n" + "="*80)
            print("MERGED RESULT")
            print("="*80)
            print(idx.merge_results(results))
    elif args.cmd == "add":
        idx.add_document(args.content, args.title)
    elif args.cmd == "test":
        run_simple_test(idx)

if __name__ == "__main__":
    main()
