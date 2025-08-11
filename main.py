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

def normalize_programming_operators(text: str) -> str:
    """Convert programming operators and symbols to natural language equivalents"""
    # Define operator mappings
    operator_mappings = {
        # Comparison operators
        '==': ' equals ',
        '!=': ' not equals ',
        '!==': ' not strictly equals ',
        '===': ' strictly equals ',
        '<=': ' less than or equal to ',
        '>=': ' greater than or equal to ',
        '<': ' less than ',
        '>': ' greater than ',
        
        # Logical operators
        '&&': ' and ',
        '||': ' or ',
        '!': ' not ',
        
        # Assignment operators
        '+=': ' plus equals ',
        '-=': ' minus equals ',
        '*=': ' multiply equals ',
        '/=': ' divide equals ',
        '%=': ' modulo equals ',
        '=': ' assign ',
        
        # Arithmetic operators
        '++': ' increment ',
        '--': ' decrement ',
        '+': ' plus ',
        '-': ' minus ',
        '*': ' multiply ',
        '/': ' divide ',
        '%': ' modulo ',
        '**': ' power ',
        
        # Bitwise operators
        '&': ' bitwise and ',
        '|': ' bitwise or ',
        '^': ' bitwise xor ',
        '~': ' bitwise not ',
        '<<': ' left shift ',
        '>>': ' right shift ',
        
        # Arrow functions and pointers
        '=>': ' arrow function ',
        '->': ' arrow pointer ',
        '::': ' scope resolution ',
        
        # Brackets and delimiters
        '{': ' open brace ',
        '}': ' close brace ',
        '[': ' open bracket ',
        ']': ' close bracket ',
        '(': ' open paren ',
        ')': ' close paren ',
        
        # Special symbols
        ';': ' semicolon ',
        ':': ' colon ',
        ',': ' comma ',
        '.': ' dot ',
        '?': ' question mark ',
        '#': ' hash ',
        '@': ' at symbol ',
        '$': ' dollar ',
        
        # Common programming patterns
        '...': ' spread operator ',
        '?.': ' optional chaining ',
        '??': ' nullish coalescing ',
    }
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_operators = sorted(operator_mappings.items(), key=lambda x: len(x[0]), reverse=True)
    
    result = text
    for operator, replacement in sorted_operators:
        # Use word boundaries where appropriate to avoid over-replacement
        if operator.isalnum():
            pattern = r'\b' + re.escape(operator) + r'\b'
        else:
            pattern = re.escape(operator)
        result = re.sub(pattern, replacement, result)
    
    return result

def separate_symbols_from_words(text: str) -> str:
    """Separate symbols from words to improve exact matching"""
    # First handle markdown-style formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'** \1 **', text)  # **word** -> ** word **
    text = re.sub(r'\*([^*]+)\*', r'* \1 *', text)        # *word* -> * word *
    text = re.sub(r'`([^`]+)`', r'` \1 `', text)          # `word` -> ` word `
    text = re.sub(r'_([^_]+)_', r'_ \1 _', text)          # _word_ -> _ word _
    
    # Handle common programming symbols attached to words
    # Function calls: word( -> word (
    text = re.sub(r'(\w)(\()', r'\1 \2', text)
    # Array access: word[ -> word [
    text = re.sub(r'(\w)(\[)', r'\1 \2', text)
    # Object access: word. -> word .
    text = re.sub(r'(\w)(\.)', r'\1 \2', text)
    # Closing brackets: )word -> ) word
    text = re.sub(r'(\))(\w)', r'\1 \2', text)
    text = re.sub(r'(\])(\w)', r'\1 \2', text)
    text = re.sub(r'(\})(\w)', r'\1 \2', text)
    
    # Handle operators attached to words
    # word= -> word =
    text = re.sub(r'(\w)(=)', r'\1 \2', text)
    # word+ -> word +
    text = re.sub(r'(\w)([+\-*/%])', r'\1 \2', text)
    # =word -> = word
    text = re.sub(r'([=+\-*/%])(\w)', r'\1 \2', text)
    
    # Handle comparison operators
    text = re.sub(r'(\w)([<>!]=?)', r'\1 \2', text)
    text = re.sub(r'([<>!]=?)(\w)', r'\1 \2', text)
    
    # Handle semicolons and commas
    text = re.sub(r'(\w)([;,])', r'\1 \2', text)
    text = re.sub(r'([;,])(\w)', r'\1 \2', text)
    
    # Handle colons (but be careful with :: and URLs)
    text = re.sub(r'(\w)(:)(?!:)', r'\1 \2', text)
    text = re.sub(r'(?<!:)(:)(\w)', r'\1 \2', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

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

# Lightweight fuzzy/substring helper to handle typos and partial tokens
def is_similar(a: str, b: str, threshold: float = 0.82) -> bool:
    """Return True if strings a and b are close enough (substring or SequenceMatcher).

    - exact equality -> True
    - substring containment -> True (helps partial tokens like "clas" -> "class")
    - SequenceMatcher ratio >= threshold -> True (helps small typos like "classs")
    """
    if not a or not b:
        return False
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return True
    # Substring containment handles short partial tokens (e.g. "clas" in "class")
    # require at least length 3 to avoid matching too aggressively on very short tokens
    if (len(a) >= 3 and a in b) or (len(b) >= 3 and b in a):
        return True
    try:
        return SequenceMatcher(None, a, b).ratio() >= threshold
    except Exception:
        return False

# Small bounded edit-distance check (early-exit, only for short strings).
def edit_distance_leq(a: str, b: str, max_dist: int = 1) -> bool:
    """Return True if edit distance between a and b is <= max_dist.
    Optimized for small max_dist (1 or 2)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    # Quick length check
    if abs(la - lb) > max_dist:
        return False
    # Ensure a is the shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    # If max_dist == 0 handled above (equality)
    # For max_dist == 1 we can do quick checks
    if max_dist == 1:
        # Try deletion/insertion substitution possibilities
        i = 0
        j = 0
        mismatch = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1
                j += 1
            else:
                mismatch += 1
                if mismatch > 1:
                    return False
                # try skip one char in b (insertion into a / deletion from b)
                j += 1
        return True
    # Fallback to dynamic programming for small strings
    # classic Levenshtein but bounded
    dp_prev = list(range(lb + 1))
    for i in range(1, la + 1):
        dp_cur = [i] + [0] * lb
        min_row = dp_cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp_cur[j] = min(dp_prev[j] + 1, dp_cur[j - 1] + 1, dp_prev[j - 1] + cost)
            if dp_cur[j] < min_row:
                min_row = dp_cur[j]
        if min_row > max_dist:
            return False
        dp_prev = dp_cur
    return dp_prev[-1] <= max_dist


# Simple correction helper: collapse long runs of the same character to up to two chars.
# Helps reduce the impact of repeated-letter typos like "classs" -> "class".
def collapse_repeated_chars(word: str, max_run: int = 2) -> str:
    if not word:
        return word
    # Replace runs of the same character longer than max_run with max_run chars
    def _repl(m):
        ch = m.group(1)
        return ch * max_run
    return re.sub(r'(.)\1{2,}', _repl, word, flags=re.IGNORECASE)

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
        'declare', 'declaring', 'declaration', 'initialize', 'initializing', 'initialization',
        'equals', 'assign', 'compare', 'comparison', 'condition', 'conditional'
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
        r'\bcreating\s+\w+', r'\bdefining\s+\w+', r'\bdeclaring\s+\w+',
        r'\bequals\s+\w+', r'\bassign\s+\w+', r'\bcompare\s+\w+'
    ]
    
    for pattern in concept_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Extract both words from the pattern
            words_in_match = match.split()
            found_concepts.update(words_in_match)
    
    return found_concepts

# ────────────────────────────────────────────────────────────
# BM25 Implementation
# ────────────────────────────────────────────────────────────

class BM25:
    """BM25 scoring implementation for lexical matching"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.5):
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.N = 0
        
    def fit(self, corpus: List[str]):
        """Fit BM25 on a corpus of documents"""
        self.corpus = corpus
        self.N = len(corpus)
        
        # Tokenize and process documents
        self.doc_freqs = []
        self.doc_len = []
        
        # Count document frequencies for IDF calculation
        df = defaultdict(int)
        
        for doc in corpus:
            # Tokenize document
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            
            # Count term frequencies in this document
            doc_freq = defaultdict(int)
            for token in tokens:
                doc_freq[token] += 1
            
            self.doc_freqs.append(doc_freq)
            
            # Count document frequency for each unique term
            for token in set(tokens):
                df[token] += 1
        
        # Calculate average document length
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate IDF for each term
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5))
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 scoring"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        
        # Filter out stop words but keep programming terms
        filtered_words = []
        for word in words:
            if word not in programming_stop_words or len(word) <= 2:
                filtered_words.append(word)
        
        return filtered_words
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a specific document"""
        if doc_idx >= len(self.doc_freqs):
            return 0.0
        
        query_tokens = self._tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_len[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token in doc_freq:
                # Term frequency in document
                tf = doc_freq[token]
                
                # IDF for this term
                idf = self.idf.get(token, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for a query against all documents"""
        scores = []
        for i in range(len(self.corpus)):
            scores.append(self.score(query, i))
        return scores

def calculate_morphological_similarity(query_words: List[str], content_words: List[str]) -> float:
    """Calculate similarity considering morphological variations"""
    # New fuzzy-aware version: count a query-normalized token as matched if any
    # content-normalized token is sufficiently similar (uses is_similar with a slightly
    # more permissive threshold to allow for short/partial tokens).
    if not query_words:
        return 0.0
    # Build normalized token lists (exclude stop words)
    query_normalized = []
    for word in query_words:
        if word not in programming_stop_words:
            # also consider a collapsed-version to fix repeated-letter typos
            query_normalized.extend(normalize_word(word))
            collapsed = collapse_repeated_chars(word)
            if collapsed and collapsed != word:
                query_normalized.extend(normalize_word(collapsed))
    content_normalized = []
    for word in content_words:
        if word not in programming_stop_words:
            content_normalized.extend(normalize_word(word))
            content_normalized.extend(normalize_word(collapse_repeated_chars(word)))
    # Deduplicate and filter empties
    query_normalized = list({w for w in query_normalized if w})
    content_normalized = list({w for w in content_normalized if w})
    if not query_normalized:
        return 0.0
    # Count how many query tokens find a fuzzy match in content tokens.
    # Treat prefix matches and small edit-distance as strong matches for short tokens.
    matched = 0
    for q in query_normalized:
        matched_flag = False
        for c in content_normalized:
            # direct fuzzy similarity
            if is_similar(q, c, threshold=0.75):
                matched_flag = True
                break
            # prefix match (e.g. "clas" -> "class") — require query token length >= 3
            if len(q) >= 3 and c.startswith(q):
                matched_flag = True
                break
            # small edit distance (allow 1 for short tokens)
            if edit_distance_leq(q, c, max_dist=1):
                matched_flag = True
                break
        if matched_flag:
            matched += 1
    # Approximate intersection and union for a Jaccard-ish score
    intersection_est = matched
    union_size = len(set(query_normalized).union(set(content_normalized)))
    jaccard = intersection_est / union_size if union_size > 0 else 0.0
    coverage = matched / len(query_normalized) if query_normalized else 0.0
    # Emphasize coverage (gives better recall for short queries)
    return (jaccard * 0.4) + (coverage * 0.6)

def calculate_concept_overlap(query_concepts: Set[str], content_concepts: Set[str]) -> float:
    """Calculate overlap between concept sets with morphological awareness"""
    if not query_concepts:
        return 0.0
    # Fuzzy-aware overlap: treat a query-normalized token as matched when any
    # content-normalized token is sufficiently similar (substring or fuzzy ratio).
    query_normalized = []
    for concept in query_concepts:
        query_normalized.extend(normalize_word(concept))
    content_normalized = []
    for concept in content_concepts:
        content_normalized.extend(normalize_word(concept))
    # Deduplicate and remove empty entries
    query_normalized = list({w for w in query_normalized if w})
    content_normalized = list({w for w in content_normalized if w})
    if not query_normalized or not content_normalized:
        return 0.0
    matched = 0
    for q in query_normalized:
        matched_flag = False
        for c in content_normalized:
            if is_similar(q, c, threshold=0.75):
                matched_flag = True
                break
            # prefix match for short query tokens
            if len(q) >= 3 and c.startswith(q):
                matched_flag = True
                break
            if edit_distance_leq(q, c, max_dist=1):
                matched_flag = True
                break
        if matched_flag:
            matched += 1
    if matched == 0:
        return 0.0
    union_size = len(set(query_normalized).union(set(content_normalized)))
    jaccard = matched / union_size if union_size > 0 else 0.0
    coverage = matched / len(query_normalized)
    # Emphasize coverage for recall (same weighting as original design)
    return (jaccard * 0.3) + (coverage * 0.5)

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
        'create', 'creating', 'define', 'defining', 'declare', 'declaring',
        'equals', 'assign', 'compare', 'comparison'
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
                  'garbage', 'collection', 'stack', 'heap', 'buffer', 'address'},
        'operators': {'equals', 'assign', 'compare', 'comparison', 'condition', 'conditional',
                     'plus', 'minus', 'multiply', 'divide', 'modulo', 'increment', 'decrement'}
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

class BM25EnhancedSimilarity:
    """Advanced similarity calculator with BM25, symbol-awareness, and morphological matching"""
    
    def __init__(self):
        # Increase weight for morphological/fuzzy lexical signals so typos/partials have more impact.
        # Keep embeddings relevant but give stronger weight to morphological matching.
        self.weights = {
            'embedding': 0.28,
            'bm25': 0.12,
            'topic_relevance': 0.18,
            'semantic_distance': 0.05,
            'morphological_match': 0.35,
            'exact_matches': 0.02
        }
        self.bm25 = None
    
    def set_bm25(self, bm25: BM25):
        """Set the BM25 instance for scoring"""
        self.bm25 = bm25
    
    def calculate_similarity(self, query: str, content: str, embedding_similarity: float, 
                           bm25_score: float = 0.0) -> Dict[str, float]:
        """Calculate comprehensive similarity score including BM25"""
        query_lower = query.lower().strip()
        content_lower = content.lower().strip()
        
        if not query_lower or not content_lower:
            return {'final_score': 0.0, 'embedding': embedding_similarity, 'bm25': bm25_score,
                   'topic_relevance': 0.0, 'semantic_distance': 0.0, 'morphological_match': 0.0, 
                   'exact_matches': 0.0}
        
        scores = {}
        
        # 1. Embedding similarity
        scores['embedding'] = embedding_similarity
        
        # 2. BM25 lexical matching (better normalization)
        scores['bm25'] = min(bm25_score / 5.0, 1.0) if bm25_score > 0 else 0.0
        
        # 3. Topic relevance (high weight, morphologically aware)
        scores['topic_relevance'] = calculate_topic_relevance(query, content)
        
        # 4. Semantic distance (topic clustering with morphological matching)
        scores['semantic_distance'] = calculate_semantic_distance(query, content)
        
        # 5. Morphological matching
        scores['morphological_match'] = self._calculate_morphological_match(query_lower, content_lower)
        
        # 6. Exact matches
        scores['exact_matches'] = self._calculate_exact_matches(query_lower, content_lower)
        
        # Calculate weighted final score
        final_score = sum(scores[component] * self.weights[component] 
                         for component in self.weights.keys())
        
        # Apply topic mismatch penalty (more lenient)
        if scores['topic_relevance'] < 0.02 and scores['semantic_distance'] < 0.05:
            final_score *= 0.7  # More lenient penalty for topic mismatch
        
        # Boost if BM25 and morphological matching are both strong
        if scores['bm25'] > 0.5 and scores['morphological_match'] > 0.5:
            final_score *= 1.15  # Boost for strong lexical + morphological matches
        
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

def preprocess_text_for_embedding(text: str) -> str:
    """Enhanced preprocessing specifically for embedding generation with symbol awareness"""
    if not text:
        return ""
    
    # Step 1: Separate symbols from words for better exact matching
    text = separate_symbols_from_words(text)
    
    # Step 2: Convert programming operators to natural language
    text = normalize_programming_operators(text)
    
    # Step 3: Standard preprocessing
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Convert to lowercase for processing
    text_lower = text.lower()
    
    # Tokenize
    try:
        tokens = word_tokenize(text_lower)
    except Exception:
        tokens = text_lower.split()
    
    # Process tokens with morphological awareness
    processed_tokens = []
    
    # Keep original tokens for exact matching
    for token in tokens:
        if token and len(token) > 1:
            processed_tokens.append(token)
    
    # Add morphologically normalized versions for important words
    for token in tokens:
        if token and len(token) > 2 and token not in programming_stop_words:
            normalized_forms = normalize_word(token)
            # Only add the most relevant normalized forms to avoid noise
            for form in normalized_forms:
                if not form.startswith('prefix_') and not form.startswith('suffix_'):
                    processed_tokens.append(form)
    
    return ' '.join(processed_tokens)

def preprocess_text(text: str, preserve_original: bool = True) -> str:
    """Standard preprocessing for search operations"""
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

# --- Fuzzy expansion: augment query with close programming vocab terms to handle typos/partials
# Small programming vocabulary used to expand likely intended tokens (can be extended)
PROGRAMMING_VOCAB = {
    'class', 'classes', 'object', 'objects', 'inheritance', 'polymorphism',
    'function', 'functions', 'method', 'methods', 'variable', 'variables',
    'array', 'arrays', 'list', 'lists', 'dictionary', 'dictionaries',
    'loop', 'loops', 'condition', 'conditions', 'if', 'else', 'while', 'for',
    'struct', 'structs', 'interface', 'interfaces', 'module', 'modules',
    'import', 'export', 'package', 'packages', 'library', 'libraries',
    'type', 'types', 'datatype', 'datatypes', 'string', 'integer', 'boolean',
    'callback', 'callbacks', 'event', 'events', 'handler', 'handlers',
    'parameter', 'parameters', 'argument', 'arguments', 'return', 'returns',
    'create', 'creating', 'define', 'defining', 'declare', 'declaring',
    'equals', 'assign', 'compare', 'comparison'
}


def enhanced_query_preprocessing(query: str) -> str:
    """Query preprocessing with symbol awareness, morphological expansion, and fuzzy expansion."""
    if not query:
        return ""
    # Step 1: Apply symbol-aware preprocessing to query
    symbol_processed = separate_symbols_from_words(query)
    operator_processed = normalize_programming_operators(symbol_processed)
    # Add original query
    all_processed = [query.lower(), operator_processed.lower()]
    # Add morphologically processed version
    processed = preprocess_text(operator_processed, preserve_original=True)
    if processed:
        all_processed.append(processed)
    # Add concept-focused expansion with morphological variants
    concepts = extract_key_concepts(operator_processed)
    for concept in concepts:
        all_processed.append(concept)
        normalized_forms = normalize_word(concept)
        all_processed.extend(normalized_forms)
        synonyms = get_synonyms(concept)
        for syn in list(synonyms)[:2]:
            if len(syn) > 2:
                all_processed.append(syn)
    # Fuzzy-expand tokens against known programming vocabulary (fix typos like "classs" or "clas")
    tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', operator_processed.lower()))
    for tok in tokens:
        # skip very short tokens and stop words
        if len(tok) < 2 or tok in programming_stop_words:
            continue
        # Add collapsed variant (handle repeated-letter typos like "classs")
        collapsed = collapse_repeated_chars(tok)
        if collapsed and collapsed != tok:
            all_processed.append(collapsed)
        # If token is already in vocab, add it directly
        if tok in PROGRAMMING_VOCAB:
            all_processed.append(tok)
            continue
        # Prefix-based expansion: if tok is a prefix of any vocab term (e.g., "clas" -> "class"),
        # add that vocab term (require tok length >= 3 to avoid over-matching).
        if len(tok) >= 3:
            for vocab_term in PROGRAMMING_VOCAB:
                if vocab_term.startswith(tok):
                    all_processed.append(vocab_term)
                    for nf in normalize_word(vocab_term):
                        if not nf.startswith('prefix_') and not nf.startswith('suffix_'):
                            all_processed.append(nf)
                    for syn in list(get_synonyms(vocab_term))[:2]:
                        if len(syn) > 2:
                            all_processed.append(syn)
        # Fallback: use difflib.get_close_matches to find likely intended vocabulary tokens
        try:
            from difflib import get_close_matches
            close = get_close_matches(tok, PROGRAMMING_VOCAB, n=3, cutoff=0.55)
        except Exception:
            close = []
        for vocab_term in close:
            all_processed.append(vocab_term)
            for nf in normalize_word(vocab_term):
                if not nf.startswith('prefix_') and not nf.startswith('suffix_'):
                    all_processed.append(nf)
            for syn in list(get_synonyms(vocab_term))[:2]:
                if len(syn) > 2:
                    all_processed.append(syn)
    return ' '.join(all_processed)


# ────────────────────────────────────────────────────────────
# Document chunking
# ────────────────────────────────────────────────────────────

def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Create chunks from text using fixed-size sliding window strategy.
    Optimized for mixed documentation and code content.
    
    Args:
        text: Input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of dictionaries with chunk info
    """
    if len(text) <= 100:
        # No chunking needed for short documents
        return [{"text": text, "chunk_id": 0, "start": 0, "end": len(text)}]
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at natural boundaries (sentence/line breaks)
        if end < len(text):
            # Look for sentence boundaries first
            sentence_break = text.rfind('.', start, end)
            line_break = text.rfind('\n', start, end)
            
            # Choose the best break point
            if sentence_break > start + chunk_size * 0.7:
                end = sentence_break + 1
            elif line_break > start + chunk_size * 0.6:
                end = line_break + 1
            # Otherwise use the hard boundary
        
        chunk_text = text[start:end].strip()
        if chunk_text:  # Only add non-empty chunks
            chunks.append({
                "text": chunk_text,
                "chunk_id": chunk_id,
                "start": start,
                "end": end
            })
            chunk_id += 1
        
        # Move start position with overlap
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    
    return chunks

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
        
        # Initialize BM25-enhanced similarity calculator
        self.similarity_calculator = BM25EnhancedSimilarity()
        self.bm25 = None
        self.bm25_corpus = []
        self.file_paths = []
        
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
    
    def _build_bm25_index(self):
        """Build BM25 index from current documents (full documents, not chunks)"""
        print("Building BM25 index...")
        
        # Get all documents from ChromaDB
        all_docs = self.collection.get(include=["documents", "metadatas"])
        
        if not all_docs["documents"]:
            print("No documents found for BM25 indexing")
            return
        
        # Prepare corpus for BM25 - use full documents, not chunks
        self.bm25_corpus = []
        self.file_paths = []
        seen_files = set()
        
        for i, doc in enumerate(all_docs["documents"]):
            file_path = all_docs["metadatas"][i].get("file_path", "")
            
            # Only add each file once to BM25 corpus (not each chunk)
            if file_path and file_path not in seen_files:
                seen_files.add(file_path)
                
                # Get original full content for BM25 (not the preprocessed version)
                original_content = self._get_original_content(file_path)
                
                if original_content:
                    self.bm25_corpus.append(original_content)
                    self.file_paths.append(file_path)
        
        # If no full documents found, fallback to using chunks
        if not self.bm25_corpus:
            print("No full documents found, using chunks for BM25")
            for i, doc in enumerate(all_docs["documents"]):
                file_path = all_docs["metadatas"][i].get("file_path", "")
                self.bm25_corpus.append(doc)
                self.file_paths.append(file_path)
        
        # Build BM25 index
        self.bm25 = BM25()
        self.bm25.fit(self.bm25_corpus)
        self.similarity_calculator.set_bm25(self.bm25)
        
        print(f"BM25 index built with {len(self.bm25_corpus)} documents ({len(seen_files)} unique files)")
    
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
        
        # Rebuild BM25 index after sync
        self._build_bm25_index()
    
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
                    content = fp.read()
                
                base_metadata = self._store_index_path_metadata(file_path, self.index_path)
                document_chunks = self._prepare_document_for_storage(content, file_path, base_metadata)
                
                for chunk_data in document_chunks:
                    documents.append(chunk_data["document"])
                    metadatas.append(chunk_data["metadata"])
                    ids.append(str(uuid.uuid4()))
                    
            except Exception as e:
                print(f"Skip {file_path}: {e}")
        
        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def _clean_content(self, content: str) -> str:
        """Symbol-aware content cleaning for embedding generation"""
        # Remove control characters
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        cleaned = content.strip()
        
        # Create symbol-aware representations
        representations = []
        
        # 1. Original cleaned content (for exact matching)
        representations.append(cleaned)
        
        # 2. Symbol-aware preprocessed content (main representation for embedding)
        symbol_processed = preprocess_text_for_embedding(cleaned)
        representations.append(symbol_processed)
        
        # 3. Extract and emphasize key concepts with morphological variants
        concepts = extract_key_concepts(symbol_processed)
        if concepts:
            concept_variants = set()
            for concept in concepts:
                concept_variants.add(concept)
                concept_variants.update(normalize_word(concept))
            concept_text = ' '.join(concept_variants)
            representations.append(concept_text)
        
        # Combine representations
        return ' '.join(representations)

    def _prepare_document_for_storage(self, content: str, file_path: str, base_metadata: dict) -> List[Dict]:
        """Prepare document for storage, creating chunks if needed"""
        cleaned_content = self._clean_content(content)
        
        # Check if document needs chunking (>100 characters)
        if len(content) <= 100:
            # Store as single document
            return [{
                "document": cleaned_content,
                "metadata": {**base_metadata, "is_chunked": False, "chunk_id": 0, "total_chunks": 1},
                "original_content": content
            }]
        
        # Create chunks from original content
        chunks = create_chunks(content)
        documents_to_store = []
        
        for chunk in chunks:
            chunk_cleaned = self._clean_content(chunk["text"])
            chunk_metadata = {
                **base_metadata,
                "is_chunked": True,
                "chunk_id": chunk["chunk_id"],
                "total_chunks": len(chunks),
                "chunk_start": chunk["start"],
                "chunk_end": chunk["end"]
            }
            
            documents_to_store.append({
                "document": chunk_cleaned,
                "metadata": chunk_metadata,
                "original_content": chunk["text"]
            })
        
        return documents_to_store

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
                            content = fp.read()
                        
                        base_metadata = self._store_index_path_metadata(p, directory)
                        document_chunks = self._prepare_document_for_storage(content, p, base_metadata)
                        
                        for chunk_data in document_chunks:
                            documents.append(chunk_data["document"])
                            metadatas.append(chunk_data["metadata"])
                            ids.append(str(uuid.uuid4()))
                        
                        chunk_count = len(document_chunks)
                        print(f"Indexed {p} ({chunk_count} chunks)")
                    except Exception as e:
                        print("Skip", f, e)
        if documents:
            self.collection.add(documents=documents,
                                metadatas=metadatas,
                                ids=ids)
            print("Added", len(documents), "document chunks")
            
            # Build BM25 index after adding documents
            self._build_bm25_index()

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

        base_metadata = self._store_index_path_metadata(path, self.index_path)
        base_metadata["title"] = title or ""
        base_metadata["size"] = len(text)  # Override with actual content size

        document_chunks = self._prepare_document_for_storage(text, path, base_metadata)
        
        documents, metadatas, ids = [], [], []
        for chunk_data in document_chunks:
            documents.append(chunk_data["document"])
            metadatas.append(chunk_data["metadata"])
            ids.append(str(uuid.uuid4()))

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        chunk_count = len(document_chunks)
        print(f"Saved & indexed {path} ({chunk_count} chunks)")
        
        # Rebuild BM25 index after adding document
        self._build_bm25_index()

    def search(self, query: str, n: int, thr: float) -> List[Dict]:
        """BM25-enhanced symbol-aware search"""
        if not query.strip():
            return []
        
        # Ensure BM25 index is built
        if self.bm25 is None:
            self._build_bm25_index()
        
        all_results = []
        
        # Strategy 1: Symbol-aware query search
        try:
            symbol_query = enhanced_query_preprocessing(query)
            res1 = self.collection.query(
                query_texts=[symbol_query], 
                n_results=min(n * 4, 100),  # Get more results for reranking
                include=['documents', 'metadatas', 'distances']
            )
            results1 = self._process_search_results(res1, query)
            all_results.extend(results1)
            print(f"Symbol-aware search: {len(results1)} results")
        except Exception as e:
            print(f"Symbol-aware search failed: {e}")
        
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
        
        # BM25-enhanced reranking and deduplication
        unique_results = self._deduplicate_and_rerank_bm25_enhanced(all_results, query, thr)
        
        # Apply threshold and limit
        filtered_results = [r for r in unique_results if r['similarity'] >= thr]
        final_results = filtered_results[:n]
        
        # Debug: show similarity scores
        if unique_results:
            print(f"Debug: Top 3 similarity scores: {[r['similarity'] for r in unique_results[:3]]}")
            print(f"Debug: Threshold applied: {thr}")
        
        print(f"Final results: {len(final_results)} (after BM25-enhanced reranking and threshold {thr})")
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
    
    def _aggregate_chunks_to_documents(self, results: List[Dict], query: str) -> List[Dict]:
        """Aggregate chunk results back to document-level results"""
        # Group results by file path
        file_groups = {}
        for result in results:
            file_path = result['metadata'].get('file_path', '')
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)
        
        aggregated_results = []
        
        for file_path, chunks in file_groups.items():
            # Find the best chunk for this document
            best_chunk = max(chunks, key=lambda x: x['embedding_similarity'])
            
            # Calculate combined score from all chunks
            chunk_scores = [chunk['embedding_similarity'] for chunk in chunks]
            max_score = max(chunk_scores)
            avg_score = sum(chunk_scores) / len(chunk_scores)
            
            # Use weighted combination: 70% best chunk + 30% average of all chunks
            combined_embedding_score = (max_score * 0.7) + (avg_score * 0.3)
            
            # Get full document content for similarity calculation
            original_content = self._get_original_content(file_path)
            if not original_content:
                original_content = str(best_chunk['document'])
            
            # Calculate BM25 score using full document
            bm25_score = 0.0
            if self.bm25 and file_path in self.file_paths:
                try:
                    doc_idx = self.file_paths.index(file_path)
                    bm25_score = self.bm25.score(query, doc_idx)
                except (ValueError, IndexError):
                    bm25_score = 0.0
            
            # Calculate BM25-enhanced similarity using full document
            similarity_scores = self.similarity_calculator.calculate_similarity(
                query, original_content, combined_embedding_score, bm25_score
            )
            
            # Create aggregated result
            aggregated_result = {
                "document": original_content,
                "metadata": best_chunk['metadata'].copy(),
                "embedding_similarity": combined_embedding_score,
                "similarity": similarity_scores['final_score'],
                "similarity_breakdown": similarity_scores,
                "bm25_score": bm25_score,
                "original_query": query,
                "chunk_count": len(chunks),
                "matching_chunks": len(chunks)
            }
            
            # Remove chunk-specific metadata for document-level result
            if 'chunk_id' in aggregated_result['metadata']:
                del aggregated_result['metadata']['chunk_id']
            if 'chunk_start' in aggregated_result['metadata']:
                del aggregated_result['metadata']['chunk_start']
            if 'chunk_end' in aggregated_result['metadata']:
                del aggregated_result['metadata']['chunk_end']
            aggregated_result['metadata']['is_chunked'] = len(chunks) > 1
            
            aggregated_results.append(aggregated_result)
        
        # Sort by final similarity score
        aggregated_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return aggregated_results

    def _deduplicate_and_rerank_bm25_enhanced(self, results: List[Dict], query: str, threshold: float) -> List[Dict]:
        """Remove duplicates and rerank using BM25-enhanced similarity with chunk aggregation"""
        # First aggregate chunks back to documents
        aggregated_results = self._aggregate_chunks_to_documents(results, query)
        
        return aggregated_results
    
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
        """Find line and column positions of query matches with symbol awareness"""
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
        """Get match information without buggy preview text"""
        if not matches:
            return []
        
        # Just return the match info without preview text
        contexts = []
        for match in matches:
            contexts.append({
                'line': match['line'],
                'column': match['column'],
                'match_type': match['match_type']
            })
        
        return contexts
    
    def _get_display_content(self, doc_content: str, file_path: str, query: str = None) -> str:
        """Get full document content"""
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as fp:
                    original_content = fp.read()
                # Clean but don't preprocess for display
                original_content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', original_content)
                original_content = original_content.replace('\r\n', '\n').replace('\r', '\n').strip()
                return original_content
            except Exception:
                return doc_content
        else:
            return doc_content

    def merge_results(self, results: List[Dict]) -> str:
        if not results:
            return ""
        merged = []
        for i, r in enumerate(results, 1):
            merged.append("\n" + "="*60)
            
            # Show chunk information if available
            chunk_info = ""
            if 'chunk_count' in r and r['chunk_count'] > 1:
                chunk_info = f" ({r['chunk_count']} chunks)"
            
            merged.append(f"DOC {i}: {self._abs(r['metadata']['file_path'])}{chunk_info} "
                        f"(sim {r['similarity']:.3f})")
            
            # Show BM25-enhanced similarity breakdown
            if 'similarity_breakdown' in r:
                breakdown = r['similarity_breakdown']
                merged.append(f"  BM25: {breakdown.get('bm25', 0):.3f}, "
                            f"Topic Relevance: {breakdown.get('topic_relevance', 0):.3f}, "
                            f"Semantic Distance: {breakdown.get('semantic_distance', 0):.3f}, "
                            f"Morphological: {breakdown.get('morphological_match', 0):.3f}, "
                            f"Embedding: {breakdown.get('embedding', 0):.3f}")
                
                # Show raw BM25 score
                if 'bm25_score' in r:
                    merged.append(f"  Raw BM25 Score: {r['bm25_score']:.3f}")
            
            # Show chunking information
            if 'chunk_count' in r:
                merged.append(f"  Chunks: {r['chunk_count']} total, {r.get('matching_chunks', r['chunk_count'])} matched")
            
            merged.append("-"*60)
            
            # Add full document content
            doc_content = r['document']
            merged.append(str(doc_content))
            merged.append("")
            
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
        thr    = float(q.get('threshold',  [self.config.get('similarity_threshold', 0.5)])[0])
        fmt    = q.get('format', ['json'])[0]

        res    = self.indexer.search(query, n, thr)
        if fmt == 'text':                       # plain-text branch
            out = ["Found %d documents" % len(res)]
            for i, r in enumerate(res, 1):
                out.append(f"{i}. {DocumentIndexer._abs(r['metadata']['file_path'])} "
                        f"(sim {r['similarity']:.3f})")
                if 'bm25_score' in r:
                    out.append(f"   BM25: {r['bm25_score']:.3f}")
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
    p.add_argument("--similarity-threshold", type=float, default=0.1)
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
                print(f"   BM25: {breakdown.get('bm25', 0):.3f}, "
                      f"Topic Relevance: {breakdown.get('topic_relevance', 0):.3f}, "
                      f"Semantic Distance: {breakdown.get('semantic_distance', 0):.3f}, "
                      f"Morphological: {breakdown.get('morphological_match', 0):.3f}")
            if 'bm25_score' in r:
                print(f"   Raw BM25 Score: {r['bm25_score']:.3f}")
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
