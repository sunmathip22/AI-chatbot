import os
import re
import json
import math
import shutil
import time
import urllib.request
import urllib.error

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional imports with graceful fallbacks
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        # pyrefly: ignore[missing-import]
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

try:
    from langchain_ollama import OllamaLLM
    HAS_LANGCHAIN_OLLAMA = True
except ImportError:
    HAS_LANGCHAIN_OLLAMA = False

try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma
    HAS_CHROMA = True
except Exception:
    HAS_CHROMA = False

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CHROMA_PATH = os.path.join(BASE_DIR, "chromadb_data")
UPLOADS_PATH = os.path.join(BASE_DIR, "uploads")
STORE_FILE = os.path.join(CHROMA_PATH, "documents_store.json")

# Try loading .env from PROJECT_ROOT or BASE_DIR
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_NAME = "gemma3n:e2b"
DEFAULT_EMBEDDING_MODEL_NAME = "embeddinggemma-2"
DEFAULT_GROQ_FALLBACK_MODEL = "gemma2-9b-it"
DEFAULT_PROJECT_NAME = "Project Documentation"
DEFAULT_PROJECT_DESCRIPTION = "Official Project Documentation"

def get_project_name() -> str:
    return os.getenv("PROJECT_NAME", DEFAULT_PROJECT_NAME).strip() or DEFAULT_PROJECT_NAME

def get_project_description() -> str:
    return os.getenv("PROJECT_DESCRIPTION", DEFAULT_PROJECT_DESCRIPTION).strip()

def get_model_type() -> str:
    return "SLM (Small Language Model)"

def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")

def get_configured_model() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip()

def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME).strip() or DEFAULT_EMBEDDING_MODEL_NAME

def get_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()

def get_groq_fallback_model() -> str:
    return os.getenv("GROQ_FALLBACK_MODEL", DEFAULT_GROQ_FALLBACK_MODEL).strip()

# Module-level variables for compatibility
PROJECT_NAME = get_project_name()
PROJECT_DESCRIPTION = get_project_description()
MODEL_TYPE = get_model_type()
OLLAMA_BASE_URL = get_ollama_base_url()
MODEL_NAME = get_configured_model()
EMBEDDING_MODEL_NAME = get_embedding_model_name()
GROQ_API_KEY = get_groq_api_key()
GROQ_FALLBACK_MODEL = get_groq_fallback_model()

# Retrieval, Routing & Generation Settings (.env)
DEFAULT_SIMILARITY_THRESHOLD = 0.35
DEFAULT_TOP_K = 2
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_NUM_CTX = 2048
DEFAULT_NUM_PREDICT = 120
DEFAULT_TEMPERATURE = 0.2
DEFAULT_KEEP_ALIVE = "30m"

def get_similarity_threshold() -> float:
    try:
        return float(os.getenv("SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD)))
    except ValueError:
        return DEFAULT_SIMILARITY_THRESHOLD

def get_top_k() -> int:
    try:
        return int(os.getenv("TOP_K", str(DEFAULT_TOP_K)))
    except ValueError:
        return DEFAULT_TOP_K

def get_chunk_size() -> int:
    try:
        return int(os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    except ValueError:
        return DEFAULT_CHUNK_SIZE

def get_chunk_overlap() -> int:
    try:
        return int(os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP)))
    except ValueError:
        return DEFAULT_CHUNK_OVERLAP

def get_num_ctx() -> int:
    try:
        return int(os.getenv("NUM_CTX", str(DEFAULT_NUM_CTX)))
    except ValueError:
        return DEFAULT_NUM_CTX

def get_num_predict() -> int:
    try:
        return int(os.getenv("NUM_PREDICT", str(DEFAULT_NUM_PREDICT)))
    except ValueError:
        return DEFAULT_NUM_PREDICT

def get_temperature() -> float:
    try:
        return float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    except ValueError:
        return DEFAULT_TEMPERATURE

def get_keep_alive() -> str:
    return os.getenv("KEEP_ALIVE", DEFAULT_KEEP_ALIVE).strip() or DEFAULT_KEEP_ALIVE

# ---------------------------------------------------------------------------
# Persistent ChromaDB Client & In-Memory Cache (Initialized ONCE at startup)
# ---------------------------------------------------------------------------
CHROMA_CLIENT = None
CHROMA_COLLECTION = None

def init_chroma():
    """Initializes ChromaDB persistent client and collection once at startup."""
    global CHROMA_CLIENT, CHROMA_COLLECTION
    if CHROMA_COLLECTION is not None:
        return CHROMA_COLLECTION
    try:
        import chromadb
        if CHROMA_CLIENT is None:
            CHROMA_CLIENT = chromadb.PersistentClient(path=CHROMA_PATH)
        CHROMA_COLLECTION = CHROMA_CLIENT.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("[ChromaDB] Initialized persistent ChromaDB client once at startup.")
        return CHROMA_COLLECTION
    except Exception as e:
        print(f"[ChromaDB Notice] Using unified document store: {e}")
        return None

# In-memory query cache: normalized question -> answer dictionary
_query_cache: dict[str, dict] = {}

def normalize_query(query: str) -> str:
    """Normalize text: lowercase, remove non-alphanumeric, collapse whitespace."""
    clean = re.sub(r"[^\w\s\u0B80-\u0BFF]", "", query.strip().lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def get_cached_response(query: str) -> dict | None:
    norm = normalize_query(query)
    if norm in _query_cache:
        return dict(_query_cache[norm])
    return None

def set_cached_response(query: str, response: dict):
    norm = normalize_query(query)
    if norm:
        if len(_query_cache) >= 500:
            first_key = next(iter(_query_cache))
            del _query_cache[first_key]
        _query_cache[norm] = response

def clear_query_cache():
    global _query_cache
    _query_cache = {}

os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(UPLOADS_PATH, exist_ok=True)
init_chroma()


# ---------------------------------------------------------------------------
# Language & Style Detection (English, Tamil script, Thanglish)
# ---------------------------------------------------------------------------

THANGLISH_KEYWORDS = {
    # Pronouns & determiners
    "indha", "intha", "yindha", "yintha", "andha", "antha", "yandha", "endha", "entha", "yendha",
    "idhu", "ithu", "yidhu", "adhu", "athu", "edhu", "ethu",
    "idhula", "ithula", "adhula", "athula", "idhu-la", "adhu-la",
    "nee", "neenga", "naan", "namma", "namakku", "enakku", "enaku", "unakku", "unaku",
    "unga", "ungalukku", "ungala", "avanga", "ivanga", "avan", "ival",
    # Question words
    "enna", "edhuku", "edhukku", "ethuku", "ethukku", "yedhuku",
    "epdi", "eppadi", "yeppadi", "yepdi",
    "enga", "enge", "engu", "eppo", "eppodhu", "yaar", "yaaru",
    "evalo", "evvalavu", "ethana", "ethanai",
    # Greetings & affirmations
    "vanakkam", "vaazhthukkal", "nandri", "aama", "aamam", "seri",
    "illa", "illai", "illana", "illaina", "illainu", "illanu",
    # Verbs & actions
    "sollu", "solren", "solluren", "sollunga", "solla", "solranga",
    "pannu", "panren", "pannanum", "pandradhu", "pannunga", "panna", "pannalam", "panniten",
    "irukku", "irukka", "irundha", "irukkum", "irundhuchu",
    "kidaikkum", "kidaikkala", "kedaikala", "kedaikkum", "kidaikuma",
    "theriyum", "theriyala", "puriyala", "puriyudhu", "puriyum",
    "vaanga", "ponga", "vanga", "varum", "varala", "podu", "podanum",
    "aagum", "aagala", "aagiduchu", "mudiyum", "mudiyadhu",
    # Particles, postpositions & colloquial slang
    "paththi", "pathi", "konjam", "romba", "mattum", "kooda",
    "bro", "machan", "machi", "thala", "pa", "nga"
}

THANGLISH_SUFFIXES = re.compile(
    r"^(.*?)(la|le|kku|ku|oda|ngra|ngura|num|pannanum|unga|ren|rom|radhu|dhu|chu|pa)$",
    re.IGNORECASE
)

# Active session language memory to preserve style for short follow-ups
_last_detected_style = "english"


def detect_language_style(text: str, default: str | None = None) -> str:
    """
    Automatically detects the language and writing style:
    - 'tamil': Tamil written in Tamil script (\u0B80-\u0BFF)
    - 'thanglish': Tamil written using English/Latin alphabet
    - 'english': Standard English
    """
    global _last_detected_style
    fallback = default or _last_detected_style or "english"

    if not text or not text.strip():
        return fallback

    # 1. Any Tamil Unicode script -> strictly 'tamil'
    if re.search(r"[\u0B80-\u0BFF]", text):
        _last_detected_style = "tamil"
        return "tamil"

    # 2. Tokenize Latin words
    raw_words = re.findall(r"[a-zA-Z0-9-]+", text.lower())
    if not raw_words:
        return fallback

    thanglish_score = 0
    for w in raw_words:
        w_clean = re.sub(r"^[^a-z]+|[^a-z]+$", "", w)
        if not w_clean:
            continue

        if w_clean in THANGLISH_KEYWORDS:
            thanglish_score += 2
            continue

        # Check hyphenated mixed tokens like 'project-la', 'server-kku', 'use aagum-ngra'
        if "-" in w:
            parts = w.split("-")
            if any(p in THANGLISH_KEYWORDS or THANGLISH_SUFFIXES.match(p) for p in parts):
                thanglish_score += 2
                continue

        # Suffix matching on words longer than 3 characters
        if len(w_clean) >= 4 and THANGLISH_SUFFIXES.match(w_clean):
            thanglish_score += 1

    # Single-word or short phrases with Thanglish indicator
    if thanglish_score >= 2 or (len(raw_words) <= 3 and thanglish_score >= 1):
        _last_detected_style = "thanglish"
        return "thanglish"

    # Very short ambiguous queries (e.g., 'and port?', 'why?', 'how?') retain conversation style
    if len(raw_words) <= 2 and _last_detected_style in ["tamil", "thanglish"] and default is None:
        return _last_detected_style

    _last_detected_style = "english"
    return "english"


# ---------------------------------------------------------------------------
# FIX 1: Language Detection in Code (No LLM)
# ---------------------------------------------------------------------------

TANGLISH_INDICATOR_WORDS = {
    "enna", "evlo", "evalo", "evvalavu", "irukku", "irukka", "venum", "vendam",
    "pannunga", "panunga", "solunga", "sollunga", "sollu", "kudunga", "ku", "la",
    "pathi", "rate", "vilai", "naal", "aagum", "aagudhu", "mudiyuma", "theriyuma",
    "anuppunga", "vaanga", "eppadi", "epdi", "enga", "neenga", "naan", "unga"
}

def detect_reply_language(text: str) -> str:
    """
    FIX 1: Rule-based language detection without LLM:
    - Tamil Unicode characters -> 'Tamil'
    - Common Tanglish words (whole-word, case-insensitive) -> 'Tanglish'
    - Otherwise -> 'English'
    """
    if not text:
        return "English"

    # 1. Tamil script check (U+0B80 to U+0BFF)
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil"

    # 2. Tanglish check using whole words
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if tokens & TANGLISH_INDICATOR_WORDS:
        return "Tanglish"

    return "English"


def is_language_retry_enabled() -> bool:
    """Returns True if LANGUAGE_RETRY is enabled in environment (default True)."""
    val = os.getenv("LANGUAGE_RETRY", "true").strip().lower()
    return val in ("true", "1", "yes", "on")

# Concept map for cross-lingual query expansion (Tamil/Thanglish -> English documentation keywords)
CONCEPT_MAP = {
    # Purpose / What is it / Use
    "edhuku": ["purpose", "what it does", "use", "overview", "about", "features"],
    "edhukku": ["purpose", "what it does", "use", "overview", "about", "features"],
    "ethuku": ["purpose", "what it does", "use", "overview", "about", "features"],
    "ethukku": ["purpose", "what it does", "use", "overview", "about", "features"],
    "use": ["purpose", "use", "features", "what it does"],
    "பயன்": ["purpose", "use", "features", "what it does"],
    "பயன்படும்": ["purpose", "use", "features", "what it does"],
    "பயன்பாடு": ["purpose", "use", "features", "what it does"],
    "நோக்கம்": ["purpose", "objective", "about", "overview"],
    # How to / Setup / Install / Run
    "epdi": ["how", "steps", "setup", "run", "install", "commands"],
    "eppadi": ["how", "steps", "setup", "run", "install", "commands"],
    "எப்படி": ["how", "steps", "setup", "run", "install"],
    "எவ்வாறு": ["how", "steps", "setup", "run"],
    "pannanum": ["steps", "setup", "create", "run", "install"],
    "pandradhu": ["steps", "setup", "create", "run"],
    "install": ["install", "requirements", "setup", "pip"],
    "நிறுவுவது": ["install", "setup", "requirements", "pip"],
    "run": ["run", "start", "launch", "uvicorn", "server"],
    "இயங்குகிறது": ["run", "server", "start", "port"],
    "இயக்குவது": ["run", "start", "launch", "server"],
    "start": ["start", "run", "launch", "serve"],
    "தொடங்குவது": ["start", "run", "launch", "serve"],
    # Port / Host / Server / Network
    "port": ["port", "host", "localhost", "8000"],
    "போர்ட்": ["port", "host", "localhost", "8000"],
    "server": ["server", "uvicorn", "fastapi", "host"],
    "சேவையகம்": ["server", "uvicorn", "fastapi", "host"],
    # Requirements / Dependencies
    "theva": ["requirements", "prerequisites", "dependencies"],
    "தேவை": ["requirements", "prerequisites", "dependencies"],
    # Architecture / Model / Stack
    "model": ["model", "slm", "gemma", "ollama"],
    "மாதிரி": ["model", "slm", "gemma", "ollama"],
    "stack": ["stack", "backend", "frontend", "docker"],
    # Errors & Troubleshooting
    "error": ["error", "fix", "troubleshooting", "failed"],
    "பிழை": ["error", "fix", "troubleshooting", "failed"],
    "பிரச்சனை": ["error", "problem", "fix"],
}


def clean_mixed_token(token: str) -> list[str]:
    """Extract root words from mixed Tamil-English or Thanglish suffixed tokens."""
    results = [token]
    # Strip Tamil case endings attached to English words (e.g., 'port-ல்' -> 'port', 'project-ன்' -> 'project')
    tamil_stripped = re.sub(r"[-–—]?[\u0B80-\u0BFF]+$", "", token)
    if tamil_stripped and tamil_stripped != token:
        results.append(tamil_stripped)

    # Strip Thanglish postpositions (e.g., 'port-la' -> 'port', 'serverla' -> 'server')
    thang_stripped = re.sub(r"[-–—]?(la|le|kku|ku|oda|ngra|num|unga|ren|radhu)$", "", token, flags=re.IGNORECASE)
    if thang_stripped and len(thang_stripped) >= 3 and thang_stripped != token:
        results.append(thang_stripped)

    return results


def expand_query_terms(query: str, history: list[dict] | None = None) -> list[str]:
    """Tokenize and expand query terms using concept mapping and recent conversation."""
    raw_tokens = re.findall(r"[\w-]+", query.lower())
    expanded = set()

    for t in raw_tokens:
        expanded.add(t)
        for cleaned in clean_mixed_token(t):
            expanded.add(cleaned)
            if cleaned in CONCEPT_MAP:
                expanded.update(CONCEPT_MAP[cleaned])
        if t in CONCEPT_MAP:
            expanded.update(CONCEPT_MAP[t])

    # Augment follow-up terms from immediate previous turn if query is concise
    if history and len(raw_tokens) <= 4:
        for turn in reversed(history[-2:]):
            prev_content = turn.get("content", "")
            prev_words = re.findall(r"[a-zA-Z0-9]+", prev_content.lower())
            for pw in prev_words:
                if len(pw) >= 4 and pw not in ["this", "that", "what", "where", "have", "with", "from"]:
                    expanded.add(pw)

    return list(expanded)


STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "and", "or", "but", "it", "its", "this", "that", "these", "those"
}


# ---------------------------------------------------------------------------
# Document Storage and Retrieval (ChromaDB compatible + BM25 indexing)
# ---------------------------------------------------------------------------

class DocChunk:
    """Represents a chunk of text with source, page metadata, and optional dense embedding."""
    def __init__(self, text: str, source: str, page: int = 0, embedding: list[float] | None = None):
        self.text = text.strip()
        self.source = os.path.basename(source)
        self.page = int(page) if page is not None and isinstance(page, (int, float)) else 0
        self.embedding = embedding

    def to_dict(self):
        d = {"text": self.text, "source": self.source, "page": self.page}
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

    @classmethod
    def from_dict(cls, data):
        return cls(
            text=data.get("text", ""),
            source=data.get("source", "unknown"),
            page=data.get("page", 0),
            embedding=data.get("embedding")
        )


def generate_embeddings(texts: list[str]) -> tuple[list[list[float]] | None, str | None]:
    """
    Generate dense embeddings using EmbeddingGemma 2 via Ollama /api/embed.
    Returns (list_of_embeddings, error_message).
    """
    if not texts:
        return [], None

    model = get_embedding_model_name()
    base_url = get_ollama_base_url()
    status = check_ollama_status()
    if not status["online"]:
        return None, f"Cannot connect to Ollama at {base_url} ({status.get('error', 'offline')})"

    try:
        url = f"{base_url}/api/embed"
        payload = {
            "model": model,
            "input": texts
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "RAG-Chatbot"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            embs = data.get("embeddings", [])
            if embs and len(embs) == len(texts):
                return embs, None
            return None, f"Embedding model '{model}' returned unexpected response structure."
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return None, f"Ollama HTTP error {e.code} while embedding with '{model}': {body[:250]}"
    except Exception as e:
        return None, f"Failed to generate embeddings with '{model}': {str(e)}"


class DocumentStore:
    """
    High-reliability in-memory & file-persisted vector and BM25 index.
    Embeddings generated with EmbeddingGemma 2; hybrid BM25 fallback.
    """
    def __init__(self):
        self.chunks: list[DocChunk] = []
        self.file_registry = {}  # {filename: chunk_count}
        self.load()

    def load(self):
        self.chunks = []
        self.file_registry = {}
        if os.path.exists(STORE_FILE):
            try:
                with open(STORE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = [DocChunk.from_dict(c) for c in data.get("chunks", [])]
                    self.file_registry = data.get("file_registry", {})
            except Exception as e:
                print(f"[Store] Warning: Could not load existing store: {e}")

    def save(self):
        try:
            with open(STORE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": [c.to_dict() for c in self.chunks],
                    "file_registry": self.file_registry
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Store] Warning: Could not save store: {e}")

    def add_chunks(self, filename: str, new_chunks: list[DocChunk]):
        clean_name = os.path.basename(filename)
        self.chunks = [c for c in self.chunks if c.source != clean_name]
        self.chunks.extend(new_chunks)
        self.file_registry[clean_name] = len(new_chunks)
        self.save()

    def clear(self):
        self.chunks = []
        self.file_registry = {}
        if os.path.exists(STORE_FILE):
            try:
                os.remove(STORE_FILE)
            except Exception:
                pass

    def count(self):
        return len(self.chunks)

    def get_files(self):
        return [{"filename": fn, "chunks": count} for fn, count in self.file_registry.items()]

    def vector_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        threshold: float = 0.30,
        source_filter: str | None = None
    ) -> tuple[list[DocChunk], float, str | None]:
        """
        Cosine similarity search using vector embeddings (from EmbeddingGemma 2).
        Supports isolating document context by source_filter.
        """
        try:
            candidates = self.chunks
            if source_filter:
                candidates = [c for c in candidates if c.source == source_filter]

            valid = [c for c in candidates if c.embedding and len(c.embedding) > 0]
            if not valid:
                return [], 0.0, None

            if HAS_NUMPY:
                q_vec = np.array(query_embedding, dtype=np.float32)
                q_norm = float(np.linalg.norm(q_vec))
                if q_norm == 0:
                    return [], 0.0, None

                c_matrix = np.array([c.embedding for c in valid], dtype=np.float32)
                c_norms = np.linalg.norm(c_matrix, axis=1)
                c_norms[c_norms == 0] = 1e-9

                sims = np.dot(c_matrix, q_vec) / (c_norms * q_norm)
                scored = sorted(zip(sims, valid), key=lambda x: x[0], reverse=True)
            else:
                # Fast pure-Python cosine similarity fallback (no compiled C-extensions required)
                q_norm_sq = sum(x * x for x in query_embedding)
                if q_norm_sq <= 0:
                    return [], 0.0, None
                q_norm = math.sqrt(q_norm_sq)

                scored = []
                for c in valid:
                    emb = c.embedding
                    c_norm_sq = sum(y * y for y in emb)
                    if c_norm_sq <= 0:
                        sim = 0.0
                    else:
                        c_norm = math.sqrt(c_norm_sq)
                        dot = sum(x * y for x, y in zip(query_embedding, emb))
                        sim = dot / (c_norm * q_norm)
                    scored.append((sim, c))

                scored.sort(key=lambda x: x[0], reverse=True)

            best_score = float(scored[0][0]) if scored else 0.0
            relevant = [chunk for score, chunk in scored if score >= threshold][:k]
            return relevant, best_score, None
        except Exception as e:
            return [], 0.0, f"Vector search error: {str(e)}"

    def similarity_search(self, query: str, k: int = 4, threshold: float = 0.2, history: list[dict] | None = None, source_filter: str | None = None):
        """Multilingual similarity search with relevance thresholding."""
        chunks, best_score, err = self.search_with_status(query, k=k, threshold=threshold, history=history, source_filter=source_filter)
        if err:
            raise RuntimeError(err)
        return chunks

    def search_with_status(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.2,
        history: list[dict] | None = None,
        source_filter: str | None = None
    ) -> tuple[list[DocChunk], float, str | None]:
        """
        Executes search:
        - Prioritizes EmbeddingGemma 2 dense vector retrieval if chunk embeddings are present.
        - Falls back gracefully to BM25 search if embeddings are not generated or if query embedding fails.
        - Distinguishes between genuine absence of information vs retrieval failure.
        """
        if not self.chunks:
            return [], 0.0, None

        candidates = self.chunks
        if source_filter:
            candidates = [c for c in candidates if c.source == source_filter]
            if not candidates:
                return [], 0.0, None

        # 1. Try vector search with EmbeddingGemma 2 if any candidate chunk has an embedding
        has_embeddings = any(c.embedding is not None and len(c.embedding) > 0 for c in candidates)
        if has_embeddings:
            q_embs, q_err = generate_embeddings([query])
            if q_embs and len(q_embs) > 0:
                rel_chunks, best_sim, v_err = self.vector_search(
                    q_embs[0], k=k, threshold=0.30, source_filter=source_filter
                )
                if not v_err:
                    return rel_chunks, best_sim, None
            elif q_err:
                print(f"[Retrieval] EmbeddingGemma 2 query embedding notice: {q_err}. Falling back to BM25.")

        # 2. BM25 similarity search
        try:
            target_chunks = candidates
            query_tokens = expand_query_terms(query, history=history)
            meaningful_query_tokens = [t for t in query_tokens if t not in STOPWORDS and len(t) > 1]
            if not meaningful_query_tokens:
                meaningful_query_tokens = query_tokens

            if not meaningful_query_tokens:
                return target_chunks[:k], 1.0, None

            def doc_tokenize(text: str):
                return re.findall(r"[\w-]+", text.lower())

            doc_tokens_list = [doc_tokenize(c.text) for c in target_chunks]
            num_docs = len(target_chunks)
            avg_doc_len = sum(len(d) for d in doc_tokens_list) / max(1, num_docs)

            # Inverted document frequency
            df = {}
            for d in doc_tokens_list:
                for t in set(d):
                    df[t] = df.get(t, 0) + 1

            k1 = 1.5
            b = 0.75
            scores = []

            for i, d in enumerate(doc_tokens_list):
                doc_len = len(d)
                score = 0.0
                term_counts = {}
                for t in d:
                    term_counts[t] = term_counts.get(t, 0) + 1

                matched_meaningful = 0
                for qt in meaningful_query_tokens:
                    if qt in term_counts:
                        matched_meaningful += 1
                        freq = term_counts[qt]
                        idf = math.log(1 + (num_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5))
                        idf = max(idf, 0.25)
                        denom = freq + k1 * (1 - b + b * (doc_len / max(1.0, avg_doc_len)))
                        score += idf * ((freq * (k1 + 1)) / max(0.001, denom))

                # Exact phrase matching bonus
                c_lower = target_chunks[i].text.lower()
                q_clean = query.lower().strip()
                if len(q_clean) > 3 and q_clean in c_lower:
                    score += 5.0
                    matched_meaningful += 1

                if matched_meaningful == 0:
                    score = 0.0

                scores.append((score, target_chunks[i]))

            scores.sort(key=lambda x: x[0], reverse=True)
            best_score = scores[0][0] if scores else 0.0

            # Filter by relevance threshold
            relevant = [item[1] for item in scores if item[0] >= threshold][:k]
            return relevant, best_score, None

        except Exception as e:
            err_msg = f"Retrieval failed: {str(e)}"
            print(f"[Store Error] {err_msg}")
            return [], 0.0, err_msg


# Global store instance
_store = DocumentStore()

# Global rolling conversation history: [{"role": "user"|"assistant", "content": "...", "lang": "..."}]
_conversation_history: list[dict] = []


def get_conversation_history() -> list[dict]:
    return list(_conversation_history)


def clear_conversation_history():
    global _conversation_history
    _conversation_history = []


def add_conversation_turn(user_msg: str, assistant_msg: str, lang: str):
    global _conversation_history
    _conversation_history.append({"role": "user", "content": user_msg, "lang": lang})
    _conversation_history.append({"role": "assistant", "content": assistant_msg, "lang": lang})
    # Keep last 6 turns (3 interactions)
    if len(_conversation_history) > 6:
        _conversation_history = _conversation_history[-6:]


def get_text_splitter():
    """Create chunk splitter configured for fast CPU SLM RAG (~300-400 tokens, ~1200-1400 chars)."""
    chunk_tokens = get_chunk_size()
    overlap_tokens = get_chunk_overlap()
    chunk_chars = max(300, chunk_tokens * 4)
    overlap_chars = max(40, overlap_tokens * 4)
    if RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_chars,
            chunk_overlap=overlap_chars,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    class FallbackSplitter:
        def split_text(self, text: str):
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_chars, len(text))
                chunks.append(text[start:end])
                start += max(100, chunk_chars - overlap_chars)
            return chunks
    return FallbackSplitter()


def process_document(file_path: str) -> int:
    """
    Extract text from PDF or TXT, split into overlapping chunks, and index.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    pages_data = []

    if ext == ".pdf":
        if HAS_PYPDF:
            try:
                reader = PdfReader(file_path)
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_data.append((page_text, page_idx))
            except Exception as e:
                raise ValueError(f"Failed to read PDF '{filename}': {str(e)}")
        else:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
                clean_text = raw_bytes.decode("latin-1", errors="ignore")
                pages_data.append((clean_text, 0))
    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
                content = f.read()
        pages_data.append((content, 0))
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    if not pages_data:
        raise ValueError(f"Document '{filename}' appears to be empty or contains no extractable text.")

    # Scanned / image-only PDF guard: require at least 50 meaningful characters total
    total_chars = sum(len(text.strip()) for text, _ in pages_data)
    if ext == ".pdf" and total_chars < 50:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "This PDF may be a scanned image. "
            "OCR software (such as Adobe Acrobat, Tesseract, or Google Docs) "
            "is required to extract text from image-based PDFs before uploading."
        )

    splitter = get_text_splitter()
    new_chunks = []

    for text, page_num in pages_data:
        sub_texts = splitter.split_text(text)
        for st in sub_texts:
            clean = st.strip()
            if clean:
                new_chunks.append(DocChunk(text=clean, source=filename, page=page_num))

    # Check if identical document is already indexed to avoid re-embedding unchanged documents
    clean_name = os.path.basename(filename)
    existing_chunks = [c for c in _store.chunks if c.source == clean_name and c.embedding is not None]
    if existing_chunks and len(existing_chunks) == len(new_chunks):
        if all(c.text == e.text for c, e in zip(new_chunks, existing_chunks)):
            print(f"[Store] Reusing existing EmbeddingGemma 2 embeddings for unchanged document '{clean_name}'.")
            return len(existing_chunks)

    # Generate embeddings using EmbeddingGemma 2 via Ollama
    texts = [c.text for c in new_chunks]
    embs, emb_err = generate_embeddings(texts)
    if embs and len(embs) == len(new_chunks):
        for idx, emb in enumerate(embs):
            new_chunks[idx].embedding = emb
        print(f"[Store] Generated {len(embs)} embeddings using '{get_embedding_model_name()}' for '{clean_name}'.")
    elif emb_err:
        print(f"[Store Warning] Embedding generation failed ({emb_err}). Storing text chunks for BM25 fallback.")

    _store.add_chunks(clean_name, new_chunks)
    clear_query_cache()
    print(f"Indexed {len(new_chunks)} chunks for '{clean_name}'")
    return len(new_chunks)


_last_invocation_error = {"ollama": None, "groq": None}


def check_ollama_status() -> dict:
    """Ping Ollama API to verify availability and list installed models."""
    base_url = get_ollama_base_url()
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "RAG-Chatbot"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            return {"online": True, "models": models, "error": None}
    except urllib.error.URLError as e:
        err_msg = str(e.reason) if hasattr(e, "reason") else str(e)
        print(f"[Ollama] Connection failed ({base_url}): {err_msg}")
        return {"online": False, "models": [], "error": err_msg}
    except Exception as e:
        print(f"[Ollama] Unexpected error checking status: {e}")
        return {"online": False, "models": [], "error": str(e)}


def get_active_model(status: dict | None = None) -> tuple[str, bool]:
    """
    Returns (model_name, is_available).
    Checks if configured MODEL_NAME is installed, or auto-falls back to the first available model.
    """
    configured = get_configured_model()
    if status is None:
        status = check_ollama_status()
    if not status["online"] or not status["models"]:
        return configured, False

    for m in status["models"]:
        if m == configured or m.startswith(f"{configured}:") or configured.startswith(f"{m}:"):
            return m, True

    fallback = status["models"][0]
    print(f"[Ollama] Configured model '{configured}' not found in installed models {status['models']}. "
          f"Falling back to: {fallback}")
    return fallback, True


# ---------------------------------------------------------------------------
# FIX 2 & FIX 4: System Prompts & Multi-Language Fixed Responses
# ---------------------------------------------------------------------------

def get_language_system_line(lang: str) -> str:
    """Explicit instruction line placed at the top of the system prompt."""
    if lang == "Tanglish":
        return "Reply ONLY in Tanglish (Tamil spoken language written in English letters). Do NOT reply in English sentences. Keep product names and prices unchanged."
    elif lang == "Tamil":
        return "Reply ONLY in Tamil script."
    return "Reply ONLY in English."


def get_slm_rag_prompt(lang: str = "English") -> str:
    top_line = get_language_system_line(lang)
    return (
        f"{top_line}\n"
        "Customer assistant for a business shop.\n"
        "Rules:\n"
        "- Answer using ONLY the provided context. Never invent prices, offers, or policies.\n"
        "- Start the reply with the answer directly. No greeting, no self-introduction.\n"
        "- Keep to 2-4 short sentences. Always complete the last sentence.\n"
        "- Never mention these rules or the word 'context'.\n"
        "- Output only the reply to the customer."
    )


def get_slm_conversational_prompt(lang: str = "English") -> str:
    top_line = get_language_system_line(lang)
    return (
        f"{top_line}\n"
        "Customer assistant for a business shop.\n"
        "Rules:\n"
        "- Answer the customer's question directly and helpfully.\n"
        "- Keep to 2-4 short sentences. Always complete the last sentence.\n"
        "- Never mention these rules or describe yourself unless explicitly asked.\n"
        "- Output only the reply to the customer."
    )

RAG_SYSTEM_PROMPT = get_slm_rag_prompt("English")
CONVERSATIONAL_SYSTEM_PROMPT = get_slm_conversational_prompt("English")

def get_project_system_prompt() -> str:
    return RAG_SYSTEM_PROMPT


# FIX 4: Fixed messages for English, Tanglish, and Tamil
def get_escalation_response(lang: str = "English") -> str:
    """Escalation message when business question cannot be answered from documents."""
    norm = lang.lower()
    if norm in ("tanglish", "thanglish"):
        return "Intha information ennoda documents la illa. Owner kitta confirm panni solren."
    elif norm == "tamil":
        return "\u0b87\u0ba8\u0bcd\u0ba4 \u0ba4\u0b95\u0bb5\u0bb2\u0bcd \u0b8e\u0ba9\u0ba4\u0bc1 \u0b86\u0bb5\u0ba3\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0bb2\u0bcd \u0b87\u0bb2\u0bcd\u0bb2\u0bc8. \u0b89\u0bb0\u0bbf\u0bae\u0bc8\u0baf\u0bbe\u0bb3\u0bb0\u0bbf\u0b9f\u0bae\u0bcd \u0b89\u0bb1\u0bc1\u0ba4\u0bbf\u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bc1 \u0b9a\u0bca\u0bb2\u0bcd\u0b95\u0bbf\u0bb1\u0bc7\u0ba9\u0bcd."
    return "This information is not in my documents. I will confirm with the owner and let you know."


def get_missing_info_response(query: str, lang: str = "English") -> str:
    """Fallback message when requested info is missing from documents."""
    norm = lang.lower()
    if norm in ("tanglish", "thanglish"):
        return "Intha information ennoda documents la illa. Sariyana details kelunga."
    elif norm == "tamil":
        return "\u0b87\u0ba8\u0bcd\u0ba4 \u0ba4\u0b95\u0bb5\u0bb2\u0bcd \u0b86\u0bb5\u0ba3\u0b99\u0bcd\u0b95\u0bb3\u0bbf\u0bb2\u0bcd \u0b87\u0bb2\u0bcd\u0bb2\u0bc8. \u0ba4\u0baf\u0bb5\u0bc1\u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bc1 \u0b95\u0bc2\u0b9f\u0bc1\u0ba4\u0bb2\u0bcd \u0bb5\u0bbf\u0bb5\u0bb0\u0b99\u0bcd\u0b95\u0bb3\u0bc8 \u0bb5\u0bb4\u0b99\u0bcd\u0b95\u0bb5\u0bc1\u0bae\u0bcd."
    return "This information is not available in the documents. Please ask with more details."


def get_retrieval_error_response(lang: str = "English") -> str:
    """Message when document retrieval encounters an error."""
    norm = lang.lower()
    if norm in ("tanglish", "thanglish"):
        return "Retrieval error kaaranamaaga documents-a access panna mudiyala. Marupadiyum try pannunga."
    elif norm == "tamil":
        return "\u0bae\u0bc0\u0b9f\u0bcd\u0b9f\u0bc6\u0b9f\u0bc1\u0baa\u0bcd\u0baa\u0bc1 \u0baa\u0bbf\u0bb4\u0bc8 \u0b95\u0bbe\u0bb0\u0ba3\u0bae\u0bbe\u0b95 \u0b86\u0bb5\u0ba3\u0b99\u0bcd\u0b95\u0bb3\u0bc8 \u0b85\u0ba3\u0bc1\u0b95 \u0bae\u0bc1\u0b9f\u0bbf\u0baf\u0bb5\u0bbf\u0bb2\u0bcd\u0bb2\u0bc8. \u0ba4\u0baf\u0bb5\u0bc1\u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bc1 \u0bae\u0bc0\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd \u0bae\u0bc1\u0baf\u0bb1\u0bcd\u0b9a\u0bbf\u0b95\u0bcd\u0b95\u0bb5\u0bc1\u0bae\u0bcd."
    return "The documents could not be accessed due to a retrieval error. Please retry your query."

# ---------------------------------------------------------------------------
# Greetings, Identity & Deterministic Fallbacks
# ---------------------------------------------------------------------------

def is_identity_question(query: str) -> bool:
    """Check if query is asking who/what the bot is or what it can do."""
    q_clean = query.strip().lower()
    patterns = [
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\bwhat model\b",
        r"\bwhat kind of model\b",
        r"\bare you (an? )?(llm|slm|large language model|small language model)\b",
        r"\b(are you|aren't you) a (llm|slm|large language model|small language model)\b",
        r"\b(slm or llm|llm or slm)\b",
        r"\btell me about yourself\b",
        r"\bwhat is your (purpose|role|name)\b",
        r"\bwhat can you do\b",
        r"\bwhat is your purpose\b",
        r"\b(nee|neenga) (yaaru?|yaar)\b",

        r"நீ யார்|நீங்கள் யார்|உன் பெயர்|உங்கள் பெயர்"
    ]
    return any(re.search(pat, q_clean) for pat in patterns)


def get_identity_response(lang: str = "english") -> str:
    """Short, direct self-introduction in the customer's language style."""
    if lang == "tamil":
        return "நான் ஒரு chatbot assistant. உங்கள் products, price, offers, delivery பற்றி கேளுங்கள், நான் help பண்றேன்."
    elif lang == "thanglish":
        return "Naan oru chatbot assistant. Engal products, price, offers, delivery pathi ennaiya kelunga, naan help panren."
    return "I am a chatbot assistant. I can help you with our products, prices, offers and delivery."


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CHANGE 1: Instant Greeting Handler (Regex with repeated letters, No LLM, < 1ms)
# ---------------------------------------------------------------------------

def get_greeting_reply(query: str) -> str | None:
    """
    Detect greetings with regex supporting repeated letters and variants:
    hi, hii, hiii, hello, helloo, hey, heyy, hai, haii, vanakkam, vanakam,
    good morning, good afternoon, good evening, gm, hlo, hola.
    Matches the whole message after trimming spaces and punctuation.
    Returns a fixed ONE-LINE reply instantly (no LLM, no retrieval, under 100ms):
      - English: "Hello! How can I help you?" (or "Hello! Welcome to {BUSINESS_NAME}. How can I help you?")
      - Tanglish / Tamil: "Vanakkam! Enna help venum?"
      - Good morning/afternoon/evening: "Good {word}! How can I help you?"
    Returns None if not a greeting.
    """
    clean = re.sub(r'^[^\w\s஀-௿]+|[^\w\s஀-௿]+$', '', query.strip()).strip().lower()
    clean = re.sub(r'\s+', ' ', clean)
    if not clean:
        return None

    # Allowed optional short suffix like 'bro', 'pa', 'there', 'all', 'team', 'sir', 'ji'
    suffix = r'(?:[\s,]+(bro|pa|there|all|team|sir|ji|friend|everyone))?$'

    # 1. Good morning / afternoon / evening / gm
    m_time = re.match(r'^(good\s+(morning|afternoon|evening)|gm)' + suffix, clean)
    if m_time:
        matched_str = m_time.group(1)
        if matched_str == 'gm' or 'morning' in matched_str:
            time_word = 'morning'
        elif 'afternoon' in matched_str:
            time_word = 'afternoon'
        else:
            time_word = 'evening'
        return f"Good {time_word}! How can I help you?"

    # 2. Tanglish / Tamil greeting (vanakkam, vanakam, வணக்கம், etc.)
    m_tamil = re.match(r'^(vanak+a+m|வணக்கம்|காலை\s+வணக்கம்|மாலை\s+வணக்கம்)' + suffix, clean)
    if m_tamil:
        return "Vanakkam! Enna help venum?"

    # 3. English greeting (hi, hii, hiii, hello, helloo, hey, heyy, hai, haii, hlo, hola)
    m_eng = re.match(r'^(h+i+|h+e+l+l*o+|h+e+y+|h+a+i+|h+l+o+|h+o+l+a+)' + suffix, clean)
    if m_eng:
        biz_name = os.getenv("BUSINESS_NAME", "").strip()
        if biz_name:
            return f"Hello! Welcome to {biz_name}. How can I help you?"
        return "Hello! How can I help you?"

    return None

# CHANGE 1: Instant Greeting & Small Talk Handler (< 100 ms, No LLM / No Store)
# ---------------------------------------------------------------------------

def classify_smalltalk(query: str) -> tuple[bool, str | None, str | None]:
    """
    Detect greetings and small talk with simple rules (regex/keyword list, no LLM call).
    Supports English, Tamil, and Tanglish.
    Returns (is_smalltalk, fixed_reply, category).
    Runs in under 1 ms with zero model or retrieval overhead.
    """
    q_raw = query.strip()
    q_clean = q_raw.lower()
    q_norm = re.sub(r"[^\w\s\u0B80-\u0BFF-]", " ", q_clean).strip()
    q_norm = re.sub(r"\s+", " ", q_norm)

    if not q_norm:
        return False, None, None

    lang = detect_language_style(q_raw)

    # 0. Primary greeting check (instant regex matching)
    greet_reply = get_greeting_reply(q_raw)
    if greet_reply:
        return True, greet_reply, "greeting"

    # 1. Thanks variations
    thanks_patterns = [
        r"^(thanks|thank you|thank u|thx|tq|many thanks|thanks a lot|thanks bro|thanks pa)$",
        r"\b(thank you|thanks a lot|many thanks|thanks so much)\b",
        r"^(nandri|romba nandri|mikka nandri|நன்றி|மிக்க நன்றி)$",
        r"\b(romba nandri|mikka nandri)\b"
    ]
    if any(re.search(p, q_norm) for p in thanks_patterns):
        if lang == "tamil":
            return True, "மிக்க மகிழ்ச்சி! (You're welcome!)", "thanks"
        return True, "You're welcome!", "thanks"

    # 2. Bye variations
    bye_patterns = [
        r"^(bye|goodbye|bye bye|see you|cya|take care|good night|bye bro|bye pa)$",
        r"\b(bye bye|good bye|see you later)\b",
        r"^(poittu varen|kelambren|varata|போய் வருகிறேன்|சென்று வருகிறேன்)$",
        r"\b(poittu varen|varata bro|bye bro)\b"
    ]
    if any(re.search(p, q_norm) for p in bye_patterns):
        if lang == "tamil":
            return True, "சென்று வாருங்கள், இனிய நாளாக அமையட்டும்!", "bye"
        return True, "Bye, have a nice day!", "bye"

    # 3. Ok / Acknowledgement variations
    ok_patterns = [
        r"^(ok|okay|okk|k|kk|alright|fine|cool|sure|got it|done)$",
        r"^(seri|seripa|seri bro|aama|sari|சரி|ஆமாம்|ஓகே)$"
    ]
    if any(re.search(p, q_norm) for p in ok_patterns):
        if lang == "tamil":
            return True, "சரி! உங்களுக்கு வேறு ஏதேனும் உதவி தேவைப்பட்டால் சொல்லுங்கள்.", "ok"
        return True, "Alright! Let me know if you need anything else.", "ok"

    # 4. Greetings
    greeting_patterns = [
        r"^(hi|hii|hiii|hello|hey|howdy|greetings|hola|heya|yo)$",
        r"^(good morning|good afternoon|good evening|good day)$",
        r"^(how are you|how are you doing|how do you do|how r u|how are u)$",
        r"^(what'?s up|sup)$",
        r"^(vanakkam|vaazhthukkal|vanakkam bro|hello bro|hi bro)$",
        r"^(epdi irukinga|eppadi irukkeenga|eppadi irukkinga|nalla irukingala)$",
        r"^(வணக்கம்|காலை வணக்கம்|மாலை வணக்கம்|எப்படி இருக்கிறீர்கள்|நலமா)$",
        r"\b(good morning|good evening|good afternoon|how are you)\b",
        r"\b(vanakkam|epdi irukinga|eppadi irukkeenga)\b"
    ]
    if any(re.search(p, q_norm) for p in greeting_patterns):
        if lang == "tamil":
            return True, "வணக்கம்! நான் உங்கள் chatbot assistant. உங்களுக்கு எப்படி உதவ முடியும்?", "greeting"
        elif lang == "thanglish" or "vanakkam" in q_norm:
            return True, "Vanakkam! I'm your chatbot assistant. How can I help you?", "greeting"
        return True, "Hello! I'm your chatbot assistant. How can I help you?", "greeting"

    return False, None, None


def is_greeting(query: str) -> bool:
    """Check if query is a greeting."""
    return get_greeting_reply(query) is not None

def is_about_question(query: str) -> bool:
    """
    Detect questions asking what the bot/document is about,
    what it can explain, or what the business does.
    These are routed to a RAG summary call, NOT a fixed response.
    """
    q = query.strip().lower()
    patterns = [
        r"\bnee edha pathi\b",
        r"\bnee ethu pathi\b",
        r"\bnee edhukku\b",
        r"\bwhat (is this|is it) about\b",
        r"\bwhat can you (tell|explain|say)\b",
        r"\bwhat do you (do|explain|cover|know about)\b",
        r"\bwhat (are you|is this)\b",
        r"\btell me about (this|yourself|your (shop|store|business))\b",
        r"\babout (this|your (shop|store|business))\b",
        r"\bivan (edha|edhula) pathi\b",
    ]
    return any(re.search(p, q) for p in patterns)


def get_greeting_response(lang: str = "english") -> str:
    """Fixed greeting response under 100ms."""
    if lang in ("tamil", "thanglish"):
        return get_greeting_reply("vanakkam") or "Vanakkam! Enna help venum?"
    return get_greeting_reply("hi") or "Hello! How can I help you?"

# ---------------------------------------------------------------------------
# CHANGE 2: Business Intent Detection (Price, Stock, Policy, Delivery, etc.)
# ---------------------------------------------------------------------------

BUSINESS_PATTERNS = [
    # English keywords
    r"\b(price|prices|pricing|rate|rates|cost|costs|charge|charges|fee|fees)\b",
    r"\b(offer|offers|deal|deals|discount|discounts|coupon|coupons|promo|promotion)\b",
    r"\b(delivery|deliver|delivering|shipping|ship|courier|dispatch)\b",
    r"\b(timing|timings|hours|schedule|opening|closing|open|close|working days)\b",
    r"\b(address|location|where is|landmark|store location|shop address|contact)\b",
    r"\b(policy|policies|terms|condition|conditions)\b",
    r"\b(refund|refunds|return|returns|cancellation|cancel|money back|exchange)\b",
    r"\b(availability|available|in stock|out of stock|stock|stocks)\b",
    r"\b(order|orders|ordering|buy|purchase|cart)\b",
    r"\b(product|products|item|items|catalog|catalogue|menu)\b",
    r"\b(warranty|guarantee|herbal|soap|oil|powder|tea|moringa|triphala|neem)\b",
    r"\b(payment|upi|gpay|cod|cash on delivery)\b",
    # Tanglish keywords
    r"\b(vilai|vila|rate-u|rateu|evvalavu|evalo|ethana|panam|kaasu|kassu)\b",
    r"\b(offera|offer-a|thallubadi|thallubadiye)\b",
    r"\b(anupa|anupuvangala|anuppu|varum|eppo varum)\b",
    r"\b(neram|thirappu|moodum|kadai|enga irukku|edam|engayavathu)\b",
    r"\b(thirumba|thirumbi|kaasu thirumba)\b",
    r"\b(vaanga|vaangalam|kedaikuma|kidaikkuma|kidaikuma|irukka|iruka|porul)\b",
    # Tamil keywords
    r"(விலை|கட்டணம்|மதிப்பு|சலுகை|தள்ளுபடி|டெலிவரி|நேரம்|முகவரி|இருப்பிடம்|பொருட்கள்|பொருள்|இருப்பு|ஆர்டர்|பணம் திரும்ப|ரீஃபண்ட்)"
]

def is_business_question(query: str) -> bool:
    """Detect whether the query is related to business matters (pricing, products, orders, timings, etc.)."""
    q = query.lower()
    return any(re.search(pat, q) for pat in BUSINESS_PATTERNS)


def get_missing_info_response(query: str, lang: str = "english") -> str:
    """
    Polite response when information cannot be found in the current documentation.
    Meets Requirement 2 with dynamic language and style matching.
    """
    if lang == "tamil":
        return "கிடைக்கக்கூடிய project documentation-ல் அந்தத் தகவல் கிடைக்கவில்லை. சம்பந்தப்பட்ட document-ஐ upload செய்யுங்கள் அல்லது கூடுதல் விவரங்களைத் தாருங்கள்."
    elif lang == "thanglish":
        return "Available project documentation-la andha information kidaikkala bro. Relevant document-a upload pannunga illa konjam details share pannunga."
    return "I couldn't find that information in the available project documentation. Please upload the relevant document or provide more details."


def get_retrieval_error_response(lang: str = "english") -> str:
    """
    Polite response when document retrieval itself fails.
    Meets Requirement 2: Do not pretend checked, explain error and suggest retry.
    """
    if lang == "tamil":
        return "ஒரு retrieval பிழை காரணமாக project documentation-ஐ அணுக முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்."
    elif lang == "thanglish":
        return "Retrieval error kaaranamaaga project documentation-a access panna mudiyala. Dayavu seidhu marupadiyum try pannunga."
    return "The project documentation could not be accessed due to a retrieval error. Please retry your query."


def clean_model_output(text: str, target_lang: str) -> str:
    """Strips parenthetical translation artifacts and prompt-leak sentences."""
    if not text:
        return text
    clean = text.strip()
    # Strip (Translation: ...) or (This project is ...) if model appended it for non-English styles
    if target_lang in ["tamil", "thanglish"]:
        clean = re.sub(r"\n*\((Translation|This project|Note):?[^\)]+\)\s*$", "", clean, flags=re.IGNORECASE)
    # Strip prompt-leak sentences
    _leak_patterns = [
        r"[^.!?\n]*I will only use[^.!?\n]*[.!?]",
        r"[^.!?\n]*designed to answer[^.!?\n]*[.!?]",
        r"[^.!?\n]*based on the provided documentation[^.!?\n]*[.!?]",
        r"[^.!?\n]*based on the provided context[^.!?\n]*[.!?]",
        r"[^.!?\n]*2-4 sentences[^.!?\n]*[.!?]",
        r"[^.!?\n]*2 to 4 sentences[^.!?\n]*[.!?]",
        r"[^.!?\n]*I will respond in[^.!?\n]*[.!?]",
        r"[^.!?\n]*I am an assistant designed[^.!?\n]*[.!?]",
        r"[^.!?\n]*I am a chatbot designed[^.!?\n]*[.!?]",
        r"[^.!?\n]*the context provided[^.!?\n]*[.!?]",
    ]
    for pattern in _leak_patterns:
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
    # Collapse extra whitespace/newlines introduced by removals
    clean = re.sub(r'\n{3,}', '\n\n', clean)
    clean = re.sub(r'  +', ' ', clean)
    return clean.strip()



# ---------------------------------------------------------------------------
# Ollama and Groq Invocation
# ---------------------------------------------------------------------------

def _invoke_ollama(prompt: str, system: str | None = None) -> tuple[str | None, str | None]:
    """
    Invoke Ollama with automatic model detection, keep_alive=60m, system prompt support,
    and graceful fallbacks. Returns (response_text, error_message).
    """
    global _last_invocation_error
    _last_invocation_error["ollama"] = None

    base_url = get_ollama_base_url()
    status = check_ollama_status()
    if not status["online"]:
        err = f"Cannot connect to Ollama at {base_url} ({status.get('error', 'connection refused')})"
        _last_invocation_error["ollama"] = err
        return None, err

    if not status["models"]:
        configured = get_configured_model()
        err = f"No models downloaded in Ollama yet. Run `ollama pull {configured}` to install it."
        _last_invocation_error["ollama"] = err
        return None, err

    model, is_available = get_active_model(status)
    if not is_available:
        err = f"Configured model '{get_configured_model()}' is not available."
        _last_invocation_error["ollama"] = err
        return None, err

    # 1. Direct HTTP call to Ollama /api/generate
    try:
        url = f"{base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": get_keep_alive(),
            "options": {
                "num_ctx": get_num_ctx(),
                "num_predict": get_num_predict(),
                "temperature": get_temperature()
            }
        }
        if system:
            payload["system"] = system

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "RAG-Chatbot"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            res = data.get("response", "").strip()
            if res:
                return res, None
            else:
                err = f"Model '{model}' returned an empty response."
                _last_invocation_error["ollama"] = err
                return None, err
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        err = f"Ollama HTTP error {e.code}: {body[:200]}"
        print(f"[Ollama] {err}")
        _last_invocation_error["ollama"] = err
    except Exception as e:
        err = f"Ollama generation error: {str(e)}"
        print(f"[Ollama] {err}")
        _last_invocation_error["ollama"] = err

    # 2. Fallback to LangChain OllamaLLM if direct HTTP failed
    if HAS_LANGCHAIN_OLLAMA:
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            llm = OllamaLLM(
                model=model,
                base_url=base_url,
                sync_client_kwargs={"timeout": 120},
                num_ctx=get_num_ctx(),
                num_predict=get_num_predict(),
                temperature=get_temperature(),
                keep_alive=get_keep_alive()
            )
            res = llm.invoke(full_prompt)
            if res and res.strip():
                return res.strip(), None
        except Exception as e:
            print(f"[Ollama] LangChain invoke notice ({e})")

    return None, _last_invocation_error["ollama"]


def _stream_ollama(prompt: str, system: str | None = None):
    """Stream tokens directly from Ollama /api/generate line-by-line."""
    base_url = get_ollama_base_url()
    status = check_ollama_status()
    if not status["online"] or not status["models"]:
        return

    model, is_available = get_active_model(status)
    if not is_available:
        return

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": get_keep_alive(),
        "options": {
            "num_ctx": get_num_ctx(),
            "num_predict": get_num_predict(),
            "temperature": get_temperature()
        }
    }
    if system:
        payload["system"] = system

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "RAG-Chatbot"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if line.strip():
                item = json.loads(line.decode("utf-8"))
                token = item.get("response", "")
                if token:
                    yield token
                if item.get("done", False):
                    break


def _invoke_groq(prompt: str) -> tuple[str | None, str | None]:
    """Fallback: call Groq API when Ollama has no models available."""
    global _last_invocation_error
    _last_invocation_error["groq"] = None

    api_key = get_groq_api_key()
    if not api_key:
        return None, None

    model = get_groq_fallback_model()
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "RAG-Chatbot"
            }
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                print(f"[Groq] Response via {model}")
                return content, None
            return None, "Empty response from Groq."
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        if e.code == 401:
            err = "Groq authentication failed (401 Unauthorized): invalid GROQ_API_KEY."
        elif e.code == 429:
            err = "Groq rate limit exceeded (429 Too Many Requests)."
        else:
            err = f"Groq HTTP error {e.code}: {body[:200]}"
        print(f"[Groq] {err}")
        _last_invocation_error["groq"] = err
        return None, err
    except Exception as e:
        err = f"Groq error: {str(e)}"
        print(f"[Groq] {err}")
        _last_invocation_error["groq"] = err
        return None, err


# ---------------------------------------------------------------------------

# FIX 3: Verification set and two-step rewrite fallback for Tanglish
_ALL_TANGLISH_RESPONSE_WORDS = TANGLISH_INDICATOR_WORDS | {
    "aama", "aamam", "illa", "illai", "irukku", "iruku", "irukkum",
    "kedaikkum", "kidaikkum", "kedaikuma", "kidaikuma", "venuma",
    "solren", "solranga", "panren", "pannuvom", "kulla", "mattum",
    "romba", "konjam", "bro", "pa", "nga", "edhu", "idhu", "adhu",
    "ithu", "athu", "idhula", "adhula", "dhan", "thaan", "varum",
    "irundha", "kooda", "engal", "kitta", "kelunga", "panni", "thara"
}

def is_fully_english_reply(text: str) -> bool:
    """
    FIX 3: Returns True if text has NO Tanglish words and NO Tamil characters.
    Used only when Tanglish was detected, to trigger Step 2 rewrite.
    """
    if not text or not text.strip():
        return False
    if re.search(r"[\u0B80-\u0BFF]", text):
        return False
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if tokens & _ALL_TANGLISH_RESPONSE_WORDS:
        return False
    return True


def ensure_tanglish_reply(answer: str, detected_lang: str) -> tuple[str, bool]:
    """
    FIX 3: Two-step fallback for reliability (only when Tanglish is detected).
    Step 1: Generated normally from context.
    Step 2: If answer is fully English, make a second short model call to rewrite into Tanglish.
    Controlled by LANGUAGE_RETRY in .env.
    """
    if detected_lang != "Tanglish" or not is_language_retry_enabled() or not answer:
        return answer, False

    if not is_fully_english_reply(answer):
        return answer, False

    # Step 2 Rewrite call
    rewrite_prompt = (
        f"Rewrite this reply in Tanglish (Tamil spoken language in English letters). "
        f"Keep numbers, prices and product names exactly the same. "
        f"Reply only with the rewritten text: {answer}"
    )
    rewrite_sys = (
        "Translate English to Tanglish (Tamil spoken language in English letters). "
        "Keep product names, numbers and prices exactly the same. "
        "Output only the rewritten text."
    )
    rewritten, _ = _invoke_ollama(rewrite_prompt, system=rewrite_sys)
    if rewritten and rewritten.strip():
        return rewritten.strip(), True

    return answer, False


# ---------------------------------------------------------------------------
# Main Question & Answering Pipeline
# ---------------------------------------------------------------------------

def build_prompt_with_context(
    question: str,
    context_text: str,
    lang: str,
    history: list[dict] | None = None
) -> str:
    """
    FIX 2: Constructs prompt with context, question, and trailing language instruction.
    Removes topic-specific few-shot examples so model never copies them.
    Places instruction at the END of user message for maximum model compliance.
    """
    history_str = ""
    if history:
        recent_turns = history[-2:]
        lines = []
        for turn in recent_turns:
            role = "Customer" if turn.get("role") == "user" else "Assistant"
            c = turn.get("content", "").replace("\n", " ").strip()
            if c:
                lines.append(f"{role}: {c}")
        if lines:
            history_str = "Recent Chat:\n" + "\n".join(lines) + "\n\n"

    reply_lang = detect_reply_language(question)

    if reply_lang == "Tanglish":
        trailing_instruction = (
            "Answer the question using only the context above. "
            "Write the answer in Tanglish: Tamil spoken language written in English letters "
            "(for example: 'Aama, irukku', 'Rs. 249 aagum', 'Trichy ku next day delivery'). "
            "Do NOT write in English sentences. Keep product names and prices unchanged."
        )
    elif reply_lang == "Tamil":
        trailing_instruction = (
            "Answer the question using only the context above in Tamil script. "
            "Keep product names and prices unchanged."
        )
    else:
        trailing_instruction = "Answer in English."

    return (
        f"Verified Context:\n{context_text}\n\n"
        f"{history_str}"
        f"Customer Question: {question}\n\n"
        f"{trailing_instruction}"
    )


def is_general_conversation(query: str) -> bool:
    """Check if query is general conversational chat rather than a business question."""
    is_st, _, _ = classify_smalltalk(query)
    if is_st:
        return True
    return not is_business_question(query)


def build_conversational_prompt(question: str, lang: str, history: list[dict] | None = None) -> str:
    """Constructs conversational prompt with trailing language instruction at the end."""
    history_str = ""
    if history:
        recent_turns = history[-2:]
        lines = []
        for turn in recent_turns:
            role = "Customer" if turn.get("role") == "user" else "Assistant"
            c = turn.get("content", "").replace("\n", " ").strip()
            if c:
                lines.append(f"{role}: {c}")
        if lines:
            history_str = "Recent Chat:\n" + "\n".join(lines) + "\n\n"

    reply_lang = detect_reply_language(question)

    if reply_lang == "Tanglish":
        trailing_instruction = (
            "Write the answer in Tanglish: Tamil spoken language written in English letters "
            "(for example: 'Aama, irukku', 'Rs. 249 aagum'). "
            "Do NOT write in English sentences. Answer helpfully and concisely."
        )
    elif reply_lang == "Tamil":
        trailing_instruction = "Answer helpfully and concisely in Tamil script."
    else:
        trailing_instruction = "Answer in English."

    return (
        f"{history_str}"
        f"Customer Question: {question}\n\n"
        f"{trailing_instruction}"
    )

# CHANGE 3: Model Warmup (Runs ONCE at server startup)
# ---------------------------------------------------------------------------

def warmup_models():
    """Warms up the embedding model and Gemma 3 once at startup with a dummy prompt."""
    print("[Startup] Warming up AI models in RAM (keep_alive=30m)...")
    t0 = time.perf_counter()
    # 1. Warm up embedding model
    try:
        generate_embeddings(["warmup"])
    except Exception as e:
        print(f"[Startup Warning] Embedding warmup notice: {e}")

    # 2. Warm up generation model
    try:
        base_url = get_ollama_base_url()
        status = check_ollama_status()
        if status["online"] and status["models"]:
            model, _ = get_active_model(status)
            url = f"{base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": "hi",
                "stream": False,
                "keep_alive": get_keep_alive(),
                "options": {
                    "num_ctx": get_num_ctx(),
                    "num_predict": 1,
                    "temperature": get_temperature()
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            print(f"[Startup] Models successfully warmed up in {time.perf_counter() - t0:.2f}s (RAM keep_alive={get_keep_alive()}).")
    except Exception as e:
        print(f"[Startup Warning] Model warmup notice: {e}")


# ---------------------------------------------------------------------------
# CHANGES 1, 2, & 3: Main ask_question Pipeline
# ---------------------------------------------------------------------------

def _safe_log_str(s: str) -> str:
    """Safe ASCII string for Windows console logging without UnicodeEncodeError."""
    return s.encode("ascii", errors="backslashreplace").decode("ascii")


def ask_question(
    question: str,
    history: list[dict] | None = None,
    mode: str = "auto",
    selected_doc: str | None = None
) -> dict:
    """
    Optimized RAG & Direct Response Pipeline:
    - Stage 0: Instant Cache check (< 1 ms).
    - CHANGE 1: Instant Greeting & Smalltalk without LLM/retrieval (< 100 ms).
    - CHANGE 2: ChromaDB search & SIMILARITY_THRESHOLD routing without extra LLM call.
      * score >= threshold -> 2-4 sentence grounded RAG answer from documents.
      * score < threshold & business query -> 'Intha information ennoda documents la illa...', escalate=True.
      * score < threshold & general query -> direct short LLM answer, escalate=False.
    - CHANGE 3: Fast CPU execution (3-8 seconds) with timing logs.
    """
    t_start = time.perf_counter()
    q = question.strip()
    if not q:
        return {"answer": "Please enter a valid question.", "sources": [], "mode": "direct", "escalate": False}

    # 1. Instant Greeting Check at the VERY START (no LLM, no retrieval, under 100ms)
    greeting_reply = get_greeting_reply(q)
    if greeting_reply:
        t_total = time.perf_counter() - t_start
        print(f"[TIMING] Query: '{_safe_log_str(q)}' | Stage: Greeting (Instant) | Total: {t_total*1000:.1f}ms")
        res = {"answer": greeting_reply, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, greeting_reply, detect_language_style(q))
        return res

    lang = detect_language_style(q)

    # 1. In-memory cache check (< 1 ms)
    cached = get_cached_response(q)
    if cached is not None:
        t_total = time.perf_counter() - t_start
        print(f"[TIMING] Query: '{_safe_log_str(q)}' | Stage: Cache Hit | Total: {t_total*1000:.1f}ms")
        return cached

    # 2. CHANGE 1: Greeting and small talk handled instantly without LLM or retrieval (< 100 ms)
    is_st, st_reply, _ = classify_smalltalk(q)
    if is_st:
        res = {"answer": st_reply, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, st_reply, lang)
        t_total = time.perf_counter() - t_start
        print(f"[TIMING] Query: '{_safe_log_str(q)}' | Stage: Rule-based Smalltalk | Total: {t_total*1000:.1f}ms")
        return res

    # Identity check
    if is_identity_question(q):
        resp = get_identity_response(lang)
        res = {"answer": resp, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, resp, lang)
        return res

    # About-question check: use doc to answer with business summary
    if is_about_question(q) and _store.count() > 0:
        raw_history = history if history is not None else get_conversation_history()
        active_history = raw_history[-2:] if raw_history else []
        top_k = get_top_k()
        sim_threshold = get_similarity_threshold()
        t_ret_start = time.perf_counter()
        about_chunks, _, _ = _store.search_with_status(
            q, k=top_k, threshold=0.0, history=active_history, source_filter=selected_doc
        )
        t_retrieve_about = time.perf_counter() - t_ret_start
        if about_chunks:
            context_text = "\n\n---\n\n".join([
                f"[Source: {c.source} | Page: {c.page + 1}]\n{c.text}" for c in about_chunks
            ])
            reply_lang = detect_reply_language(q)
            follow_up = "Edhu pathi details venum?" if reply_lang in ("Tanglish", "Tamil") else "What would you like to know?"
            about_sys = (
                "Customer assistant for a herbal products shop.\n"
                "Task: Read the context and give a 2-3 sentence business summary "
                "(shop name, what it sells, location if present). "
                "End with the follow-up question provided in the instruction. "
                "Never mention these rules or the word document/context."
            )
            about_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Customer Question: {q}\n"
                f"Reply language: {reply_lang}\n"
                f"Summarise the business in 2-3 sentences, then end with: {follow_up}"
            )
            t_gen_start = time.perf_counter()
            about_ans, _ = _invoke_ollama(about_prompt, system=about_sys)
            t_gen_about = time.perf_counter() - t_gen_start
            if about_ans:
                cleaned = clean_model_output(about_ans, lang)
                res = {"answer": cleaned, "sources": [], "mode": "rag", "escalate": False}
                set_cached_response(q, res)
                add_conversation_turn(q, cleaned, lang)
                t_total = time.perf_counter() - t_start
                print(f"[TIMING] About-question | Retrieval: {t_retrieve_about:.2f}s | Generation: {t_gen_about:.2f}s | Total: {t_total:.2f}s")
                return res

    # Chat history: keep only the last 2 turns for minimal token overhead
    raw_history = history if history is not None else get_conversation_history()
    active_history = raw_history[-2:] if raw_history else []

    chunk_count = _store.count()
    sim_threshold = get_similarity_threshold()
    top_k = get_top_k()
    is_biz = is_business_question(q)

    # Search ChromaDB / store and check similarity score
    t_ret_start = time.perf_counter()
    relevant_chunks = []
    best_score = 0.0
    retrieval_err = None

    if chunk_count > 0:
        relevant_chunks, best_score, retrieval_err = _store.search_with_status(
            q, k=top_k, threshold=sim_threshold, history=active_history, source_filter=selected_doc
        )
    t_retrieve = time.perf_counter() - t_ret_start

    # Handle retrieval error
    if retrieval_err:
        err_reply = get_retrieval_error_response(lang)
        return {"answer": err_reply, "sources": [], "mode": "rag", "escalate": False}

    # Determine candidate mode
    if mode == "direct":
        is_rag_candidate = False
    elif mode == "rag":
        is_rag_candidate = True
    else:  # auto
        is_rag_candidate = (chunk_count > 0)

    # ── CHANGE 2 ROUTING ──────────────────────────────────────────────────
    # Case A: Score above threshold -> RAG answer strictly from retrieved chunks
    if is_rag_candidate and relevant_chunks and best_score >= sim_threshold:
        sources = []
        for c in relevant_chunks:
            page_str = f" (page {c.page + 1})" if isinstance(c.page, int) and c.page >= 0 else ""
            source_label = f"{c.source}{page_str}"
            if source_label not in sources:
                sources.append(source_label)

        context_text = "\n\n---\n\n".join([
            f"[Source: {c.source} | Page: {c.page + 1}]\n{c.text}" for c in relevant_chunks
        ])
        detected_lang = detect_reply_language(q)
        sys_prompt = get_slm_rag_prompt(detected_lang)
        prompt = build_prompt_with_context(q, context_text, detected_lang, history=active_history)

        t_gen_start = time.perf_counter()
        ollama_answer, ollama_err = _invoke_ollama(prompt, system=sys_prompt)
        
        # FIX 3: Two-step fallback for Tanglish reliability
        ollama_answer, retried = ensure_tanglish_reply(ollama_answer, detected_lang)
        t_gen = time.perf_counter() - t_gen_start
        print(f"[LANGUAGE] Query: '{_safe_log_str(q)}' | Detected: {detected_lang} | Step 2 Retry: {retried}")

        if ollama_answer:
            cleaned = clean_model_output(ollama_answer, lang)
            res = {"answer": cleaned, "sources": sources, "mode": "rag", "escalate": False}
            set_cached_response(q, res)
            add_conversation_turn(q, cleaned, lang)
            t_total = time.perf_counter() - t_start
            print(f"[TIMING] Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Generation: {t_gen:.2f}s | Total: {t_total:.2f}s | Top Score: {best_score:.3f} >= {sim_threshold}")
            return res

        # Fallback to groq if local ollama failed
        groq_ans, _ = _invoke_groq(f"System: {sys_prompt}\n\n{prompt}")
        if groq_ans:
            cleaned = clean_model_output(groq_ans, lang)
            res = {"answer": cleaned, "sources": sources, "mode": "rag", "escalate": False}
            set_cached_response(q, res)
            add_conversation_turn(q, cleaned, lang)
            return res

    # Case B: Score below threshold but looks business-related -> Escalate immediately in detected language
    if is_biz:
        detected_lang = detect_reply_language(q)
        escalate_msg = get_escalation_response(detected_lang)
        res = {"answer": escalate_msg, "sources": [], "mode": "rag", "escalate": True}
        set_cached_response(q, res)
        add_conversation_turn(q, escalate_msg, detected_lang)
        t_total = time.perf_counter() - t_start
        print(f"[TIMING] Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Generation: 0.00s | Total: {t_total:.2f}s | Score: {best_score:.3f} < {sim_threshold} (Escalated in {detected_lang})")
        return res

    # Explicit RAG requested but information not found
    if mode == "rag":
        dynamic_prompt = (
            f"The user asked: {q}\n"
            f"STATUS: This specific topic was not found in the verified project documentation.\n"
            f"TARGET LANGUAGE STYLE: {lang}.\n"
            f"INSTRUCTION: In 1 or 2 sentences in {lang}, politely inform the user that this information "
            f"is not available in the project documentation and suggest uploading the relevant document or providing more details."
        )
        ollama_res, _ = _invoke_ollama(dynamic_prompt, system=get_slm_rag_prompt())
        missing_reply = clean_model_output(ollama_res, lang) if ollama_res else get_missing_info_response(q, lang)
        res = {"answer": missing_reply, "sources": [], "mode": "rag", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, missing_reply, lang)
        return res

    # Case C: Score below threshold & general question (not business) -> Direct model answer
    detected_lang = detect_reply_language(q)
    conv_prompt = build_conversational_prompt(q, detected_lang, history=active_history)
    conv_sys = get_slm_conversational_prompt(detected_lang)

    t_gen_start = time.perf_counter()
    ollama_ans, ollama_err = _invoke_ollama(conv_prompt, system=conv_sys)
    ollama_ans, retried = ensure_tanglish_reply(ollama_ans, detected_lang)
    t_gen = time.perf_counter() - t_gen_start
    print(f"[LANGUAGE] Query: '{_safe_log_str(q)}' | Detected: {detected_lang} | Step 2 Retry: {retried}")

    if ollama_ans:
        cleaned = clean_model_output(ollama_ans, lang)
        res = {"answer": cleaned, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, cleaned, lang)
        t_total = time.perf_counter() - t_start
        print(f"[TIMING] Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Generation: {t_gen:.2f}s | Total: {t_total:.2f}s (Direct LLM)")
        return res

    # Fallback to groq
    groq_ans, _ = _invoke_groq(f"System: {conv_sys}\n\n{conv_prompt}")
    if groq_ans:
        cleaned = clean_model_output(groq_ans, lang)
        res = {"answer": cleaned, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, cleaned, lang)
        return res

    missing_resp = get_missing_info_response(q, lang)
    res = {"answer": missing_resp, "sources": [], "mode": "direct", "escalate": False}
    return res


# ---------------------------------------------------------------------------
# Streaming Pipeline (Server-Sent Events)
# ---------------------------------------------------------------------------

def stream_ask_question(
    question: str,
    history: list[dict] | None = None,
    mode: str = "auto",
    selected_doc: str | None = None
):
    """
    Generator yielding Server-Sent Events (SSE) for streaming responses.
    - Yields initial token instantly for small-talk, cached queries, and escalations.
    - Yields model tokens as generated (max 120 tokens, CPU response within 3-8s).
    """
    t_start = time.perf_counter()
    q = question.strip()
    if not q:
        yield f"data: {json.dumps({'error': 'Please enter a valid question.'})}\n\n"
        return

    # 1. Instant Greeting Check at the VERY START (no LLM, no retrieval, under 100ms)
    greeting_reply = get_greeting_reply(q)
    if greeting_reply:
        t_total = time.perf_counter() - t_start
        print(f"[TIMING - Stream] Query: '{_safe_log_str(q)}' | Stage: Greeting (Instant) | Total: {t_total*1000:.1f}ms")
        res = {"answer": greeting_reply, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, greeting_reply, detect_language_style(q))
        yield f"data: {json.dumps({'token': greeting_reply})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
        return

    lang = detect_language_style(q)

    # 1. In-memory cache check
    cached = get_cached_response(q)
    if cached is not None:
        t_total = time.perf_counter() - t_start
        print(f"[TIMING - Stream] Query: '{_safe_log_str(q)}' | Cache Hit | Total: {t_total*1000:.1f}ms")
        yield f"data: {json.dumps({'token': cached.get('answer', '')})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': cached.get('sources', []), 'mode': cached.get('mode', 'direct'), 'escalate': cached.get('escalate', False)})}\n\n"
        return

    # 2. CHANGE 1: Instant Greeting & Small Talk (< 100 ms)
    is_st, st_reply, _ = classify_smalltalk(q)
    if is_st:
        res = {"answer": st_reply, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, st_reply, lang)
        t_total = time.perf_counter() - t_start
        print(f"[TIMING - Stream] Query: '{_safe_log_str(q)}' | Instant Smalltalk | Total: {t_total*1000:.1f}ms")
        yield f"data: {json.dumps({'token': st_reply})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
        return

    # Identity check
    if is_identity_question(q):
        resp = get_identity_response(lang)
        res = {"answer": resp, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, resp, lang)
        yield f"data: {json.dumps({'token': resp})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
        return

    # About-question check: stream business summary from doc
    if is_about_question(q) and _store.count() > 0:
        raw_history = history if history is not None else get_conversation_history()
        active_history = raw_history[-2:] if raw_history else []
        top_k = get_top_k()
        about_chunks, _, _ = _store.search_with_status(
            q, k=top_k, threshold=0.0, history=active_history, source_filter=selected_doc
        )
        if about_chunks:
            context_text = "\n\n---\n\n".join([
                f"[Source: {c.source} | Page: {c.page + 1}]\n{c.text}" for c in about_chunks
            ])
            reply_lang = detect_reply_language(q)
            follow_up = "Edhu pathi details venum?" if reply_lang in ("Tanglish", "Tamil") else "What would you like to know?"
            about_sys = (
                "Customer assistant for a herbal products shop.\n"
                "Task: Read the context and give a 2-3 sentence business summary "
                "(shop name, what it sells, location if present). "
                "End with the follow-up question provided in the instruction. "
                "Never mention these rules or the word document/context."
            )
            about_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Customer Question: {q}\n"
                f"Reply language: {reply_lang}\n"
                f"Summarise the business in 2-3 sentences, then end with: {follow_up}"
            )
            about_ans, _ = _invoke_ollama(about_prompt, system=about_sys)
            if about_ans:
                cleaned = clean_model_output(about_ans, lang)
                set_cached_response(q, {"answer": cleaned, "sources": [], "mode": "rag", "escalate": False})
                add_conversation_turn(q, cleaned, lang)
                yield f"data: {json.dumps({'token': cleaned})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': False})}\n\n"
                return

    raw_history = history if history is not None else get_conversation_history()
    active_history = raw_history[-2:] if raw_history else []

    chunk_count = _store.count()
    sim_threshold = get_similarity_threshold()
    top_k = get_top_k()
    is_biz = is_business_question(q)

    t_ret_start = time.perf_counter()
    relevant_chunks = []
    best_score = 0.0
    retrieval_err = None

    if chunk_count > 0:
        relevant_chunks, best_score, retrieval_err = _store.search_with_status(
            q, k=top_k, threshold=sim_threshold, history=active_history, source_filter=selected_doc
        )
    t_retrieve = time.perf_counter() - t_ret_start

    # Handle retrieval error
    if retrieval_err:
        err_msg = get_retrieval_error_response(lang)
        yield f"data: {json.dumps({'token': err_msg})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': False})}\n\n"
        return

    # Determine candidate mode
    if mode == "direct":
        is_rag_candidate = False
    elif mode == "rag":
        is_rag_candidate = True
    else:  # auto
        is_rag_candidate = (chunk_count > 0)

    # ── CHANGE 2 STREAM ROUTING ─────────────────────────────────────────────
    # Case A: Score above threshold -> Stream RAG answer from documents
    if is_rag_candidate and relevant_chunks and best_score >= sim_threshold:
        sources = []
        for c in relevant_chunks:
            page_str = f" (page {c.page + 1})" if isinstance(c.page, int) and c.page >= 0 else ""
            source_label = f"{c.source}{page_str}"
            if source_label not in sources:
                sources.append(source_label)

        context_text = "\n\n---\n\n".join([
            f"[Source: {c.source} | Page: {c.page + 1}]\n{c.text}" for c in relevant_chunks
        ])
        detected_lang = detect_reply_language(q)
        prompt = build_prompt_with_context(q, context_text, detected_lang, history=active_history)
        sys_prompt = get_slm_rag_prompt(detected_lang)
        print(f"[LANGUAGE - Stream] Query: '{_safe_log_str(q)}' | Detected: {detected_lang}")

        has_streamed = False
        accumulated = []
        t_gen_start = time.perf_counter()
        try:
            for token in _stream_ollama(prompt, system=sys_prompt):
                has_streamed = True
                accumulated.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception:
            pass
        t_gen = time.perf_counter() - t_gen_start

        if has_streamed:
            full_ans = "".join(accumulated).strip()
            res = {"answer": full_ans, "sources": sources, "mode": "rag", "escalate": False}
            set_cached_response(q, res)
            add_conversation_turn(q, full_ans, lang)
            t_total = time.perf_counter() - t_start
            print(f"[TIMING - Stream] Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Generation: {t_gen:.2f}s | Total: {t_total:.2f}s (RAG Stream)")
            yield f"data: {json.dumps({'done': True, 'sources': sources, 'mode': 'rag', 'escalate': False})}\n\n"
            return

        # Fallback to ask_question
        fallback_res = ask_question(q, history=active_history, mode="rag", selected_doc=selected_doc)
        yield f"data: {json.dumps({'token': fallback_res.get('answer', '')})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': sources, 'mode': 'rag', 'escalate': False})}\n\n"
        return

    # Case B: Score below threshold but looks business-related -> Escalate immediately in detected language
    if is_biz:
        detected_lang = detect_reply_language(q)
        escalate_msg = get_escalation_response(detected_lang)
        res = {"answer": escalate_msg, "sources": [], "mode": "rag", "escalate": True}
        set_cached_response(q, res)
        add_conversation_turn(q, escalate_msg, detected_lang)
        t_total = time.perf_counter() - t_start
        print(f"[TIMING - Stream] Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Generation: 0.00s | Total: {t_total:.2f}s | Score: {best_score:.3f} < {sim_threshold} (Escalated in {detected_lang})")
        yield f"data: {json.dumps({'token': escalate_msg})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': True})}\n\n"
        return

    # Explicit RAG mode requested but not found
    if mode == "rag":
        fallback_res = ask_question(q, history=active_history, mode="rag", selected_doc=selected_doc)
        yield f"data: {json.dumps({'token': fallback_res.get('answer', '')})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': False})}\n\n"
        return

    # Case C: Score below threshold & general question -> Direct model streaming
    conv_prompt = build_conversational_prompt(q, lang, history=active_history)
    conv_sys = get_slm_conversational_prompt()

    has_streamed = False
    accumulated = []
    t_gen_start = time.perf_counter()
    try:
        for token in _stream_ollama(conv_prompt, system=conv_sys):
            has_streamed = True
            accumulated.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
    except Exception:
        pass
    t_gen = time.perf_counter() - t_gen_start

    if has_streamed:
        full_ans = "".join(accumulated).strip()
        res = {"answer": full_ans, "sources": [], "mode": "direct", "escalate": False}
        set_cached_response(q, res)
        add_conversation_turn(q, full_ans, lang)
        t_total = time.perf_counter() - t_start
        print(f"[TIMING - Stream] Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Generation: {t_gen:.2f}s | Total: {t_total:.2f}s (Direct Stream)")
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
        return

    fallback_res = ask_question(q, history=active_history, mode="direct")
    yield f"data: {json.dumps({'token': fallback_res.get('answer', '')})}\n\n"
    yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"


def get_system_status() -> dict:
    """Return live system state, loaded documents, and Ollama connection status."""
    ollama_info = check_ollama_status()
    active_model, is_available = get_active_model(ollama_info)
    configured_model = get_configured_model()
    groq_key = get_groq_api_key()
    return {
        "status": "ok",
        "project_name": get_project_name(),
        "project_description": get_project_description(),
        "model_type": get_model_type(),
        "mode": "rag" if _store.count() > 0 else "direct",
        "chunks_indexed": _store.count(),
        "files": _store.get_files(),
        "model_configured": configured_model,
        "embedding_model": get_embedding_model_name(),
        "active_model": active_model if is_available else None,
        "ollama_base_url": get_ollama_base_url(),
        "ollama_online": ollama_info["online"],
        "ollama_error": ollama_info.get("error"),
        "available_models": ollama_info["models"],
        "groq_fallback_active": bool(groq_key),
        "groq_fallback_model": get_groq_fallback_model() if groq_key else None,
        "last_error": _last_invocation_error,
        "last_detected_language": _last_detected_style
    }


def clear_all_documents() -> dict:
    """Delete all stored embeddings, indexed files, uploads, and conversation history."""
    _store.clear()
    clear_conversation_history()
    clear_query_cache()

    if os.path.exists(UPLOADS_PATH):
        for f in os.listdir(UPLOADS_PATH):
            file_path = os.path.join(UPLOADS_PATH, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    return {"message": "All documents and conversation history cleared successfully"}


def remove_document(filename: str) -> dict:
    """
    Remove a single document's chunks from the BM25 index and delete it from disk.
    Raises FileNotFoundError if the document is not found in the index.
    """
    clean_name = os.path.basename(filename)
    if clean_name not in _store.file_registry:
        raise FileNotFoundError(f"Document '{clean_name}' not found in index.")

    # Remove chunks belonging to this file
    _store.chunks = [c for c in _store.chunks if c.source != clean_name]
    del _store.file_registry[clean_name]
    _store.save()
    clear_query_cache()

    # Remove file from disk
    file_path = os.path.join(UPLOADS_PATH, clean_name)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"[remove_document] Could not delete file from disk: {e}")

    print(f"[Store] Removed document '{clean_name}' from index.")
    return {"status": "ok", "message": f"Document '{clean_name}' removed successfully."}


def list_documents() -> list:
    """Return the list of indexed documents with filename and chunk counts."""
    return _store.get_files()

