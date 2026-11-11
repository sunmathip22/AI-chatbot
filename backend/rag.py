import os
import re
import json
import math
import shutil
import time
import difflib
import traceback
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
DEFAULT_MODEL_NAME = ""
DEFAULT_EMBEDDING_MODEL_NAME = ""
DEFAULT_GROQ_FALLBACK_MODEL = ""
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
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME).strip()

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
DEFAULT_TOP_K_LIST = 4
DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_NUM_CTX = 1536
DEFAULT_NUM_PREDICT = 80
DEFAULT_NUM_PREDICT_LIST = 150
DEFAULT_TEMPERATURE = 0.2
DEFAULT_KEEP_ALIVE = "30m"
DEFAULT_OLLAMA_TIMEOUT = 180

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

def get_top_k_list() -> int:
    try:
        return int(os.getenv("TOP_K_LIST", str(DEFAULT_TOP_K_LIST)))
    except ValueError:
        return DEFAULT_TOP_K_LIST

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

def get_num_predict_list() -> int:
    try:
        return int(os.getenv("NUM_PREDICT_LIST", str(DEFAULT_NUM_PREDICT_LIST)))
    except ValueError:
        return DEFAULT_NUM_PREDICT_LIST

def get_temperature() -> float:
    try:
        return float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    except ValueError:
        return DEFAULT_TEMPERATURE

def get_keep_alive() -> str:
    return os.getenv("KEEP_ALIVE", DEFAULT_KEEP_ALIVE).strip() or DEFAULT_KEEP_ALIVE

def get_ollama_timeout() -> int:
    try:
        return int(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_OLLAMA_TIMEOUT)))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT

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

def get_cached_response(query: str, doc_key: str = "") -> dict | None:
    norm = normalize_query(query)
    key = f"{doc_key}::{norm}" if doc_key else norm
    if key in _query_cache:
        return dict(_query_cache[key])
    return None

def set_cached_response(query: str, response: dict, doc_key: str = ""):
    norm = normalize_query(query)
    if norm:
        key = f"{doc_key}::{norm}" if doc_key else norm
        if len(_query_cache) >= 500:
            first_key = next(iter(_query_cache))
            del _query_cache[first_key]
        _query_cache[key] = response

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
    "model": ["model", "slm", "llm", "ollama"],
    "மாதிரி": ["model", "slm", "llm", "ollama"],
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
    Generate dense embeddings using the configured Ollama embedding model via /api/embed.
    Returns (list_of_embeddings, error_message).
    """
    if not texts:
        return [], None

    model = get_embedding_model_name()
    if not model:
        return None, "EMBEDDING_MODEL_NAME is not configured in .env"
    base_url = get_ollama_base_url()
    status = check_ollama_status()
    if not status["online"]:
        return None, f"Cannot connect to Ollama at {base_url} ({status.get('error', 'offline')})"

    try:
        url = f"{base_url}/api/embed"
        payload = {
            "model": model,
            "input": texts,
            "keep_alive": get_keep_alive()
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "RAG-Chatbot"}
        )
        with urllib.request.urlopen(req, timeout=get_ollama_timeout()) as resp:
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
        self.last_uploaded_doc: str | None = None
        self.load()

    def load(self):
        self.chunks = []
        self.file_registry = {}
        self.last_uploaded_doc = None
        if os.path.exists(STORE_FILE):
            try:
                with open(STORE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chunks = [DocChunk.from_dict(c) for c in data.get("chunks", [])]
                    self.file_registry = data.get("file_registry", {})
                    last_doc = data.get("last_uploaded_doc")
                    if last_doc and last_doc in self.file_registry:
                        self.last_uploaded_doc = last_doc
                    elif self.file_registry:
                        self.last_uploaded_doc = list(self.file_registry.keys())[-1]
            except Exception as e:
                print(f"[Store] Warning: Could not load existing store: {e}")

    def save(self):
        try:
            with open(STORE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": [c.to_dict() for c in self.chunks],
                    "file_registry": self.file_registry,
                    "last_uploaded_doc": self.last_uploaded_doc
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Store] Warning: Could not save store: {e}")

    def add_chunks(self, filename: str, new_chunks: list[DocChunk], replace_existing: bool = True):
        clean_name = os.path.basename(filename)
        if replace_existing:
            self.chunks = []
            self.file_registry = {}
            if os.path.exists(UPLOADS_PATH):
                for existing_file in os.listdir(UPLOADS_PATH):
                    if existing_file != clean_name:
                        old_path = os.path.join(UPLOADS_PATH, existing_file)
                        try:
                            if os.path.isfile(old_path):
                                os.remove(old_path)
                        except OSError:
                            pass
        else:
            self.chunks = [c for c in self.chunks if c.source != clean_name]
        self.chunks.extend(new_chunks)
        self.file_registry[clean_name] = len(new_chunks)
        self.last_uploaded_doc = clean_name
        self.save()

    def clear(self):
        self.chunks = []
        self.file_registry = {}
        self.last_uploaded_doc = None
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
        source_filter: str | None = None,
        return_scores: bool = False,
    ):
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
            relevant_pairs = [(chunk, float(score)) for score, chunk in scored if score >= threshold][:k]
            if return_scores:
                return relevant_pairs, best_score, None
            log_pairs = relevant_pairs if relevant_pairs else [(chunk, float(score)) for score, chunk in scored[:min(2, len(scored))]]
            for chunk, sc in log_pairs:
                print(f"[Retrieval] Chunk: {chunk.source} (page {chunk.page + 1}) | similarity={sc:.4f}", flush=True)
            relevant = [chunk for chunk, _ in relevant_pairs]
            return relevant, best_score, None
        except Exception as e:
            return [], 0.0, f"Vector search error: {str(e)}"

    def similarity_search(self, query: str, k: int = 4, threshold: float = 0.2, history: list[dict] | None = None, source_filter: str | None = None):
        """Multilingual similarity search with relevance thresholding."""
        chunks, best_score, err = self.search_with_status(query, k=k, threshold=threshold, history=history, source_filter=source_filter)
        if err:
            raise RuntimeError(err)
        return chunks

    _GENERIC_TOPIC_STOPWORDS = {
        "what", "when", "where", "which", "who", "whom", "how", "why", "is", "are", "was", "were",
        "do", "does", "did", "can", "could", "will", "would", "should", "have", "has", "had",
        "you", "your", "yours", "we", "our", "they", "their", "this", "that", "these", "those",
        "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "at", "by", "with", "from",
        "about", "tell", "me", "give", "show", "any", "some", "please", "pls",
        "price", "prices", "rate", "rates", "cost", "costs", "charge", "charges", "fee", "fees",
        "detail", "details", "info", "information", "available", "availability", "stock",
        "enna", "evlo", "evalo", "evvalavu", "ethana", "irukku", "irukka", "iruka", "irukkum",
        "unga", "ungala", "ungalukku", "kita", "kitta", "ta", "la", "le", "ku", "kku", "oda",
        "pathi", "paththi", "sollu", "sollunga", "pannunga", "panunga", "vaanga", "vaangalam",
        "kedaikuma", "kidaikkuma", "kidaikuma", "aagum", "aagudhu", "venum", "eppadi", "epdi",
        "eppo", "enga", "edhu", "ethu", "idhu", "ithu", "adhu", "athu", "lam", "ellam",
    }

    @classmethod
    def _extract_topic_tokens(cls, query: str) -> list[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", (query or "").lower())
        topic = []
        for t in tokens:
            if len(t) < 3 or t in cls._GENERIC_TOPIC_STOPWORDS:
                continue
            topic.append(t)
        return topic

    @staticmethod
    def _chunk_keyword_hits(chunk_text: str, topic_tokens: list[str]) -> int:
        text_low = (chunk_text or "").lower()
        hits = 0
        for tok in topic_tokens:
            stem = tok[:5] if len(tok) >= 6 else tok
            if re.search(rf"\b{re.escape(tok)}\b", text_low) or (
                len(stem) >= 4 and re.search(rf"\b{re.escape(stem)}", text_low)
            ):
                hits += 1
        return hits

    @classmethod
    def _chunk_rerank_score(cls, chunk_text: str, query: str, topic_tokens: list[str]) -> tuple[int, int, int]:
        text_low = (chunk_text or "").lower()
        distinct_hits = cls._chunk_keyword_hits(text_low, topic_tokens)
        # Prefer chunks matching a multi-word phrase (e.g. 'cotton saree', 'silk saree', 'return policy')
        phrase_hits = 0
        if len(topic_tokens) >= 2:
            for i in range(len(topic_tokens) - 1):
                big = f"{topic_tokens[i]} {topic_tokens[i + 1]}"
                if re.search(rf"\b{re.escape(big)}", text_low):
                    phrase_hits += 1
        # Prefer structured catalog/policy rows when multiple chunks match equally
        is_row = 1 if ("product:" in text_low or "columns:" in text_low or "section:" in text_low) else 0
        return (distinct_hits + phrase_hits * 2, phrase_hits, is_row)

    def search_with_status(
        self,
        query: str,
        k: int = 4,
        threshold: float = 0.2,
        history: list[dict] | None = None,
        source_filter: str | None = None
    ) -> tuple[list[DocChunk], float, str | None]:
        """
        Executes hybrid search:
        - Prioritizes EmbeddingGemma 2 dense vector retrieval + word-boundary keyword reranking.
        - If a specific item/entity question has zero whole-word topic matches across the entire
          filtered document, treats similarity as below threshold to prevent false retrieval.
        - Falls back gracefully to BM25 search if embeddings fail.
        """
        if not self.chunks:
            return [], 0.0, None

        candidates = self.chunks
        if source_filter:
            candidates = [c for c in candidates if c.source == source_filter]
            if not candidates:
                return [], 0.0, None

        topic_tokens = self._extract_topic_tokens(query)
        expanded_tokens = [t for t in expand_query_terms(query, history=None) if len(t) >= 3 and t not in self._GENERIC_TOPIC_STOPWORDS]
        check_tokens = list(dict.fromkeys(topic_tokens + expanded_tokens))

        if check_tokens and len(topic_tokens) > 0:
            total_topic_hits = sum(self._chunk_keyword_hits(c.text, check_tokens) for c in candidates)
            if total_topic_hits == 0:
                below_score = max(0.0, threshold - 0.10)
                print(f"[Retrieval] No document match for topic tokens {topic_tokens} in '{source_filter or 'ALL'}' -> below threshold ({below_score:.3f} < {threshold})", flush=True)
                return [], below_score, None

        # 1. Try vector search with EmbeddingGemma 2 if any candidate chunk has an embedding
        has_embeddings = any(c.embedding is not None and len(c.embedding) > 0 for c in candidates)
        if has_embeddings:
            q_embs, q_err = generate_embeddings([query])
            if q_embs and len(q_embs) > 0:
                fetch_n = len(candidates) if check_tokens else k
                rel_pairs, best_sim, v_err = self.vector_search(
                    q_embs[0], k=fetch_n, threshold=threshold, source_filter=source_filter, return_scores=True
                )
                if not v_err:
                    if check_tokens and rel_pairs:
                        rel_pairs = sorted(
                            rel_pairs,
                            key=lambda pair: (self._chunk_rerank_score(pair[0].text, query, check_tokens), pair[1]),
                            reverse=True,
                        )[:k]
                    else:
                        rel_pairs = rel_pairs[:k]
                    for chunk, sc in rel_pairs:
                        print(f"[Retrieval] Chunk: {chunk.source} (page {chunk.page + 1}) | similarity={sc:.4f}", flush=True)
                    rel_chunks = [chunk for chunk, _ in rel_pairs]
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
            relevant_pairs = [(item[1], float(item[0])) for item in scores if item[0] >= threshold][:k]
            log_pairs = relevant_pairs if relevant_pairs else [(item[1], float(item[0])) for item in scores[:min(2, len(scores))]]
            for chunk, sc in log_pairs:
                print(f"[Retrieval BM25] Chunk: {chunk.source} (page {chunk.page + 1}) | score={sc:.4f}", flush=True)
            relevant = [chunk for chunk, _ in relevant_pairs]
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
    """Create chunk splitter configured for fast CPU SLM RAG using CHUNK_SIZE from .env."""
    chunk_tokens = get_chunk_size()
    overlap_tokens = get_chunk_overlap()
    chunk_chars = max(100, chunk_tokens * 4)
    overlap_chars = max(10, overlap_tokens * 4)
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
                start += max(10, chunk_chars - overlap_chars)
            return chunks
    return FallbackSplitter()


def _split_table_columns(line: str) -> list[str]:
    """Split a table-like line into columns by pipe, tab, or 2+ spaces."""
    s = line.strip()
    if not s:
        return []
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        if s.startswith("|") and parts and parts[0] == "":
            parts = parts[1:]
        if s.endswith("|") and parts and parts[-1] == "":
            parts = parts[:-1]
        return [p for p in parts if p]
    if "\t" in s:
        return [p.strip() for p in re.split(r"\t+", s) if p.strip()]
    return [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]


def _get_column_offsets(raw_line: str, cols: list[str]) -> list[int]:
    """Find character start offsets of each column in a space-aligned layout line."""
    offsets = []
    pos = 0
    for col in cols:
        idx = raw_line.find(col, pos)
        if idx == -1:
            idx = pos
        offsets.append(idx)
        pos = idx + len(col)
    return offsets


def _is_section_heading(stripped: str) -> bool:
    """
    Detect generic section headings:
    - Lines starting with a number and a title (e.g., '1. About the Shop')
    - Lines in ALL CAPS (with >= 3 letters)
    - Lines ending with a colon (e.g., 'Current Offers:')
    """
    if not stripped or len(stripped) > 90:
        return False
    if re.match(r"^[-*•]\s+", stripped):
        return False
    if re.match(r"^(q|a)\s*:", stripped, re.IGNORECASE):
        return False
    # Avoid treating multi-column lines as headings
    if len(_split_table_columns(stripped)) >= 2:
        return False

    # 1. Numbered heading (e.g., '1. About the Shop', '2.1 Pricing')
    if re.match(r"^\d+(?:\.\d+)*[.)]\s+[A-Za-z\u0B80-\u0BFF]", stripped):
        after_num = stripped.split(" ", 1)[1] if " " in stripped else ""
        if not stripped.endswith(("?", ".")) and "(Answer:" not in stripped and "(Expected:" not in stripped and ". " not in after_num:
            return True

    # 2. ALL CAPS heading
    alpha_count = sum(1 for c in stripped if c.isalpha())
    if alpha_count >= 3 and stripped == stripped.upper() and not stripped.endswith("?"):
        return True

    # 3. Ends with a colon
    if stripped.endswith(":") and len(stripped) >= 3:
        return True

    return False


def _is_table_header(cols: list[str], stripped: str) -> bool:
    """Detect column header lines generically without hardcoding column names."""
    if len(cols) < 2:
        return False
    if any(re.search(r"\d", c) for c in cols):
        return False
    if any(len(c) > 35 for c in cols):
        return False
    if stripped.endswith((".", "?", "!")):
        return False
    return True


def split_document_structured(pages_data: list[tuple[str, int]], splitter, filename: str) -> list[DocChunk]:
    """
    Generic structure-aware chunker:
    - Splits by section headings and prefixes every chunk with 'Section: <title>'.
    - Keeps each table/row-like line as its own chunk prefixed with section title and column headers.
    - Splits remaining prose with the normal text splitter, keeping Q:/A: pairs together.
    """
    chunks: list[DocChunk] = []
    current_section = ""
    prose_lines: list[str] = []
    prose_page = 0
    table_headers: list[str] | None = None
    header_offsets: list[int] | None = None
    active_row_cols: list[str] | None = None
    active_row_page = 0
    blank_since_row = False

    def flush_prose():
        nonlocal prose_lines
        if not prose_lines:
            return
        # Normalize prose lines, keeping Q: and A: together
        normalized_blocks: list[str] = []
        buf: list[str] = []
        for ln in prose_lines:
            if not ln:
                if buf:
                    normalized_blocks.append(" ".join(buf))
                    buf = []
            elif re.match(r"^q\s*:", ln, re.IGNORECASE):
                if buf:
                    normalized_blocks.append(" ".join(buf))
                    buf = []
                buf.append(ln)
            elif re.match(r"^a\s*:", ln, re.IGNORECASE):
                if buf:
                    buf.append("\n" + ln)
                else:
                    buf.append(ln)
            elif re.match(r"^[-*•]\s+", ln) or re.match(r"^\d+[.)]\s+", ln):
                if buf:
                    normalized_blocks.append(" ".join(buf))
                    buf = []
                buf.append(ln)
            else:
                buf.append(ln)
        if buf:
            normalized_blocks.append(" ".join(buf))

        prose_text = "\n\n".join(b.strip() for b in normalized_blocks if b.strip())
        if prose_text:
            for st in splitter.split_text(prose_text):
                clean = st.strip()
                if clean:
                    text_out = f"Section: {current_section}\n{clean}" if current_section else clean
                    chunks.append(DocChunk(text=text_out, source=filename, page=prose_page))
        prose_lines = []

    def flush_row():
        nonlocal active_row_cols
        if not active_row_cols:
            return
        cleaned_cols = [c.strip() for c in active_row_cols if c.strip()]
        if cleaned_cols:
            if table_headers and len(cleaned_cols) == len(table_headers):
                row_str = " | ".join(f"{h}: {v}" for h, v in zip(table_headers, cleaned_cols))
            else:
                row_str = " | ".join(cleaned_cols)
            parts = []
            if current_section:
                parts.append(f"Section: {current_section}")
            if table_headers:
                parts.append(f"Columns: {' | '.join(table_headers)}")
            parts.append(row_str)
            chunks.append(DocChunk(text="\n".join(parts), source=filename, page=active_row_page))
        active_row_cols = None

    for page_text, page_num in pages_data:
        for raw_line in page_text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if active_row_cols is not None:
                    blank_since_row = True
                elif prose_lines and prose_lines[-1] != "":
                    prose_lines.append("")
                continue

            # Skip markdown divider lines (e.g. |---|---|)
            if re.match(r"^[\|\-\+\=\s:]+$", stripped):
                continue

            # Check if previous section heading had an unclosed parenthesis
            if current_section and current_section.count("(") > current_section.count(")") and len(stripped) <= 40:
                current_section = f"{current_section} {stripped}".strip()
                continue

            cols = _split_table_columns(raw_line)
            pipe_count = stripped.count("|")
            tab_count = len(re.findall(r"\t+", stripped))
            multispace_count = len(re.findall(r"\s{2,}", stripped))
            has_2plus_seps = (pipe_count >= 2) or (tab_count >= 2) or (multispace_count >= 2)
            has_any_sep = (pipe_count >= 1) or (tab_count >= 1) or (multispace_count >= 1)

            # Check continuation line for an active multi-column table row (no blank line before it)
            is_bullet_or_qa = bool(re.match(r"^([-*•]\s+|(q|a)\s*:|\d+[.)]\s+)", stripped, re.IGNORECASE))
            if (
                active_row_cols is not None
                and len(active_row_cols) >= 2
                and not blank_since_row
                and not has_2plus_seps
                and not is_bullet_or_qa
                and not _is_section_heading(stripped)
            ):
                if header_offsets and len(active_row_cols) == len(header_offsets):
                    indent = len(raw_line) - len(raw_line.lstrip())
                    col_idx = min(range(len(header_offsets)), key=lambda i: abs(header_offsets[i] - indent))
                    active_row_cols[col_idx] = f"{active_row_cols[col_idx]} {stripped}".strip()
                else:
                    active_row_cols[-1] = f"{active_row_cols[-1]} {stripped}".strip()
                continue

            # Close any open row before handling a new line
            flush_row()

            # 1. Section heading
            if _is_section_heading(stripped):
                flush_prose()
                table_headers = None
                header_offsets = None
                current_section = stripped
                continue

            # 2. Table header line
            if has_any_sep and _is_table_header(cols, stripped):
                flush_prose()
                table_headers = cols
                header_offsets = _get_column_offsets(raw_line, cols) if ("|" not in stripped and "\t" not in stripped) else None
                continue

            # 3. Table-like or row-like line
            in_prose_paragraph = bool(prose_lines and prose_lines[-1] != "")
            has_price_or_num_pattern = bool(
                re.search(r"(?:Rs\.?|INR|₹|\$|€|£)\s*\d|\b\d[\d,]*\s*-\s*\d[\d,]*\b", stripped, re.IGNORECASE)
            )
            is_row_line = False
            if not is_bullet_or_qa:
                if has_2plus_seps:
                    is_row_line = True
                elif table_headers is not None and has_any_sep:
                    is_row_line = True
                elif not in_prose_paragraph and has_price_or_num_pattern and len(stripped) <= 140 and not re.search(r"[.?!]\s+[A-Z]", stripped):
                    is_row_line = True

            if is_row_line:
                flush_prose()
                active_row_cols = cols if len(cols) >= 2 else [stripped]
                active_row_page = page_num
                blank_since_row = False
                continue

            # 4. Regular prose line
            if table_headers is not None and not has_any_sep:
                # Prose resumed after table
                table_headers = None
                header_offsets = None
            if not prose_lines:
                prose_page = page_num
            prose_lines.append(stripped)

    flush_row()
    flush_prose()
    return chunks


def process_document(file_path: str, replace_existing: bool = True) -> int:
    """
    Extract text from PDF or TXT, split into structure-aware chunks, and index.
    By default, uploading a new document replaces previous documents unless replace_existing=False.
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
                    try:
                        page_text = page.extract_text(extraction_mode="layout") or ""
                    except TypeError:
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
    new_chunks = split_document_structured(pages_data, splitter, filename)

    # Check if identical document is already indexed to avoid re-embedding unchanged documents
    clean_name = os.path.basename(filename)
    existing_chunks = [c for c in _store.chunks if c.source == clean_name and c.embedding is not None]
    if existing_chunks and len(existing_chunks) == len(new_chunks):
        if all(c.text == e.text for c, e in zip(new_chunks, existing_chunks)):
            if replace_existing:
                _store.add_chunks(clean_name, existing_chunks, replace_existing=True)
            else:
                _store.last_uploaded_doc = clean_name
                _store.save()
            clear_query_cache()
            print(f"[Store] Reusing existing embeddings for unchanged document '{clean_name}'. Indexed documents: {list(_store.file_registry.keys())}", flush=True)
            return len(existing_chunks)

    # Generate embeddings using configured Ollama embedding model
    texts = [c.text for c in new_chunks]
    embs, emb_err = generate_embeddings(texts)
    if embs and len(embs) == len(new_chunks):
        for idx, emb in enumerate(embs):
            new_chunks[idx].embedding = emb
        print(f"[Store] Generated {len(embs)} embeddings using '{get_embedding_model_name()}' for '{clean_name}'.")
    elif emb_err:
        print(f"[Store Warning] Embedding generation failed ({emb_err}). Storing text chunks for BM25 fallback.")

    _store.add_chunks(clean_name, new_chunks, replace_existing=replace_existing)
    clear_query_cache()
    print(f"Indexed {len(new_chunks)} chunks for '{clean_name}'. Indexed documents: {list(_store.file_registry.keys())}", flush=True)
    return len(new_chunks)


_last_invocation_error = {"ollama": None, "groq": None}
_ollama_status_cache = {"result": None, "timestamp": 0.0}
_STATUS_CACHE_TTL = 30.0


def check_ollama_status(force: bool = False) -> dict:
    """Ping Ollama API to verify availability and list installed models (cached for 30s)."""
    now = time.time()
    if not force and _ollama_status_cache["result"] is not None:
        if (now - _ollama_status_cache["timestamp"]) < _STATUS_CACHE_TTL:
            return _ollama_status_cache["result"]

    base_url = get_ollama_base_url()
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "RAG-Chatbot"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            res = {"online": True, "models": models, "error": None}
            _ollama_status_cache["result"] = res
            _ollama_status_cache["timestamp"] = now
            return res
    except urllib.error.URLError as e:
        err_msg = str(e.reason) if hasattr(e, "reason") else str(e)
        print(f"[Ollama] Connection failed ({base_url}): {err_msg}")
        res = {"online": False, "models": [], "error": err_msg}
        _ollama_status_cache["result"] = res
        _ollama_status_cache["timestamp"] = now
        return res
    except Exception as e:
        print(f"[Ollama] Unexpected error checking status: {e}")
        res = {"online": False, "models": [], "error": str(e)}
        _ollama_status_cache["result"] = res
        _ollama_status_cache["timestamp"] = now
        return res


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

    if configured:
        for m in status["models"]:
            if m == configured or m.startswith(f"{configured}:") or configured.startswith(f"{m}:"):
                return m, True
        print(f"[Ollama WARNING] Configured MODEL_NAME='{configured}' not found in installed models {status['models']}. "
              f"Falling back to: {status['models'][0]}")
    else:
        print(f"[Ollama WARNING] MODEL_NAME is not set in .env. Falling back to: {status['models'][0]}")

    return status["models"][0], True


# ---------------------------------------------------------------------------
# FIX 2 & FIX 4: System Prompts & Multi-Language Fixed Responses
# ---------------------------------------------------------------------------

def get_language_system_line(lang: str) -> str:
    """Explicit instruction line placed at the top of the system prompt."""
    if lang == "Tanglish":
        return "Reply ONLY in Tanglish (Tamil spoken language written in English letters). Do NOT reply in English sentences. Keep item names, numbers, and prices unchanged."
    elif lang == "Tamil":
        return "Reply ONLY in Tamil script. Keep item names, numbers, and prices unchanged."
    return "Reply ONLY in English."


def get_missing_info_response(query: str, lang: str = "English") -> str:
    """Generic fallback message when requested detail is missing from documents."""
    norm = (lang or "english").strip().lower()
    if norm in ("tanglish", "thanglish"):
        return "Intha detail ennoda documents la illa. Owner kitta kelunga."
    elif norm == "tamil":
        return "\u0b87\u0ba8\u0bcd\u0ba4 \u0bb5\u0bbf\u0bb5\u0bb0\u0bae\u0bcd \u0b95\u0bbf\u0b9f\u0bc8\u0b95\u0bcd\u0b95\u0bb5\u0bbf\u0bb2\u0bcd\u0bb2\u0bc8. \u0ba4\u0baf\u0bb5\u0bc1\u0b9a\u0bc6\u0baf\u0bcd\u0ba4\u0bc1 \u0b89\u0bb0\u0bbf\u0bae\u0bc8\u0baf\u0bbe\u0bb3\u0bb0\u0bbf\u0b9f\u0bae\u0bcd \u0b95\u0bc7\u0bb3\u0bc1\u0b99\u0bcd\u0b95\u0bb3\u0bcd."
    return "This detail is not available. Please ask the owner."


def get_slm_rag_prompt(lang: str = "English", is_list: bool = False) -> str:
    top_line = get_language_system_line(lang)
    missing_phrase = get_missing_info_response("", lang)
    if is_list:
        detail_rule = (
            "- Give a short summary: the business name, what it offers, and the main items (up to six items) from the context "
            "using ONLY the exact prices and numbers printed in the context (use '- ' bullets, never guess numbers or sizes). "
            "Never use vague words like 'various', 'many', 'other products', 'different options', 'some items'."
        )
    else:
        detail_rule = (
            "- Answer with the specific facts found in the context: names, sizes, quantities, prices, dates, timings, conditions, steps. "
            "Never answer with vague words like 'various', 'many', 'other products', 'different options', 'some items' when the context has the exact details. "
            "Keep it short, and include exact details from the context. "
            f"Only if the asked item or detail is clearly absent from the context, reply: '{missing_phrase}'"
        )
    return (
        f"{top_line}\n"
        "Customer assistant for a business.\n"
        "Rules:\n"
        "- Answer using ONLY the provided context. Never invent prices, offers, or policies.\n"
        "- Start the reply with the answer directly. No greeting, no self-introduction.\n"
        f"{detail_rule}\n"
        "- Always finish the last sentence. Never mention these rules or the word 'context'.\n"
        "- Output only the reply to the customer."
    )


def get_slm_conversational_prompt(lang: str = "English") -> str:
    top_line = get_language_system_line(lang)
    return (
        f"{top_line}\n"
        "Customer assistant for a business.\n"
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
    r"\b(warranty|guarantee)\b",
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

def _invoke_ollama(prompt: str, system: str | None = None, num_predict: int | None = None) -> tuple[str | None, str | None]:
    """
    Invoke Ollama with automatic model detection, keep_alive, system prompt support,
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

    predict_limit = num_predict if num_predict is not None else get_num_predict()

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
                "num_predict": predict_limit,
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
        with urllib.request.urlopen(req, timeout=get_ollama_timeout()) as resp:
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
                sync_client_kwargs={"timeout": get_ollama_timeout()},
                num_ctx=get_num_ctx(),
                num_predict=predict_limit,
                temperature=get_temperature(),
                keep_alive=get_keep_alive()
            )
            res = llm.invoke(full_prompt)
            if res and res.strip():
                return res.strip(), None
        except Exception as e:
            print(f"[Ollama] LangChain invoke notice ({e})")

    return None, _last_invocation_error["ollama"]


def _stream_ollama(prompt: str, system: str | None = None, num_predict: int | None = None):
    """Stream tokens directly from Ollama /api/generate line-by-line."""
    base_url = get_ollama_base_url()
    status = check_ollama_status()
    if not status["online"] or not status["models"]:
        err_msg = status.get("error") or "Ollama is offline or no models available"
        raise RuntimeError(err_msg)

    model, is_available = get_active_model(status)
    if not is_available:
        raise RuntimeError(f"Configured model '{get_configured_model()}' is not available.")

    predict_limit = num_predict if num_predict is not None else get_num_predict()

    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": get_keep_alive(),
        "options": {
            "num_ctx": get_num_ctx(),
            "num_predict": predict_limit,
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
    timeout_val = get_ollama_timeout()
    try:
        with urllib.request.urlopen(req, timeout=timeout_val) as resp:
            for line in resp:
                if line.strip():
                    item = json.loads(line.decode("utf-8"))
                    token = item.get("response", "")
                    if token:
                        yield token
                    if item.get("done", False):
                        break
    except Exception:
        _ollama_status_cache["result"] = None
        raise


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

LIST_OVERVIEW_PHRASES = (
    "what do you have", "what do you offer", "what do you sell",
    "what is this about", "what is it about",
    "tell me about your shop", "tell me about this", "tell me about your store", "tell me about your business",
    "what products", "what services", "what items",
    "un ta edhu pathi details irukku", "unta edhu pathi details irukku",
    "edhu pathi details irukku", "enna details irukku",
    "enna productslam irukku", "enna productlam irukku", "enna products lam irukku",
    "productslam", "productlam", "products lam",
    "enna irukku", "unga kita enna irukku", "unga kitta enna irukku",
    "details sollu", "enna enna",
)
LIST_OVERVIEW_WORDS = {
    "list", "types", "catalog", "catalogue", "menu", "all",
    "available", "options", "sizes", "quantities", "varieties", "overview",
}
_OVERVIEW_FUZZY_TARGETS = (
    "what do you have", "what is this about", "tell me about your shop",
    "edhu pathi details irukku", "enna details irukku",
    "enna productslam irukku", "unga kita enna irukku", "details sollu",
)
_UPLOAD_ACK_EXACT = {
    "ok", "okay", "okk", "k", "kk", "done", "thanks", "thank you", "thx", "ty", "nandri",
    "uploaded", "doc uploaded", "document uploaded", "file uploaded", "upload done",
    "i have upload the doc", "i have uploaded the doc", "i uploaded the doc",
    "i have uploaded the document", "i uploaded the document", "uploaded the doc",
}


_UPLOAD_ACK_WORDS = {
    "ok", "okay", "okk", "k", "kk", "done", "thanks", "thank", "you", "thx", "ty", "nandri",
    "uploaded", "upload", "got", "it", "alright", "fine", "cool", "sure", "seri", "doc", "document",
}


def is_upload_ack_message(query: str) -> bool:
    """Detect non-question upload notifications and short acknowledgements (ok, done, thanks)."""
    if not query or not query.strip():
        return False
    q_raw = query.strip()
    q_norm = re.sub(r"[^\w\s]", "", q_raw.lower())
    q_norm = re.sub(r"\s+", " ", q_norm).strip()
    if q_norm in _UPLOAD_ACK_EXACT:
        return True
    words = q_norm.split()
    if 1 <= len(words) <= 4 and all(w in _UPLOAD_ACK_WORDS for w in words):
        return True
    if "?" not in q_raw and ("upload" in q_norm or "uploaded" in q_norm):
        if not is_business_question(q_norm) and not is_list_or_overview_question(q_norm):
            return True
    return False


def get_upload_ack_reply(query: str) -> str:
    """Short friendly acknowledgement in English or Tanglish without retrieval or fallback."""
    lang_label = detect_reply_language(query)
    q_low = (query or "").lower()
    if lang_label in ("Tanglish", "Tamil") or any(w in q_low for w in ("nandri", "panniten", "aachu", "ayiduchu", "seri")):
        return "Document vandhuduchu. Edhu venum nalum kelunga."
    return "Document received. You can ask me anything about it."


def is_list_or_overview_question(query: str) -> bool:
    """Detect list or overview intent using English/Tanglish phrases, keywords, and fuzzy matching."""
    if not query or not query.strip():
        return False
    q_lower = re.sub(r"\s+", " ", query.strip().lower())
    if is_about_question(q_lower):
        return True
    if any(phrase in q_lower for phrase in LIST_OVERVIEW_PHRASES):
        return True
    words = re.findall(r"\b[a-zA-Z0-9]+\b", q_lower)
    word_set = set(words)
    if word_set & LIST_OVERVIEW_WORDS:
        return True
    # Small-typo fuzzy matching on overview words/phrases
    for w in words:
        if len(w) >= 6 and difflib.get_close_matches(w, ["productslam", "productlam", "products", "catalog", "catalogue", "overview", "varieties"], n=1, cutoff=0.80):
            return True
    for target in _OVERVIEW_FUZZY_TARGETS:
        if difflib.SequenceMatcher(None, q_lower, target).ratio() >= 0.80:
            return True
    return False


def resolve_active_document(selected_doc: str | None = None) -> str | None:
    """Return the active document filename: explicit selection if valid, else most recently uploaded."""
    if selected_doc:
        clean_sel = os.path.basename(selected_doc)
        if clean_sel in _store.file_registry:
            return clean_sel
    if _store.last_uploaded_doc and _store.last_uploaded_doc in _store.file_registry:
        return _store.last_uploaded_doc
    if _store.file_registry:
        return list(_store.file_registry.keys())[-1]
    return None


def get_overview_chunks(active_doc: str | None = None, top_k: int = 4) -> list[DocChunk]:
    """Retrieve chunks from the beginning of the active document (title, about, key catalog items)."""
    candidates = _store.chunks
    if active_doc:
        candidates = [c for c in candidates if c.source == active_doc]
    if not candidates:
        return []
    opening = candidates[:2]
    catalog = [c for c in candidates[2:] if "Columns:" in (c.text or "")]
    # Take first several catalog rows directly from the top of the catalog plus diverse picks
    sampled_catalog = catalog[:5]
    combined: list[DocChunk] = []
    seen = set()
    limit = max(top_k + 3, 7)
    for c in opening + sampled_catalog + candidates:
        cid = (c.source, c.page, c.text)
        if cid not in seen:
            seen.add(cid)
            combined.append(c)
        if len(combined) >= limit:
            break
    for c in combined:
        print(f"[Retrieval Overview] Chunk: {c.source} (page {c.page + 1}) | score=1.0000", flush=True)
    return combined


def format_overview_chunks(chunks: list, max_total_chars: int = 2000) -> str:
    """Compact overview context deduplicating repeated table headers so diverse items + prices fit cleanly."""
    parts = []
    seen_headers = set()
    total_len = 0
    for c in chunks:
        lines = (c.text or "").splitlines()
        cleaned_lines = []
        for ln in lines:
            s = ln.strip()
            if s.startswith("Section:") or s.startswith("Columns:"):
                if s in seen_headers:
                    continue
                seen_headers.add(s)
            cleaned_lines.append(ln)
        body = "\n".join(cleaned_lines).strip()[:600]
        if not body:
            continue
        if not parts:
            part = f"[Source: {c.source} | Page: {c.page + 1}]\n{body}"
        else:
            part = body
        if total_len + len(part) > max_total_chars:
            break
        parts.append(part)
        total_len += len(part)
    return "\n\n".join(parts)


def check_number_groundedness(reply: str, context_text: str) -> tuple[bool, list[str]]:
    """
    Verify without an extra LLM call that every number (price, size, quantity)
    in the reply exists in the retrieved context text.
    """
    if not reply or not reply.strip():
        return True, []
    # Ignore ordered bullet markers at line starts like "1. Cotton Saree"
    reply_for_check = re.sub(r"(?m)^\s*\d+\.\s+(?=[A-Za-z\u0B80-\u0BFF])", "", reply)
    num_pat = r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?|(?<![A-Za-z])\d+(?:\.\d+)?"
    reply_nums = re.findall(num_pat, reply_for_check)
    if not reply_nums:
        return True, []
    ctx_nums = re.findall(num_pat, context_text or "")
    ctx_norm = set()
    for cn in ctx_nums:
        stripped = cn.replace(",", "")
        ctx_norm.add(stripped)
        if "." in stripped:
            try:
                fval = float(stripped)
                if fval.is_integer():
                    ctx_norm.add(str(int(fval)))
            except ValueError:
                pass
    unverified = []
    for rn in reply_nums:
        rn_stripped = rn.replace(",", "")
        if rn_stripped in ctx_norm:
            continue
        matched_float = False
        if "." in rn_stripped:
            try:
                fval = float(rn_stripped)
                if fval.is_integer() and str(int(fval)) in ctx_norm:
                    matched_float = True
            except ValueError:
                pass
        if not matched_float:
            unverified.append(rn)
    return (len(unverified) == 0), unverified


def build_retrieval_query(question: str, history: list[dict] | None = None) -> str:
    """
    For short follow-up questions WITHOUT a topic (< 6 words and no specific topic words,
    e.g. 'how much', 'price?', 'sizes?'), prepend the last user turn from history.
    Questions with their own topic (e.g. 'silk saree rate enna', 'aloe vera gel irukka')
    keep their own query clean so prior topics do not pollute retrieval.
    """
    q_clean = (question or "").strip()
    if is_list_or_overview_question(q_clean) and len(q_clean.split()) <= 5:
        return f"{q_clean} products items services menu catalog"
    has_own_topic = len(DocumentStore._extract_topic_tokens(q_clean)) > 0
    if not has_own_topic and len(q_clean.split()) < 6 and isinstance(history, list) and history:
        for turn in reversed(history):
            if isinstance(turn, dict) and turn.get("role") == "user":
                prev_q = str(turn.get("content") or "").strip()
                if prev_q and prev_q.lower() != q_clean.lower():
                    return f"{prev_q} {q_clean}"
    return q_clean


def select_list_chunks(chunks: list[DocChunk], top_k: int) -> list[DocChunk]:
    """For list/overview queries, prefer table/row chunks ('Columns:') and intro chunks over disclaimers."""
    if not chunks:
        return []
    preferred = [c for c in chunks if "Columns:" in c.text]
    others = [c for c in chunks if "Columns:" not in c.text]
    return (preferred + others)[:top_k]


def format_context_chunks(chunks: list, max_total_chars: int = 1400) -> str:
    """Format retrieved chunks: max 700 chars per chunk, capped at max_total_chars total."""
    parts = []
    total_len = 0
    for c in chunks:
        chunk_body = (c.text or "")[:700]
        part = f"[Source: {c.source} | Page: {c.page + 1}]\n{chunk_body}"
        parts.append(part)
        total_len += len(part)
        if total_len >= max_total_chars:
            break
    return "\n\n---\n\n".join(parts)[:max_total_chars]


def build_prompt_with_context(
    question: str,
    context_text: str,
    lang: str,
    history: list[dict] | None = None
) -> str:
    """
    Constructs prompt with context, last 2 history turns, question, and trailing language instruction.
    Places language and specificity instructions at the END of user message for maximum compliance.
    """
    history_str = ""
    has_standalone_topic = len(DocumentStore._extract_topic_tokens(question)) > 0 and len((question or "").split()) >= 2
    if isinstance(history, list) and history and not has_standalone_topic:
        recent_turns = history[-2:]
        lines = []
        for turn in recent_turns:
            if not isinstance(turn, dict):
                continue
            role = "Customer" if turn.get("role") == "user" else "Assistant"
            c = str(turn.get("content") or "").replace("\n", " ").strip()
            c = c[:150]
            if c:
                lines.append(f"{role}: {c}")
        if lines:
            history_str = "Recent Chat:\n" + "\n".join(lines) + "\n\n"

    reply_lang = detect_reply_language(question)
    is_list_q = is_list_or_overview_question(question)

    if reply_lang == "Tanglish":
        if is_list_q:
            trailing_instruction = (
                "Verified Context-la irukkura business name, enna offer panranga matrum key items-a (maximum six items) "
                "context-la irukkura exact prices-oda '- ' bullets-la short-a sollunga. "
                "New numbers ethuvum guess pannadhinga. Write in Tanglish without '|' symbols."
            )
        else:
            trailing_instruction = (
                "Answer ONLY the Customer Question directly in 1-2 short Tanglish sentences using ONLY the exact facts, "
                "names, and numbers printed in Verified Context above (for example: 'Aama, irukku', 'Idhoda price ...'). "
                "Never repeat the question. Do NOT copy '|' symbols. Do NOT write English sentences."
            )
    elif reply_lang == "Tamil":
        if is_list_q:
            trailing_instruction = (
                "Verified Context-ல் உள்ள வணிக பெயர், சேவைகள் மற்றும் 4-6 முக்கிய பொருட்களை அவற்றின் விலை மற்றும் அளவுடன் '- ' புள்ளிகளில் சுருக்கமாகக் கூறவும்."
            )
        else:
            trailing_instruction = (
                "Answer the question directly from the Verified Context above in Tamil script with exact item names, sizes, prices, and details. "
                "Keep item names, numbers, and prices unchanged."
            )
    else:
        if is_list_q:
            trailing_instruction = (
                "Give a short summary of the business name and what it offers, followed by up to six key items "
                "from the Verified Context above with their exact prices and details using '- ' bullets. Copy only numbers shown in the context."
            )
        elif len(question.split()) < 4:
            trailing_instruction = (
                "Answer in English directly from the Verified Context above with exact names, sizes, prices, and details."
            )
        else:
            trailing_instruction = (
                "Answer ONLY the current Customer Question in English using ONLY the Verified Context above "
                "with exact names, sizes, prices, and details. Ignore unrelated numbers from Recent Chat."
            )

    return (
        f"{history_str}"
        f"Verified Context:\n{context_text}\n\n"
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
    if isinstance(history, list) and history:
        recent_turns = history[-2:]
        lines = []
        for turn in recent_turns:
            if not isinstance(turn, dict):
                continue
            role = "Customer" if turn.get("role") == "user" else "Assistant"
            c = str(turn.get("content") or "").replace("\n", " ").strip()
            c = c[:150]
            if c:
                lines.append(f"{role}: {c}")
        if lines:
            history_str = "Recent Chat:\n" + "\n".join(lines) + "\n\n"

    reply_lang = detect_reply_language(question)

    if reply_lang == "Tanglish":
        trailing_instruction = (
            "Write the answer in Tanglish: Tamil spoken language written in English letters "
            "(for example: 'Aama, irukku', 'Sollunga, enna help venum?'). "
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
    """Warms up the configured embedding and generation models once at startup with a dummy prompt."""
    print(f"[Startup] Warming up AI models in RAM (keep_alive={get_keep_alive()})...")
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
            with urllib.request.urlopen(req, timeout=get_ollama_timeout()) as resp:
                resp.read()
            print(f"[Startup] Models successfully warmed up in {time.perf_counter() - t0:.2f}s (RAM keep_alive={get_keep_alive()}).")
    except Exception as e:
        print(f"[Startup Warning] Model warmup notice: {e}")
    active_m, _ = get_active_model()
    print(f"[Startup] Ready | Model: {active_m or get_configured_model() or 'none'} | Embedding: {get_embedding_model_name() or 'none'} | Documents: {len(_store.file_registry)} ({_store.count()} chunks)", flush=True)


# ---------------------------------------------------------------------------
# CHANGES 1, 2, & 3: Main ask_question Pipeline
# ---------------------------------------------------------------------------

def _safe_log_str(s: str) -> str:
    """Safe ASCII string for Windows console logging without UnicodeEncodeError."""
    return s.encode("ascii", errors="backslashreplace").decode("ascii")


def _is_explicit_escalation_query(query: str) -> bool:
    """Detect explicit customer requests to speak with or complain to the owner."""
    q = (query or "").lower()
    return bool(re.search(
        r"\b(complaint|refund|owner|manager|support|contact|call me|talk to|speak to|escalate)\b",
        q
    ))


def ask_question(
    question: str,
    history: list[dict] | None = None,
    mode: str = "auto",
    selected_doc: str | None = None
) -> dict:
    """
    Optimized RAG & Direct Response Pipeline with strict single-doc isolation,
    upload-ack detection, overview summary routing, groundedness check, and route logging.
    """
    t_start = time.perf_counter()
    try:
        q = (question or "").strip()
        if not q:
            return {"answer": "Please enter a valid question.", "sources": [], "mode": "direct", "escalate": False}

        active_doc = resolve_active_document(selected_doc)
        doc_key = active_doc or ""

        # 1. Instant Greeting Check at the VERY START
        greeting_reply = get_greeting_reply(q)
        if greeting_reply:
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] greeting | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            res = {"answer": greeting_reply, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, res, doc_key=doc_key)
            add_conversation_turn(q, greeting_reply, detect_language_style(q))
            return res

        # 2. Upload / Acknowledgement check (never trigger retrieval or fallback)
        if is_upload_ack_message(q):
            ack_reply = get_upload_ack_reply(q)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] upload-ack | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            res = {"answer": ack_reply, "sources": [], "mode": "direct", "escalate": False}
            add_conversation_turn(q, ack_reply, detect_language_style(q))
            return res

        lang = detect_language_style(q)

        # 3. In-memory cache check (< 1 ms)
        cached = get_cached_response(q, doc_key=doc_key)
        if cached is not None:
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] {cached.get('route', 'rag-answer')} (cache) | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            return cached

        # 4. Smalltalk handled without LLM or retrieval
        is_st, st_reply, _ = classify_smalltalk(q)
        if is_st:
            res = {"answer": st_reply, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, res, doc_key=doc_key)
            add_conversation_turn(q, st_reply, lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] greeting (smalltalk) | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            return res

        # 5. Identity check
        if is_identity_question(q):
            resp = get_identity_response(lang)
            res = {"answer": resp, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, res, doc_key=doc_key)
            add_conversation_turn(q, resp, lang)
            print(f"[ROUTE] greeting (identity) | Query: '{_safe_log_str(q)}'", flush=True)
            return res

        raw_history = history if isinstance(history, list) else get_conversation_history()
        active_history = raw_history[-2:] if isinstance(raw_history, list) and raw_history else []
        chunk_count = _store.count()

        # 6. Overview / List question -> retrieve from beginning + catalog chunks of active doc
        if is_list_or_overview_question(q) and chunk_count > 0 and mode != "direct":
            top_k_list = get_top_k_list()
            t_ret_start = time.perf_counter()
            overview_chunks = get_overview_chunks(active_doc, top_k=top_k_list)
            t_retrieve = time.perf_counter() - t_ret_start
            if overview_chunks:
                sources = []
                for c in overview_chunks:
                    page_str = f" (page {c.page + 1})" if isinstance(c.page, int) and c.page >= 0 else ""
                    label = f"{c.source}{page_str}"
                    if label not in sources:
                        sources.append(label)
                context_text = format_overview_chunks(overview_chunks, max_total_chars=2000)
                detected_lang = detect_reply_language(q)
                sys_prompt = get_slm_rag_prompt(detected_lang, is_list=True)
                prompt = build_prompt_with_context(q, context_text, detected_lang, history=None)
                t_gen_start = time.perf_counter()
                ans, _ = _invoke_ollama(prompt, system=sys_prompt, num_predict=get_num_predict_list())
                ans, _ = ensure_tanglish_reply(ans, detected_lang)
                t_gen = time.perf_counter() - t_gen_start
                if ans:
                    cleaned = clean_model_output(ans, lang)
                    ok_num, bad_nums = check_number_groundedness(cleaned, context_text)
                    route_tag = "overview"
                    if not ok_num:
                        print(f"[GROUNDEDNESS FAIL] Overview unverified numbers {bad_nums} -> fallback", flush=True)
                        cleaned = get_missing_info_response(q, detected_lang)
                        route_tag = "fallback"
                    res = {"answer": cleaned, "sources": sources, "mode": "rag", "escalate": False}
                    set_cached_response(q, {**res, "route": route_tag}, doc_key=doc_key)
                    add_conversation_turn(q, cleaned, lang)
                    t_total = time.perf_counter() - t_start
                    print(f"[ROUTE] {route_tag} | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Gen: {t_gen:.2f}s | Total: {t_total:.2f}s | Prompt chars: {len(prompt)}", flush=True)
                    return res

        sim_threshold = get_similarity_threshold()
        top_k = get_top_k()
        num_predict_val = get_num_predict()
        retrieval_query = build_retrieval_query(q, history=active_history)
        is_biz = is_business_question(q)

        t_ret_start = time.perf_counter()
        relevant_chunks = []
        best_score = 0.0
        retrieval_err = None

        if chunk_count > 0:
            relevant_chunks, best_score, retrieval_err = _store.search_with_status(
                retrieval_query, k=top_k, threshold=sim_threshold, history=active_history, source_filter=active_doc
            )
        t_retrieve = time.perf_counter() - t_ret_start

        if retrieval_err:
            err_reply = get_retrieval_error_response(lang)
            print(f"[ROUTE] fallback (retrieval error) | Query: '{_safe_log_str(q)}'", flush=True)
            return {"answer": err_reply, "sources": [], "mode": "rag", "escalate": False}

        if mode == "direct":
            is_rag_candidate = False
        elif mode == "rag":
            is_rag_candidate = True
        else:
            is_rag_candidate = (chunk_count > 0)

        # Case A: Score above threshold -> RAG answer strictly from active document chunks
        if is_rag_candidate and relevant_chunks and best_score >= sim_threshold:
            sources = []
            for c in relevant_chunks:
                page_str = f" (page {c.page + 1})" if isinstance(c.page, int) and c.page >= 0 else ""
                source_label = f"{c.source}{page_str}"
                if source_label not in sources:
                    sources.append(source_label)

            context_text = format_context_chunks(relevant_chunks, max_total_chars=1400)
            detected_lang = detect_reply_language(q)
            sys_prompt = get_slm_rag_prompt(detected_lang, is_list=False)
            prompt = build_prompt_with_context(q, context_text, detected_lang, history=active_history)

            t_gen_start = time.perf_counter()
            ollama_answer, ollama_err = _invoke_ollama(prompt, system=sys_prompt, num_predict=num_predict_val)
            ollama_answer, retried = ensure_tanglish_reply(ollama_answer, detected_lang)
            t_gen = time.perf_counter() - t_gen_start
            print(f"[LANGUAGE] Query: '{_safe_log_str(q)}' | Detected: {detected_lang} | Step 2 Retry: {retried}", flush=True)

            if ollama_answer:
                cleaned = clean_model_output(ollama_answer, lang)
                ok_num, bad_nums = check_number_groundedness(cleaned, context_text)
                route_tag = "rag-answer"
                if not ok_num:
                    print(f"[GROUNDEDNESS FAIL] Unverified numbers {bad_nums} -> replacing with missing info", flush=True)
                    cleaned = get_missing_info_response(q, detected_lang)
                    route_tag = "fallback"
                elif "not available" in cleaned.lower() or "la illa" in cleaned.lower():
                    route_tag = "fallback"
                res = {"answer": cleaned, "sources": sources, "mode": "rag", "escalate": False}
                set_cached_response(q, {**res, "route": route_tag}, doc_key=doc_key)
                add_conversation_turn(q, cleaned, lang)
                t_total = time.perf_counter() - t_start
                print(f"[ROUTE] {route_tag} | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Gen: {t_gen:.2f}s | Total: {t_total:.2f}s | Prompt chars: {len(prompt)}", flush=True)
                return res

            groq_ans, _ = _invoke_groq(f"System: {sys_prompt}\n\n{prompt}")
            if groq_ans:
                cleaned = clean_model_output(groq_ans, lang)
                ok_num, bad_nums = check_number_groundedness(cleaned, context_text)
                route_tag = "rag-answer"
                if not ok_num:
                    cleaned = get_missing_info_response(q, detected_lang)
                    route_tag = "fallback"
                res = {"answer": cleaned, "sources": sources, "mode": "rag", "escalate": False}
                set_cached_response(q, {**res, "route": route_tag}, doc_key=doc_key)
                add_conversation_turn(q, cleaned, lang)
                print(f"[ROUTE] {route_tag} (groq) | Query: '{_safe_log_str(q)}'", flush=True)
                return res

        # Case B: Below threshold -> fallback for specific detail queries against indexed docs; escalation otherwise
        detected_lang = detect_reply_language(q)
        if _is_explicit_escalation_query(q) or (is_biz and chunk_count == 0):
            escalate_msg = get_escalation_response(detected_lang)
            res = {"answer": escalate_msg, "sources": [], "mode": "rag", "escalate": True}
            set_cached_response(q, {**res, "route": "escalation"}, doc_key=doc_key)
            add_conversation_turn(q, escalate_msg, detected_lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] escalation | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Total: {t_total:.2f}s | Score: {best_score:.3f} < {sim_threshold}", flush=True)
            return res

        if (is_biz and chunk_count > 0) or mode == "rag":
            missing_reply = get_missing_info_response(q, detected_lang)
            res = {"answer": missing_reply, "sources": [], "mode": "rag", "escalate": False}
            set_cached_response(q, {**res, "route": "fallback"}, doc_key=doc_key)
            add_conversation_turn(q, missing_reply, detected_lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] fallback | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Total: {t_total:.2f}s | Score: {best_score:.3f} < {sim_threshold}", flush=True)
            return res

        # Case C: General question (not business) -> Direct model answer
        conv_prompt = build_conversational_prompt(q, detected_lang, history=active_history)
        conv_sys = get_slm_conversational_prompt(detected_lang)

        t_gen_start = time.perf_counter()
        ollama_ans, ollama_err = _invoke_ollama(conv_prompt, system=conv_sys)
        ollama_ans, retried = ensure_tanglish_reply(ollama_ans, detected_lang)
        t_gen = time.perf_counter() - t_gen_start
        print(f"[LANGUAGE] Query: '{_safe_log_str(q)}' | Detected: {detected_lang} | Step 2 Retry: {retried}", flush=True)

        if ollama_ans:
            cleaned = clean_model_output(ollama_ans, lang)
            res = {"answer": cleaned, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, {**res, "route": "rag-answer"}, doc_key=doc_key)
            add_conversation_turn(q, cleaned, lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] rag-answer (direct) | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Gen: {t_gen:.2f}s | Total: {t_total:.2f}s", flush=True)
            return res

        groq_ans, _ = _invoke_groq(f"System: {conv_sys}\n\n{conv_prompt}")
        if groq_ans:
            cleaned = clean_model_output(groq_ans, lang)
            res = {"answer": cleaned, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, {**res, "route": "rag-answer"}, doc_key=doc_key)
            add_conversation_turn(q, cleaned, lang)
            return res

        print(f"[ROUTE] fallback (no llm response) | Query: '{_safe_log_str(q)}'", flush=True)
        return {"answer": get_missing_info_response(q, detected_lang), "sources": [], "mode": "direct", "escalate": False}
    except Exception as e:
        print(f"[Chat Error] ask_question unhandled exception: {e}", flush=True)
        traceback.print_exc()
        return {
            "answer": "Sorry, a temporary error occurred while processing your question. Please try again.",
            "sources": [],
            "mode": "direct",
            "escalate": False,
        }


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
    Wraps the entire generator in try/except, resolves single active document,
    handles upload-ack & overview routes, checks number groundedness, and logs route tags.
    """
    t_start = time.perf_counter()
    try:
        q = (question or "").strip()
        if not q:
            yield f"data: {json.dumps({'error': 'Please enter a valid question.'})}\n\n"
            return

        active_doc = resolve_active_document(selected_doc)
        doc_key = active_doc or ""

        # 1. Instant Greeting Check at the VERY START
        greeting_reply = get_greeting_reply(q)
        if greeting_reply:
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] greeting | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            res = {"answer": greeting_reply, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, {**res, "route": "greeting"}, doc_key=doc_key)
            add_conversation_turn(q, greeting_reply, detect_language_style(q))
            yield f"data: {json.dumps({'token': greeting_reply})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
            return

        # 2. Upload / Acknowledgement check (never trigger retrieval or fallback)
        if is_upload_ack_message(q):
            ack_reply = get_upload_ack_reply(q)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] upload-ack | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            add_conversation_turn(q, ack_reply, detect_language_style(q))
            yield f"data: {json.dumps({'token': ack_reply})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
            return

        lang = detect_language_style(q)

        # 3. In-memory cache check
        cached = get_cached_response(q, doc_key=doc_key)
        if cached is not None:
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] {cached.get('route', 'rag-answer')} (cache) | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            yield f"data: {json.dumps({'token': cached.get('answer', '')})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': cached.get('sources', []), 'mode': cached.get('mode', 'direct'), 'escalate': cached.get('escalate', False)})}\n\n"
            return

        # 4. Instant small talk
        is_st, st_reply, _ = classify_smalltalk(q)
        if is_st:
            res = {"answer": st_reply, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, {**res, "route": "greeting"}, doc_key=doc_key)
            add_conversation_turn(q, st_reply, lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] greeting (smalltalk) | Query: '{_safe_log_str(q)}' | Total: {t_total*1000:.1f}ms", flush=True)
            yield f"data: {json.dumps({'token': st_reply})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
            return

        # 5. Identity check
        if is_identity_question(q):
            resp = get_identity_response(lang)
            res = {"answer": resp, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, {**res, "route": "greeting"}, doc_key=doc_key)
            add_conversation_turn(q, resp, lang)
            print(f"[ROUTE] greeting (identity) | Query: '{_safe_log_str(q)}'", flush=True)
            yield f"data: {json.dumps({'token': resp})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
            return

        raw_history = history if isinstance(history, list) else get_conversation_history()
        active_history = raw_history[-2:] if isinstance(raw_history, list) and raw_history else []
        chunk_count = _store.count()

        # 6. Overview / List question -> retrieve opening + catalog chunks from active doc
        if is_list_or_overview_question(q) and chunk_count > 0 and mode != "direct":
            top_k_list = get_top_k_list()
            t_ret_start = time.perf_counter()
            overview_chunks = get_overview_chunks(active_doc, top_k=top_k_list)
            t_retrieve = time.perf_counter() - t_ret_start
            if overview_chunks:
                sources = []
                for c in overview_chunks:
                    page_str = f" (page {c.page + 1})" if isinstance(c.page, int) and c.page >= 0 else ""
                    label = f"{c.source}{page_str}"
                    if label not in sources:
                        sources.append(label)
                context_text = format_overview_chunks(overview_chunks, max_total_chars=2000)
                detected_lang = detect_reply_language(q)
                sys_prompt = get_slm_rag_prompt(detected_lang, is_list=True)
                prompt = build_prompt_with_context(q, context_text, detected_lang, history=None)

                accumulated = []
                t_gen_start = time.perf_counter()
                t_first_token = None
                stream_err = None
                try:
                    for token in _stream_ollama(prompt, system=sys_prompt, num_predict=get_num_predict_list()):
                        if t_first_token is None:
                            t_first_token = time.perf_counter() - t_gen_start
                        accumulated.append(token)
                except Exception as e:
                    stream_err = e
                    print(f"[Ollama Stream Error] Overview stream error: {e}", flush=True)
                    traceback.print_exc()

                full_ans = clean_model_output("".join(accumulated).strip(), lang)
                if full_ans:
                    ok_num, bad_nums = check_number_groundedness(full_ans, context_text)
                    route_tag = "overview"
                    if not ok_num:
                        print(f"[GROUNDEDNESS FAIL] Overview unverified numbers {bad_nums} -> fallback", flush=True)
                        full_ans = get_missing_info_response(q, detected_lang)
                        route_tag = "fallback"
                    res = {"answer": full_ans, "sources": sources, "mode": "rag", "escalate": False}
                    set_cached_response(q, {**res, "route": route_tag}, doc_key=doc_key)
                    add_conversation_turn(q, full_ans, lang)
                    t_total = time.perf_counter() - t_start
                    print(f"[ROUTE] {route_tag} | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | First token: {t_first_token or 0:.2f}s | Total: {t_total:.2f}s | Prompt chars: {len(prompt)}", flush=True)
                    yield f"data: {json.dumps({'token': full_ans})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': sources, 'mode': 'rag', 'escalate': False})}\n\n"
                    return
                if stream_err:
                    err_reply = "Sorry, a temporary error occurred while summarizing the document. Please try again."
                    yield f"data: {json.dumps({'token': err_reply})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': False})}\n\n"
                    return

        sim_threshold = get_similarity_threshold()
        top_k = get_top_k()
        num_predict_val = get_num_predict()
        retrieval_query = build_retrieval_query(q, history=active_history)
        is_biz = is_business_question(q)

        t_ret_start = time.perf_counter()
        relevant_chunks = []
        best_score = 0.0
        retrieval_err = None

        if chunk_count > 0:
            relevant_chunks, best_score, retrieval_err = _store.search_with_status(
                retrieval_query, k=top_k, threshold=sim_threshold, history=active_history, source_filter=active_doc
            )
        t_retrieve = time.perf_counter() - t_ret_start

        if retrieval_err:
            err_msg = get_retrieval_error_response(lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] fallback (retrieval error) | Query: '{_safe_log_str(q)}' | Total: {t_total:.2f}s", flush=True)
            yield f"data: {json.dumps({'token': err_msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': False})}\n\n"
            return

        if mode == "direct":
            is_rag_candidate = False
        elif mode == "rag":
            is_rag_candidate = True
        else:
            is_rag_candidate = (chunk_count > 0)

        # Case A: Score above threshold -> Stream RAG answer from active document
        if is_rag_candidate and relevant_chunks and best_score >= sim_threshold:
            sources = []
            for c in relevant_chunks:
                page_str = f" (page {c.page + 1})" if isinstance(c.page, int) and c.page >= 0 else ""
                source_label = f"{c.source}{page_str}"
                if source_label not in sources:
                    sources.append(source_label)

            context_text = format_context_chunks(relevant_chunks, max_total_chars=1400)
            detected_lang = detect_reply_language(q)
            prompt = build_prompt_with_context(q, context_text, detected_lang, history=active_history)
            sys_prompt = get_slm_rag_prompt(detected_lang, is_list=False)

            accumulated = []
            t_gen_start = time.perf_counter()
            t_first_token = None
            stream_err = None
            try:
                for token in _stream_ollama(prompt, system=sys_prompt, num_predict=num_predict_val):
                    if t_first_token is None:
                        t_first_token = time.perf_counter() - t_gen_start
                    accumulated.append(token)
            except Exception as e:
                stream_err = e
                print(f"[Ollama Stream Error] RAG stream error: {e}", flush=True)
                traceback.print_exc()

            full_ans = clean_model_output("".join(accumulated).strip(), lang)
            if full_ans:
                ok_num, bad_nums = check_number_groundedness(full_ans, context_text)
                route_tag = "rag-answer"
                if not ok_num:
                    print(f"[GROUNDEDNESS FAIL] Unverified numbers {bad_nums} -> replacing with missing info", flush=True)
                    full_ans = get_missing_info_response(q, detected_lang)
                    route_tag = "fallback"
                elif "not available" in full_ans.lower() or "la illa" in full_ans.lower():
                    route_tag = "fallback"
                res = {"answer": full_ans, "sources": sources, "mode": "rag", "escalate": False}
                set_cached_response(q, {**res, "route": route_tag}, doc_key=doc_key)
                add_conversation_turn(q, full_ans, lang)
                t_total = time.perf_counter() - t_start
                print(f"[ROUTE] {route_tag} | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | First token: {t_first_token or 0:.2f}s | Total: {t_total:.2f}s | Prompt chars: {len(prompt)}", flush=True)
                yield f"data: {json.dumps({'token': full_ans})}\n\n"
                yield f"data: {json.dumps({'done': True, 'sources': sources, 'mode': 'rag', 'escalate': False})}\n\n"
                return

            if stream_err:
                print(f"[Ollama] Generation failed with error: {stream_err}", flush=True)
            else:
                print("[Ollama] Generation yielded 0 tokens", flush=True)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] fallback (stream error) | Query: '{_safe_log_str(q)}' | Total: {t_total:.2f}s", flush=True)
            err_reply = "Sorry, Ollama could not generate a reply right now. Please try again."
            yield f"data: {json.dumps({'token': err_reply})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': sources, 'mode': 'rag', 'escalate': False})}\n\n"
            return

        # Case B: Score below threshold
        detected_lang = detect_reply_language(q)
        if _is_explicit_escalation_query(q) or (is_biz and chunk_count == 0):
            escalate_msg = get_escalation_response(detected_lang)
            res = {"answer": escalate_msg, "sources": [], "mode": "rag", "escalate": True}
            set_cached_response(q, {**res, "route": "escalation"}, doc_key=doc_key)
            add_conversation_turn(q, escalate_msg, detected_lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] escalation | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Total: {t_total:.2f}s | Score: {best_score:.3f} < {sim_threshold}", flush=True)
            yield f"data: {json.dumps({'token': escalate_msg})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': True})}\n\n"
            return

        if (is_biz and chunk_count > 0) or mode == "rag":
            missing_reply = get_missing_info_response(q, detected_lang)
            res = {"answer": missing_reply, "sources": [], "mode": "rag", "escalate": False}
            set_cached_response(q, {**res, "route": "fallback"}, doc_key=doc_key)
            add_conversation_turn(q, missing_reply, detected_lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] fallback | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | Total: {t_total:.2f}s | Score: {best_score:.3f} < {sim_threshold}", flush=True)
            yield f"data: {json.dumps({'token': missing_reply})}\n\n"
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'rag', 'escalate': False})}\n\n"
            return

        # Case C: General question -> Direct model streaming
        conv_prompt = build_conversational_prompt(q, lang, history=active_history)
        conv_sys = get_slm_conversational_prompt(detected_lang)

        has_streamed = False
        accumulated = []
        t_gen_start = time.perf_counter()
        t_first_token = None
        stream_err = None
        try:
            for token in _stream_ollama(conv_prompt, system=conv_sys, num_predict=num_predict_val):
                if not has_streamed:
                    t_first_token = time.perf_counter() - t_gen_start
                has_streamed = True
                accumulated.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            stream_err = e
            print(f"[Ollama Stream Error] Direct stream error: {e}", flush=True)
            traceback.print_exc()

        if has_streamed:
            full_ans = "".join(accumulated).strip()
            res = {"answer": full_ans, "sources": [], "mode": "direct", "escalate": False}
            set_cached_response(q, {**res, "route": "rag-answer"}, doc_key=doc_key)
            add_conversation_turn(q, full_ans, lang)
            t_total = time.perf_counter() - t_start
            print(f"[ROUTE] rag-answer (direct) | Query: '{_safe_log_str(q)}' | Retrieval: {t_retrieve:.2f}s | First token: {t_first_token or 0:.2f}s | Total: {t_total:.2f}s | Prompt chars: {len(conv_prompt)}", flush=True)
            yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
            return

        if stream_err:
            print(f"[Ollama] Direct stream generation failed: {stream_err}", flush=True)
        else:
            print("[Ollama] Direct stream yielded 0 tokens", flush=True)
        t_total = time.perf_counter() - t_start
        print(f"[ROUTE] fallback (direct stream error) | Query: '{_safe_log_str(q)}' | Total: {t_total:.2f}s", flush=True)
        err_reply = "Sorry, Ollama could not respond right now. Please try again."
        yield f"data: {json.dumps({'token': err_reply})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
        return
    except Exception as e:
        print(f"[Chat Stream Error] Unhandled exception in stream_ask_question: {e}", flush=True)
        traceback.print_exc()
        yield f"data: {json.dumps({'token': 'Sorry, a temporary error occurred. Please try again.'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'sources': [], 'mode': 'direct', 'escalate': False})}\n\n"
        return


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
    if _store.last_uploaded_doc == clean_name:
        _store.last_uploaded_doc = list(_store.file_registry.keys())[-1] if _store.file_registry else None
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

