import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_CHAT_DIR = PROJECT_ROOT / "main_chat"
DATA_INGESTION_DIR = MAIN_CHAT_DIR / "data_ingestion"
RAG_ROUTE_DIR = MAIN_CHAT_DIR / "rag_pipeline"
SQL_ROUTE_DIR = MAIN_CHAT_DIR / "sql_pipeline"
API_DIR = PROJECT_ROOT / "api"

# Load .env from project root
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

# ============================================================================
# rethinkAI API Configuration
# ============================================================================
API_VERSION = os.getenv("API_VERSION", "2.0")
_raw_keys = os.getenv("RETHINKAI_API_KEYS", "").split(",")
RETHINKAI_API_KEYS = [k.strip() for k in _raw_keys if k.strip()]
HOST = os.getenv("API_HOST", "127.0.0.1")
PORT = int(os.getenv("API_PORT", "8888"))

# Flask settings
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")
SESSION_COOKIE_SECURE = os.getenv("FLASK_SESSION_COOKIE_SECURE", "False").lower() == "true"


# ============================================================================
# MySQL Configuration
# ============================================================================

MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "rethink_ai_boston")
MYSQL_MAX_RETRIES = 3

METADATA_CATALOG_PATH = os.getenv("METADATA_CATALOG_PATH", "")
METADATA_DIR = os.getenv("METADATA_DIR", "")
SCHEMA_METADATA_PATH = os.getenv("SCHEMA_METADATA_PATH", "")
PGSCHEMA = os.getenv("PGSCHEMA", "public")

# ============================================================================
# Gemini AI Configuration
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", GEMINI_MODEL)
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# Lazy-loaded client instance
_genai_client = None


def get_genai_client():
    """
    Get or create the singleton Gemini client.

    Returns:
        google.genai.Client: Configured Gemini client

    Raises:
        RuntimeError: If GEMINI_API_KEY is not configured
    """
    global _genai_client

    if _genai_client is not None:
        return _genai_client

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured in environment")

    try:
        from google import genai

        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
        return _genai_client
    except ImportError:
        raise RuntimeError("google-genai package not installed. " "Run: pip install google-genai")


def generate_content(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Generate content using Gemini.

    Args:
        prompt: The user prompt/question
        model: Model name (defaults to GEMINI_MODEL)
        temperature: Sampling temperature (0 = deterministic)
        system_instruction: Optional system prompt

    Returns:
        str: Generated text response
    """
    client = get_genai_client()
    model_name = model or GEMINI_MODEL

    from google.genai import types

    config_obj = types.GenerateContentConfig(
        temperature=temperature,
    )

    if system_instruction:
        config_obj.system_instruction = system_instruction

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config_obj,
    )

    return get_response_text(response).strip()


def generate_content_with_history(
    messages: list,
    model: Optional[str] = None,
    temperature: float = 0,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Generate content with conversation history.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  Roles should be 'user' or 'model' (not 'assistant')
        model: Model name (defaults to GEMINI_MODEL)
        temperature: Sampling temperature
        system_instruction: Optional system prompt

    Returns:
        str: Generated text response
    """
    client = get_genai_client()
    model_name = model or GEMINI_MODEL

    from google.genai import types

    # Convert messages to Content objects
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        # Map 'assistant' to 'model' for Gemini API
        if role == "assistant":
            role = "model"
        content = msg.get("content", "")
        contents.append(types.Content(role=role, parts=[types.Part(text=content)]))

    config_obj = types.GenerateContentConfig(
        temperature=temperature,
    )

    if system_instruction:
        config_obj.system_instruction = system_instruction

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config_obj,
    )

    return get_response_text(response).strip()


def embed_content(text: str, model: Optional[str] = None) -> list:
    """
    Generate embeddings for text.

    Args:
        text: Text to embed
        model: Embedding model name (defaults to GEMINI_EMBED_MODEL)

    Returns:
        list: Embedding vector
    """
    client = get_genai_client()
    model_name = model or GEMINI_EMBED_MODEL

    response = client.models.embed_content(
        model=model_name,
        contents=text,
    )

    if hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values
    elif hasattr(response, "embedding"):
        return response.embedding.values if hasattr(response.embedding, "values") else response.embedding

    raise RuntimeError("Unexpected embedding response format")


def embed_content_batch(texts: list, model: Optional[str] = None) -> list:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model: Embedding model name (defaults to GEMINI_EMBED_MODEL)

    Returns:
        list: List of embedding vectors
    """
    return [embed_content(text, model) for text in texts]


def get_response_text(response):
    """
    Extract text from new google-genai response format

    Args:
        response: Model response

    Returns:
        Text from the response
    """
    if hasattr(response, "candidates") and response.candidates:
        if response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        print("Warning: Empty model response")
    return ""


# ============================================================================
# Google Drive Configuration
# ============================================================================
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "")

# ============================================================================
# Email Configuration (Gmail OAuth 2.0)
# ============================================================================
EMAIL_ADDRESS = os.getenv("NEWSLETTER_EMAIL_ADDRESS", "")
GMAIL_CREDENTIALS_PATH = os.getenv("GMAIL_CREDENTIALS_PATH", "")
GMAIL_TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", "")

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))

# ============================================================================
# File Paths Configuration
# ============================================================================
_VECTORDB_DIR_RAW = os.getenv("VECTORDB_DIR", "vectordb_new")
if Path(_VECTORDB_DIR_RAW).is_absolute():
    VECTORDB_DIR = Path(_VECTORDB_DIR_RAW)
else:
    VECTORDB_DIR = (PROJECT_ROOT / _VECTORDB_DIR_RAW).resolve()

_TEMP_DIR_RAW = os.getenv("TEMP_DOWNLOAD_DIR", "main_chat/data_ingestion/temp_downloads")
if Path(_TEMP_DIR_RAW).is_absolute():
    TEMP_DOWNLOAD_DIR = Path(_TEMP_DIR_RAW)
else:
    TEMP_DOWNLOAD_DIR = (PROJECT_ROOT / _TEMP_DIR_RAW).resolve()

TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Processing Configuration
# ============================================================================
EMAIL_LOOKBACK_DAYS = int(os.getenv("EMAIL_LOOKBACK_DAYS", "7"))
MAX_FILES_PER_RUN = int(os.getenv("MAX_FILES_PER_RUN", "100"))
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() in ("true", "1", "yes")

# ============================================================================
# Sync State Files
# ============================================================================
SYNC_STATE_FILE = DATA_INGESTION_DIR / ".sync_state.json"
EMAIL_SYNC_STATE_FILE = DATA_INGESTION_DIR / ".email_sync_state.json"

# ============================================================================
# Supported File Extensions
# ============================================================================
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".txt", ".md"}

# ============================================================================
# Data URLs:
# Boston 311 data portal
# Dotnews (Dorchester Reporter
# ============================================================================
BOSTON_CKAN_API = os.getenv("BOSTON_CKAN_API", "https://data.boston.gov/api/3/action")
DOTNEWS_URL = os.getenv("DOTNEWS_URL", "https://www.dotnews.com/inprint/")


# ============================================================================
# Validation
# ============================================================================
def validate_config() -> list:
    """Validate that required configuration is present."""
    errors = []

    if not GOOGLE_DRIVE_FOLDER_ID:
        errors.append("GOOGLE_DRIVE_FOLDER_ID is not set")

    if GOOGLE_CREDENTIALS_PATH and not Path(GOOGLE_CREDENTIALS_PATH).exists():
        errors.append(f"Google credentials file not found: {GOOGLE_CREDENTIALS_PATH}")

    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set")

    return errors


def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 80)
    print("Configuration Summary")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Gemini Model: {GEMINI_MODEL}")
    print(f"Gemini Summary Model: {GEMINI_SUMMARY_MODEL}")
    print(f"Gemini Embed Model: {GEMINI_EMBED_MODEL}")
    print(f"Vector DB Directory: {VECTORDB_DIR}")
    print(f"Temp Download Directory: {TEMP_DOWNLOAD_DIR}")
    print("=" * 80)

    errors = validate_config()
    if errors:
        print("\n⚠️  Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        print()


if __name__ == "__main__":
    print_config_summary()
