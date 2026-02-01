import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# ============================================================================
# Path Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env from project root
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

# ============================================================================
# Environment Detection
# ============================================================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "local").lower()
IS_PRODUCTION = ENVIRONMENT == "production"
IS_LOCAL = ENVIRONMENT == "local"


def _env(key: str, default: str = "") -> str:
    """
    Get environment-specific value.

    Looks for LOCAL_<key> or PROD_<key> based on current environment,
    falls back to <key> if prefixed version not found.
    Empty strings are treated as "not set" and fall through to default.
    """
    prefix = "PROD_" if IS_PRODUCTION else "LOCAL_"
    # Try prefixed version first
    value = os.getenv(f"{prefix}{key}")
    if value:  # Truthy check - empty string falls through
        return value
    # Fall back to unprefixed
    value = os.getenv(key)
    if value:  # Truthy check - empty string falls through
        return value
    return default


# ============================================================================
# Base Paths (environment-specific)
# ============================================================================
_base_path_str = _env("BASE_PATH", str(PROJECT_ROOT))
BASE_PATH = Path(_base_path_str) if _base_path_str else PROJECT_ROOT

# Standard project directories (derived from PROJECT_ROOT, not BASE_PATH)
MAIN_CHAT_DIR = PROJECT_ROOT / "main_chat"
DATA_INGESTION_DIR = MAIN_CHAT_DIR / "data_ingestion"
RAG_ROUTE_DIR = MAIN_CHAT_DIR / "rag_pipeline"
SQL_ROUTE_DIR = MAIN_CHAT_DIR / "sql_pipeline"
API_DIR = PROJECT_ROOT / "api"

# ============================================================================
# rethinkAI API Configuration
# ============================================================================
API_VERSION = os.getenv("API_VERSION", "2.0")
_raw_keys = os.getenv("RETHINKAI_API_KEYS", "").split(",")
RETHINKAI_API_KEYS = [k.strip() for k in _raw_keys if k.strip()]
HOST = _env("API_HOST", "127.0.0.1")
PORT = int(_env("API_PORT", "8888"))

# Flask settings
SECRET_KEY = _env("FLASK_SECRET_KEY", "")
SESSION_COOKIE_SECURE = _env("FLASK_SESSION_COOKIE_SECURE", "False").lower() == "true"

# ============================================================================
# MySQL Configuration (environment-specific)
# ============================================================================
MYSQL_HOST = _env("MYSQL_HOST", "")
MYSQL_PORT = _env("MYSQL_PORT", "3306")
MYSQL_USER = _env("MYSQL_USER", "")
MYSQL_PASSWORD = _env("MYSQL_PASSWORD", "")
MYSQL_DB = _env("MYSQL_DB", "rethink_ai_boston")
MYSQL_MAX_RETRIES = _env("MYSQL_MAX_RETRIES", "3")

# Connection pooling settings
MYSQL_POOL_SIZE = int(_env("MYSQL_POOL_SIZE", "5"))
MYSQL_POOL_RECYCLE = int(_env("MYSQL_POOL_RECYCLE", "3600"))

METADATA_CATALOG_PATH = BASE_PATH / "main_chat/metadata/tables_catalog.json"
METADATA_DIR = BASE_PATH / "/main_chat/metadata"
SCHEMA_METADATA_PATH = os.getenv("SCHEMA_METADATA_PATH", "")
PGSCHEMA = os.getenv("PGSCHEMA", "public")

# ============================================================================
# Google Drive Configuration
# ============================================================================
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_CREDENTIALS_PATH = BASE_PATH / "credentials/gdrive_client_secret.json"

# ============================================================================
# Email Configuration (Gmail OAuth 2.0)
# ============================================================================
EMAIL_ADDRESS = os.getenv("NEWSLETTER_EMAIL_ADDRESS", "")
GMAIL_CREDENTIALS_PATH = BASE_PATH / "credentials/gmail_oauth_credentials.json"
GMAIL_TOKEN_PATH = BASE_PATH / "credentials/gmail_token.json"

IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))

# ============================================================================
# File Paths Configuration (environment-specific via BASE_PATH)
# ============================================================================
# These can be overridden individually or will use BASE_PATH as root
VECTORDB_DIR = BASE_PATH / "vectordb"
DATA_DOWNLOAD_DIR = BASE_PATH / "data"
# Ensure directories exist (with validation)
if DATA_DOWNLOAD_DIR and str(DATA_DOWNLOAD_DIR) != ".":
    try:
        DATA_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create DATA_DOWNLOAD_DIR '{DATA_DOWNLOAD_DIR}': {e}")

SYNC_STATE_FILE = DATA_DOWNLOAD_DIR / ".sync_state_gdrive.json"
EMAIL_SYNC_STATE_FILE = DATA_DOWNLOAD_DIR / ".sync_state_gmail.json"
DOTNEWS_SYNC_STATE_FILENAME = DATA_DOWNLOAD_DIR / ".sync_state_dotnews.json"


# ============================================================================
# Processing Configuration
# ============================================================================
EMAIL_LOOKBACK_DAYS = int(os.getenv("EMAIL_LOOKBACK_DAYS", "7"))
MAX_FILES_PER_RUN = int(os.getenv("MAX_FILES_PER_RUN", "100"))

# LLM processing settings
LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "3"))

# ============================================================================
# Supported File Extensions
# ============================================================================
SUPPORTED_EXTENSIONS = frozenset({".pdf", ".doc", ".docx", ".txt", ".md"})

# ============================================================================
# External Data URLs
# ============================================================================
BOSTON_CKAN_API = os.getenv("BOSTON_CKAN_API", "https://data.boston.gov/api/3/action")

# ============================================================================
# Gemini AI Configuration (typically same for both environments)
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_SUMMARY_MODEL = os.getenv("GEMINI_SUMMARY_MODEL", GEMINI_MODEL)
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

# Lazy-loaded client instance
_genai_client = None


def get_genai_client():
    """Get or create the singleton Gemini client."""
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
        raise RuntimeError("google-genai package not installed. Run: pip install google-genai")


def generate_content(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0,
    system_instruction: Optional[str] = None,
) -> str:
    """Generate content using Gemini."""
    client = get_genai_client()
    model_name = model or GEMINI_MODEL

    from google.genai import types

    config_obj = types.GenerateContentConfig(temperature=temperature)

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
    """Generate content with conversation history."""
    client = get_genai_client()
    model_name = model or GEMINI_MODEL

    from google.genai import types

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"
        content = msg.get("content", "")
        contents.append(types.Content(role=role, parts=[types.Part(text=content)]))

    config_obj = types.GenerateContentConfig(temperature=temperature)

    if system_instruction:
        config_obj.system_instruction = system_instruction

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config_obj,
    )

    return get_response_text(response).strip()


def embed_content(text: str, model: Optional[str] = None) -> list:
    """Generate embeddings for text."""
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
    """Generate embeddings for multiple texts."""
    return [embed_content(text, model) for text in texts]


def get_response_text(response):
    """Extract text from google-genai response format."""
    if hasattr(response, "candidates") and response.candidates:
        if response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        print("Warning: Empty model response")
    return ""


# ============================================================================
# Validation
# ============================================================================
def validate_config(test_connections: bool = True) -> list:
    errors = []
    warnings = []

    # Check required environment variables
    if not GOOGLE_DRIVE_FOLDER_ID:
        errors.append("GOOGLE_DRIVE_FOLDER_ID is not set")

    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set")

    if not MYSQL_HOST:
        errors.append(f"MYSQL_HOST is not set (looked for {'PROD' if IS_PRODUCTION else 'LOCAL'}_MYSQL_HOST)")

    if not MYSQL_USER:
        errors.append("MYSQL_USER is not set")

    if not MYSQL_PASSWORD:
        warnings.append("MYSQL_PASSWORD is empty (may be intentional for local dev)")

    # Check credential files exist
    if GOOGLE_CREDENTIALS_PATH:
        if not GOOGLE_CREDENTIALS_PATH.exists():
            errors.append(f"Google Drive credentials file not found: {GOOGLE_CREDENTIALS_PATH}")
    else:
        errors.append("GOOGLE_CREDENTIALS_PATH is not configured")

    if GMAIL_CREDENTIALS_PATH:
        if not GMAIL_CREDENTIALS_PATH.exists():
            warnings.append(f"Gmail OAuth credentials file not found: {GMAIL_CREDENTIALS_PATH}")
    else:
        warnings.append("GMAIL_CREDENTIALS_PATH is not configured")

    if GMAIL_TOKEN_PATH and not GMAIL_TOKEN_PATH.exists():
        warnings.append(f"Gmail token not found: {GMAIL_TOKEN_PATH} (run auth flow to create)")

    # Check directories exist or can be created
    for name, path in [
        ("VECTORDB_DIR", VECTORDB_DIR),
        ("DATA_DOWNLOAD_DIR", DATA_DOWNLOAD_DIR),
    ]:
        if path:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"  ✔ Created {name}: {path}")
                except Exception as e:
                    errors.append(f"Cannot create {name} '{path}': {e}")
            elif not os.access(path, os.W_OK):
                errors.append(f"{name} is not writable: {path}")

    # Check metadata paths
    if METADATA_CATALOG_PATH and not METADATA_CATALOG_PATH.exists():
        warnings.append(f"Metadata catalog not found: {METADATA_CATALOG_PATH}")

    # Test connections (if requested)
    if test_connections:
        print("Testing connections...")

        # Test Google Drive connection
        gdrive_error = _test_google_drive_connection()
        if gdrive_error:
            errors.append(f"Google Drive: {gdrive_error}")

        # Test Gmail connection
        gmail_result = _test_gmail_connection()
        if gmail_result:
            if gmail_result.startswith("AUTH_REQUIRED:"):
                warnings.append(f"Gmail: {gmail_result.replace('AUTH_REQUIRED:', '')}")
            else:
                errors.append(f"Gmail: {gmail_result}")

        # Test MySQL connection
        mysql_error = _test_mysql_connection()
        if mysql_error:
            errors.append(f"MySQL: {mysql_error}")

        # Test Gemini API
        gemini_error = _test_gemini_connection()
        if gemini_error:
            errors.append(f"Gemini API: {gemini_error}")

    # Print warnings (non-fatal)
    if warnings:
        print("\n⚠  Configuration Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return errors


def _test_google_drive_connection() -> Optional[str]:
    if not GOOGLE_CREDENTIALS_PATH or not GOOGLE_CREDENTIALS_PATH.exists():
        return "Credentials file not found"

    if not GOOGLE_DRIVE_FOLDER_ID:
        return "GOOGLE_DRIVE_FOLDER_ID not set"

    try:
        from google.oauth2.service_account import Credentials as ServiceAccountCredentials
        from googleapiclient.discovery import build

        creds = ServiceAccountCredentials.from_service_account_file(str(GOOGLE_CREDENTIALS_PATH), scopes=["https://www.googleapis.com/auth/drive.readonly"])
        service = build("drive", "v3", credentials=creds)

        # Try to get folder metadata (validates both auth and folder access)
        folder = service.files().get(fileId=GOOGLE_DRIVE_FOLDER_ID, fields="id, name").execute()

        print(f"  ✔ Google Drive: Connected to folder '{folder.get('name', 'unknown')}'")
        return None

    except ImportError:
        return "google-auth or google-api-python-client not installed"
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            return f"Folder not found or not shared with service account: {GOOGLE_DRIVE_FOLDER_ID}"
        elif "403" in error_msg:
            return "Permission denied - check service account has access to folder"
        elif "invalid_grant" in error_msg.lower():
            return "Invalid credentials - check service account JSON file"
        else:
            return f"Connection failed: {error_msg}"


def _test_gmail_connection() -> Optional[str]:
    if not GMAIL_CREDENTIALS_PATH or not GMAIL_CREDENTIALS_PATH.exists():
        return "OAuth credentials file not found"

    if not GMAIL_TOKEN_PATH:
        return "AUTH_REQUIRED: Token path not configured"

    if not GMAIL_TOKEN_PATH.exists():
        return "AUTH_REQUIRED: Token not found - run 'python email_to_calendar_sql.py --auth' to authenticate"

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        # Load existing token
        creds = Credentials.from_authorized_user_file(str(GMAIL_TOKEN_PATH), scopes=["https://www.googleapis.com/auth/gmail.readonly"])

        # Check if token needs refresh
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save refreshed token
                with open(GMAIL_TOKEN_PATH, "w") as token_file:
                    token_file.write(creds.to_json())
                print("  ℹ Gmail: Token refreshed")
            except Exception as refresh_error:
                return f"AUTH_REQUIRED: Token expired and refresh failed: {refresh_error}"
        elif creds.expired:
            return "AUTH_REQUIRED: Token expired with no refresh token - re-authenticate"

        # Test connection by getting user profile
        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        email = profile.get("emailAddress", "unknown")

        print(f"  ✔ Gmail: Connected as {email}")
        return None

    except ImportError:
        return "google-auth or google-api-python-client not installed"
    except Exception as e:
        error_msg = str(e)
        if "invalid_grant" in error_msg.lower():
            return "AUTH_REQUIRED: Token invalid - re-authenticate"
        elif "401" in error_msg:
            return "AUTH_REQUIRED: Unauthorized - re-authenticate"
        elif "403" in error_msg:
            return "Permission denied - check OAuth scopes include gmail.readonly"
        else:
            return f"Connection failed: {error_msg}"


def _test_mysql_connection() -> Optional[str]:
    if not MYSQL_HOST or not MYSQL_USER:
        return "Host or user not configured"

    try:
        import pymysql

        conn = pymysql.connect(host=MYSQL_HOST, port=int(MYSQL_PORT), user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DB, connect_timeout=10)

        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()

        conn.close()
        print(f"  ✔ MySQL: Connected to {MYSQL_HOST}/{MYSQL_DB}")
        return None

    except ImportError:
        return "pymysql not installed"
    except Exception as e:
        error_msg = str(e)
        if "Access denied" in error_msg:
            return f"Access denied for user '{MYSQL_USER}'"
        elif "Can't connect" in error_msg or "Connection refused" in error_msg:
            return f"Cannot connect to {MYSQL_HOST}:{MYSQL_PORT}"
        elif "Unknown database" in error_msg:
            return f"Database '{MYSQL_DB}' does not exist"
        else:
            return f"Connection failed: {error_msg}"


def _test_gemini_connection() -> Optional[str]:
    if not GEMINI_API_KEY:
        return "API key not configured"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Test with a minimal generation request
        response = client.models.generate_content(model=GEMINI_MODEL, contents="Reply with only: All Good!", config=types.GenerateContentConfig(temperature=0, max_output_tokens=10))

        # Verify we got a valid response
        if response and hasattr(response, "candidates") and response.candidates:
            text = ""
            if response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text.strip().lower()

            if text:
                print(f"  ✔ Gemini API: Model '{GEMINI_MODEL}' responded with: {text}")
                return None
            else:
                return "API responded but returned empty content"
        else:
            return "API responded but returned no candidates"

    except ImportError:
        return "google-genai not installed. Run: pip install google-genai"
    except Exception as e:
        error_msg = str(e)

        # Parse common error types
        if "API_KEY_INVALID" in error_msg or "401" in error_msg:
            return "Invalid API key"
        elif "API key not valid" in error_msg:
            return "API key not valid. Check it was copied correctly."
        elif "PERMISSION_DENIED" in error_msg or "403" in error_msg:
            return f"API key doesn't have permission for model '{GEMINI_MODEL}'"
        elif "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            # Rate limited means the key works!
            print("  ✔ Gemini API: Key valid (rate limited, but working)")
            return None
        elif "not found" in error_msg.lower() or "404" in error_msg:
            return f"Model '{GEMINI_MODEL}' not found - check model name"
        elif "INVALID_ARGUMENT" in error_msg:
            return f"Invalid request - model '{GEMINI_MODEL}' may not support this operation"
        else:
            return f"API call failed: {error_msg}"


def print_config_summary(test_connections: bool = False):
    print("=" * 80)
    print(f"Configuration Summary  [ENVIRONMENT: {ENVIRONMENT.upper()}]")
    print("=" * 80)
    print(f"Base Path: {BASE_PATH}")
    print()
    print("Database:")
    print(f"  MySQL Host: {MYSQL_HOST}:{MYSQL_PORT}")
    print(f"  MySQL DB: {MYSQL_DB}")
    print(f"  MySQL User: {MYSQL_USER}")
    print()
    print("AI/LLM:")
    print(f"  Gemini Model: {GEMINI_MODEL}")
    print(f"  Gemini Embed Model: {GEMINI_EMBED_MODEL}")
    print(f"  Gemini API Key: {'*' * 10}...{GEMINI_API_KEY[-4:] if GEMINI_API_KEY else 'NOT SET'}")
    print()
    print("Google Services:")
    print(f"  Drive Folder ID: {GOOGLE_DRIVE_FOLDER_ID or 'NOT SET'}")
    print(f"  Drive Credentials: {GOOGLE_CREDENTIALS_PATH}")
    print(f"  Gmail Credentials: {GMAIL_CREDENTIALS_PATH}")
    print(f"  Gmail Token: {GMAIL_TOKEN_PATH} {'✔' if GMAIL_TOKEN_PATH and GMAIL_TOKEN_PATH.exists() else '(not found)'}")
    print()
    print("Directories:")
    print(f"  Vector DB: {VECTORDB_DIR}")
    print(f"  Data Downloads: {DATA_DOWNLOAD_DIR}")
    print()
    print(f"Processing: LLM Max Workers = {LLM_MAX_WORKERS}")
    print("=" * 80)

    errors = validate_config(test_connections=test_connections)

    if errors:
        print("\n✗ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        print()
    else:
        print("\n✔ Configuration valid!")
        if test_connections:
            print("   All connections successful.")
    print()

    return len(errors) == 0


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate configuration")
    parser.add_argument("--test-connections", "-t", action="store_true", help="Test actual connections to external services")
    args = parser.parse_args()

    success = print_config_summary(test_connections=args.test_connections)
    sys.exit(0 if success else 1)
