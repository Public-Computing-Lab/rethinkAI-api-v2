"""
Google Drive to Vector DB Ingestion
Downloads new files from a shared Google Drive folder and adds them to the vector database.
For newsletters: extracts events page-by-page using LLM and stores in SQL.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import concurrent.futures

# Google Drive API
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Document processing
from langchain_chroma import Chroma

# Local imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from main_chat.rag_pipeline.rag_retrieval import GeminiEmbeddings
import main_chat.sql_pipeline.sql_retrieval as sql_retrieval
from main_chat.data_ingestion.utils.document_processor import process_file_to_documents, extract_pages_from_pdf
from main_chat.data_ingestion.utils.log_util import log_debug, log_info, log_error, log_success, log_warning

# =============================================================================
# Pre-compiled regex patterns (compiled once at module load)
# =============================================================================

# Date patterns for PDF content extraction
_PDF_DATE_PATTERNS = [
    # Full format: "Day, Month Day, Year"
    re.compile(r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+" r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+" r"(\d{1,2}),\s+(\d{4})", re.IGNORECASE),
    # Without day of week: "November 20, 2025"
    re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+" r"(\d{1,2}),\s+(\d{4})", re.IGNORECASE),
    # Alternative: "Nov 20, 2025"
    re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+" r"(\d{1,2}),\s+(\d{4})", re.IGNORECASE),
]

# Date patterns for filename extraction with their group order (year, month, day)
_FILENAME_DATE_PATTERNS = [
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), lambda m: (m.group(1), m.group(2), m.group(3))),  # YYYY-MM-DD
    (re.compile(r"(\d{2})/(\d{2})/(\d{4})"), lambda m: (m.group(3), m.group(1), m.group(2))),  # MM/DD/YYYY
    (re.compile(r"(\d{4})(\d{2})(\d{2})"), lambda m: (m.group(1), m.group(2), m.group(3))),  # YYYYMMDD
    (re.compile(r"(\d{2})_(\d{2})_(\d{4})"), lambda m: (m.group(3), m.group(1), m.group(2))),  # MM_DD_YYYY
]

# JSON cleanup patterns
_JSON_MARKDOWN_FENCE_START = re.compile(r"^```(?:json|javascript)?\s*", re.MULTILINE)
_JSON_MARKDOWN_FENCE_END = re.compile(r"\s*```$")
_JSON_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_JSON_TRAILING_COMMAS = re.compile(r",\s*([}\]])")
_JSON_ARRAY_OR_OBJECT = re.compile(r"[\[{].*[}\]]", re.DOTALL)

# Time validation pattern
_TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")

# Month name to number mapping
_MONTH_MAP = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


# =============================================================================
# Sync State Management
# =============================================================================


def load_sync_state() -> dict:
    """Load the state of what files have been synced."""
    if config.SYNC_STATE_FILE.exists():
        try:
            return json.loads(config.SYNC_STATE_FILE.read_text())
        except Exception:
            return {"processed_files": {}, "last_sync": None}
    return {"processed_files": {}, "last_sync": None}


def save_sync_state(state: dict) -> None:
    """Save the sync state to track processed files."""
    state["last_sync"] = datetime.now().isoformat()
    config.SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


# =============================================================================
# Google Drive API
# =============================================================================


def get_drive_service():
    """Authenticate and return Google Drive service."""
    creds_path = Path(config.GOOGLE_CREDENTIALS_PATH)
    if not creds_path.exists():
        raise FileNotFoundError(f"Google credentials file not found: {config.GOOGLE_CREDENTIALS_PATH}")

    creds = ServiceAccountCredentials.from_service_account_file(config.GOOGLE_CREDENTIALS_PATH, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds)


def list_subfolders(service, folder_id: str) -> List[dict]:
    """List all subfolders in a Google Drive folder."""
    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    try:
        results = service.files().list(q=query, fields="files(id, name)", pageSize=100).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to list subfolders from Google Drive: {e}")
    return results.get("files", [])


def list_files_in_folder(service, folder_id: str, folder_name: str, processed_files: dict) -> List[dict]:
    """List all files in a specific Google Drive folder that haven't been processed yet."""
    query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
    try:
        results = service.files().list(q=query, fields="files(id, name, mimeType, modifiedTime, md5Checksum)", pageSize=1000).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to list files from folder '{folder_name}': {e}")

    all_files = results.get("files", [])
    new_files = []

    for file in all_files:
        file_id = file["id"]
        modified_time = file.get("modifiedTime", "")

        # Check if we've seen this file with the same modification time
        if file_id in processed_files:
            if processed_files[file_id].get("modifiedTime") == modified_time:
                continue

        # Check file extension
        ext = Path(file["name"]).suffix.lower()
        if ext in config.SUPPORTED_EXTENSIONS:
            file["folder_category"] = folder_name
            new_files.append(file)

    return new_files


def list_new_files_from_drive(service, folder_id: str, processed_files: dict) -> List[dict]:
    """List all files in the Google Drive folder and subfolders that haven't been processed yet."""
    if not folder_id:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID is not set")

    all_new_files = []

    # Get files directly in the root folder
    root_files = list_files_in_folder(service, folder_id, "root", processed_files)
    all_new_files.extend(root_files)
    if root_files:
        log_debug(f"  Found {len(root_files)} new files in root folder")

    # Scan each subfolder
    subfolders = list_subfolders(service, folder_id)
    log_debug(f"  Found {len(subfolders)} subfolders: {[f['name'] for f in subfolders]}")

    for subfolder in subfolders:
        subfolder_files = list_files_in_folder(service, subfolder["id"], subfolder["name"], processed_files)
        all_new_files.extend(subfolder_files)
        if subfolder_files:
            log_debug(f"  Found {len(subfolder_files)} new files in '{subfolder['name']}'")

    return all_new_files


def download_file(service, file_id: str, file_name: str, folder_category: str = None) -> Path:
    """Download a file from Google Drive to temp directory, organized by category."""
    request = service.files().get_media(fileId=file_id)

    # Determine target directory based on folder category
    if folder_category and folder_category != "root":
        target_dir = config.DATA_DOWNLOAD_DIR / folder_category
    else:
        target_dir = config.DATA_DOWNLOAD_DIR

    # Ensure directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    local_path = target_dir / file_name

    try:
        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    log_debug(f"    Download {int(status.progress() * 100)}%")
    except Exception as e:
        raise RuntimeError(f"Failed to download file {file_name}: {e}")

    return local_path


# =============================================================================
# Date Extraction Utilities
# =============================================================================


def _extract_date_from_pdf_content(file_path: Path) -> Optional[str]:
    """
    Extract publication date from the first page of the PDF.
    Returns date in YYYY-MM-DD format or None if not found.
    """
    try:
        pages = extract_pages_from_pdf(file_path)
        if not pages:
            return None

        first_page_text = pages[0]["text"]

        for pattern in _PDF_DATE_PATTERNS:
            match = pattern.search(first_page_text)
            if match:
                try:
                    month_name, day, year = match.groups()
                    month = _MONTH_MAP.get(month_name.lower())
                    if month:
                        date_str = f"{year}-{month}-{day.zfill(2)}"
                        datetime.strptime(date_str, "%Y-%m-%d")  # Validate
                        return date_str
                except (ValueError, AttributeError):
                    continue

        return None
    except Exception as e:
        log_debug(f"    ⚠ Could not extract date from PDF content: {e}")
        return None


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Try to extract a publication date from newsletter filename.
    Returns date in YYYY-MM-DD format or None if not found.
    """
    for pattern, extractor in _FILENAME_DATE_PATTERNS:
        match = pattern.search(filename)
        if match:
            try:
                year, month, day = extractor(match)
                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                datetime.strptime(date_str, "%Y-%m-%d")  # Validate
                return date_str
            except (ValueError, IndexError):
                continue
    return None


# =============================================================================
# JSON Cleanup Utilities
# =============================================================================


def _clean_llm_json(text: str) -> str:
    """
    Clean LLM JSON response in a single pass.
    Handles markdown fences, control characters, and trailing commas.
    """
    if not text:
        return ""

    text = text.strip()

    # Strip markdown code fences
    text = _JSON_MARKDOWN_FENCE_START.sub("", text)
    text = _JSON_MARKDOWN_FENCE_END.sub("", text)

    # Remove control characters
    text = _JSON_CONTROL_CHARS.sub(" ", text)

    # Remove trailing commas before } or ]
    text = _JSON_TRAILING_COMMAS.sub(r"\1", text)

    # Extract JSON array/object if embedded in other text
    text = text.strip()
    if not (text.startswith("[") or text.startswith("{")):
        match = _JSON_ARRAY_OR_OBJECT.search(text)
        if match:
            text = match.group(0)

    return text


# =============================================================================
# Event Extraction
# =============================================================================


def extract_events_from_page(page_text: str, page_num: int, source: str, publication_date: str = None) -> List[Dict]:
    """
    Use LLM to extract structured events from a single newsletter page.
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    if not page_text.strip() or len(page_text.strip()) < 50:
        return []

    # Truncate if too long
    max_chars = 8000
    if len(page_text) > max_chars:
        page_text = page_text[:max_chars] + "\n\n[... text truncated ...]"

    # Build date context for the prompt
    date_context = ""
    if publication_date:
        try:
            pub_dt = datetime.strptime(publication_date, "%Y-%m-%d")
            date_context = f"""
IMPORTANT DATE CONTEXT:
- Newsletter publication date: {publication_date} ({pub_dt.strftime('%A, %B %d, %Y')})
- Use this date to convert day-of-week references (Monday, Tuesday, etc.) to exact dates
- For example, if the newsletter is dated {publication_date} and an event says "Monday", calculate which Monday that refers to
- Events typically occur in the week following the newsletter publication date
- Always prefer exact dates (YYYY-MM-DD) over day names when possible
"""
        except Exception:
            pass

    # NOTE: This prompt must not be modified per user requirements
    prompt = f"""
You are reading PAGE {page_num} of a community newsletter.
{date_context}
Extract ALL events with their dates and times from this page.

CRITICAL: Convert day-of-week references (Monday, Tuesday, Wednesday, etc.) to EXACT dates (YYYY-MM-DD) using the newsletter publication date as reference.
- If an event says "Monday" and the newsletter is dated {publication_date or 'unknown'}, calculate the exact Monday date
- Events in newsletters typically refer to the upcoming week
- Always provide start_date and end_date in ISO format (YYYY-MM-DD) when possible
- Only use null if the date truly cannot be determined

Return ONLY valid JSON (no explanations, no markdown, no code fences), in this exact format:
[
  {{
    "event_name": "...",
    "event_date": "...",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "start_time": "HH:MM or null",
    "end_time": "HH:MM or null",
    "raw_text": "...",
    "location": "... or null",
    "category": "... or null"
  }},
  ...
]

Field rules:
- "event_name": Short descriptive name (REQUIRED - must always be provided)
- "event_date": Date label as written (REQUIRED - e.g., "Monday", "June 3-5", "All week", "Ongoing", "TBA" - always provide something, even if approximate)
- "start_date": ISO date YYYY-MM-DD when event starts - CONVERT day names to exact dates using publication date
- "end_date": ISO date YYYY-MM-DD when event ends (or null if same day) - CONVERT day names to exact dates
- "start_time": 24-hour time HH:MM (or null)
- "end_time": 24-hour time HH:MM (or null)
- "raw_text": Original text describing the event (include full details)
- "location": Where the event takes place (or null)
- "category": Choose one best fit: "Youth/Family", "Public Meeting", "Arts/Culture", "Health/Wellness", "Housing", "Safety", "Education", "Other"

If there are NO events on this page, return an empty array: []

Be thorough but conservative: extract all clear events but never invent information, but DO convert day names to exact dates when the publication date is provided.

Page {page_num} text:
\"\"\"
{page_text}
\"\"\"
"""

    try:
        text_response = config.generate_content(
            prompt=prompt,
            model=config.GEMINI_MODEL,
            temperature=0,
        )

        if not text_response:
            log_debug(f"    ⚠ Empty response from LLM for page {page_num}")
            return []

        # Clean JSON in one pass
        text_response = _clean_llm_json(text_response)

        if not text_response or not (text_response.startswith("[") or text_response.startswith("{")):
            log_debug(f"    ⚠ Invalid JSON response format for page {page_num}")
            return []

        try:
            events = json.loads(text_response)
        except json.JSONDecodeError as json_err:
            log_debug(f"    ⚠ JSON parse error for page {page_num}: {json_err}")
            return []

        if not isinstance(events, list):
            log_debug(f"    ⚠ Expected list but got {type(events).__name__} for page {page_num}")
            return []

        # Validate and clean events
        return _validate_events(events, source, page_num)

    except Exception as e:
        log_debug(f"    ⚠ Error extracting events from page {page_num}: {e}")
        return []


def _validate_events(events: List[Dict], source: str, page_num: int) -> List[Dict]:
    """Validate and clean a list of extracted events."""
    validated = []

    for event in events:
        if not isinstance(event, dict):
            continue

        event_name = event.get("event_name", "").strip() if event.get("event_name") else ""
        if not event_name:
            continue

        # Ensure event_date exists
        event_date = event.get("event_date", "").strip() if event.get("event_date") else ""
        if not event_date:
            event_date = event.get("start_date") or event.get("end_date") or "no info"

        # Clean string fields
        cleaned_event = {
            "event_name": event_name,
            "event_date": event_date,
            "raw_text": (event.get("raw_text") or "").strip(),
            "location": (event.get("location") or "").strip() or None,
            "category": (event.get("category") or "Other").strip(),
            "source": source,
            "page_number": page_num,
        }

        # Validate date formats
        for date_key in ["start_date", "end_date"]:
            date_val = event.get(date_key)
            if date_val and str(date_val).strip().lower() not in ("null", "none", ""):
                try:
                    datetime.strptime(str(date_val).strip(), "%Y-%m-%d")
                    cleaned_event[date_key] = str(date_val).strip()
                except ValueError:
                    cleaned_event[date_key] = None
            else:
                cleaned_event[date_key] = None

        # Validate time formats
        for time_key in ["start_time", "end_time"]:
            time_val = event.get(time_key)
            if time_val and str(time_val).strip().lower() not in ("null", "none", ""):
                if _TIME_PATTERN.match(str(time_val).strip()):
                    cleaned_event[time_key] = str(time_val).strip()
                else:
                    cleaned_event[time_key] = None
            else:
                cleaned_event[time_key] = None

        validated.append(cleaned_event)

    return validated


# =============================================================================
# Database Operations (Batched)
# =============================================================================


def insert_events_to_db(events: List[Dict]) -> int:
    """Insert events into the weekly_events SQL table using batch inserts."""
    if not events:
        return 0

    conn = sql_retrieval._get_db_connection()
    inserted_count = 0

    try:
        with conn.cursor() as cur:
            # Ensure table exists
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS weekly_events (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    source_pdf VARCHAR(255) NULL,
                    page_number INT NULL,
                    event_name VARCHAR(255) NOT NULL,
                    event_date VARCHAR(255) NOT NULL,
                    start_date DATE NULL,
                    end_date DATE NULL,
                    start_time TIME NULL,
                    end_time TIME NULL,
                    raw_text TEXT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            )

            # Check if category column exists
            cur.execute("SHOW COLUMNS FROM weekly_events LIKE 'category'")
            if not cur.fetchone():
                try:
                    log_debug("  ℹ Adding 'category' column to weekly_events table...")
                    cur.execute("ALTER TABLE weekly_events ADD COLUMN category TEXT")
                except Exception as e:
                    log_debug(f"  ⚠ Could not add category column: {e}")

            # Prepare batch values - filter invalid events upfront
            values = []
            for event in events:
                event_name = (event.get("event_name") or "").strip()
                if not event_name:
                    continue

                event_date = (event.get("event_date") or "").strip()
                if not event_date:
                    event_date = event.get("start_date") or event.get("end_date") or "no info"

                values.append(
                    (
                        event.get("source", "google_drive_newsletter"),
                        event.get("page_number"),
                        event_name,
                        event_date,
                        event.get("start_date"),
                        event.get("end_date"),
                        event.get("start_time"),
                        event.get("end_time"),
                        event.get("raw_text", ""),
                        event.get("category", "Other"),
                    )
                )

            if values:
                # Batch insert using executemany
                cur.executemany(
                    """
                    INSERT INTO weekly_events (
                        source_pdf, page_number, event_name, event_date,
                        start_date, end_date, start_time, end_time, raw_text, category
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    values,
                )
                inserted_count = len(values)

            conn.commit()
    finally:
        conn.close()

    return inserted_count


# =============================================================================
# Newsletter Processing (with concurrent LLM calls)
# =============================================================================


def _process_single_page(args: tuple) -> List[Dict]:
    """Process a single page - wrapper for concurrent execution."""
    page_text, page_num, source_name, publication_date = args
    return extract_events_from_page(page_text, page_num, source_name, publication_date)


def process_newsletter_pdf(file_path: Path, file_metadata: Dict) -> Dict:
    """
    Process a newsletter PDF page by page with concurrent LLM calls.
    """
    source_name = file_metadata.get("name", "unknown")
    log_debug(f"     Processing newsletter page-by-page: {source_name}")

    # Determine publication date (try multiple sources)
    publication_date = _extract_date_from_pdf_content(file_path)
    if not publication_date:
        publication_date = _extract_date_from_filename(source_name)
    if not publication_date:
        try:
            modified_time = file_metadata.get("modifiedTime", "")
            if modified_time:
                pub_dt = datetime.fromisoformat(modified_time.replace("Z", "+00:00"))
                publication_date = pub_dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    if publication_date:
        log_debug(f"       Using publication date: {publication_date}")
    else:
        log_debug("    ⚠ Could not determine publication date")

    # Extract pages
    try:
        pages = extract_pages_from_pdf(file_path)
    except Exception as e:
        log_debug(f"    ⚠ Failed to extract pages: {e}")
        return {"documents": [], "events": []}

    if not pages:
        log_debug("    ⚠ No pages extracted from PDF")
        return {"documents": [], "events": []}

    log_debug(f"    Found {len(pages)} pages")

    # Filter pages with content
    pages_to_process = [(p["text"], p["page_num"], source_name, publication_date) for p in pages if p["text"].strip()]

    all_events = []

    # Process pages concurrently (limit workers to avoid rate limiting)
    max_workers = min(config.LLM_MAX_WORKERS, len(pages_to_process))

    if max_workers > 1 and len(pages_to_process) > 1:
        log_debug(f"      Processing {len(pages_to_process)} pages concurrently (max {max_workers} workers)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {executor.submit(_process_single_page, args): args[1] for args in pages_to_process}  # args[1] is page_num

            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    events = future.result()
                    if events:
                        log_debug(f"    📄 Page {page_num}: found {len(events)} events")
                        all_events.extend(events)
                    else:
                        log_debug(f"    📄 Page {page_num}: no events")
                except Exception as e:
                    log_debug(f"    ⚠ Page {page_num} failed: {e}")
    else:
        # Sequential processing for single page or single worker
        for args in pages_to_process:
            page_text, page_num, src, pub_date = args
            log_debug(f"    📄 Page {page_num}/{len(pages)}: ", end="")
            events = extract_events_from_page(page_text, page_num, src, pub_date)
            if events:
                log_debug(f"found {len(events)} events")
                all_events.extend(events)
            else:
                log_debug("no events")

    log_debug(f"    ✔ Total: {len(all_events)} events extracted")

    return {"documents": [], "events": all_events}


# =============================================================================
# Vector Database Operations
# =============================================================================


def _get_vectordb_instance(embeddings=None):
    """Get or create vectordb instance."""
    if embeddings is None:
        embeddings = GeminiEmbeddings()

    if config.VECTORDB_DIR.exists():
        return Chroma(persist_directory=str(config.VECTORDB_DIR), embedding_function=embeddings)
    return None


def add_documents_to_vectordb(documents: List) -> None:
    """Add new documents to the existing vector database."""
    if not documents:
        log_debug("No documents to add.")
        return

    embeddings = GeminiEmbeddings()

    if config.VECTORDB_DIR.exists():
        vectordb = Chroma(persist_directory=str(config.VECTORDB_DIR), embedding_function=embeddings)
        vectordb.add_documents(documents)
        log_debug(f"✔ Added {len(documents)} new document chunks to existing vector DB.")
    else:
        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=str(config.VECTORDB_DIR))
        log_debug(f"✔ Created new vector DB with {len(documents)} document chunks.")


def delete_chunks_by_file_id(file_id: str, vectordb=None) -> int:
    """Delete all chunks from vector database that belong to a specific Google Drive file ID."""
    if not config.VECTORDB_DIR.exists():
        return 0

    try:
        if vectordb is None:
            embeddings = GeminiEmbeddings()
            vectordb = Chroma(persist_directory=str(config.VECTORDB_DIR), embedding_function=embeddings)

        results = vectordb.get(where={"drive_file_id": file_id})

        if results and results.get("ids") and len(results["ids"]) > 0:
            vectordb.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    except Exception as e:
        log_debug(f"    ⚠ Error deleting chunks for file ID {file_id}: {e}")
        return 0


def get_all_current_file_ids(service, folder_id: str) -> set:
    """Get all file IDs currently in Google Drive folder (including subfolders)."""
    current_file_ids = set()

    try:
        # Get files in root folder
        query = f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, fields="files(id)", pageSize=1000).execute()

        for file in results.get("files", []):
            current_file_ids.add(file["id"])

        # Get subfolders and their files
        subfolders = list_subfolders(service, folder_id)
        for subfolder in subfolders:
            subfolder_query = f"'{subfolder['id']}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
            subfolder_results = service.files().list(q=subfolder_query, fields="files(id)", pageSize=1000).execute()

            for file in subfolder_results.get("files", []):
                current_file_ids.add(file["id"])

    except Exception as e:
        log_debug(f"    ⚠ Error getting current file IDs: {e}")

    return current_file_ids


def remove_deleted_files_from_vectordb(service, folder_id: str, processed_files: dict) -> dict:
    """Detect files deleted from Google Drive and remove their chunks from vector DB."""
    deletion_stats = {"files_deleted": 0, "chunks_removed": 0, "errors": []}

    if not processed_files:
        return deletion_stats

    log_debug("\nChecking for deleted files in Google Drive...")

    try:
        current_file_ids = get_all_current_file_ids(service, folder_id)
        processed_file_ids = set(processed_files.keys())
        deleted_file_ids = processed_file_ids - current_file_ids

        if not deleted_file_ids:
            log_debug("✔ No deleted files detected")
            return deletion_stats

        log_debug(f"  Found {len(deleted_file_ids)} deleted file(s) to remove from vector DB")

        # Load vectordb once for all deletions
        vectordb = _get_vectordb_instance()

        for file_id in deleted_file_ids:
            file_name = processed_files[file_id].get("name", "unknown")

            try:
                log_debug(f"  🗑️  Removing chunks for deleted file: {file_name}")
                chunks_deleted = delete_chunks_by_file_id(file_id, vectordb)

                if chunks_deleted > 0:
                    log_debug(f"    ✔ Deleted {chunks_deleted} chunk(s)")
                else:
                    log_debug("    ⚠ No chunks found to delete")

                del processed_files[file_id]
                deletion_stats["files_deleted"] += 1
                deletion_stats["chunks_removed"] += chunks_deleted

            except Exception as e:
                error_msg = f"Error removing chunks for {file_name}: {str(e)}"
                log_debug(f"    ✗ {error_msg}")
                deletion_stats["errors"].append(error_msg)

        if deletion_stats["files_deleted"] > 0:
            log_debug(f"  ✔ Removed {deletion_stats['files_deleted']} deleted file(s) from vector DB")

    except Exception as e:
        error_msg = f"Error detecting deleted files: {str(e)}"
        log_debug(f"  ✗ {error_msg}")
        deletion_stats["errors"].append(error_msg)

    return deletion_stats


def cleanup_temp_files() -> None:
    """Clean up temporary downloaded files."""
    for file in config.DATA_DOWNLOAD_DIR.glob("*"):
        if file.is_file() and not file.name.startswith("."):
            try:
                file.unlink()
            except Exception:
                pass


# =============================================================================
# Main Sync Function
# =============================================================================


def sync_google_drive_to_vectordb() -> dict:
    """Main function to sync Google Drive files to vector database."""
    stats = {"files_processed": 0, "chunks_added": 0, "files_deleted": 0, "chunks_removed": 0, "events_extracted": 0, "events_sql_inserted": 0, "errors": []}

    try:
        # Validate configuration
        # errors = config.validate_config()
        # drive_errors = [e for e in errors if "GOOGLE_DRIVE" in e or "Google credentials" in e]
        # if drive_errors:
        #     for error in drive_errors:
        #         log_debug(f"✗ Configuration error: {error}")
        #         stats["errors"].append(error)
        #     return stats

        # Load sync state
        state = load_sync_state()
        processed_files = state.get("processed_files", {})

        # Get Drive service
        log_debug("Authenticating with Google Drive...")
        service = get_drive_service()
        log_debug("✔ Authenticated successfully")

        # Check for and remove deleted files
        deletion_stats = remove_deleted_files_from_vectordb(service, config.GOOGLE_DRIVE_FOLDER_ID, processed_files)
        stats["files_deleted"] = deletion_stats["files_deleted"]
        stats["chunks_removed"] = deletion_stats["chunks_removed"]
        stats["errors"].extend(deletion_stats["errors"])

        # List new files
        log_debug(f"\nScanning folder {config.GOOGLE_DRIVE_FOLDER_ID}...")
        new_files = list_new_files_from_drive(service, config.GOOGLE_DRIVE_FOLDER_ID, processed_files)

        if not new_files:
            log_debug("No new files to process.")
            state["processed_files"] = processed_files
            save_sync_state(state)
            if stats["files_deleted"] > 0:
                log_debug("✔ Sync state updated (deleted files removed)")
            return stats

        log_debug(f"Found {len(new_files)} new or updated files to process.")

        all_documents = []
        all_events = []

        # Process each new file
        for i, file_meta in enumerate(new_files[: config.MAX_FILES_PER_RUN], 1):
            try:
                folder_cat = file_meta.get("folder_category", "root")
                file_ext = Path(file_meta["name"]).suffix.lower()
                log_debug(f"\n[{i}/{len(new_files)}] Processing: {file_meta['name']} (from '{folder_cat}')")

                # Download file
                local_path = download_file(service, file_meta["id"], file_meta["name"], file_meta.get("folder_category"))
                log_debug(f"  ✔ Downloaded to {local_path.name}")

                # Special processing for newsletters
                if folder_cat == "newsletters" and file_ext == ".pdf":
                    result = process_newsletter_pdf(local_path, file_meta)
                    documents = result["documents"]
                    events = result["events"]

                    all_documents.extend(documents)
                    all_events.extend(events)
                    stats["events_extracted"] += len(events)

                    processed_files[file_meta["id"]] = {"name": file_meta["name"], "folder_category": folder_cat, "modifiedTime": file_meta.get("modifiedTime", ""), "processed_at": datetime.now().isoformat(), "chunks": len(documents), "events": len(events)}
                else:
                    # Standard processing for non-newsletter files
                    documents = process_file_to_documents(local_path, file_meta)
                    all_documents.extend(documents)

                    processed_files[file_meta["id"]] = {"name": file_meta["name"], "folder_category": folder_cat, "modifiedTime": file_meta.get("modifiedTime", ""), "processed_at": datetime.now().isoformat(), "chunks": len(documents)}
                    log_debug(f"  ✔ Extracted {len(documents)} chunks")

                stats["files_processed"] += 1

            except Exception as e:
                error_msg = f"Error processing {file_meta['name']}: {str(e)}"
                log_debug(f"  ✗ {error_msg}")
                stats["errors"].append(error_msg)

        # Add documents to vector DB
        if all_documents:
            log_debug(f"\nAdding {len(all_documents)} chunks to vector database...")
            add_documents_to_vectordb(all_documents)
            stats["chunks_added"] = len(all_documents)

        # Insert events to SQL (batched)
        if all_events:
            log_debug(f"\nInserting {len(all_events)} events into SQL database...")
            inserted = insert_events_to_db(all_events)
            stats["events_sql_inserted"] = inserted
            log_debug(f"✔ Inserted {inserted} events to SQL")

        # Save sync state
        state["processed_files"] = processed_files
        save_sync_state(state)
        log_debug("✔ Sync state saved")

        # Cleanup
        # cleanup_temp_files()
        # log_debug("✔ Temporary files cleaned up")

    except Exception as e:
        error_msg = f"Fatal error during sync: {str(e)}"
        log_debug(f"\n✗ {error_msg}")
        stats["errors"].append(error_msg)

    log_debug("\n" + "=" * 80)
    log_debug("Google Drive Sync Complete")
    log_debug(f"Files processed: {stats['files_processed']}")
    log_debug(f"Document chunks added: {stats['chunks_added']}")
    if stats["files_deleted"] > 0:
        log_debug(f"Files deleted: {stats['files_deleted']}")
        log_debug(f"Chunks removed: {stats['chunks_removed']}")
    log_debug(f"Events extracted: {stats['events_extracted']}")
    log_debug(f"Events inserted (SQL): {stats['events_sql_inserted']}")
    log_debug(f"Errors: {len(stats['errors'])}")
    log_debug("=" * 80)

    return stats


if __name__ == "__main__":
    try:
        sync_google_drive_to_vectordb()
    except KeyboardInterrupt:
        log_debug("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        log_debug(f"\n\nFatal error: {e}")
        sys.exit(1)
