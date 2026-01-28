"""
Email Newsletter to Calendar SQL Ingestion
Reads newsletters from a dedicated email inbox and extracts events to SQL database.
Uses Gmail API with OAuth 2.0 authentication (more reliable than IMAP).
"""

import json
import sys
import base64
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta
import email

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import config
import main_chat.sql_pipeline.sql_retrieval as sql_retrieval

from utils.email_parser import extract_text_from_email, extract_pdf_attachments, get_email_subject, get_email_date

# Gmail OAuth scopes - using Gmail API (readonly)
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class AuthenticationRequiredError(Exception):
    """Raised when user interaction is needed for OAuth."""

    def __init__(self, auth_url: str):
        self.auth_url = auth_url
        super().__init__(f"Authentication required. Visit: {auth_url}")


def load_email_sync_state() -> dict:
    """Load state of which emails have been processed."""
    if config.EMAIL_SYNC_STATE_FILE.exists():
        try:
            return json.loads(config.EMAIL_SYNC_STATE_FILE.read_text())
        except Exception:
            return {"processed_email_ids": [], "last_sync": None}
    return {"processed_email_ids": [], "last_sync": None}


def save_email_sync_state(state: dict) -> None:
    """Save email sync state."""
    state["last_sync"] = datetime.now().isoformat()
    config.EMAIL_SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


def get_gmail_credentials(interactive: bool = True) -> Credentials:
    """
    Get or refresh Gmail OAuth 2.0 credentials.

    Args:
        interactive: If True, opens browser for OAuth if needed.
                    If False, raises AuthenticationRequiredError instead.
                    Set to False for cron/automated runs.

    Returns:
        Valid Credentials object

    Raises:
        AuthenticationRequiredError: When interactive=False and user auth is needed
        FileNotFoundError: When OAuth credentials file is missing
    """
    creds = None
    token_path = Path(config.GMAIL_TOKEN_PATH)

    # Load existing token if available
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as e:
            print(f"  ⚠ Could not load existing token: {e}")

    # If valid credentials exist, return them
    if creds and creds.valid:
        return creds

    # Try to refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            print("  ✓ Token refreshed successfully")
            return creds
        except Exception as e:
            print(f"  ⚠ Token refresh failed: {e}")
            creds = None

    # No valid credentials - need user interaction
    credentials_path = Path(config.GMAIL_CREDENTIALS_PATH)
    if not credentials_path.exists():
        raise FileNotFoundError(f"Gmail OAuth credentials not found: {config.GMAIL_CREDENTIALS_PATH}\n" "Please download credentials from Google Cloud Console.")

    flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)

    if not interactive:
        # Non-interactive mode: generate URL and raise error
        flow.redirect_uri = "http://localhost:8080/"
        auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")  # Request refresh token  # Force consent to ensure refresh token
        raise AuthenticationRequiredError(auth_url)

    # Interactive mode: open browser for user to authorize
    print("  Opening browser for Gmail authorization...")
    creds = flow.run_local_server(port=8080, access_type="offline", prompt="consent")
    token_path.write_text(creds.to_json())
    print("  ✓ Authorization complete, token saved")
    return creds


def get_gmail_service(interactive: bool = True):
    """
    Get authenticated Gmail API service.

    Args:
        interactive: If True, allows browser-based OAuth if needed.
                    If False, raises AuthenticationRequiredError instead.

    Returns:
        Gmail API service object

    Raises:
        AuthenticationRequiredError: When interactive=False and user auth is needed
        ValueError: When EMAIL_ADDRESS is not configured
        RuntimeError: When connection to Gmail API fails
    """
    if not config.EMAIL_ADDRESS:
        raise ValueError("NEWSLETTER_EMAIL_ADDRESS not configured")

    # Get OAuth credentials (may raise AuthenticationRequiredError)
    creds = get_gmail_credentials(interactive=interactive)

    # Build Gmail API service
    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as e:
        raise RuntimeError(f"Failed to build Gmail API service: {e}")


def get_recent_newsletters(service, processed_ids: List[str], days_back: int = 7) -> List[tuple]:
    """
    Fetch recent newsletters using Gmail API.
    Returns list of (email_id, email_message) tuples.
    """
    try:
        # Calculate date for query
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y/%m/%d")

        # Search for recent emails
        query = f"after:{since_date}"
        results = service.users().messages().list(userId="me", q=query, maxResults=100).execute()

        messages = results.get("messages", [])

        newsletters = []
        for msg_ref in messages:
            msg_id = msg_ref["id"]

            # Skip if already processed
            if msg_id in processed_ids:
                continue

            # Get full message
            try:
                message = service.users().messages().get(userId="me", id=msg_id, format="raw").execute()

                # Decode the raw message
                raw_email = base64.urlsafe_b64decode(message["raw"])
                msg = email.message_from_bytes(raw_email)

                newsletters.append((msg_id, msg))
            except Exception:
                continue

        return newsletters

    except Exception as e:
        raise RuntimeError(f"Failed to fetch emails from Gmail API: {e}")


def extract_events_with_llm(text: str, source: str, publication_date: str = None) -> List[Dict]:
    """
    Use LLM to extract structured events from newsletter text.

    Args:
        text: Newsletter text content
        source: Source identifier
        publication_date: Newsletter publication date in YYYY-MM-DD format
    """
    # Truncate very long text to avoid token limits
    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... text truncated ...]"

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

    prompt = f"""
You are reading a community newsletter or calendar listing events.
{date_context}
Extract ALL events with their dates and times from the text below.

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
    "location": "... or null"
  }},
  ...
]

Field rules:
- "event_name": Short descriptive name
- "event_date": Date label as written (e.g., "Monday", "June 3-5", "All week") - keep original text
- "start_date": ISO date YYYY-MM-DD when event starts - CONVERT day names to exact dates using publication date
- "end_date": ISO date YYYY-MM-DD when event ends (or null if same day) - CONVERT day names to exact dates
- "start_time": 24-hour time HH:MM (or null)
- "end_time": 24-hour time HH:MM (or null)
- "raw_text": Original text describing the event
- "location": Where the event takes place (or null)

Be conservative: only extract clear events with dates. Never invent information, but DO convert day names to exact dates when the publication date is provided.

Text:
\"\"\"
{text}
\"\"\"
"""

    try:
        text_response = config.generate_content(
            prompt=prompt,
            model=config.GEMINI_MODEL,
            temperature=0,
        )

        # Clean up potential code fences
        if text_response.startswith("```"):
            text_response = text_response.strip("`").strip()
            lines = text_response.splitlines()
            if lines and lines[0].strip().lower() in ("json", "javascript"):
                text_response = "\n".join(lines[1:]).strip()

        events = json.loads(text_response)

        if not isinstance(events, list):
            return []

        for event in events:
            event["source"] = source

        return events

    except Exception as e:
        print(f"  ✗ Error extracting events with LLM: {e}")
        return []


def insert_events_to_db(events: List[Dict]) -> int:
    """Insert events into the weekly_events table."""
    if not events:
        return 0

    conn = sql_retrieval._get_db_connection()
    inserted_count = 0

    try:
        with conn.cursor() as cur:
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

            for event in events:
                try:
                    cur.execute(
                        """
                        INSERT INTO weekly_events (
                            source_pdf,
                            page_number,
                            event_name,
                            event_date,
                            start_date,
                            end_date,
                            start_time,
                            end_time,
                            raw_text
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            event.get("source", "email_newsletter"),
                            None,
                            event.get("event_name", ""),
                            event.get("event_date", ""),
                            event.get("start_date"),
                            event.get("end_date"),
                            event.get("start_time"),
                            event.get("end_time"),
                            event.get("raw_text", ""),
                        ),
                    )
                    inserted_count += 1
                except Exception as e:
                    if config.VERBOSE_LOGGING:
                        print(f"  ⚠ Could not insert event '{event.get('event_name')}': {e}")

        conn.commit()
    finally:
        conn.close()

    return inserted_count


def sync_email_newsletters_to_sql(interactive: bool = True) -> dict:
    """
    Main function to sync email newsletters to calendar SQL database.

    Args:
        interactive: If True, allows browser-based OAuth if needed.
                    If False, returns error in stats instead of blocking.

    Returns:
        Dictionary with summary statistics
    """
    print("=" * 80)
    print("Starting Email Newsletter → Calendar SQL Sync")
    print("=" * 80)

    stats = {
        "emails_processed": 0,
        "events_extracted": 0,
        "events_inserted": 0,
        "errors": [],
        "auth_required": False,
        "auth_url": None,
    }

    try:
        # Validate configuration
        errors = config.validate_config()
        email_errors = [e for e in errors if "EMAIL" in e.upper()]
        if email_errors:
            for error in email_errors:
                print(f"✗ Configuration error: {error}")
                stats["errors"].append(error)
            return stats

        # Load sync state
        state = load_email_sync_state()
        processed_ids = state.get("processed_email_ids", [])

        # Connect to Gmail API
        print("Connecting to Gmail API...")
        try:
            service = get_gmail_service(interactive=interactive)
        except AuthenticationRequiredError as e:
            # Non-interactive mode and auth needed
            stats["auth_required"] = True
            stats["auth_url"] = e.auth_url
            error_msg = f"Gmail authentication required. Visit: {e.auth_url}"
            print(f"⚠ {error_msg}")
            stats["errors"].append(error_msg)
            return stats

        print("✓ Connected successfully")

        # Get recent newsletters
        print(f"Scanning inbox for newsletters from last {config.EMAIL_LOOKBACK_DAYS} days...")
        newsletters = get_recent_newsletters(service, processed_ids, days_back=config.EMAIL_LOOKBACK_DAYS)

        print(f"Found {len(newsletters)} new newsletters to process.")

        if not newsletters:
            print("No new newsletters. Exiting.")
            return stats

        all_events = []

        # Process each newsletter
        for i, (email_id, msg) in enumerate(newsletters, 1):
            try:
                subject = get_email_subject(msg)
                email_date = get_email_date(msg)

                from email.utils import parsedate_to_datetime

                try:
                    pub_date = parsedate_to_datetime(email_date).strftime("%Y-%m-%d")
                except Exception:
                    pub_date = datetime.now().strftime("%Y-%m-%d")

                print(f"\n[{i}/{len(newsletters)}] Processing: {subject[:60]}...")

                email_text = extract_text_from_email(msg)
                pdf_texts = extract_pdf_attachments(msg)

                full_text = email_text
                if pdf_texts:
                    full_text += "\n\n" + "\n\n".join(pdf_texts)
                    print(f"  ✓ Found {len(pdf_texts)} PDF attachment(s)")

                if not full_text.strip():
                    print("  ⚠ No text content found")
                    continue

                events = extract_events_with_llm(full_text, source=f"Email: {subject}", publication_date=pub_date)

                if events:
                    print(f"  ✓ Extracted {len(events)} events")
                    all_events.extend(events)
                else:
                    print("  ⚠ No events found")

                processed_ids.append(email_id)
                stats["emails_processed"] += 1
                stats["events_extracted"] += len(events)

            except Exception as e:
                error_msg = f"Error processing email {email_id}: {str(e)}"
                print(f"  ✗ {error_msg}")
                stats["errors"].append(error_msg)

        if all_events:
            print(f"\nInserting {len(all_events)} events into database...")
            inserted = insert_events_to_db(all_events)
            stats["events_inserted"] = inserted
            print(f"✓ Inserted {inserted} events successfully")

        state["processed_email_ids"] = processed_ids[-1000:]
        save_email_sync_state(state)
        print("✓ Sync state saved")

    except Exception as e:
        error_msg = f"Fatal error during sync: {str(e)}"
        print(f"\n✗ {error_msg}")
        stats["errors"].append(error_msg)

    print("\n" + "=" * 80)
    print("Email Newsletter Sync Complete")
    print(f"Emails processed: {stats['emails_processed']}")
    print(f"Events extracted: {stats['events_extracted']}")
    print(f"Events inserted (SQL): {stats['events_inserted']}")
    print(f"Errors: {len(stats['errors'])}")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync email newsletters to calendar database")
    parser.add_argument("--auth", action="store_true", help="Run interactive OAuth flow to authenticate with Gmail")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode (for cron jobs)")
    args = parser.parse_args()

    try:
        if args.auth:
            # Just do authentication
            print("Running Gmail OAuth authentication...")
            get_gmail_credentials(interactive=True)
            print("✓ Authentication successful! Token saved.")
        else:
            # Run full sync
            interactive = not args.non_interactive
            sync_email_newsletters_to_sql(interactive=interactive)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
    except AuthenticationRequiredError as e:
        print(f"\n⚠ Authentication required!")
        print(f"Visit this URL to authorize: {e.auth_url}")
        print("\nOr run: python email_to_calendar_sql.py --auth")
        sys.exit(2)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)
