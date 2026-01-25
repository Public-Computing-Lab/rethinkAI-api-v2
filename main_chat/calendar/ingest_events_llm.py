import json
import sys
from pathlib import Path

from pypdf import PdfReader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import config

import main_chat.sql_pipeline.sql_retrieval as sql_retrieval  # type: ignore  # reuse MySQL connection helper


def _extract_events_from_pdf(pdf_path: str, page_index: int = 1):
    """
    Use Gemini to turn text from one PDF page into structured events.

    Returns a list of dicts like:
      [{
         "event_name": "...",
         "event_date": "...",          # display label as written
         "start_date": "YYYY-MM-DD" or null,
         "end_date": "YYYY-MM-DD" or null,
         "start_time": "HH:MM" or null,
         "end_time": "HH:MM" or null,
         "raw_text": "..."
      }, ...]
    """
    reader = PdfReader(pdf_path)
    page_text = reader.pages[page_index].extract_text() or ""

    prompt = f"""
You are reading text extracted from page {page_index + 1} of a PDF. It contains a box listing events happening throughout a week.

From the text below, extract ALL events and their dates/times.

Return ONLY valid JSON, no explanations, no backticks, in this exact shape:
[
  {{
    "event_name": "...",
    "event_date": "...",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "start_time": "HH:MM or null",
    "end_time": "HH:MM or null",
    "raw_text": "..."
  }},
  ...
]

Field rules:
- "event_name": short human-readable name for the event.
- "event_date": the date or label exactly as written (e.g. "Monday", "June 3–5, 2025", "All week").
- "start_date": normalized calendar date when the event starts, in ISO format YYYY-MM-DD, or null if truly unknown.
- "end_date": normalized calendar date when the event ends, in ISO format YYYY-MM-DD, or null if same as start_date or unknown.
- "start_time": 24-hour time HH:MM when the event starts (e.g. "09:30"), or null if not specified.
- "end_time": 24-hour time HH:MM when the event ends, or null if not specified.
- "raw_text": the original line or snippet from the schedule describing this event.

Be conservative: never invent dates or times not clearly implied by the text. Use null when uncertain.

Text:
\"\"\"{page_text}\"\"\""""

    text = config.generate_content(
        prompt=prompt,
        model=config.GEMINI_MODEL,
        temperature=0,
    )

    # Strip accidental code fences if the model adds them
    if text.startswith("```"):
        text = text.strip("`").strip()
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in ("json", "javascript", "js"):
            text = "\n".join(lines[1:]).strip()

    try:
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse Gemini JSON: {exc}\nRaw text:\n{text}") from exc

    if not isinstance(data, list):
        raise RuntimeError(f"Gemini JSON is not a list: {data!r}")

    events = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = (item.get("event_name") or "").strip()
        date = (item.get("event_date") or "").strip()
        start_date = (item.get("start_date") or "").strip() or None
        end_date = (item.get("end_date") or "").strip() or None
        start_time = (item.get("start_time") or "").strip() or None
        end_time = (item.get("end_time") or "").strip() or None
        raw = (item.get("raw_text") or "").strip()
        if not name or not date:
            continue
        events.append(
            {
                "event_name": name,
                "event_date": date,
                "start_date": start_date,
                "end_date": end_date,
                "start_time": start_time,
                "end_time": end_time,
                "raw_text": raw,
            }
        )
    return events


def _insert_events_into_db(
    events,
    source_pdf: str,
    page_number: int,
) -> None:
    """
    Insert parsed events into the weekly_events table.
    """
    if not events:
        print("No events to insert.")
        return

    conn = sql_retrieval._get_db_connection()
    try:
        with conn.cursor() as cur:
            for ev in events:
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
                        source_pdf,
                        page_number,
                        ev["event_name"],
                        ev["event_date"],
                        ev.get("start_date"),
                        ev.get("end_date"),
                        ev.get("start_time"),
                        ev.get("end_time"),
                        ev.get("raw_text", ""),
                    ),
                )
        print(f"Inserted {len(events)} events into weekly_events.")
    finally:
        conn.close()


def main() -> None:
    """
    1) Read ALL pages of REP 46_25web.pdf.
    2) Use Gemini to extract events + dates from each page as JSON.
    3) Write them into the weekly_events table.
    """
    pdf_path = "REP 46_25web.pdf"

    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        total_events = 0
        for page_index in range(num_pages):
            print(f"\n=== Processing page {page_index + 1} / {num_pages} ===")
            events = _extract_events_from_pdf(pdf_path, page_index=page_index)
            print(f"LLM extracted {len(events)} events from page {page_index + 1}:")
            for ev in events:
                print(f"- {ev['event_name']} | {ev['event_date']}")
            _insert_events_into_db(events, source_pdf=pdf_path, page_number=page_index + 1)
            total_events += len(events)
        print(f"\nDone. Inserted a total of {total_events} events from {num_pages} pages.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error ingesting events: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
