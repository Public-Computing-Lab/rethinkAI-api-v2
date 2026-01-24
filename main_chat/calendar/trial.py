import sys
from pathlib import Path

from pypdf import PdfReader

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT))

import config


def main() -> None:
    # 1) Extract text from page 2 of the PDF (0-based index -> pages[1])
    reader = PdfReader("REP 46_25web.pdf")
    page2_text = reader.pages[1].extract_text() or ""

    # 2) Call Gemini the same way sql_pipeline does (GenerativeModel + generate_content)

    prompt = f"""
You are reading text extracted from page 2 of a PDF. It contains a box listing events happening throughout a week.

From the text below, extract ALL events and their dates.

Return ONLY plain text, one event per line, in this exact format:
EVENT_NAME | DATE

If you are unsure about a date, make a best-effort guess based on the text (e.g. 'Monday', 'June 3–5', etc.).
Do not add any explanations or extra text.

Text:
\"\"\"{page2_text}\"\"\"
"""

    response = config.generate_content(
        prompt=prompt,
        model=config.GEMINI_MODEL,
        temperature=0,
    )

    # 3) Just print the LLM's response (event name + date per line)
    print(response)


if __name__ == "__main__":
    main()
