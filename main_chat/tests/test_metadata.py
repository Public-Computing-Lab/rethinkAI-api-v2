from pathlib import Path
import sys

# test_policy_retrieval.py

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
# check_metadata.py
from main_chat.rag_pipeline.rag_retrieval import load_vectordb

vectordb = load_vectordb()
all_docs = vectordb.get(limit=10000)

# Find CLIENT_UPLOAD docs and show their metadata
print("=" * 80)
print("CLIENT_UPLOAD DOCUMENTS - ACTUAL METADATA")
print("=" * 80)

for i, meta in enumerate(all_docs["metadatas"]):
    doc_type = meta.get("doc_type", "")

    # Check all possible variations
    if "CLIENT" in str(doc_type).upper() or "UPLOAD" in str(doc_type).upper():
        print(f"\nDocument {i+1}:")
        print(f"  doc_type: '{doc_type}' (type: {type(doc_type).__name__})")
        print(f"  folder_category: '{meta.get('folder_category')}' (exists: {('folder_category' in meta)})")
        print(f"  source: '{meta.get('source')}'")
        print(f"  All keys: {list(meta.keys())}")

        # Only show first 3
        if i >= 2:
            break
