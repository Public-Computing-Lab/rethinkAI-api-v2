from pathlib import Path
import sys

# test_policy_retrieval.py

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from main_chat.rag_pipeline.rag_retrieval import retrieve_policies

# Test general policy query
result = retrieve_policies("housing policy", k=10)
print(f"Found {len(result['chunks'])} chunks")

for i, meta in enumerate(result["metadata"][:5], 1):
    print(f"\n{i}. {meta.get('source')}")
    print(f"   Folder: {meta.get('folder_category')}")
    print(f"   Preview: {result['chunks'][i-1][:150]}...")
