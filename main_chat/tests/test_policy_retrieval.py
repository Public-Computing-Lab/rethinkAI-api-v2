"""
Test direct policy retrieval to verify the full RAG pipeline works.
"""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from main_chat.chat_route import _run_rag


def test_policy_retrieval():
    """Test that _run_rag actually retrieves policies when told to."""

    question = "What does the Anti-Displacement Plan say about housing?"

    # Test 1: Without specifying policy sources (should only get transcripts)
    print("=" * 80)
    print("Test 1: RAG without policy sources specified")
    print("=" * 80)

    plan1 = {"mode": "rag", "transcript_tags": ["displacement", "community"], "policy_sources": None, "folder_categories": None, "k": 5}  # Not specified

    print(f"\nPlan: {plan1}")
    print(f"\nQuestion: {question}\n")

    result1 = _run_rag(question, plan1)

    print(f"\nTotal chunks returned: {len(result1.get('chunks', []))}")

    metadata1 = result1.get("metadata", [])
    doc_types1 = {}
    for meta in metadata1:
        dtype = meta.get("doc_type", "unknown")
        doc_types1[dtype] = doc_types1.get(dtype, 0) + 1

    print("Chunks by type:")
    for dtype, count in doc_types1.items():
        print(f"  - {dtype}: {count}")

    # Test 2: WITH policy sources specified
    print("\n" + "=" * 80)
    print("Test 2: RAG WITH policy sources specified")
    print("=" * 80)

    plan2 = {"mode": "rag", "transcript_tags": ["displacement", "community"], "policy_sources": ["Boston Anti-Displacement Plan Analysis.txt"], "folder_categories": None, "k": 5}

    print(f"\nPlan: {plan2}")
    print(f"\nQuestion: {question}\n")

    result2 = _run_rag(question, plan2)

    print(f"\nTotal chunks returned: {len(result2.get('chunks', []))}")

    metadata2 = result2.get("metadata", [])
    doc_types2 = {}
    sources2 = {}
    for meta in metadata2:
        dtype = meta.get("doc_type", "unknown")
        source = meta.get("source", "unknown")
        doc_types2[dtype] = doc_types2.get(dtype, 0) + 1
        sources2[source] = sources2.get(source, 0) + 1

    print("Chunks by type:")
    for dtype, count in doc_types2.items():
        print(f"  - {dtype}: {count}")

    print("\nChunks by source:")
    for source, count in sources2.items():
        print(f"  - {source}: {count}")

    # Show if policies were actually retrieved
    policy_chunks = sum(1 for m in metadata2 if m.get("doc_type") == "policy")
    if policy_chunks > 0:
        print(f"\n✅ SUCCESS: Retrieved {policy_chunks} policy chunks!")
    else:
        print("\n❌ PROBLEM: No policy chunks retrieved even with policy_sources specified!")


if __name__ == "__main__":
    test_policy_retrieval()
