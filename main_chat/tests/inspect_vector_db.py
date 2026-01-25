"""
Diagnostic script to check what's in the vector database.
Run this to see what policy documents are actually stored.
"""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import config
from main_chat.rag_pipeline.rag_retrieval import load_vectordb


def inspect_vectordb():
    """Inspect what's in the vector database."""
    print("=" * 80)
    print("Vector Database Inspection")
    print("=" * 80)

    vectordb = load_vectordb()

    # Get all documents (up to 1000)
    try:
        # Chroma's get() method returns all documents when called without filters
        all_docs = vectordb.get(limit=1000)

        if not all_docs or not all_docs.get("metadatas"):
            print("❌ No documents found in vector database!")
            return

        metadatas = all_docs["metadatas"]
        print(f"\n✅ Total documents in database: {len(metadatas)}\n")

        # Count by doc_type
        doc_types = {}
        sources = {}

        for meta in metadatas:
            doc_type = meta.get("doc_type", "unknown")
            source = meta.get("source", "unknown")

            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            if doc_type == "policy":
                sources[source] = sources.get(source, 0) + 1

        print("📊 Documents by type:")
        for dtype, count in sorted(doc_types.items()):
            print(f"  - {dtype}: {count}")

        print("\n📋 Policy documents by source:")
        if sources:
            for source, count in sorted(sources.items()):
                print(f"  - {source}: {count} chunks")
        else:
            print("  ❌ NO POLICY DOCUMENTS FOUND!")

        # Test retrieval
        print("\n" + "=" * 80)
        print("Testing Policy Retrieval")
        print("=" * 80)

        test_query = "anti-displacement"
        print(f"\nTest query: '{test_query}'")

        # Try retrieving policies
        from main_chat.rag_pipeline.rag_retrieval import retrieve_policies

        result = retrieve_policies(test_query, k=5)
        chunks = result.get("chunks", [])
        metadata = result.get("metadata", [])

        print(f"✅ Retrieved {len(chunks)} chunks")

        if chunks:
            print("\nFirst 2 results:")
            for i, (chunk, meta) in enumerate(zip(chunks[:2], metadata[:2]), 1):
                print(f"\n  Result {i}:")
                print(f"  Source: {meta.get('source', 'unknown')}")
                print(f"  Type: {meta.get('doc_type', 'unknown')}")
                print(f"  Preview: {chunk[:200]}...")
        else:
            print("❌ No chunks retrieved! This is the problem.")

            # Try a more specific source filter
            print("\nTrying with specific source filter...")
            result2 = retrieve_policies(test_query, k=5, source="Boston Anti-Displacement Plan Analysis.txt")
            chunks2 = result2.get("chunks", [])
            print(f"Retrieved {len(chunks2)} chunks with source filter")

    except Exception as e:
        print(f"❌ Error inspecting database: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_vectordb()
