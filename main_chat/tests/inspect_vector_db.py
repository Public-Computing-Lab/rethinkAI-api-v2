"""
Diagnostic script to check what's in the vector database.
Run this to see what policy documents are actually stored.
"""

from pathlib import Path
import sys
from collections import defaultdict

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

    try:
        # Chroma's get() method returns all documents when called without filters
        all_docs = vectordb.get(limit=10000)

        if not all_docs or not all_docs.get("metadatas"):
            print("❌ No documents found in vector database!")
            return

        metadatas = all_docs["metadatas"]
        print(f"\n✅ Total documents in database: {len(metadatas)}\n")

        # Count by doc_type
        doc_types = {}
        # Track all sources/filenames grouped by doc_type
        sources_by_type = defaultdict(lambda: defaultdict(int))
        # Track all unique metadata keys seen
        all_meta_keys = set()

        for meta in metadatas:
            all_meta_keys.update(meta.keys())
            doc_type = meta.get("doc_type", "unknown")
            # Try multiple possible source field names
            source = meta.get("source") or meta.get("filename") or meta.get("file_name") or meta.get("drive_file_id") or meta.get("name") or "unknown"
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            sources_by_type[doc_type][source] += 1

        # Show metadata keys found (helpful for debugging)
        print("🔑 Metadata keys found across all documents:")
        print(f"   {sorted(all_meta_keys)}\n")

        print("📊 Documents by type:")
        for dtype, count in sorted(doc_types.items()):
            print(f"  - {dtype}: {count}")

        # Show ALL files/sources grouped by document type
        print("\n" + "=" * 80)
        print("📁 ALL FILES IN VECTOR DATABASE (by type)")
        print("=" * 80)

        for doc_type in sorted(sources_by_type.keys()):
            sources = sources_by_type[doc_type]
            print(f"\n📂 {doc_type.upper()} ({sum(sources.values())} total chunks)")
            print("-" * 60)
            for source, count in sorted(sources.items(), key=lambda x: (-x[1], x[0])):
                # Truncate long filenames for display
                display_name = source if len(source) <= 55 else source[:52] + "..."
                print(f"   {count:4d} chunks │ {display_name}")

        # Show a flat list of all unique filenames
        print("\n" + "=" * 80)
        print("📋 UNIQUE FILENAMES (flat list)")
        print("=" * 80)

        all_sources = set()
        for sources in sources_by_type.values():
            all_sources.update(sources.keys())

        print(f"\nTotal unique files: {len(all_sources)}\n")
        for source in sorted(all_sources):
            print(f"  • {source}")

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
