from langchain_chroma import Chroma
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import config


class GeminiEmbeddings:
    """
    Minimal embeddings wrapper using Gemini's embedding API,
    compatible with LangChain's interface.
    Uses centralized config for client setup.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or config.GEMINI_EMBED_MODEL

    def embed_documents(self, texts):
        return config.embed_content_batch(texts, self.model)

    def embed_query(self, text):
        return config.embed_content(text, self.model)


def load_vectordb():
    """Load the unified vector database (policies, transcripts, client uploads, etc.)."""
    embeddings = GeminiEmbeddings()
    vectordb = Chroma(
        persist_directory=str(config.VECTORDB_DIR),
        embedding_function=embeddings,
    )
    return vectordb


def retrieve(query, k=5, doc_type=None, tags=None, source=None, min_score=None, vectordb=None):
    """
    Universal retrieval with flexible metadata filtering.
    [... rest of function unchanged ...]
    """
    # Defensive clamp: Chroma requires k >= 1
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 5
    if k <= 0:
        k = 5

    if vectordb is None:
        vectordb = load_vectordb()

    # Build filter dictionary
    filter_dict = None

    doc_filter = None
    if isinstance(doc_type, (list, tuple)):
        doc_types = [dt for dt in doc_type if dt]
        if len(doc_types) == 1:
            doc_filter = {"doc_type": doc_types[0]}
        elif len(doc_types) > 1:
            doc_filter = {"$or": [{"doc_type": dt} for dt in doc_types]}
    elif doc_type:
        doc_filter = {"doc_type": doc_type}

    if doc_filter and source:
        filter_dict = {
            "$and": [
                doc_filter,
                {"source": source},
            ]
        }
    elif doc_filter:
        filter_dict = doc_filter
    elif source:
        filter_dict = {"source": source}

    if min_score is not None:
        results_with_scores = vectordb.similarity_search_with_score(query, k=k * 3 if tags else k, filter=filter_dict if filter_dict else None)

        if tags:
            filtered_results = []
            for doc, score in results_with_scores:
                if "tags" in doc.metadata:
                    doc_tags = [t.strip() for t in doc.metadata["tags"].split(",")]
                    if any(tag in doc_tags for tag in tags):
                        filtered_results.append((doc, score))
                        if len(filtered_results) >= k:
                            break
            if filtered_results:
                results_with_scores = filtered_results

        filtered_results = [(doc, score) for doc, score in results_with_scores if score <= min_score]

        return {"chunks": [doc.page_content for doc, _ in filtered_results[:k]], "metadata": [doc.metadata for doc, _ in filtered_results[:k]], "scores": [score for _, score in filtered_results[:k]], "query": query}
    else:
        results = vectordb.similarity_search(query, k=k * 3 if tags else k, filter=filter_dict if filter_dict else None)

        if tags:
            filtered_results = []
            for doc in results:
                if "tags" in doc.metadata:
                    doc_tags = [t.strip() for t in doc.metadata["tags"].split(",")]
                    if any(tag in doc_tags for tag in tags):
                        filtered_results.append(doc)
                        if len(filtered_results) >= k:
                            break
            if filtered_results:
                results = filtered_results

        return {"chunks": [doc.page_content for doc in results[:k]], "metadata": [doc.metadata for doc in results[:k]], "scores": None, "query": query}


def retrieve_transcripts(query, tags=None, k=5):
    """Convenience function for transcript-only search."""
    return retrieve(query, k=k, doc_type="transcript", tags=tags)


def retrieve_policies(query, k=5, source=None):
    """Convenience function for policy-only search."""
    return retrieve(query, k=k, doc_type="policy", source=source)


def format_results(result_dict):
    """Format retrieval results for display."""
    formatted = []
    formatted.append(f"Query: {result_dict['query']}")
    formatted.append(f"Total results: {len(result_dict['chunks'])}\n")

    for i, (chunk, meta) in enumerate(zip(result_dict["chunks"], result_dict["metadata"]), 1):
        formatted.append(f"\n{'='*80}")
        formatted.append(f"Result {i}")
        formatted.append(f"Source: {meta.get('source', 'Unknown')}")
        formatted.append(f"Type: {meta.get('doc_type', 'Unknown')}")

        if meta.get("tags"):
            formatted.append(f"Tags: {meta['tags']}")
        if meta.get("Heading"):
            formatted.append(f"Section: {meta['Heading']}")

        if result_dict.get("scores"):
            formatted.append(f"Score: {result_dict['scores'][i-1]:.4f}")

        formatted.append(f"\nContent:\n{chunk[:300]}...")

    return "\n".join(formatted)
