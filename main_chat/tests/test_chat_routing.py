"""
Test script to see what the router decides for various questions.
This will help identify if the routing logic is the problem.
"""

from pathlib import Path
import sys
import json

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from main_chat.chat_route import _route_question


def test_routing():
    """Test routing for various questions."""

    test_questions = [
        # Should clearly use policies
        "What does the Anti-Displacement Plan say about housing?",
        "What strategies does the Slow Streets plan propose?",
        "Tell me about the goals in Imagine Boston 2030",
        # Should use transcripts (opinions)
        "What do residents think about displacement?",
        "How do people feel about the neighborhood?",
        # Ambiguous - could use either
        "Tell me about displacement in Dorchester",
        "What is being done about housing affordability?",
    ]

    print("=" * 80)
    print("Routing Test - Policy Questions")
    print("=" * 80)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 80)

        try:
            plan = _route_question(question)

            print(f"Mode: {plan.get('mode')}")
            print(f"Policy sources: {plan.get('policy_sources')}")
            print(f"Transcript tags: {plan.get('transcript_tags')}")
            print(f"k: {plan.get('k')}")

            # Check if policies will be used
            mode = plan.get("mode")
            has_policy_sources = plan.get("policy_sources") is not None

            if mode == "rag" and not has_policy_sources:
                print("⚠️  RAG mode but NO policy sources specified!")
            elif mode == "rag" and has_policy_sources:
                print("✅ Will query policies")
            elif mode == "sql":
                print("📊 SQL only - no policy retrieval")
            elif mode == "hybrid":
                if has_policy_sources:
                    print("✅ Hybrid with policies")
                else:
                    print("⚠️  Hybrid but NO policy sources!")

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_routing()
