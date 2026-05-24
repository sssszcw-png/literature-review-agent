"""Quick E2E test — runs the full pipeline non-interactively."""
import asyncio
import time
from pathlib import Path

from src.config.settings import get_settings
from src.agent.graph import build_graph
from src.agent.state import AgentState


async def main():
    settings = get_settings()
    graph = build_graph()

    question = "how does attention mechanism work in transformers"
    print(f"Question: {question}")
    print(f"Max rounds: 2")
    print()

    initial_state: AgentState = {
        "research_question": question,
        "max_rounds": 2,
        "output_dir": "reports",
        "current_round": 0,
        "phase": "broad",
        "round_history": [],
        "search_queries": [],
        "paper_index": {},
        "search_results": [],
        "papers_read": {},
        "read_failures": {},
        "knowledge_nodes": [],
        "gaps": [],
        "outline": None,
        "outline_feedback": "",
        "outline_approved": False,
        "chapters": {},
        "final_report": "",
        "messages": [],
        "errors": [],
        "iteration_count": 0,
        "last_saturation_scores": {},
        "consecutive_no_improvement": 0,
        "checkpoint_metadata": {},
        "user_action": "approve",
        "output_zh": False,
        "final_report_zh": "",
    }

    config = {"configurable": {"thread_id": str(int(time.time()))}}

    t0 = time.time()
    node_order = []

    try:
        async for event in graph.astream(initial_state, config=config):
            for node_name, node_state in event.items():
                node_order.append(node_name)

                if node_name == "plan_queries":
                    print(f"[Round {node_state.get('current_round', '?')}] Phase: {node_state.get('phase', '?')}")
                    print(f"  Queries: {node_state.get('search_queries', [])}")

                elif node_name == "search":
                    print(f"  Papers found: {len(node_state.get('search_results', []))}")
                    total = len(node_state.get('paper_index', {}))
                    print(f"  Total paper index: {total}")

                elif node_name == "read":
                    print(f"  Papers read: {len(node_state.get('papers_read', {}))}")

                elif node_name == "update_knowledge_map":
                    print(f"  Knowledge nodes: {len(node_state.get('knowledge_nodes', []))}")

                elif node_name == "detect_gaps":
                    gaps = node_state.get("gaps", [])
                    print(f"  Gaps: {len(gaps)}")
                    for g in gaps:
                        print(f"    [{g.get('severity', '?')}] {g.get('description', '')[:80]}...")

                elif node_name == "evaluate_saturation":
                    gaps = node_state.get("gaps", [])
                    for g in gaps:
                        print(f"    Saturation: {g.get('description', '')[:50]}... = {g.get('saturation', 0):.2f}")

                elif node_name == "generate_outline":
                    outline = node_state.get("outline", {})
                    print(f"  Outline: {outline.get('title', '?')}")
                    for ch in outline.get("chapters", []):
                        print(f"    - {ch.get('title', '')}")

                elif node_name == "write_report":
                    final = node_state.get("final_report", "")
                    print(f"\n  Report written! ({len(final)} chars)")
                    print(f"  Chapters: {list(node_state.get('chapters', {}).keys())}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Node path: {' → '.join(node_order)}")

    # Check report
    reports_dir = Path("reports")
    for f in sorted(reports_dir.glob("*.md")):
        if f.name != ".gitkeep":
            size = f.stat().st_size
            print(f"\nReport: {f.name} ({size} bytes)")
            preview = f.read_text(encoding="utf-8")[:500]
            print(preview)


if __name__ == "__main__":
    asyncio.run(main())
