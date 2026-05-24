"""Node: generate_outline — report outline generation with self-reflection and HITL."""

from __future__ import annotations

import asyncio
import json
import logging

from src.agent.state import AgentState
from src.config.settings import get_settings
from src.knowledge.map import summarize_knowledge_map
from src.llm.client import get_llm_client
from src.llm.prompts import get_prompt_manager

logger = logging.getLogger(__name__)


async def generate_outline_node(state: AgentState) -> AgentState:
    """Generate a report outline with self-reflection critique and HITL confirmation.

    Reads: knowledge_nodes, research_question, gaps, outline_feedback
    Writes: outline, outline_approved, user_action
    """
    km_data = state.get("knowledge_nodes", [])
    gaps_data = state.get("gaps", [])
    question = state.get("research_question", "")
    feedback = state.get("outline_feedback", "")

    settings = get_settings()
    llm = get_llm_client()
    pm = get_prompt_manager()

    km_summary = summarize_knowledge_map(km_data)
    gaps_summary = json.dumps(gaps_data, indent=2, ensure_ascii=False)

    try:
        response = await llm.complete_json(
            system_prompt="You create academic literature review outlines.",
            user_prompt=pm.get(
                "generate_outline",
                research_question=question,
                knowledge_map_summary=km_summary,
                gaps_summary=gaps_summary,
            ),
        )
    except Exception as e:
        logger.error(f"Outline generation failed: {e}")
        state.setdefault("errors", []).append({
            "node": "generate_outline",
            "error": str(e),
        })
        # Create a minimal outline
        response = {
            "title": question,
            "chapters": [
                {"title": "Introduction", "topics": [], "description": ""},
                {"title": "Literature Review", "topics": [], "description": ""},
                {"title": "Discussion", "topics": [], "description": ""},
                {"title": "Conclusion", "topics": [], "description": ""},
            ],
        }

    # Self-reflection critique
    try:
        critique_response = await llm.complete_json(
            system_prompt="You critically review academic outlines.",
            user_prompt=pm.get(
                "critique_outline",
                research_question=question,
                outline_json=json.dumps(response, indent=2, ensure_ascii=False),
                knowledge_map_summary=km_summary,
            ),
        )
        revised = critique_response.get("revised_outline", response)
        critique_text = critique_response.get("critique", "")
    except Exception as e:
        logger.warning(f"Outline critique failed: {e}")
        revised = response
        critique_text = ""

    # Apply user feedback if editing
    if feedback:
        revised = await _apply_feedback(question, revised, feedback, llm, pm)

    state["outline"] = revised
    state["outline_feedback"] = feedback

    # Display for HITL
    _display_outline(revised, critique_text)

    # HITL: check for pre-set action (non-interactive mode), otherwise prompt
    pre_action = state.get("user_action", "")
    if pre_action and pre_action != "awaiting":
        logger.info(f"[generate_outline] Using pre-set user_action: {pre_action}")
        user_input = pre_action
    else:
        try:
            user_input = (await asyncio.to_thread(input, "\n[a]pprove / [e]dit / [q]uit? ")).strip().lower()
        except EOFError:
            user_input = ""  # non-interactive → approve

    if user_input in ("e", "edit"):
        edit_feedback = (await asyncio.to_thread(input, "Enter your feedback for outline revision: ")).strip()
        state["outline_feedback"] = edit_feedback
        state["outline_approved"] = False
        state["user_action"] = "edit"
    elif user_input in ("q", "quit", "abort"):
        state["outline_approved"] = False
        state["user_action"] = "abort"
        logger.info("User aborted outline review")
    else:
        # "a", "approve", empty string (non-interactive default) → approve
        state["outline_approved"] = True
        state["user_action"] = "approve"
        logger.info("Outline approved")

    return state


async def _apply_feedback(
    question: str,
    outline: dict,
    feedback: str,
    llm: LLMClient,
    pm: PromptManager,
) -> dict:
    """Regenerate outline with user feedback."""
    try:
        response = await llm.complete_json(
            system_prompt="You revise academic outlines based on feedback.",
            user_prompt=pm.get(
                "revise_outline",
                research_question=question,
                outline_json=json.dumps(outline, indent=2, ensure_ascii=False),
                feedback=feedback,
            ),
        )
        return response
    except Exception:
        return outline


def _display_outline(outline: dict, critique: str = "") -> None:
    """Display outline using Rich formatting in the terminal."""
    # Print to stdout for HITL
    title = outline.get("title", "Report Outline")
    chapters = outline.get("chapters", [])

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown

        console = Console()
        console.print()
        console.rule("[bold blue]Report Outline")
        console.print(Panel(f"[bold]{title}[/bold]", style="blue"))
        if critique:
            console.print(Panel(critique, title="Self-Reflection Critique", style="yellow"))
        console.print()
        for i, ch in enumerate(chapters, 1):
            topics_str = ", ".join(ch.get("topics", []))
            console.print(
                f"  [bold cyan]{i}. {ch.get('title', 'Untitled')}[/bold cyan]"
            )
            if ch.get("description"):
                console.print(f"     [dim]{ch['description']}[/dim]")
            if topics_str:
                console.print(f"     [dim]Topics: {topics_str}[/dim]")
        console.print()
    except ImportError:
        # Fallback: plain print
        print(f"\n=== {title} ===")
        if critique:
            print(f"[Critique]: {critique}")
        for i, ch in enumerate(chapters, 1):
            print(f"  {i}. {ch.get('title', 'Untitled')}")
            if ch.get("description"):
                print(f"     {ch['description']}")
