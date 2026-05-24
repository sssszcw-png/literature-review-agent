"""Rich-based progress display for the research pipeline."""

from __future__ import annotations

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box


class ResearchProgress:
    """Displays research progress using Rich Live display with auto-refreshing layout."""

    def __init__(self):
        self.console = Console()
        self.live: Live | None = None
        self._layout: Layout | None = None
        self._current_node = ""
        self._state: dict = {}

    def start(self, question: str):
        """Begin the progress display with a Live layout."""
        self.console.print()
        self.console.rule("[bold blue]Academic Deep Research Agent")
        self.console.print(f"[bold]Research Question:[/bold] {question}")
        self.console.print()

        self._layout = Layout()
        self._layout.split(
            Layout(name="status", size=3),
            Layout(name="body"),
        )
        self._layout["status"].update(Panel("Initializing...", style="cyan"))

        self.live = Live(
            self._layout,
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self.live.start()

    def update(self, node_name: str, state: dict):
        """Update the live display with the current node's state."""
        self._current_node = node_name
        self._state = state

        if self._layout is None or self.live is None:
            return

        node_display = {
            "plan_queries": "Planning search queries...",
            "search": "Searching academic databases...",
            "read": "Reading papers...",
            "update_knowledge_map": "Updating knowledge map...",
            "detect_gaps": "Detecting knowledge gaps...",
            "evaluate_saturation": "Evaluating saturation...",
            "generate_outline": "Generating report outline...",
            "write_report": "Writing report...",
        }

        msg = node_display.get(node_name, f"Processing: {node_name}")
        paper_count = len(state.get("paper_index", {}))
        node_count = len(state.get("knowledge_nodes", []))
        gap_count = len(state.get("gaps", []))
        round_num = state.get("current_round", "?")
        phase = state.get("phase", "")

        # Build status bar
        status_text = (
            f"[bold yellow]→ {msg}[/bold yellow]  "
            f"Round: {round_num} ({phase})  "
            f"Papers: {paper_count}  "
            f"Nodes: {node_count}  "
            f"Gaps: {gap_count}"
        )
        self._layout["status"].update(Panel(status_text, style="cyan"))

        # Build body content
        body_parts = []

        # Gaps table
        gaps = state.get("gaps", [])
        if gaps and node_name in ("detect_gaps", "evaluate_saturation"):
            gap_table = Table(title="Knowledge Gaps", box=box.SIMPLE)
            gap_table.add_column("Severity", style="red")
            gap_table.add_column("Description")
            gap_table.add_column("Saturation", justify="right")
            for g in gaps[:5]:
                severity = g.get("severity", "?")
                color = {
                    "critical": "red",
                    "important": "yellow",
                    "nice_to_have": "green",
                }.get(severity, "white")
                gap_table.add_row(
                    f"[{color}]{severity}[/{color}]",
                    g.get("description", "")[:80],
                    f"{g.get('saturation', 0.0):.2f}",
                )
            body_parts.append(gap_table)

        # Outline
        outline = state.get("outline")
        if outline and node_name == "generate_outline":
            chapters = outline.get("chapters", [])
            chapter_lines = []
            for i, ch in enumerate(chapters, 1):
                chapter_lines.append(f"  {i}. {ch.get('title', 'Untitled')}")
            body_parts.append(Panel(
                "\n".join(chapter_lines),
                title=f"Outline: {outline.get('title', '')}",
                style="blue",
            ))

        # Completion
        final = state.get("final_report", "")
        if final and node_name == "write_report":
            body_parts.append(Panel(
                f"[bold green]Report written successfully![/bold green]\n"
                f"[dim]Length: {len(final)} characters[/dim]",
                style="green",
            ))

        if body_parts:
            from rich.columns import Columns
            self._layout["body"].update(Columns(body_parts))
        else:
            self._layout["body"].update(
                Panel("Working...", style="dim")
            )

    def stop(self, message: str = ""):
        """End the Live display and print final message."""
        if self.live:
            self.live.stop()
            self.live = None
        if message:
            self.console.print(f"\n[bold]{message}[/bold]")
        self.console.rule()
