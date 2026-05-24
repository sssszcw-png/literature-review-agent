"""Rich-based formatted display utilities."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

console = Console()


def display_knowledge_map_summary(nodes: list[dict]) -> None:
    """Render a summary table of the knowledge map."""
    if not nodes:
        console.print("[dim]No knowledge nodes yet[/dim]")
        return

    table = Table(title="Knowledge Map", box=box.SIMPLE)
    table.add_column("Topic", style="cyan", max_width=25)
    table.add_column("Claim", max_width=50)
    table.add_column("Evidence", justify="right")
    table.add_column("Confidence", justify="right")

    for n in nodes[:15]:
        table.add_row(
            n.get("topic", "")[:25],
            n.get("claim", "")[:50],
            str(len(n.get("evidence", []))),
            f"{n.get('confidence', 0.0):.2f}",
        )

    if len(nodes) > 15:
        table.caption = f"... and {len(nodes) - 15} more nodes"

    console.print(table)


def display_gaps_table(gaps: list[dict]) -> None:
    """Render a formatted table of knowledge gaps."""
    if not gaps:
        console.print("[dim]No gaps detected[/dim]")
        return

    table = Table(title="Knowledge Gaps", box=box.SIMPLE)
    table.add_column("Severity")
    table.add_column("Description", max_width=60)
    table.add_column("Saturation", justify="right")

    for g in gaps:
        severity = g.get("severity", "?")
        color = {
            "critical": "red",
            "important": "yellow",
            "nice_to_have": "green",
        }.get(severity, "white")
        table.add_row(
            f"[{color}]{severity}[/{color}]",
            g.get("description", ""),
            f"{g.get('saturation', 0.0):.2f}",
        )

    console.print(table)


def display_outline(outline: dict, critique: str = "") -> None:
    """Render a formatted report outline."""
    title = outline.get("title", "Report Outline")
    chapters = outline.get("chapters", [])

    console.print(Panel(f"[bold]{title}[/bold]", style="blue"))

    if critique:
        console.print(Panel(critique, title="Self-Reflection", style="yellow"))

    for i, ch in enumerate(chapters, 1):
        console.print(f"  [bold cyan]{i}. {ch.get('title', 'Untitled')}[/bold cyan]")
        if ch.get("description"):
            console.print(f"     [dim]{ch['description']}[/dim]")

    console.print()


def display_report_preview(final_report: str, lines: int = 50, title: str = "Report Preview") -> None:
    """Show a preview of the final report."""
    preview = "\n".join(final_report.split("\n")[:lines])
    console.print(Panel(
        Markdown(preview),
        title=title,
        style="green",
    ))
    if len(final_report.split("\n")) > lines:
        console.print(f"[dim]... ({len(final_report.split(chr(10)))} total lines)[/dim]")
