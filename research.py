"""Academic Deep Research Agent — CLI entry point.

Usage:
    python research.py "<research question>" [--max-rounds 5] [--output-dir reports/] [--resume <thread_id>]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config.settings import get_settings
from src.cli.app import ResearchCLI
from src.cli.display import display_report_preview
from src.utils.logging import setup_logging
from src.utils.checkpoints import CheckpointManager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academic Deep Research Agent — iterative literature review with provenance tracking",
    )
    parser.add_argument(
        "question",
        type=str,
        help="The research question to investigate",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum deep-dive rounds (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for the report (default: reports/)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="THREAD_ID",
        help="Resume from a previous checkpoint",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint saving",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List saved checkpoints and exit",
    )
    parser.add_argument(
        "--cleanup",
        type=str,
        default=None,
        metavar="THREAD_ID",
        help="Delete checkpoints for a thread",
    )
    parser.add_argument(
        "--zh",
        action="store_true",
        help="Generate a Chinese translation of the final report",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Load settings
    settings = get_settings()

    # Setup logging
    log_level = "DEBUG" if args.verbose else settings.log_level
    setup_logging(level=log_level, log_file=settings.log_file)

    max_rounds = args.max_rounds or settings.default_max_rounds

    if args.no_checkpoint:
        settings.checkpoint_enabled = False

    # Checkpoint management commands
    ckmgr = CheckpointManager(settings.checkpoint_dir)

    if args.list_checkpoints:
        threads = ckmgr.list_threads()
        if threads:
            print("Saved checkpoints:")
            for t in threads:
                print(f"  {t}")
        else:
            print("No checkpoints found.")
        return

    if args.cleanup:
        count = ckmgr.cleanup(args.cleanup)
        print(f"Deleted {count} checkpoint(s) for thread: {args.cleanup}")
        return

    # Validate research question
    question = args.question.strip()
    if not question:
        print("Error: research question cannot be empty.", file=sys.stderr)
        sys.exit(1)
    if len(question) > 2000:
        print("Error: research question is too long (max 2000 characters).", file=sys.stderr)
        sys.exit(1)

    # Run research
    cli = ResearchCLI(settings)

    try:
        final_report = await cli.run(
            question=question,
            max_rounds=max_rounds,
            output_dir=args.output_dir,
            resume_thread_id=args.resume,
            zh=args.zh,
        )

        if final_report:
            display_report_preview(final_report)
        else:
            print("\nResearch completed. Check the output directory for the report.")

        if args.zh and cli.final_report_zh:
            display_report_preview(cli.final_report_zh, title="Chinese Translation Preview")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Use --resume to continue from the last checkpoint.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
