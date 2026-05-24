"""PromptManager: loads and renders .txt prompt templates."""

import re
from functools import lru_cache
from pathlib import Path


class PromptManager:
    """Loads all .txt files from the prompts directory at startup.

    Usage:
        pm = PromptManager(Path("prompts"))
        rendered = pm.get("plan_queries_broad", question="...", num_queries=5)
    """

    def __init__(self, prompt_dir: Path):
        self._templates: dict[str, str] = {}
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
        for file in sorted(prompt_dir.glob("*.txt")):
            name = file.stem
            self._templates[name] = file.read_text(encoding="utf-8")

    def get(self, name: str, **kwargs) -> str:
        """Render a prompt template with the given variables.

        Uses regex-based substitution: only replaces {known_key} patterns that
        match passed kwargs. Unknown braces and literal text containing braces
        (e.g. LaTeX, JSON examples) are left untouched.

        Raises KeyError if the prompt name is unknown.
        """
        if name not in self._templates:
            available = ", ".join(sorted(self._templates.keys()))
            raise KeyError(f"Unknown prompt '{name}'. Available: {available}")

        template = self._templates[name]

        def _replace(match: re.Match) -> str:
            key = match.group(1)
            if key in kwargs:
                return str(kwargs[key])
            return match.group(0)

        rendered = re.sub(r"\{(\w+)\}", _replace, template)

        # Check for unreplaced placeholders — these indicate missing kwargs
        unreplaced = re.findall(r"\{(\w+)\}", rendered)
        if unreplaced:
            missing = ", ".join(sorted(set(unreplaced)))
            raise KeyError(
                f"Missing variables for prompt '{name}': {missing}. "
                f"Available: {', '.join(sorted(self._templates.keys()))}"
            )

        return rendered

    def list_prompts(self) -> list[str]:
        """Return all available prompt names."""
        return sorted(self._templates.keys())

    def __repr__(self) -> str:
        return f"PromptManager({len(self._templates)} prompts)"


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """Return a cached PromptManager singleton that loads prompts once."""
    return PromptManager(Path("prompts"))
