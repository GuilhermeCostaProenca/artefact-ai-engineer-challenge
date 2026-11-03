"""Section-based retrieval over the store policy manual.

The manual is short (~3k tokens) and already organized by topic, so retrieval is a
deterministic lookup by section key instead of embeddings: the model picks the
topic(s), we return the exact source text.
"""

from __future__ import annotations

import re
from pathlib import Path

POLICIES_FILE = Path(__file__).parent / "knowledge" / "store_policies.md"
_SECTION_MARKER = re.compile(r"<!--\s*section:\s*([a-z_]+)\s*-->")


def load_policy_sections(path: Path = POLICIES_FILE) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = _SECTION_MARKER.split(text)
    # parts = [preamble, key1, body1, key2, body2, ...]
    return {key: body.strip() for key, body in zip(parts[1::2], parts[2::2])}


POLICY_SECTIONS = load_policy_sections()
POLICY_TOPICS = tuple(POLICY_SECTIONS)
