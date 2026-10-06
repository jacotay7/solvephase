"""Execute the Python code blocks of every documentation page.

Each page's blocks run in order in one namespace, so a page reads as one
script. Pages are marked slow because some run full retrievals.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
PAGES = sorted(p for p in DOCS.rglob("*.md") if BLOCK.search(p.read_text(encoding="utf-8")))


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(DOCS)) for p in PAGES])
def test_docs_page_runs(page: Path, tmp_path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    text = page.read_text(encoding="utf-8")
    if "solvephase.interop" in text:
        pytest.importorskip("pyturb")
        pytest.importorskip("getframes")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for i, block in enumerate(BLOCK.findall(text)):
        if "# doctest: +SKIP" in block:
            continue
        code = compile(block, f"{page.name}[block {i}]", "exec")
        exec(code, namespace)
