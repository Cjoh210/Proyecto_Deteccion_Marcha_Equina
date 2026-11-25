"""Convierte scripts en formato 'percent' (# %% / # %% [markdown]) a notebooks .ipynb."""
import sys
from pathlib import Path

import nbformat as nbf


def convert(src: Path, dst: Path):
    cells, buf, kind = [], [], "code"

    def flush():
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append(nbf.v4.new_markdown_cell(md))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            buf = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buf.append(line)
    flush()
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, dst)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        p = Path(p)
        convert(p, p.with_suffix(".ipynb"))
        print("->", p.with_suffix(".ipynb"))
