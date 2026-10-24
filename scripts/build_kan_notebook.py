"""Build the standalone, unexecuted KAN notebook using only the standard library."""

import ast
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/Transformer_KAN_BiLSTM_GPU.py"
DESTINATION = SOURCE.with_suffix(".ipynb")


def build_notebook(source=SOURCE):
    cells = []
    parts = re.split(r"^# %% (\[markdown\] )?id=([a-z0-9-]+)\s*$", source.read_text(), flags=re.M)
    if parts[0].strip():
        raise ValueError("Notebook source must start with a cell marker")
    for index in range(1, len(parts), 3):
        markdown, identifier, body = parts[index:index + 3]
        body = body.strip("\n")
        if markdown:
            lines = body.splitlines()
            if any(line and not line.startswith("#") for line in lines):
                raise ValueError(f"Non-comment text in markdown cell {identifier}")
            body = "\n".join(line[2:] if line.startswith("# ") else line[1:] for line in lines)
        else:
            ast.parse(body, filename=f"{source.name}:{identifier}")
        cell = {"cell_type": "markdown" if markdown else "code", "id": identifier,
                "metadata": {}, "source": body.rstrip() + "\n"}
        if not markdown:
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    ids = [cell["id"] for cell in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate notebook cell IDs")
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3 (CUDA)", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"},
                         "aeroguard": {"architecture": "Transformer-KAN-BiLSTM", "executed": False,
                                      "paper_doi": "10.3390/aerospace12110998", "standalone": True}}}


if __name__ == "__main__":
    notebook = build_notebook()
    DESTINATION.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(f"Created {DESTINATION} ({len(notebook['cells'])} cells, all unexecuted)")
