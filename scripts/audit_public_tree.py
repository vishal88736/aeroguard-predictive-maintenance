"""Check Git publication paths, common credential patterns, links and notebook outputs.

This targeted guard is not a comprehensive secret scanner or security review.
It never prints matched credential text. Use --worktree before staging and
--history before first publication to include every reachable source blob.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BINARY_ASSETS = {".png", ".jpg", ".jpeg", ".webp", ".woff", ".woff2"}
PRIVATE_ROOTS = {"artifacts", "runtime", "output", "tmp", "mlruns", "mlartifacts", "node_modules", ".cache"}
PRIVATE_SUFFIXES = {".pdf", ".zip", ".db", ".sqlite", ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".safetensors", ".log"}
PATTERNS = {
    "GitHub credential": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{20,255})"),
    "AWS access identifier": re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "Slack credential": re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,255}"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----"),
    "personal home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}


def git(*arguments, text=False):
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, check=True)
    return result.stdout.decode() if text else result.stdout


def path_problem(name):
    path = PurePosixPath(name)
    if path.parts[0] in PRIVATE_ROOTS or any(part.startswith(".venv") or part == "__pycache__" for part in path.parts):
        return "local/generated directory"
    if path.name.startswith(".env") and path.name != ".env.example":
        return "actual environment file"
    if path.suffix.lower() in PRIVATE_SUFFIXES:
        return "data, model, database, document or generated artifact"
    if path.parts[0] == "data" and name != "data/cmapss/README.md":
        return "dataset content"
    if "golden-gate-reference" in name:
        return "original user reference image"
    return None


def check_content(name, content):
    failures = []
    if len(content) > 5 * 1024 * 1024:
        failures.append(f"{name}: exceeds the 5 MB source/asset limit")
    if PurePosixPath(name).suffix.lower() in BINARY_ASSETS:
        return failures
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return failures + [f"{name}: unexpected binary file"]
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            failures.append(f"{name}: {label} pattern (value withheld)")
    if name.endswith(".ipynb"):
        try:
            book = json.loads(text)
            if any(cell.get("outputs", []) or cell.get("execution_count") is not None for cell in book["cells"]):
                failures.append(f"{name}: notebook has saved outputs/execution counts")
        except (ValueError, KeyError, TypeError):
            failures.append(f"{name}: invalid notebook JSON")
    return failures


def check_links(name, content, tracked):
    if not name.endswith(".md"):
        return []
    failures = []
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content.decode("utf-8")):
        target = target.strip().strip("<>").split("#", 1)[0]
        if not target or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            continue
        resolved = (ROOT / PurePosixPath(name).parent / target).resolve()
        if not resolved.is_relative_to(ROOT):
            failures.append(f"{name}: local link escapes the repository")
            continue
        relative = resolved.relative_to(ROOT).as_posix()
        if relative not in tracked and not any(path.startswith(relative.rstrip("/") + "/") for path in tracked):
            failures.append(f"{name}: link target is not published: {relative}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", action="store_true")
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    names = {item.decode() for item in git("ls-files", "-z").split(b"\0") if item}
    if args.worktree:
        names.update(item.decode() for item in git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0") if item)
        names.difference_update(item.decode() for item in git("ls-files", "--deleted", "-z").split(b"\0") if item)
    if not names:
        raise SystemExit("No source files selected for publication")
    failures, seen = [], set()
    for name in sorted(names):
        problem = path_problem(name)
        if problem:
            failures.append(f"{name}: forbidden publication path ({problem})")
            continue
        if args.worktree:
            content = (ROOT / name).read_bytes()
        else:
            content = git("show", ":" + name)
        failures.extend(check_content(name, content))
        failures.extend(check_links(name, content, names))
        seen.add(content)
    history_blobs = 0
    if args.history:
        objects = git("rev-list", "--objects", "--all", text=True).splitlines()
        for line in objects:
            pieces = line.split(" ", 1)
            if len(pieces) != 2:
                continue
            oid, name = pieces
            if git("cat-file", "-t", oid, text=True).strip() != "blob":
                continue
            problem = path_problem(name)
            if problem:
                failures.append(f"History: forbidden source path: {name}")
            content = git("cat-file", "blob", oid)
            if content not in seen:
                failures.extend(check_content("history:" + name, content))
                seen.add(content)
            history_blobs += 1
    if failures:
        for failure in sorted(set(failures)):
            print("FAIL " + failure, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"public_files_checked": len(names), "history_blobs_checked": history_blobs,
                      "common_secret_patterns": "passed", "private_paths_and_artifacts": "excluded",
                      "local_documentation_links": "passed", "notebook_outputs": "empty"}))


if __name__ == "__main__":
    main()
