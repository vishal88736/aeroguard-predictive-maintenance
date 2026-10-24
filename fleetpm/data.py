"""Strict C-MAPSS parsing and ZIP import; no numerical dependencies required."""

from collections import Counter
import io
import math
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

from .config import COUNTS, DATA_DIR, SUBSETS, file_hash

ALLOWED = {f"{part}_{subset}.txt" for subset in SUBSETS for part in ("train", "test", "RUL")}
CANONICAL = {name.lower(): name for name in ALLOWED}


def parse_rows(path):
    rows, seen = [], set()
    with open(path, encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 26:
                raise ValueError(f"{Path(path).name}:{number}: expected 26 fields, got {len(fields)}")
            values = [float(x) for x in fields]
            unit, cycle = values[:2]
            if not all(math.isfinite(x) and x > 0 and x.is_integer() for x in (unit, cycle)):
                raise ValueError(f"{Path(path).name}:{number}: invalid engine/cycle")
            if cycle>10000:
                raise ValueError(f"{Path(path).name}:{number}: cycle exceeds the 10000-cycle benchmark allocation limit")
            if not all(math.isfinite(x) for x in values[2:5]):
                raise ValueError(f"{Path(path).name}:{number}: missing operating setting")
            if any(math.isinf(x) for x in values[5:]):
                raise ValueError(f"{Path(path).name}:{number}: infinite sensor value")
            key = (int(unit), int(cycle))
            if key in seen:
                raise ValueError(f"{Path(path).name}:{number}: duplicate engine/cycle {key}")
            seen.add(key)
            rows.append(values)
    if not rows:
        raise ValueError(f"{Path(path).name}: empty file")
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def parse_truth(path, engine_count):
    values = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        if len(line.split()) != 1:
            raise ValueError(f"Truth line {number}: expected one endpoint RUL")
        value = float(line.strip())
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid endpoint RUL")
        values.append(value)
    if len(values) != engine_count:
        raise ValueError(f"Truth has {len(values)} entries, expected {engine_count}")
    return values


def inventory(directory=DATA_DIR, deep=False):
    directory = Path(directory)
    output = []
    for subset in SUBSETS:
        required = [f"{part}_{subset}.txt" for part in ("train", "test", "RUL")]
        item = {"subset": subset, "expected_train": COUNTS[subset][0],
                "expected_test": COUNTS[subset][1], "conditions": COUNTS[subset][2],
                "fault_modes": COUNTS[subset][3], "files": [], "ready": False,
                "train_engines": 0, "test_engines": 0, "rows": 0, "errors": []}
        for filename in required:
            path = directory / filename
            if path.is_file():
                item["files"].append({"name": filename, "bytes": path.stat().st_size})
        item["ready"] = len(item["files"]) == 3
        if deep and item["ready"]:
            try:
                train = parse_rows(directory / required[0]); test = parse_rows(directory / required[1])
                item["train_engines"] = len({r[0] for r in train})
                item["test_engines"] = len({r[0] for r in test})
                test_ids = sorted({int(r[0]) for r in test})
                if test_ids != list(range(1, len(test_ids) + 1)):
                    raise ValueError("Official truth mapping requires consecutive test engine IDs starting at 1")
                parse_truth(directory / required[2], len(test_ids))
                item["rows"] = len(train) + len(test)
                item["official_counts_match"] = (item["train_engines"], item["test_engines"]) == COUNTS[subset][:2]
            except (ValueError, OSError) as error:
                item["errors"].append(str(error)); item["ready"] = False
        output.append(item)
    return output


def import_zip(payload, directory=DATA_DIR):
    """Validate before replacing any input; never extract arbitrary archive paths."""
    directory = Path(directory)
    try:
        archive_file=zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("Choose a valid ZIP archive") from error
    with archive_file as archive:
        members = {}
        total = 0
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = PurePosixPath(member.filename.replace("\\", "/")).name
            canonical = CANONICAL.get(name.lower())
            if canonical is None:
                continue
            if canonical in members:
                raise ValueError(f"Duplicate dataset filename in archive: {canonical}")
            if member.file_size > 80 * 1024 * 1024:
                raise ValueError("Dataset member is too large")
            total += member.file_size
            if total > 200 * 1024 * 1024:
                raise ValueError("Dataset archive exceeds the uncompressed size limit")
            members[canonical] = member
        if set(members) != ALLOWED:
            raise ValueError("ZIP must contain all 12 NASA train/test/RUL files for FD001-FD004")
        directory.mkdir(parents=True, exist_ok=True)
        stage = directory / (".import-" + uuid.uuid4().hex)
        stage.mkdir()
        try:
            for name, member in members.items():
                with archive.open(member) as stream:
                    content = stream.read(80 * 1024 * 1024 + 1)
                if len(content) != member.file_size:
                    raise ValueError("Archive member size mismatch")
                (stage / name).write_bytes(content)
            report = inventory(stage, deep=True)
            if not all(x["ready"] for x in report):
                raise ValueError("; ".join(e for x in report for e in x["errors"]))
            # Do not overwrite a different corpus silently, including during training.
            for name in members:
                target = directory / name
                if target.exists() and file_hash(target) != file_hash(stage / name):
                    raise ValueError(f"{name} already exists with different contents. Use a separate data directory.")
            for name in members:
                target = directory / name
                if not target.exists():
                    (stage / name).replace(target)
            return report
        finally:
            for path in stage.iterdir():
                path.unlink()
            stage.rmdir()
