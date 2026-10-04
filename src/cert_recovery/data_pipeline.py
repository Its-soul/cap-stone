from __future__ import annotations

import json
import math
import random
import unicodedata
from pathlib import Path

from .config import require_manual, resolve_path
from .crypto import canonical_json, sha256


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Pair text must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalized:
        raise ValueError("Empty normalized text")
    return normalized


def clean_rows(rows: list[dict], group_key: str = "world_id") -> tuple[list[dict], dict]:
    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    seen_pairs: dict[tuple[str, str], dict] = {}
    duplicate_count = 0
    for index, row in enumerate(rows):
        for key in ("pair_id", group_key, "premise", "hypothesis", "label", "source"):
            if key not in row:
                raise ValueError(f"Row {index} missing {key}")
        if (type(row["label"]) is not int or row["label"] not in (0, 1)
                or not isinstance(row["pair_id"], str) or not row["pair_id"].strip()
                or not isinstance(row[group_key], str) or not row[group_key].strip()
                or not isinstance(row["source"], str) or not row["source"].strip()):
            raise ValueError(f"Row {index} has invalid ID/group/source/label")
        if row["pair_id"] in seen_ids:
            raise ValueError("Duplicate pair_id")
        seen_ids.add(row["pair_id"])
        item = {"pair_id": row["pair_id"], group_key: row[group_key],
                "premise": normalize_text(row["premise"]), "hypothesis": normalize_text(row["hypothesis"]),
                "label": row["label"], "source": row["source"]}
        pair = (item["premise"], item["hypothesis"])
        if pair in seen_pairs:
            old = seen_pairs[pair]
            if old["label"] != item["label"]:
                raise ValueError("Contradictory labels for the same normalized pair")
            if old[group_key] != item[group_key]:
                raise ValueError("Repeated pair spans worlds; regroup connected worlds before splitting")
            duplicate_count += 1
            continue
        seen_pairs[pair] = item
        cleaned.append(item)
    if not cleaned:
        raise ValueError("Dataset is empty")
    return cleaned, {"raw_rows": len(rows), "clean_rows": len(cleaned), "removed_duplicates": duplicate_count}


def grouped_split(rows: list[dict], fractions: tuple[float, float, float], seed: int,
                  group_key: str = "world_id") -> dict[str, list[dict]]:
    if (len(fractions) != 3 or any(not math.isfinite(x) or not 0 < x < 1 for x in fractions)
            or not math.isclose(sum(fractions), 1.0)):
        raise ValueError("Three positive split fractions must sum to one")
    groups = sorted({row[group_key] for row in rows})
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")
    random.Random(seed).shuffle(groups)
    train_count = min(len(groups) - 2, max(1, round(len(groups) * fractions[0])))
    validation_count = min(len(groups) - train_count - 1, max(1, round(len(groups) * fractions[1])))
    assignments = {group: "train" if index < train_count else "validation"
                   if index < train_count + validation_count else "test"
                   for index, group in enumerate(groups)}
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in rows:
        splits[assignments[row[group_key]]].append(dict(row))
    return splits


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_data(config: dict, *, manual: bool = False) -> dict:
    require_manual(config, "allow_data_preparation", manual)
    rows, cleaning = clean_rows(read_jsonl(resolve_path(config, "raw_pairs")), config["data"]["group_key"])
    fractions = tuple(config["data"][f"{name}_fraction"] for name in ("train", "validation", "test"))
    splits = grouped_split(rows, fractions, config["data"]["seed"], config["data"]["group_key"])
    target = resolve_path(config, "processed")
    if any((target / f"{name}.jsonl").exists() for name in splits):
        raise FileExistsError("Split files already exist; preserve them or choose a new processed directory")
    target.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (target / f"{name}.jsonl").write_text("".join(canonical_json(item) + "\n" for item in items), encoding="utf-8")
    manifest = {"stage": "DATA_PREPARED_MANUALLY", "cleaning": cleaning,
                "seed": config["data"]["seed"], "fractions": fractions, "group_key": config["data"]["group_key"],
                "normalized_dataset_hash": sha256(rows),
                "splits": {name: {"rows": len(items), "hash": sha256(items),
                                   "groups": sorted({row[config["data"]["group_key"]] for row in items})}
                           for name, items in splits.items()}}
    write_json(target / "split_manifest.json", manifest)
    return manifest


def load_verified_splits(config: dict) -> tuple[dict[str, list[dict]], dict]:
    target = resolve_path(config, "processed")
    manifest = json.loads((target / "split_manifest.json").read_text(encoding="utf-8"))
    group_key = config["data"]["group_key"]
    if manifest["group_key"] != group_key:
        raise ValueError("Split group key differs from config")
    splits = {name: read_jsonl(target / f"{name}.jsonl") for name in ("train", "validation", "test")}
    groups_seen: set[str] = set()
    pairs_seen: set[tuple[str, str]] = set()
    for name, rows in splits.items():
        if not rows or sha256(rows) != manifest["splits"][name]["hash"]:
            raise ValueError(f"Empty or altered split: {name}")
        groups = {row[group_key] for row in rows}
        pairs = {(row["premise"], row["hypothesis"]) for row in rows}
        if groups & groups_seen or pairs & pairs_seen:
            raise ValueError("Cross-split group or pair leakage")
        groups_seen.update(groups)
        pairs_seen.update(pairs)
    return splits, manifest

