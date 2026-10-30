"""Video discovery and leak-free, group-level train/val/test splitting.

Why groups? A fake video is usually made FROM a real video. If the original
lands in the training set and its manipulated copy lands in the test set, the
test score is inflated. We therefore assign related videos to the same
"source group" and split by group, never by individual file.

Group assignment, in priority order:
1. dataset/raw/groups.csv (columns: path,group). `path` is relative to
   dataset/raw, e.g. `fake/000_003.mp4`.
2. Filename heuristics for common datasets, linked with union-find:
   - FaceForensics++: real `000.mp4`, fake `000_003.mp4` -> links 000 and 003
   - Celeb-DF: real `id0_0003.mp4`, fake `id0_id16_0003.mp4` -> links id0 and id16
3. Otherwise each video is its own group (a warning is recorded).
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import DATASET_DIR, VIDEO_EXTENSIONS

_ID_TOKEN = re.compile(r"^id\d+$", re.IGNORECASE)
_NUM_TOKEN = re.compile(r"^\d+$")


def discover_videos(root: Path = DATASET_DIR, max_per_class: int | None = None, seed: int = 42) -> list[dict]:
    """Find videos in root/real (label 0) and root/fake (label 1)."""
    items: list[dict] = []
    for label, sub in ((0, "real"), (1, "fake")):
        folder = root / sub
        if not folder.exists():
            continue
        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if max_per_class and len(files) > max_per_class:
            rng = random.Random(seed + label)
            files = sorted(rng.sample(files, max_per_class))
        for p in files:
            items.append({
                "path": str(p),
                "rel_path": p.relative_to(root).as_posix(),
                "label": label,
            })
    return items


def validate_dataset(items: list[dict], min_frames: int = 1) -> dict:

    """Validate all items: check readability, corruption, duplicates, and label consistency.

    Reports corrupted, unreadable, duplicate, and invalid videos separately.
    Never silently drops videos.
    """
    from .video_utils import validate_video_file

    valid: list[dict] = []
    corrupted: list[dict] = []
    unreadable: list[dict] = []
    duplicates: list[dict] = []
    invalid: list[dict] = []

    seen_paths: set[str] = set()

    for it in items:
        p_str = it["path"]
        if p_str in seen_paths:
            duplicates.append({"path": it["rel_path"], "reason": "Duplicate video path detected"})
            continue
        seen_paths.add(p_str)

        if it.get("label") not in (0, 1):
            invalid.append({"path": it["rel_path"], "reason": f"Invalid class label: {it.get('label')}"})
            continue

        ok, reason, meta = validate_video_file(p_str, min_frames=min_frames)
        if not ok:
            if "could not open" in reason.lower() or "0 bytes" in reason.lower():
                corrupted.append({"path": it["rel_path"], "reason": reason})
            else:
                unreadable.append({"path": it["rel_path"], "reason": reason})
        else:
            item_valid = dict(it)
            item_valid.update(meta)
            valid.append(item_valid)

    real_cnt = sum(1 for it in valid if it["label"] == 0)
    fake_cnt = sum(1 for it in valid if it["label"] == 1)

    return {
        "valid": valid,
        "corrupted": corrupted,
        "unreadable": unreadable,
        "duplicates": duplicates,
        "invalid": invalid,
        "counts": {
            "real": real_cnt,
            "fake": fake_cnt,
            "total": len(valid),
        },
    }




def _source_keys(stem: str) -> list[str] | None:
    """Identify the original source video, not just the person ID."""

    tokens = stem.lower().split("_")

    # Celeb-DF:
    # Real: id0_0000
    # Fake: id0_id1_0000
    #
    # Both belong to source video id0_0000.

    if len(tokens) == 2:
        if _ID_TOKEN.match(tokens[0]) and _NUM_TOKEN.match(tokens[1]):
            return [f"celeb:{tokens[0]}_{tokens[1]}"]

    if len(tokens) == 3:
        if (
            _ID_TOKEN.match(tokens[0])
            and _ID_TOKEN.match(tokens[1])
            and _NUM_TOKEN.match(tokens[2])
        ):
            return [f"celeb:{tokens[0]}_{tokens[2]}"]

    # FaceForensics++:
    # Real: 000
    # Fake: 000_003
    # The first number identifies the source video.

    if len(tokens) in (1, 2):
        if all(_NUM_TOKEN.match(t) for t in tokens):
            return [f"num:{int(tokens[0])}"]

    return None



def _load_groups_csv(root: Path) -> dict[str, str]:
    csv_path = root / "groups.csv"
    if not csv_path.exists():
        return {}
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("path") and row.get("group"):
                mapping[row["path"].strip().replace("\\", "/")] = row["group"].strip()
    return mapping


def assign_groups(items: list[dict], root: Path = DATASET_DIR) -> tuple[list[dict], str]:
    """Adds a 'group' key to every item. Returns (items, method_description)."""
    explicit = _load_groups_csv(root)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    item_keys: list[list[str]] = []
    used = Counter()
    for it in items:
        if it["rel_path"] in explicit:
            keys = [f"csv:{explicit[it['rel_path']]}"]
            used["groups.csv"] += 1
        else:
            keys = _source_keys(Path(it["rel_path"]).stem)
            if keys:
                used["filename-pattern"] += 1
            else:
                keys = [f"file:{it['rel_path']}"]
                used["per-video"] += 1
        for k in keys:
            find(k)
        for k in keys[1:]:
            union(keys[0], k)
        item_keys.append(keys)

    for it, keys in zip(items, item_keys):
        it["group"] = find(keys[0])

    method = ", ".join(f"{k}: {v}" for k, v in used.items()) or "none"
    return items, method


def split_by_group(items: list[dict], val_fraction: float, test_fraction: float, seed: int) -> dict:
    """Group-level split, stratified by each group's label composition."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        groups[it["group"]].append(it)

    def kind(members: list[dict]) -> str:
        labels = {m["label"] for m in members}
        return "mixed" if len(labels) == 2 else ("real" if 0 in labels else "fake")

    by_kind: dict[str, list[str]] = defaultdict(list)
    for g in sorted(groups):
        by_kind[kind(groups[g])].append(g)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for k in sorted(by_kind):
        gs = by_kind[k][:]
        rng.shuffle(gs)
        n = len(gs)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 3:
            n_test = max(n_test, 1)
            n_val = max(n_val, 1)
        n_test = min(n_test, max(n - 1, 0))
        n_val = min(n_val, max(n - 1 - n_test, 0))
        for i, g in enumerate(gs):
            assignment[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for it in items:
        splits[assignment[it["group"]]].append(it)

    warnings = []
    for name in ("train", "val", "test"):
        labels = {it["label"] for it in splits[name]}
        if labels != {0, 1}:
            warnings.append(
                f"The {name} split does not contain both Real and Fake videos "
                f"({len(splits[name])} videos). Metrics for it are not meaningful."
            )
    group_sizes = Counter(it["group"] for it in items)
    largest = max(group_sizes.values()) if group_sizes else 0
    if items and largest > 0.4 * len(items):
        warnings.append(
            f"One source group contains {largest} of {len(items)} videos; the split may be unbalanced."
        )

    summary = {
        name: {
            "videos": len(v),
            "real": sum(1 for it in v if it["label"] == 0),
            "fake": sum(1 for it in v if it["label"] == 1),
            "groups": len({it["group"] for it in v}),
        }
        for name, v in splits.items()
    }
    # Sanity check: no group may appear in more than one split.
    seen: dict[str, str] = {}
    for name, v in splits.items():
        for it in v:
            assert seen.setdefault(it["group"], name) == name, "group leaked across splits"

    return {
        "splits": splits,
        "summary": summary,
        "num_groups": len(groups),
        "warnings": warnings,
        "meaningful": not any("not meaningful" in w for w in warnings),
    }


def class_distribution(items: list[dict]) -> dict:
    c = Counter(it["label"] for it in items)
    return {"real": c.get(0, 0), "fake": c.get(1, 0), "total": len(items)}
