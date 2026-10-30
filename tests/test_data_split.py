from pathlib import Path

from src.data_split import assign_groups, discover_videos, split_by_group


def _items(names):
    out = []
    for rel in names:
        out.append({"path": rel, "rel_path": rel, "label": 0 if rel.startswith("real/") else 1})
    return out


def test_faceforensics_names_are_linked():
    items, _ = assign_groups(_items(["real/000.mp4", "real/003.mp4", "fake/000_003.mp4", "real/010.mp4"]),
                             root=Path("/nonexistent"))
    g = {it["rel_path"]: it["group"] for it in items}
    assert g["real/000.mp4"] == g["fake/000_003.mp4"]
    assert g["real/010.mp4"] != g["real/000.mp4"]


def test_celebdf_names_are_linked():
    items, _ = assign_groups(_items(["real/id0_0001.mp4", "fake/id0_id16_0001.mp4", "real/id16_0004.mp4",
                                     "real/id5_0000.mp4"]), root=Path("/nonexistent"))
    g = {it["rel_path"]: it["group"] for it in items}
    assert g["real/id0_0001.mp4"] == g["fake/id0_id16_0001.mp4"]
    assert g["real/id5_0000.mp4"] != g["real/id0_0001.mp4"]



def test_groups_csv_overrides(tmp_path):
    (tmp_path / "groups.csv").write_text("path,group\nreal/a.mp4,s1\nfake/b.mp4,s1\n")
    items, method = assign_groups(_items(["real/a.mp4", "fake/b.mp4", "real/c.mp4"]), root=tmp_path)
    g = {it["rel_path"]: it["group"] for it in items}
    assert g["real/a.mp4"] == g["fake/b.mp4"] != g["real/c.mp4"]
    assert "groups.csv" in method


def test_split_has_no_group_leakage_and_is_reproducible():
    names = [f"real/{i:03d}.mp4" for i in range(30)] + [f"fake/{i:03d}_{(i + 1) % 30:03d}.mp4" for i in range(0, 30, 2)]
    names += [f"real/x{i}.mp4" for i in range(20)] + [f"fake/y{i}.mp4" for i in range(20)]
    items, _ = assign_groups(_items(names), root=Path("/nonexistent"))
    s1 = split_by_group(items, 0.15, 0.15, seed=7)
    s2 = split_by_group([dict(i) for i in items], 0.15, 0.15, seed=7)
    where = {}
    for name, vids in s1["splits"].items():
        for v in vids:
            assert where.setdefault(v["group"], name) == name
    assert {k: [v["rel_path"] for v in vs] for k, vs in s1["splits"].items()} == \
           {k: [v["rel_path"] for v in vs] for k, vs in s2["splits"].items()}
    assert sum(s["videos"] for s in s1["summary"].values()) == len(items)


def test_too_small_dataset_is_flagged():
    items, _ = assign_groups(_items(["real/a.mp4", "fake/b.mp4"]), root=Path("/nonexistent"))
    s = split_by_group(items, 0.15, 0.15, seed=1)
    assert s["meaningful"] is False
    assert s["warnings"]


def test_discover_videos(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "fake").mkdir()
    for n in ("a.mp4", "b.MOV", "notes.txt"):
        (tmp_path / "real" / n).write_bytes(b"x")
    (tmp_path / "fake" / "c.mkv").write_bytes(b"x")
    items = discover_videos(tmp_path)
    assert sorted(i["rel_path"] for i in items) == ["fake/c.mkv", "real/a.mp4", "real/b.MOV"]
