from src.metrics import compute_metrics, save_confusion_matrix_png


def test_metrics_values():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.8, 0.9])
    assert m["accuracy"] == 0.75
    assert m["recall"] == 1.0
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["roc_auc"] == 1.0


def test_roc_auc_not_reported_for_single_class():
    m = compute_metrics([1, 1], [1, 0], [0.9, 0.2])
    assert m["roc_auc"] is None and m["roc_auc_note"]


def test_empty():
    m = compute_metrics([], [], [])
    assert m["accuracy"] is None and m["num_samples"] == 0


def test_confusion_png(tmp_path):
    p = tmp_path / "cm.png"
    save_confusion_matrix_png([[3, 1], [0, 4]], p)
    assert p.exists() and p.stat().st_size > 1000
