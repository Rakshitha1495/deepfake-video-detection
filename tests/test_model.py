import pytest

torch = pytest.importorskip("torch")

from src.model import CNNLSTM, count_parameters, load_checkpoint, save_checkpoint  # noqa: E402


def _tiny_model():
    # pretrained_backbone=False: no download needed for tests
    return CNNLSTM(hidden_size=16, pretrained_backbone=False, freeze_backbone=True)


def test_forward_shapes():
    m = _tiny_model().eval()
    with torch.inference_mode():
        out = m(torch.randn(2, 5, 3, 224, 224))
        feats = m.extract_features(torch.randn(1, 5, 3, 224, 224))
    assert out.shape == (2, 2)
    assert feats.shape == (1, 5, 512)


def test_only_lstm_and_head_are_trainable():
    c = count_parameters(_tiny_model())
    assert 0 < c["trainable"] < c["total"] / 50


def test_checkpoint_roundtrip_without_download(tmp_path):
    m = _tiny_model().eval()
    cfg = {"hidden_size": 16, "num_layers": 1, "dropout": 0.3, "freeze_backbone": True, "num_frames": 5}
    p = tmp_path / "m.pt"
    save_checkpoint(p, m, cfg, {"best_epoch": 1})
    m2, ckpt = load_checkpoint(p)
    x = torch.randn(1, 5, 3, 224, 224)
    with torch.inference_mode():
        assert torch.allclose(m(x), m2(x), atol=1e-5)
    assert ckpt["best_epoch"] == 1
