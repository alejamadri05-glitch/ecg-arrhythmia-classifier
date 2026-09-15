import numpy as np
import pytest
import torch

from ecg import config
from ecg.models.cnn import (
    CNNClassifier,
    CNNConfig,
    ECGNet,
    augment_batch,
    loss_class_weights,
)

W = config.WINDOW


@pytest.fixture(scope="module")
def toy():
    """Latidos sintéticos separables: la clase cambia la forma y el RR."""
    rng = np.random.default_rng(0)
    n_per, n_groups = 60, 8
    t = np.linspace(-1, 1, W)
    classes = [("N", 0.03, 1.0), ("S", 0.03, 0.6), ("V", 0.12, 0.6), ("F", 0.07, 1.0)]
    X, F, y, g = [], [], [], []
    for grp in range(n_groups):
        for cls, width, pre in classes:
            n = n_per if cls == "N" else n_per // 3
            beat = np.exp(-0.5 * (t / width) ** 2)
            X.append(beat + 0.1 * rng.normal(size=(n, W)))
            rr = np.array([pre, 1.0, pre, 1 / pre]) + 0.05 * rng.normal(size=(n, 4))
            F.append(rr)
            y += [cls] * n
            g += [grp] * n
    X, F = np.vstack(X).astype(np.float32), np.vstack(F).astype(np.float32)
    return X, F, np.array(y), np.array(g)


def small_cfg(**kw):
    return CNNConfig(**{"device": "cpu", "max_epochs": 3, "patience": 2, "batch_size": 64, **kw})


@pytest.mark.parametrize("n_rr", [0, 2, 4])
def test_ecgnet_output_shape(n_rr):
    out = ECGNet(n_rr=n_rr)(torch.randn(5, 1, W), torch.randn(5, n_rr))
    assert out.shape == (5, 4)


def test_augment_keeps_shape_and_invert_flips_sign():
    torch.manual_seed(0)
    x = torch.ones(200, 1, W)
    a = augment_batch(x, invert=False)
    assert a.shape == x.shape and (a.mean(dim=2) > 0).all()
    b = augment_batch(x, invert=True)
    frac_neg = (b.mean(dim=2) < 0).float().mean().item()
    assert 0.3 < frac_neg < 0.7


def test_loss_weights_zero_for_absent_class():
    w = loss_class_weights(np.array([0] * 90 + [2] * 10))
    assert w[1] == 0 and w[3] == 0
    assert w[2] > w[0]


def test_inner_validation_patients_are_disjoint_and_scaler_uses_train_only(toy):
    X, F, y, g = toy
    clf = CNNClassifier(small_cfg()).fit(X, F, y, groups=g)
    tr, va = clf.train_idx_, clf.val_idx_
    assert set(g[tr]).isdisjoint(g[va]) and len(va) > 0
    np.testing.assert_allclose(clf.scaler_.mean_, F[tr].mean(axis=0), rtol=1e-5)


def test_fit_requires_groups_or_epochs(toy):
    X, F, y, g = toy
    with pytest.raises(ValueError):
        CNNClassifier(small_cfg()).fit(X, F, y)
    with pytest.raises(ValueError):
        CNNClassifier(small_cfg()).fit(X, F, y, groups=g, epochs=2)


@pytest.mark.parametrize("cfg_kw", [{}, {"rr": "none"}, {"rr": "ratios", "balance": "sampler"},
                                    {"augment": True, "invert": True}])  # fmt: skip
def test_predict_proba_valid_for_variants(toy, cfg_kw):
    X, F, y, _ = toy
    clf = CNNClassifier(small_cfg(**cfg_kw)).fit(X, F, y, epochs=2)
    p = clf.predict_proba(X[:50], F[:50])
    assert p.shape == (50, 4)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


def test_learns_separable_toy_problem(toy):
    X, F, y, _ = toy
    clf = CNNClassifier(small_cfg(max_epochs=8)).fit(X, F, y, epochs=8)
    assert (clf.predict(X, F) == y).mean() > 0.9


def test_same_seed_same_output_on_cpu(toy):
    X, F, y, _ = toy
    a = CNNClassifier(small_cfg()).fit(X, F, y, epochs=2).predict_proba(X, F)
    b = CNNClassifier(small_cfg()).fit(X, F, y, epochs=2).predict_proba(X, F)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_save_load_roundtrip(toy, tmp_path):
    X, F, y, _ = toy
    clf = CNNClassifier(small_cfg(rr="ratios")).fit(X, F, y, epochs=2)
    clf.save(tmp_path / "cnn.pt")
    loaded = CNNClassifier.load(tmp_path / "cnn.pt")
    np.testing.assert_allclose(loaded.predict_proba(X, F), clf.predict_proba(X, F), atol=1e-5)
    assert loaded.model_version == "1.0.0"
