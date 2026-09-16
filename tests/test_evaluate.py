import numpy as np
import pytest

from ecg.evaluate import confusion, per_class_metrics, per_record_metrics, summary

Y_TRUE = np.array(["N"] * 6 + ["V"] * 3 + ["S"])
Y_PRED = np.array(["N", "N", "N", "N", "V", "S"] + ["V", "V", "N"] + ["S"])


def test_confusion_counts_and_row_normalization():
    cm = confusion(Y_TRUE, Y_PRED)
    assert cm.loc["N", "N"] == 4 and cm.loc["N", "V"] == 1 and cm.loc["V", "N"] == 1
    assert cm.to_numpy().sum() == len(Y_TRUE)
    norm = confusion(Y_TRUE, Y_PRED, normalize=True)
    np.testing.assert_allclose(norm.loc[["N", "S", "V"]].sum(axis=1), 1.0)
    assert norm.loc["F"].isna().all()  # clase sin casos reales


def test_se_ppv_f1_by_hand():
    m = per_class_metrics(Y_TRUE, Y_PRED)
    # V: TP=2, FN=1, FP=1
    assert m.loc["V", "Se"] == pytest.approx(2 / 3)
    assert m.loc["V", "+P"] == pytest.approx(2 / 3)
    assert m.loc["V", "F1"] == pytest.approx(2 / 3)
    # S: TP=1, FN=0, FP=1
    assert m.loc["S", "Se"] == 1.0
    assert m.loc["S", "+P"] == 0.5
    # F: sin casos reales ni predichos
    assert np.isnan(m.loc["F", "Se"]) and np.isnan(m.loc["F", "+P"])


def test_all_normal_model_has_high_accuracy_but_low_macro_f1():
    y = np.array(["N"] * 90 + ["V"] * 10)
    s = summary(y, np.array(["N"] * 100))
    assert s["accuracy"] == 0.9
    assert s["V_Se"] == 0.0
    assert s["macro_f1"] < 0.25


def test_per_record_metrics():
    records = np.array([1] * 5 + [2] * 5)
    pr = per_record_metrics(Y_TRUE, Y_PRED, records)
    assert pr.loc[1, "latidos"] == 5 and pr["errores"].sum() == int((Y_TRUE != Y_PRED).sum())


def test_metrics_on_empty_input_do_not_crash():
    """Un subconjunto vacío (p. ej. un registro ausente) devuelve NaN, no una excepción."""
    empty = np.array([], dtype="<U1")
    assert confusion(empty, empty).to_numpy().sum() == 0
    assert per_class_metrics(empty, empty)["Se"].isna().all()


def test_unknown_label_raises_a_clear_error():
    with pytest.raises(ValueError, match="Etiquetas fuera de"):
        confusion(np.array(["N", "S"]), np.array(["N", "F"]), classes=["N", "S", "V"])
