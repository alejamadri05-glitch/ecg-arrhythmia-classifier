"""Features para el modelo baseline, calculadas sobre las ventanas de `segment.beat_windows`.

Las ventanas ya vienen normalizadas con z-score, así que las features de morfología son de
forma (ancho, asimetría, energía por región), no de amplitud absoluta en mV.
Todo está vectorizado: ~100 000 latidos en menos de un segundo.
"""

import numpy as np
from scipy.stats import kurtosis, skew

from ecg import config
from ecg.segment import RR_FEATURE_NAMES

FS = config.FS
R = config.PRE_SAMPLES  # índice del pico R dentro de la ventana
WAVE_BINS = 21  # 252 / 12 muestras ≈ 33 ms por bin


def _ms(ms: float) -> int:
    return round(ms * FS / 1000)


def qrs_width(X: np.ndarray) -> np.ndarray:
    """Ancho aproximado del QRS (ms): extensión donde la pendiente supera el 20 % de la máxima."""
    d = np.abs(np.diff(X[:, R - _ms(100) : R + _ms(120)], axis=1))
    above = d > 0.2 * d.max(axis=1, keepdims=True)
    first = above.argmax(axis=1)
    last = above.shape[1] - 1 - above[:, ::-1].argmax(axis=1)
    return (last - first + 1) * 1000 / FS


def morphology_features(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features de forma del latido: QRS, onda P, onda T y estadísticos de la ventana."""
    base = np.median(X, axis=1, keepdims=True)  # la mayor parte de la ventana es isoeléctrica
    a = np.abs(X - base)

    qrs = slice(R - _ms(100), R + _ms(100))
    p_region = slice(0, R - _ms(100))  # -250 a -100 ms
    t_region = slice(R + _ms(150), X.shape[1])  # +150 a +450 ms

    qrs_width_ms = qrs_width(X)

    seg_qrs = X[:, qrs]
    offset = np.arange(seg_qrs.shape[1]) - _ms(100)
    feats = {
        "r_amp": X[:, R] - base[:, 0],
        "qrs_max": seg_qrs.max(axis=1),
        "qrs_min": seg_qrs.min(axis=1),
        "qrs_argmax_ms": offset[seg_qrs.argmax(axis=1)] * 1000 / FS,
        "qrs_argmin_ms": offset[seg_qrs.argmin(axis=1)] * 1000 / FS,
        "qrs_width_ms": qrs_width_ms,
        "qrs_area": a[:, qrs].sum(axis=1) / FS,
        "total_area": a.sum(axis=1) / FS,
        "qrs_energy_ratio": (a[:, qrs] ** 2).sum(axis=1) / ((a**2).sum(axis=1) + 1e-8),
        "p_std": X[:, p_region].std(axis=1),
        "p_max": X[:, p_region].max(axis=1) - base[:, 0],
        "t_std": X[:, t_region].std(axis=1),
        "t_max": X[:, t_region].max(axis=1) - base[:, 0],
        "t_min": X[:, t_region].min(axis=1) - base[:, 0],
        "skewness": skew(X, axis=1),
        "kurtosis": kurtosis(X, axis=1),
    }
    return np.column_stack(list(feats.values())).astype(np.float32), list(feats)


def waveform_features(X: np.ndarray, bins: int = WAVE_BINS) -> tuple[np.ndarray, list[str]]:
    """Latido submuestreado (promedio por bloques): la forma completa en pocas dimensiones."""
    n, w = X.shape
    step = w // bins
    wave = X[:, : bins * step].reshape(n, bins, step).mean(axis=2)
    names = [f"wave_{round((i * step + step / 2 - R) * 1000 / FS):+d}ms" for i in range(bins)]
    return wave.astype(np.float32), names


def normalized_rr(F: np.ndarray, records: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """RR dividido por la mediana del RR del propio registro.

    Hace la feature comparable entre pacientes con frecuencias cardíacas distintas: un RR de
    0.73 s es normal en alguien a 80 lpm y claramente prematuro en un bradicárdico a 32 lpm
    (registro 232 de DS2). La mediana se calcula con la señal de entrada, no con las etiquetas,
    así que la API puede calcularla igual con el ECG que recibe.

    `records`: identificador de registro por latido. Si es None, todos los latidos se toman como
    una sola grabación (el caso de la API).
    """
    pre, post = F[:, 0], F[:, 1]
    median = np.empty(len(F), dtype=np.float64)
    if records is None:
        median[:] = np.median(pre)
    else:
        for rec in np.unique(records):
            m = records == rec
            median[m] = np.median(pre[m])
    median = np.maximum(median, 1e-3)
    out = np.column_stack([pre / median, post / median]).astype(np.float32)
    return out, ["pre_rr_over_median", "post_rr_over_median"]


def patient_template(
    X: np.ndarray,
    records: np.ndarray | None = None,
    max_beats: int | None = None,
    block: int | None = None,
) -> np.ndarray:
    """Latido dominante de cada paciente: la mediana de sus latidos.

    No usa etiquetas, así que la API puede calcularlo con el ECG que recibe. La mediana es
    robusta mientras la morfología habitual sea mayoría; en un paciente con más de la mitad de
    latidos anormales la plantilla se contamina (limitación documentada en el model card).

    `max_beats` usa solo los primeros N latidos de cada registro, como haría un sistema que
    calcula la plantilla una sola vez al inicio. `block` la recalcula cada N latidos (plantilla
    local), que rinde mejor: la morfología deriva a lo largo de una grabación de 30 minutos, así
    que una plantilla del arranque queda vieja.
    """
    if max_beats and block:
        raise ValueError("`max_beats` y `block` son alternativas, no se combinan")
    template = np.empty_like(X)
    grupos = (
        [np.arange(len(X))]
        if records is None
        else [np.flatnonzero(records == rec) for rec in np.unique(records)]
    )
    for idx in grupos:
        if block:
            for i in range(0, len(idx), block):
                b = idx[i : i + block]
                template[b] = np.median(X[b], axis=0)
        else:
            template[idx] = np.median(X[idx[:max_beats]], axis=0)
    return template


def _row_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1)) + 1e-8
    return (a * b).sum(axis=1) / denom


def relative_morphology_features(
    X: np.ndarray,
    records: np.ndarray | None = None,
    max_beats: int | None = None,
    template: np.ndarray | None = None,
    block: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Describe cada latido **relativo al latido dominante de su paciente**.

    Es la misma idea de `normalized_rr`, pero aplicada a la forma en vez del ritmo: lo que
    distingue a un latido anormal no es su morfología absoluta (que cambia entre personas y entre
    derivaciones) sino en cuánto se aparta de la morfología habitual de esa persona.
    """
    if template is None:
        template = patient_template(X, records, max_beats, block)
    residual = X - template
    qrs = slice(R - _ms(100), R + _ms(100))
    ancho_latido, ancho_plantilla = qrs_width(X), qrs_width(template)
    feats = {
        "corr_plantilla": _row_correlation(X, template),
        "corr_plantilla_qrs": _row_correlation(X[:, qrs], template[:, qrs]),
        "rms_residuo": np.sqrt((residual**2).mean(axis=1)),
        "rms_residuo_qrs": np.sqrt((residual[:, qrs] ** 2).mean(axis=1)),
        "max_residuo": np.abs(residual).max(axis=1),
        "r_amp_menos_plantilla": X[:, R] - template[:, R],
        "qrs_ancho_sobre_plantilla": ancho_latido / np.maximum(ancho_plantilla, 1e-3),
    }
    return np.column_stack(list(feats.values())).astype(np.float32), list(feats)


def residual_waveform_features(
    X: np.ndarray,
    records: np.ndarray | None = None,
    bins: int = WAVE_BINS,
    max_beats: int | None = None,
    template: np.ndarray | None = None,
    block: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """El latido residual (latido − plantilla del paciente), submuestreado."""
    if template is None:
        template = patient_template(X, records, max_beats, block)
    wave, names = waveform_features(X - template, bins)
    return wave, [n.replace("wave_", "residuo_") for n in names]


FEATURE_SETS = {
    "rr": ["rr"],
    "rr_ratios": ["rr_ratios"],
    "morph": ["morph"],
    "rr+morph": ["rr", "morph"],
    "rr_ratios+morph": ["rr_ratios", "morph"],
    "rr+morph+wave": ["rr", "morph", "wave"],
    "rr_ratios+morph+wave": ["rr_ratios", "morph", "wave"],
    # Versión 2: RR normalizado por la mediana del registro + cocientes
    "rr_norm+morph+wave": ["rr_norm", "rr_ratios", "morph", "wave"],
    # Idea 3: morfología relativa al latido dominante del paciente
    "rr_norm+morph+rel+wave": ["rr_norm", "rr_ratios", "morph", "rel", "wave"],
    "rr_norm+rel": ["rr_norm", "rr_ratios", "rel"],
    "rr_norm+morph+rel+wave_rel": ["rr_norm", "rr_ratios", "morph", "rel", "wave_rel"],
}


def build_features(
    X: np.ndarray,
    F: np.ndarray,
    feature_set: str = "rr+morph+wave",
    records: np.ndarray | None = None,
    template_beats: int | None = None,
    template: np.ndarray | None = None,
    template_block: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Matriz de features para un conjunto con nombre (ver FEATURE_SETS).

    `records` se usa en los conjuntos con `rr_norm` (normalizar el ritmo por registro) y con
    `rel`/`wave_rel` (comparar cada latido con la plantilla de su paciente). `template_beats`
    limita la plantilla a los primeros N latidos de cada registro y `template_block` la recalcula
    cada N latidos (lo que usa la v4).
    """
    blocks, names = [], []
    for part in FEATURE_SETS[feature_set]:
        if part == "rr_norm":
            n_block, n_names = normalized_rr(F, records)
            blocks.append(n_block), names.extend(n_names)
        elif part == "rr":
            blocks.append(F), names.extend(RR_FEATURE_NAMES)
        elif part == "rr_ratios":
            # sin RR absolutos: solo cocientes, que generalizan mejor entre pacientes
            blocks.append(F[:, 2:]), names.extend(RR_FEATURE_NAMES[2:])
        elif part == "morph":
            m, n = morphology_features(X)
            blocks.append(m), names.extend(n)
        elif part == "rel":
            r, n = relative_morphology_features(
                X, records, template_beats, template, template_block
            )
            blocks.append(r), names.extend(n)
        elif part == "wave":
            w, n = waveform_features(X)
            blocks.append(w), names.extend(n)
        elif part == "wave_rel":
            w, n = residual_waveform_features(
                X, records, max_beats=template_beats, template=template, block=template_block
            )
            blocks.append(w), names.extend(n)
    return np.hstack(blocks).astype(np.float32), names
