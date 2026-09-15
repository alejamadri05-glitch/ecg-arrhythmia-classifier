"""Configuración central: splits inter-paciente, mapeo AAMI y parámetros de señal.

Todo lo que define "qué datos van dónde" vive acá, para que haya una sola fuente de verdad
y las pruebas anti-fuga puedan verificarlo.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DB_DIR = DATA_DIR / "mitdb"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42

# --- Señal -----------------------------------------------------------------------------
FS = 360  # Hz, frecuencia nativa de MIT-BIH
LEAD = "MLII"  # siempre se busca por nombre (el registro 114 tiene los canales invertidos)
BANDPASS_LO = 0.5  # Hz, elimina deriva de línea base
BANDPASS_HI = 40.0  # Hz, reduce ruido muscular y de red
BANDPASS_ORDER = 4
PRE_SAMPLES = 90  # 0.25 s antes del pico R
POST_SAMPLES = 162  # 0.45 s después del pico R
WINDOW = PRE_SAMPLES + POST_SAMPLES  # 252 muestras
LOCAL_RR_BEATS = 10  # latidos previos para el RR local promedio

# --- Clases AAMI (EC57) ----------------------------------------------------------------
CLASSES = ["N", "S", "V", "F"]
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}  # fmt: skip
Q_SYMBOLS = {"/", "f", "Q"}  # marcapasos / no clasificable: cuentan para el RR pero se excluyen
BEAT_SYMBOLS = set(AAMI_MAP) | Q_SYMBOLS

# --- Split inter-paciente (de Chazal et al., 2004) -------------------------------------
DS1 = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
    124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
]  # fmt: skip
DS2 = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
    212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
]  # fmt: skip
# Registros con marcapasos: excluidos según AAMI EC57
PACED = [102, 104, 107, 217]
