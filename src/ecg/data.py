"""Descarga y lectura de registros MIT-BIH."""

import hashlib
import shutil
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from ecg import config

MITDB_ZIP_URL = (
    "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
)
_MITDB_SUFFIXES = {".dat", ".hea", ".atr"}


@dataclass
class Record:
    rec_id: int
    fs: float
    signal: np.ndarray  # canal MLII, en mV, sin filtrar
    ann_samples: np.ndarray  # posición de cada anotación (muestras)
    ann_symbols: np.ndarray  # símbolo de cada anotación


def with_retry(
    action: Callable[[], None],
    what: str,
    attempts: int = 5,
    wait_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Reintenta errores de red con espera exponencial. Los errores de programación no se
    reintentan: solo OSError, del que heredan URLError, los timeouts y NetFileError de wfdb."""
    for attempt in range(1, attempts + 1):
        try:
            action()
            return
        except OSError as exc:
            if attempt == attempts:
                raise
            delay = wait_s * 2 ** (attempt - 1)
            print(f"{what} falló ({exc}); intento {attempt + 1}/{attempts} en {delay:.0f} s")
            sleep(delay)


def dl_database_with_retry(
    db: str, dl_dir: Path, attempts: int = 5, wait_s: float = 5.0, sleep=time.sleep
) -> None:
    """``wfdb.dl_database`` con reintentos. Como wfdb no vuelve a bajar los archivos que ya
    están completos, cada intento sigue donde quedó el anterior."""
    with_retry(
        lambda: wfdb.dl_database(db, dl_dir=str(dl_dir)),
        f"Descarga de {db}",
        attempts=attempts,
        wait_s=wait_s,
        sleep=sleep,
    )


def download(db_dir: Path = config.DB_DIR, url: str = MITDB_ZIP_URL, sleep=time.sleep) -> None:
    """Descarga MIT-BIH (zip oficial de 77 MB) si todavía no está.

    No usa ``wfdb.dl_database``: esa función pide cientos de archivos seguidos y PhysioNet corta
    la ráfaga con 502 Bad Gateway (pasó en CI y se reprodujo en un contenedor Linux, incluso con
    reintentos), mientras que una petición suelta pasa sin problema. Cada archivo se verifica
    contra el SHA256SUMS.txt que publica PhysioNet dentro del mismo zip.
    """
    db_dir = Path(db_dir)
    if (db_dir / "234.atr").exists():
        return
    db_dir.mkdir(parents=True, exist_ok=True)
    zip_path = db_dir / "mitdb.zip.part"
    try:
        with_retry(lambda: _fetch(url, zip_path), "Descarga de MIT-BIH", sleep=sleep)
        extract_verified(zip_path, db_dir)
    finally:
        zip_path.unlink(missing_ok=True)


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def extract_verified(zip_path: Path, db_dir: Path) -> list[str]:
    """Extrae los registros (.dat, .hea, .atr y RECORDS) verificando su SHA-256.

    Todo se verifica antes de escribir: un zip corrupto no deja datos a medias que después
    pasen por buenos (``download`` usa la existencia de 234.atr para no volver a bajar).
    Solo se aceptan archivos del primer nivel de la carpeta raíz del zip, lo que descarta
    subcarpetas (x_mitdb, mitdbdir) y cualquier ruta con "..".
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = {}
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) == 2 and parts[1] and "\\" not in parts[1]:
                members[parts[1]] = name
        if "SHA256SUMS.txt" not in members:
            raise ValueError("El zip no trae SHA256SUMS.txt: no se puede verificar la integridad")
        expected = {}
        for line in zf.read(members["SHA256SUMS.txt"]).decode().splitlines():
            if line.strip():
                digest, fname = line.split(maxsplit=1)
                expected[fname] = digest

        wanted = sorted(f for f in members if f == "RECORDS" or Path(f).suffix in _MITDB_SUFFIXES)
        contents = {}
        for fname in wanted:
            content = zf.read(members[fname])
            if hashlib.sha256(content).hexdigest() != expected.get(fname):
                raise ValueError(f"{fname}: el SHA-256 no coincide con SHA256SUMS.txt")
            contents[fname] = content

    if "RECORDS" not in contents:
        raise ValueError("El zip no trae RECORDS")
    missing = [
        f"{rec}{suffix}"
        for rec in contents["RECORDS"].decode().split()
        for suffix in sorted(_MITDB_SUFFIXES)
        if f"{rec}{suffix}" not in contents
    ]
    if missing:
        raise ValueError(f"Faltan archivos de registros en el zip: {missing[:5]}")

    for fname, content in contents.items():
        (db_dir / fname).write_bytes(content)
    return wanted


def lead_index(sig_names: list[str], lead: str = config.LEAD) -> int:
    """Índice del canal por nombre. Nunca asumir índice 0 (el registro 114 está invertido)."""
    if lead not in sig_names:
        raise ValueError(f"Derivación {lead!r} no está en el registro: {sig_names}")
    return sig_names.index(lead)


def rhythm_at(rec_id: int, samples: np.ndarray, db_dir: Path = config.DB_DIR) -> np.ndarray:
    """Ritmo anotado vigente en cada muestra: "(N", "(AFIB", "(SVTA", "(VT", etc.

    MIT-BIH marca los cambios de ritmo con anotaciones "+" cuyo aux_note empieza con "(".
    Sirve para explicar errores (p. ej. latidos N que parecen prematuros durante una FA).
    """
    ann = wfdb.rdann(str(Path(db_dir) / str(rec_id)), "atr")
    idx = [i for i, note in enumerate(ann.aux_note) if note.startswith("(")]
    starts = ann.sample[idx]
    names = np.array([ann.aux_note[i].rstrip("\x00") for i in idx] + ["?"])
    pos = np.searchsorted(starts, np.asarray(samples), side="right") - 1
    return names[np.where(pos >= 0, pos, -1)]


def load_record(rec_id: int, db_dir: Path = config.DB_DIR, lead: str = config.LEAD) -> Record:
    path = str(Path(db_dir) / str(rec_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    ch = lead_index(rec.sig_name, lead)
    return Record(
        rec_id=rec_id,
        fs=float(rec.fs),
        signal=rec.p_signal[:, ch].astype(np.float64),
        ann_samples=np.asarray(ann.sample),
        ann_symbols=np.asarray(ann.symbol),
    )
