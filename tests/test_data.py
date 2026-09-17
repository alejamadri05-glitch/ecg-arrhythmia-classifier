"""Descarga de PhysioNet: reintentos ante errores transitorios y verificación de integridad."""

import hashlib
import urllib.error
import zipfile

import pytest
from wfdb.io._url import NetFileError

from ecg import data


def test_reintenta_tras_errores_transitorios(monkeypatch, tmp_path):
    llamadas, esperas = [], []

    def dl_falla_dos_veces(db, dl_dir):
        llamadas.append(db)
        if len(llamadas) <= 2:
            raise NetFileError("502 Error: Bad Gateway for url: .../mitdb/1.0.0/221.atr")

    monkeypatch.setattr(data.wfdb, "dl_database", dl_falla_dos_veces)
    data.dl_database_with_retry("mitdb", tmp_path, attempts=5, wait_s=5, sleep=esperas.append)
    assert len(llamadas) == 3
    assert esperas == [5, 10]  # espera exponencial


def test_se_rinde_tras_agotar_los_intentos(monkeypatch, tmp_path):
    esperas = []

    def dl_siempre_falla(db, dl_dir):
        raise NetFileError("502 Error: Bad Gateway")

    monkeypatch.setattr(data.wfdb, "dl_database", dl_siempre_falla)
    with pytest.raises(NetFileError):
        data.dl_database_with_retry("mitdb", tmp_path, attempts=3, wait_s=1, sleep=esperas.append)
    assert esperas == [1, 2]  # no espera después del último intento


def test_no_reintenta_errores_de_programacion(monkeypatch, tmp_path):
    esperas = []

    def dl_con_bug(db, dl_dir):
        raise TypeError("argumento inválido")

    monkeypatch.setattr(data.wfdb, "dl_database", dl_con_bug)
    with pytest.raises(TypeError):
        data.dl_database_with_retry("mitdb", tmp_path, sleep=esperas.append)
    assert esperas == []


# --- descarga de MIT-BIH desde el zip oficial ---


ROOT = "mit-bih-arrhythmia-database-1.0.0"


def _zip_falso(path, archivos, sums_override=None, extra=None):
    """Zip con la estructura del oficial: carpeta raíz, archivos y SHA256SUMS.txt."""
    sums = {f: hashlib.sha256(c).hexdigest() for f, c in archivos.items()}
    sums.update(sums_override or {})
    with zipfile.ZipFile(path, "w") as zf:
        for f, c in archivos.items():
            zf.writestr(f"{ROOT}/{f}", c)
        for nombre, c in (extra or {}).items():
            zf.writestr(nombre, c)
        zf.writestr(f"{ROOT}/SHA256SUMS.txt", "".join(f"{h} {f}\n" for f, h in sums.items()))
    return path


def _registro(rec="100"):
    return {f"{rec}.dat": b"senal", f"{rec}.hea": b"cabecera", f"{rec}.atr": b"anotaciones"}


def test_extrae_solo_registros_verificados(tmp_path):
    archivos = {"RECORDS": b"100\n", **_registro(), "100.xws": b"no se usa"}
    z = _zip_falso(tmp_path / "m.zip", archivos, extra={f"{ROOT}/x_mitdb/x_100.dat": b"subcarpeta"})
    out = tmp_path / "db"
    out.mkdir()
    escritos = data.extract_verified(z, out)
    assert escritos == ["100.atr", "100.dat", "100.hea", "RECORDS"]
    assert (out / "100.dat").read_bytes() == b"senal"
    assert not (out / "100.xws").exists()
    assert not (out / "x_mitdb").exists()


def test_zip_adulterado_no_escribe_nada(tmp_path):
    archivos = {"RECORDS": b"100\n", **_registro()}
    z = _zip_falso(tmp_path / "m.zip", archivos, sums_override={"100.dat": "0" * 64})
    out = tmp_path / "db"
    out.mkdir()
    with pytest.raises(ValueError, match="100.dat"):
        data.extract_verified(z, out)
    assert list(out.iterdir()) == []  # ni siquiera los archivos que sí estaban bien


def test_falta_un_archivo_de_registro(tmp_path):
    archivos = {"RECORDS": b"100\n101\n", **_registro("100"), "101.dat": b"x", "101.hea": b"y"}
    z = _zip_falso(tmp_path / "m.zip", archivos)
    out = tmp_path / "db"
    out.mkdir()
    with pytest.raises(ValueError, match="101.atr"):
        data.extract_verified(z, out)
    assert list(out.iterdir()) == []


def test_sin_sha256sums_no_se_acepta(tmp_path):
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(f"{ROOT}/100.dat", b"senal")
    with pytest.raises(ValueError, match="SHA256SUMS"):
        data.extract_verified(z, tmp_path)


def test_ignora_rutas_que_escapan_de_la_carpeta(tmp_path):
    archivos = {"RECORDS": b"100\n", **_registro()}
    malicioso = f"{ROOT}/../../evil.dat"
    z = _zip_falso(tmp_path / "m.zip", archivos, extra={malicioso: b"x"})
    out = tmp_path / "db"
    out.mkdir()
    data.extract_verified(z, out)
    assert not (tmp_path / "evil.dat").exists()
    assert not list(tmp_path.parent.glob("evil.dat"))


def test_download_reintenta_y_limpia_el_zip(monkeypatch, tmp_path):
    archivos = {"RECORDS": b"234\n", **_registro("234")}
    fuente = _zip_falso(tmp_path / "fuente.zip", archivos)
    intentos, esperas = [], []

    def fetch_con_502(url, dest):
        intentos.append(url)
        if len(intentos) == 1:
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", None, None)
        dest.write_bytes(fuente.read_bytes())

    monkeypatch.setattr(data, "_fetch", fetch_con_502)
    db = tmp_path / "mitdb"
    data.download(db, url="https://ejemplo/mitdb.zip", sleep=esperas.append)
    assert len(intentos) == 2 and esperas == [5.0]
    assert (db / "234.atr").read_bytes() == b"anotaciones"
    assert not (db / "mitdb.zip.part").exists()


def test_download_no_hace_nada_si_ya_esta(monkeypatch, tmp_path):
    (tmp_path / "234.atr").write_bytes(b"ya descargado")
    monkeypatch.setattr(data, "_fetch", lambda url, dest: pytest.fail("no debía descargar"))
    data.download(tmp_path)
