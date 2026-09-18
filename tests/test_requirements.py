"""Verificación de la documentación de la Fase 7.

Una matriz de trazabilidad que nadie comprueba se desactualiza sola: alcanza con renombrar una
prueba para que la trazabilidad mienta sin que nadie se entere. Estas pruebas la mantienen honesta.
"""

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "docs"

V_SE_MINIMA = 0.90  # REQ-003


def texto(nombre: str) -> str:
    return (DOCS / nombre).read_text(encoding="utf-8")


def ids(patron: str, contenido: str) -> set[str]:
    return set(re.findall(patron, contenido))


def test_v_sensitivity_meets_the_declared_threshold():
    """REQ-003: la sensibilidad de V del modelo servido, en la validación inter-paciente de DS1."""
    cv = json.loads((RAIZ / "reports" / "baseline_v5_cv.json").read_text())["cv"]
    assert cv["V_Se"] >= V_SE_MINIMA, (
        f"La sensibilidad de V bajó a {cv['V_Se']:.3f}, por debajo del umbral declarado "
        f"{V_SE_MINIMA} (RISK-01). Si el cambio es deliberado, actualizá REQ-003."
    )


def test_every_requirement_appears_in_the_traceability_matrix():
    requisitos = ids(r"REQ-\d{3}", texto("requirements.md"))
    trazados = ids(r"REQ-\d{3}", texto("traceability.md"))
    assert requisitos, "No se encontraron requisitos en requirements.md"
    assert requisitos == trazados, (
        f"Sin trazar: {sorted(requisitos - trazados)}; "
        f"trazados pero inexistentes: {sorted(trazados - requisitos)}"
    )


def test_every_risk_referenced_is_defined():
    definidos = ids(r"RISK-\d{2}", texto("risk_analysis.md"))
    assert definidos, "No se encontraron riesgos en risk_analysis.md"
    for doc in ("requirements.md", "traceability.md", "intended_use.md", "model_card.md"):
        referidos = ids(r"RISK-\d{2}", texto(doc))
        assert referidos <= definidos, (
            f"{doc} cita riesgos inexistentes: {sorted(referidos - definidos)}"
        )


@pytest.mark.parametrize(
    "referencia", sorted(ids(r"`([\w/\.]+\.py)::(\w+)`", texto("traceability.md")) or {("", "")})
)
def test_traceability_references_point_to_real_code(referencia):
    """Cada `archivo.py::nombre` de la matriz tiene que existir de verdad."""
    archivo, nombre = referencia
    ruta = RAIZ / archivo
    assert ruta.exists(), f"La matriz cita {archivo}, que no existe"
    fuente = ruta.read_text(encoding="utf-8")
    definicion = rf"^(async def|def|class) {re.escape(nombre)}\b"
    assert re.search(definicion, fuente, re.MULTILINE), (
        f"La matriz cita {archivo}::{nombre}, que no está definido en el archivo"
    )


def test_traceability_files_exist():
    """Los módulos citados sin `::` (archivos enteros) también tienen que existir."""
    citados = set(re.findall(r"`([\w/\.]+\.(?:py|yml|toml))`", texto("traceability.md")))
    faltan = [c for c in citados if not (RAIZ / c).exists()]
    assert not faltan, f"La matriz cita archivos inexistentes: {sorted(faltan)}"


def test_soup_lists_every_dependency():
    """IEC 62304 pide inventariar el software de terceros. Si se agrega una dependencia y no se
    documenta, esta prueba lo detecta.

    Las *versiones* del SOUP son las verificadas al publicar y no se comprueban acá: hacerlo
    pondría el CI en rojo cada vez que cualquier paquete publique una versión nueva, por algo
    ajeno a este repositorio.
    """
    import tomllib

    pyproject = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declaradas = list(pyproject["dependencies"])
    for extra in ("api", "app", "dl"):
        declaradas += pyproject["optional-dependencies"][extra]
    nombres = {re.split(r"[><=!\[]", d)[0].strip().lower() for d in declaradas}

    soup = texto("soup.md").lower()
    faltan = sorted(n for n in nombres if f"`{n}`" not in soup)
    assert not faltan, f"Dependencias sin documentar en soup.md: {faltan}"
