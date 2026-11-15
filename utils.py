"""
utils.py - Funciones compartidas por el notebook y la app de Streamlit.

Contiene:
  * Conexión / carga del dataset xAPI-Edu-Data (Google Drive con respaldo local).
  * Limpieza y estandarización de nombres y categorías.
  * Definición de la red neuronal (MLP) en PyTorch.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Conexión a la base de datos (archivo CSV en Google Drive)
# ---------------------------------------------------------------------------
DRIVE_FILE_ID = "1cIeVbKI3NpUvt3GHW7yuZibyADUCq3uw"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
LOCAL_CSV = Path(__file__).resolve().parent / "data" / "xAPI-Edu-Data.csv"


def cargar_datos(url: str = DRIVE_URL, ruta_local: Path = LOCAL_CSV, verbose: bool = True):
    """Lee el CSV desde Google Drive; si no hay conexión usa la copia local.

    Devuelve (DataFrame, origen) donde origen es 'Google Drive' o 'Copia local'.
    El archivo de Drive debe estar compartido como "Cualquier persona con el enlace".
    """
    try:
        import requests

        r = requests.get(url, timeout=20)
        r.raise_for_status()
        texto = r.content.decode("utf-8")
        if not texto.lstrip().startswith("gender"):  # Drive devolvió HTML (permisos)
            raise ValueError("La respuesta de Drive no es el CSV esperado")
        df = pd.read_csv(io.StringIO(texto))
        origen = "Google Drive"
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[aviso] No se pudo leer desde Google Drive ({type(e).__name__}: {e}).")
            print(f"[aviso] Usando copia local: {ruta_local}")
        df = pd.read_csv(ruta_local)
        origen = "Copia local"
    if verbose:
        print(f"Datos cargados desde: {origen} -> {df.shape[0]} filas x {df.shape[1]} columnas")
    return df, origen


# ---------------------------------------------------------------------------
# 2. Limpieza / estandarización
# ---------------------------------------------------------------------------
RENOMBRAR = {
    "NationalITy": "Nationality",
    "PlaceofBirth": "PlaceOfBirth",
    "VisITedResources": "VisitedResources",
    "ParentschoolSatisfaction": "ParentSchoolSatisfaction",
}

NORMALIZAR_PAISES = {
    "KW": "Kuwait", "KuwaIT": "Kuwait", "lebanon": "Lebanon", "venzuela": "Venezuela",
    "Tunis": "Tunisia", "Lybia": "Libya", "SaudiArabia": "SaudiArabia",
}

NUMERICAS = ["raisedhands", "VisitedResources", "AnnouncementsView", "Discussion", "Grade"]
BINARIAS = {
    "gender": {"M": 1, "F": 0},
    "Semester": {"S": 1, "F": 0},
    "Relation": {"Mum": 1, "Father": 0},
    "ParentAnsweringSurvey": {"Yes": 1, "No": 0},
    "ParentSchoolSatisfaction": {"Good": 1, "Bad": 0},
    "StudentAbsenceDays": {"Above-7": 1, "Under-7": 0},
}
CATEGORICAS = ["Nationality", "StageID", "SectionID", "Topic"]

CLASES = ["L", "M", "H"]  # orden ordinal: bajo, medio, alto
NOMBRE_CLASE = {"L": "Bajo (0-69)", "M": "Medio (70-89)", "H": "Alto (90-100)"}


def limpiar(df: pd.DataFrame, min_freq_pais: int = 10, paises_validos=None) -> pd.DataFrame:
    """Renombra columnas, normaliza países, crea 'Grade' numérico y agrupa países raros.

    - Entrenamiento: nacionalidades con menos de `min_freq_pais` registros pasan a 'Other'.
    - Predicción: se pasa `paises_validos` (categorías aprendidas) y lo que no esté ahí pasa a 'Other'.
    """
    d = df.rename(columns=RENOMBRAR).copy()
    for c in ["Nationality", "PlaceOfBirth"]:
        if c in d:
            d[c] = d[c].replace(NORMALIZAR_PAISES)
    d["Grade"] = d["GradeID"].astype(str).str.replace("G-", "", regex=False).astype(int)
    if paises_validos is None:
        frec = d["Nationality"].value_counts()
        raros = frec[frec < min_freq_pais].index
        d["Nationality"] = d["Nationality"].where(~d["Nationality"].isin(raros), "Other")
    else:
        d["Nationality"] = d["Nationality"].where(d["Nationality"].isin(paises_validos), "Other")
    return d


def preparar_X(d: pd.DataFrame) -> pd.DataFrame:
    """Selecciona columnas de entrada y codifica las binarias a 0/1.
    (PlaceOfBirth y GradeID se descartan: redundantes con Nationality y Grade.)"""
    X = d[NUMERICAS + list(BINARIAS) + CATEGORICAS].copy()
    for c, m in BINARIAS.items():
        X[c] = X[c].map(m).astype(float)
    return X


# ---------------------------------------------------------------------------
# 3. Red neuronal: Perceptrón Multicapa (MLP) profundo
# ---------------------------------------------------------------------------
class MLPRendimiento(nn.Module):
    """MLP con bloques Linear -> BatchNorm -> ReLU -> Dropout."""

    def __init__(self, n_entradas: int, capas=(64, 32, 16), n_clases: int = 3, dropout: float = 0.3):
        super().__init__()
        bloques, prev = [], n_entradas
        for h in capas:
            bloques += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        bloques.append(nn.Linear(prev, n_clases))
        self.red = nn.Sequential(*bloques)

    def forward(self, x):
        return self.red(x)
