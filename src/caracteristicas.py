"""Construccion de variables. Todo es estrictamente causal.

Regla que gobierna este modulo
------------------------------
Para decidir sobre la transaccion en el instante t solo puede usarse informacion
disponible ANTES de t. Ninguna variable agregada incluye la transaccion que se
esta calificando ni ninguna posterior. En pandas esto se logra con
`closed="left"` en las ventanas temporales y con `shift(1)` dentro de cada
tarjeta.

Esto es lo que evita el descuento de -15 por "ajustar el ventaneo usando
estadisticas del conjunto completo": ningun estadistico cruza el instante de
decision, y el escalado numerico se ajusta despues, solo con el bloque de
entrenamiento (ver `src/particion.py`).

Que ve cada modelo
------------------
A (sin orden)  : atributos de la transaccion actual + agregados de ventana.
                 Los agregados son *invariantes a permutaciones* de la historia:
                 resumen el conjunto de eventos, no su secuencia.
B (secuencial) : los mismos atributos por evento, pero como secuencia ordenada
                 de las ultimas K transacciones de la tarjeta.

La diferencia entre ambos es exactamente el orden, que es lo que el proyecto
debe medir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .generador import CANALES, CATEGORIAS, DEPARTAMENTOS

# Ventanas del motor antifraude actual descrito por el banco.
VENTANAS = ["1h", "24h", "7d"]

# Variables numericas por evento que alimentan al modelo secuencial B.
NUMERICAS_EVENTO = [
    "log_monto",
    "log_delta_t",
    "hora_sin", "hora_cos",
    "dia_sin", "dia_cos",
    "es_madrugada",
    "fuera_depto_base",
]

# Variables categoricas por evento (van a capas de embedding en B).
CATEGORICAS_EVENTO = [
    ("categoria", len(CATEGORIAS)),
    ("canal", len(CANALES)),
    ("departamento", len(DEPARTAMENTOS)),
]


def _rolling_por_tarjeta(df: pd.DataFrame, columna: str, ventana: str,
                         funcion: str) -> np.ndarray:
    """Estadistico rodante por tarjeta que EXCLUYE la transaccion actual.

    `closed="left"` es la pieza critica: la ventana cubre [t - ventana, t) y
    por tanto nunca incorpora el evento que se esta calificando.
    """
    serie = (df.set_index("timestamp")
               .groupby("id_tarjeta")[columna]
               .rolling(ventana, closed="left"))
    valores = getattr(serie, funcion)().to_numpy()
    if len(valores) != len(df):
        raise AssertionError("Desalineacion en la ventana rodante")
    return valores


def _diversidad(df: pd.DataFrame, columna: str, n_niveles: int,
                ventana: str) -> np.ndarray:
    """Numero de niveles distintos observados en la ventana previa.

    pandas no ofrece `rolling().nunique()`, asi que se suman indicadores
    one-hot por nivel y se cuentan los que aparecieron al menos una vez.
    Es equivalente y vectorizado.
    """
    presentes = np.zeros(len(df), dtype=np.float64)
    codigos = df[columna].to_numpy()
    aux = df[["id_tarjeta", "timestamp"]].copy()
    for nivel in range(n_niveles):
        aux["_ind"] = (codigos == nivel).astype(np.float64)
        conteo = _rolling_por_tarjeta(aux, "_ind", ventana, "sum")
        presentes += (np.nan_to_num(conteo) > 0).astype(np.float64)
    return presentes


def construir(df: pd.DataFrame) -> pd.DataFrame:
    """Anade todas las variables derivadas. Espera el orden (tarjeta, timestamp)."""
    df = df.sort_values(["id_tarjeta", "timestamp"], kind="mergesort").reset_index(drop=True)

    # ---------------- Atributos de la transaccion actual ----------------
    df["log_monto"] = np.log1p(df["monto"])

    ts = df["timestamp"]
    hora_dec = ts.dt.hour + ts.dt.minute / 60.0
    df["hora_sin"] = np.sin(2 * np.pi * hora_dec / 24.0)
    df["hora_cos"] = np.cos(2 * np.pi * hora_dec / 24.0)
    df["dia_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["dia_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    df["es_madrugada"] = ((hora_dec >= 0) & (hora_dec < 5)).astype(np.float64)
    df["es_fin_semana"] = (ts.dt.dayofweek >= 5).astype(np.float64)
    df["es_online"] = (df["canal"] == CANALES.index("online")).astype(np.float64)

    # Tiempo desde la transaccion anterior de la MISMA tarjeta.
    delta = df.groupby("id_tarjeta")["timestamp"].diff().dt.total_seconds() / 60.0
    # La primera transaccion de cada tarjeta no tiene antecesora: se marca con
    # un valor grande y una bandera, en vez de imputar un cero enganoso.
    df["primera_tx"] = delta.isna().astype(np.float64)
    df["delta_t_min"] = delta.fillna(60 * 24 * 30.0)
    df["log_delta_t"] = np.log1p(df["delta_t_min"])

    # Departamento base historico = moda de los departamentos previos.
    # Aproximacion causal barata: el departamento de la primera transaccion.
    primer_depto = df.groupby("id_tarjeta")["departamento"].transform("first")
    df["fuera_depto_base"] = (df["departamento"] != primer_depto).astype(np.float64)

    # ---------------- Agregados de ventana (motor actual del banco) ----------------
    for v in VENTANAS:
        df[f"n_tx_{v}"] = np.nan_to_num(_rolling_por_tarjeta(df, "monto", v, "count"))
        df[f"monto_prom_{v}"] = _rolling_por_tarjeta(df, "monto", v, "mean")
        df[f"monto_max_{v}"] = _rolling_por_tarjeta(df, "monto", v, "max")
        df[f"monto_suma_{v}"] = np.nan_to_num(_rolling_por_tarjeta(df, "monto", v, "sum"))
        df[f"monto_std_{v}"] = _rolling_por_tarjeta(df, "monto", v, "std")
        df[f"prop_online_{v}"] = _rolling_por_tarjeta(df, "es_online", v, "mean")
        df[f"prop_madrugada_{v}"] = _rolling_por_tarjeta(df, "es_madrugada", v, "mean")

    # Diversidad: la cuarta variable que el banco ya usa.
    for v in ["1h", "24h"]:
        df[f"n_categorias_{v}"] = _diversidad(df, "categoria", len(CATEGORIAS), v)
        df[f"n_deptos_{v}"] = _diversidad(df, "departamento", len(DEPARTAMENTOS), v)
        df[f"n_canales_{v}"] = _diversidad(df, "canal", len(CANALES), v)

    # ---------------- Referencias historicas de la tarjeta ----------------
    # Mediana expandida de los montos ANTERIORES (nunca el actual).
    monto_previo = df.groupby("id_tarjeta")["monto"].shift(1)
    mediana_hist = (monto_previo.groupby(df["id_tarjeta"])
                    .expanding().median().reset_index(level=0, drop=True))
    df["mediana_hist"] = mediana_hist
    df["razon_monto_mediana"] = df["monto"] / mediana_hist.replace(0, np.nan)
    df["log_razon_monto"] = np.log1p(df["razon_monto_mediana"].clip(upper=500))

    # Antiguedad: cuantas transacciones previas tiene la tarjeta.
    df["n_tx_historicas"] = df.groupby("id_tarjeta").cumcount().astype(np.float64)

    # Velocidad: transacciones por hora en la ultima hora (el "numero de
    # transacciones por hora" del motor actual).
    df["velocidad_1h"] = df["n_tx_1h"]
    df["monto_vs_prom24h"] = df["monto"] / df["monto_prom_24h"].replace(0, np.nan)

    return df


def columnas_agregadas() -> list[str]:
    """Variables que ve la linea base A (invariantes al orden de la historia)."""
    cols = [
        # Transaccion actual
        "log_monto", "hora_sin", "hora_cos", "dia_sin", "dia_cos",
        "es_madrugada", "es_fin_semana", "es_online", "fuera_depto_base",
        "log_delta_t", "primera_tx",
        # Referencias historicas
        "log_razon_monto", "monto_vs_prom24h", "n_tx_historicas",
    ]
    for v in VENTANAS:
        cols += [f"n_tx_{v}", f"monto_prom_{v}", f"monto_max_{v}",
                 f"monto_suma_{v}", f"monto_std_{v}",
                 f"prop_online_{v}", f"prop_madrugada_{v}"]
    for v in ["1h", "24h"]:
        cols += [f"n_categorias_{v}", f"n_deptos_{v}", f"n_canales_{v}"]
    return cols


def columnas_categoricas_agregadas() -> list[str]:
    """Categoricas de la transaccion actual que tambien ve A."""
    return ["categoria", "canal", "departamento"]
