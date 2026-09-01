"""Particion temporal, escalado sin fuga y construccion de secuencias.

Protocolo temporal
------------------
La particion es por FECHA GLOBAL, no aleatoria y no por tarjeta:

    entrenamiento : primeras 70 % de las fechas
    validacion    : siguientes 15 %
    prueba        : ultimas 15 %

Cada transaccion se asigna a un bloque segun SU PROPIA fecha. El conjunto de
prueba se mira una sola vez, al final.

Sobre la historia que cruza el corte
------------------------------------
Una transaccion de prueba puede tener, en su ventana de historia, transacciones
ocurridas antes del corte. Esto NO es fuga: en produccion el historial del
cliente existe y esta disponible en el instante de la decision. Lo prohibido es
lo contrario -- usar informacion posterior al instante de decision -- y eso no
ocurre en ningun punto de este pipeline. Se deja constancia explicita porque es
una decision de diseno que el comite podria cuestionar.

Escalado
--------
Medias, desviaciones y medianas de imputacion se estiman UNICAMENTE con el
bloque de entrenamiento y se aplican sin recalcular a validacion y prueba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .caracteristicas import CATEGORICAS_EVENTO, NUMERICAS_EVENTO


@dataclass
class CortesTemporales:
    fin_train: pd.Timestamp
    fin_val: pd.Timestamp

    def __str__(self) -> str:
        return (f"train < {self.fin_train:%Y-%m-%d %H:%M} <= val < "
                f"{self.fin_val:%Y-%m-%d %H:%M} <= test")


def particion_temporal(df: pd.DataFrame, prop_train: float = 0.70,
                       prop_val: float = 0.15) -> tuple[pd.Series, CortesTemporales]:
    """Asigna cada transaccion a train/val/test segun su propia fecha."""
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    span = t_max - t_min

    fin_train = t_min + span * prop_train
    fin_val = t_min + span * (prop_train + prop_val)

    split = pd.Series("test", index=df.index, dtype=object)
    split[df["timestamp"] < fin_train] = "train"
    split[(df["timestamp"] >= fin_train) & (df["timestamp"] < fin_val)] = "val"

    return split, CortesTemporales(fin_train, fin_val)


def resumen_particion(df: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Tabla de control del protocolo temporal, para el informe."""
    tabla = df.assign(split=split).groupby("split", observed=True).agg(
        transacciones=("id_transaccion", "size"),
        fraudes=("es_fraude", "sum"),
        desde=("timestamp", "min"),
        hasta=("timestamp", "max"),
        episodios=("id_episodio", lambda s: s[s >= 0].nunique()),
    )
    tabla["tasa_fraude_%"] = 100 * tabla["fraudes"] / tabla["transacciones"]
    return tabla.reindex(["train", "val", "test"])


# --------------------------------------------------------------------------
# Escalado ajustado solo con entrenamiento
# --------------------------------------------------------------------------

class Escalador:
    """Estandarizacion e imputacion estimadas exclusivamente en entrenamiento."""

    def __init__(self, columnas: list[str]) -> None:
        self.columnas = columnas
        self.medianas: np.ndarray | None = None
        self.medias: np.ndarray | None = None
        self.escalas: np.ndarray | None = None

    def ajustar(self, df_train: pd.DataFrame) -> "Escalador":
        bloque = df_train[self.columnas].to_numpy(dtype=np.float64)
        self.medianas = np.nanmedian(bloque, axis=0)
        # Columnas enteramente NaN en train: se imputan con 0.
        self.medianas = np.nan_to_num(self.medianas)
        lleno = np.where(np.isnan(bloque), self.medianas, bloque)
        self.medias = lleno.mean(axis=0)
        escalas = lleno.std(axis=0)
        # Evita division por cero en variables constantes.
        self.escalas = np.where(escalas < 1e-8, 1.0, escalas)
        return self

    def transformar(self, df: pd.DataFrame) -> np.ndarray:
        if self.medias is None:
            raise RuntimeError("Escalador sin ajustar")
        bloque = df[self.columnas].to_numpy(dtype=np.float64)
        bloque = np.where(np.isnan(bloque), self.medianas, bloque)
        bloque = np.clip(bloque, -1e12, 1e12)
        return ((bloque - self.medias) / self.escalas).astype(np.float32)

    def estado(self) -> dict:
        """Parametros serializables (entregable `artefactos/`)."""
        return {"columnas": self.columnas,
                "medianas": self.medianas,
                "medias": self.medias,
                "escalas": self.escalas}

    @classmethod
    def desde_estado(cls, estado: dict) -> "Escalador":
        obj = cls(estado["columnas"])
        obj.medianas = estado["medianas"]
        obj.medias = estado["medias"]
        obj.escalas = estado["escalas"]
        return obj


# --------------------------------------------------------------------------
# Secuencias
# --------------------------------------------------------------------------

def indices_secuencia(df: pd.DataFrame, k: int) -> np.ndarray:
    """Matriz (N, k) con los indices de las ultimas k transacciones de la tarjeta.

    La columna k-1 es SIEMPRE la transaccion que se esta calificando; las
    anteriores son su historia, en orden cronologico. Las posiciones sin
    historia suficiente se rellenan con -1 (padding a la izquierda).

    Requiere que `df` este ordenado por (id_tarjeta, timestamp), que es como lo
    entrega el generador: las filas de cada tarjeta quedan contiguas, lo que
    permite construir la matriz con aritmetica de indices en vez de bucles.
    """
    n = len(df)
    posicion = df.groupby("id_tarjeta").cumcount().to_numpy()
    global_idx = np.arange(n)
    inicio_tarjeta = global_idx - posicion  # primera fila de cada tarjeta

    idx = np.full((n, k), -1, dtype=np.int64)
    for j in range(k):
        desplazamiento = k - 1 - j          # j = k-1 -> la transaccion actual
        candidato = global_idx - desplazamiento
        valido = candidato >= inicio_tarjeta
        idx[valido, j] = candidato[valido]
    return idx


def matriz_eventos(df: pd.DataFrame, escalador: Escalador
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Atributos por evento que consume el modelo secuencial.

    Devuelve (numericas escaladas, categoricas como indices enteros).
    Son los MISMOS atributos que resume la linea base A; lo unico que cambia
    es que aqui conservan su posicion en la secuencia.
    """
    num = escalador.transformar(df)
    cat = df[[c for c, _ in CATEGORICAS_EVENTO]].to_numpy(dtype=np.int64)
    return num, cat


def escalador_eventos(df_train: pd.DataFrame) -> Escalador:
    return Escalador(NUMERICAS_EVENTO).ajustar(df_train)
