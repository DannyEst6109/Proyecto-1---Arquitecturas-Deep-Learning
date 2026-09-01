"""Ensamblado del experimento: datos -> variables -> particion -> tensores.

Concentra aqui todo lo que debe ocurrir en el MISMO orden para los tres
modelos, de modo que la comparacion A vs B vs C sea justa por construccion:
mismos datos, misma particion, mismo horizonte de prediccion y mismo escalado.

Horizonte de prediccion
-----------------------
Para cada transaccion, en el instante en que ocurre, se predice si ESA
transaccion es fraudulenta, usando unicamente su propia informacion y la de
las K-1 transacciones anteriores de la misma tarjeta. Es la decision que un
motor antifraude toma en linea: autorizar o bloquear, ahora.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import caracteristicas as ca
from . import particion as pa
from .generador import ConfigGenerador, generar
from .modelos import BaseEventos

RAIZ = pathlib.Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "datos"
DIR_ARTEFACTOS = RAIZ / "artefactos"
DIR_FIGURAS = RAIZ / "figuras"

LONGITUD_SECUENCIA = 20  # K: transacciones de historia que ve el modelo B


@dataclass
class Experimento:
    df: pd.DataFrame
    cortes: pa.CortesTemporales
    base: BaseEventos
    filas: dict[str, np.ndarray]
    tensores: dict[str, object]
    escalador_evento: pa.Escalador
    escalador_agregadas: pa.Escalador
    cols_agregadas: list[str]
    cols_categoricas: list[str]
    k: int
    cfg_generador: ConfigGenerador = field(default_factory=ConfigGenerador)

    def bloque(self, nombre: str) -> pd.DataFrame:
        """Filas del DataFrame correspondientes a una particion."""
        return self.df.iloc[self.filas[nombre]]

    def objetivo(self, nombre: str) -> np.ndarray:
        return self.df["es_fraude"].to_numpy()[self.filas[nombre]]

    def matriz_a(self, nombre: str) -> pd.DataFrame:
        """Entrada de la linea base A: agregados + categoricas actuales."""
        return self.bloque(nombre)[self.cols_agregadas + self.cols_categoricas]

    @property
    def dias_prueba(self) -> float:
        b = self.bloque("test")
        return (b["timestamp"].max() - b["timestamp"].min()).total_seconds() / 86400.0


def preparar(cfg: ConfigGenerador | None = None, k: int = LONGITUD_SECUENCIA,
             usar_cache: bool = True, verboso: bool = True) -> Experimento:
    """Construye el experimento completo de forma determinista."""
    cfg = cfg or ConfigGenerador()
    DIR_DATOS.mkdir(exist_ok=True)
    cache = DIR_DATOS / f"transacciones_s{cfg.semilla}_n{cfg.n_tarjetas}_d{cfg.dias}.parquet"

    # Columnas que el resto del pipeline da por hechas. Un cache escrito por una
    # version anterior del generador puede no tenerlas; en ese caso se regenera
    # en vez de fallar mas adelante con un KeyError dificil de rastrear.
    requeridas = {"id_transaccion", "id_tarjeta", "timestamp", "monto", "categoria",
                  "canal", "departamento", "es_fraude", "mecanismo", "id_episodio",
                  "rol", "mecanismo_nom", "categoria_nom"}

    df = None
    if usar_cache and cache.exists():
        candidato = pd.read_parquet(cache)
        faltantes = requeridas - set(candidato.columns)
        if faltantes:
            if verboso:
                print(f"Cache obsoleto (faltan {sorted(faltantes)}); se regenera.")
        else:
            if verboso:
                print(f"Leyendo cache {cache.name}")
            df = candidato

    if df is None:
        if verboso:
            print("Generando transacciones...")
        df = generar(cfg)
        df.to_parquet(cache, index=False)
        if verboso:
            print(f"Guardado {cache.name}")

    if verboso:
        print("Construyendo variables causales...")
    df = ca.construir(df)

    split, cortes = pa.particion_temporal(df)
    df["split"] = split.to_numpy()

    filas = {nombre: np.flatnonzero((df["split"] == nombre).to_numpy())
             for nombre in ("train", "val", "test")}
    df_train = df.iloc[filas["train"]]

    # Los escaladores se ajustan SOLO con entrenamiento.
    esc_evento = pa.Escalador(ca.NUMERICAS_EVENTO).ajustar(df_train)
    cols_agg = ca.columnas_agregadas()
    esc_agg = pa.Escalador(cols_agg).ajustar(df_train)

    num = esc_evento.transformar(df)
    cat = df[[c for c, _ in ca.CATEGORICAS_EVENTO]].to_numpy(dtype=np.int64)
    agg = esc_agg.transformar(df)
    y = df["es_fraude"].to_numpy(dtype=np.float32)

    idx = pa.indices_secuencia(df, k)
    base = BaseEventos(num, cat, idx, agg, y)
    tensores = {nombre: base.vista(f) for nombre, f in filas.items()}

    if verboso:
        print(f"Listo. {len(df):,} transacciones, K={k}, {len(cols_agg)} agregadas")

    return Experimento(df=df, cortes=cortes, base=base, filas=filas,
                       tensores=tensores, escalador_evento=esc_evento,
                       escalador_agregadas=esc_agg, cols_agregadas=cols_agg,
                       cols_categoricas=ca.columnas_categoricas_agregadas(),
                       k=k, cfg_generador=cfg)
