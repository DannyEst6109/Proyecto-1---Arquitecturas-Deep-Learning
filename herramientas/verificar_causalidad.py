"""Verificacion independiente de que NINGUNA variable usa informacion futura.

Afirmar que no hay fuga es barato. Este script lo demuestra de dos formas
distintas, y falla con codigo de salida 1 si alguna no pasa.

    python herramientas/verificar_causalidad.py

Prueba 1 - Recalculo por fuerza bruta
    Para un subconjunto pequeno se recalculan las variables fila por fila con
    un bucle ingenuo que solo puede mirar hacia atras, y se comparan contra las
    que produce el pipeline vectorizado. Si difieren, el pipeline mira adelante.

Prueba 2 - Truncar el futuro
    La prueba decisiva. Se recorta el dataset a su primer 60 % temporal y se
    reconstruyen las variables desde cero. Si alguna variable de una fila
    cambia al haber eliminado las transacciones POSTERIORES, entonces esa
    variable dependia del futuro. Ninguna debe cambiar.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import caracteristicas as ca
from src.generador import ConfigGenerador, generar

COLUMNAS_CRUDAS = ["id_transaccion", "id_tarjeta", "timestamp", "monto",
                   "categoria", "canal", "departamento", "es_fraude",
                   "mecanismo", "id_episodio", "rol"]

# Variables representativas de cada familia construida en caracteristicas.py.
A_VERIFICAR = ["n_tx_1h", "n_tx_24h", "monto_prom_24h", "monto_max_24h",
               "monto_std_24h", "monto_suma_7d", "prop_online_24h",
               "n_categorias_24h", "n_deptos_1h", "n_canales_1h",
               "mediana_hist", "log_razon_monto", "monto_vs_prom24h",
               "n_tx_historicas", "fuera_depto_base", "log_delta_t"]


def prueba_fuerza_bruta(df: pd.DataFrame, n_tarjetas: int = 12) -> int:
    """Recalcula variables con un bucle que solo mira el pasado."""
    sub = df[df["id_tarjeta"] < n_tarjetas]
    fallos = 0

    for _, g in sub.groupby("id_tarjeta"):
        g = g.sort_values("timestamp")
        montos = g["monto"].to_numpy()
        deptos = g["departamento"].to_numpy()
        ts = g["timestamp"].to_numpy()

        for i in range(len(g)):
            previos = ts[:i]

            # n_tx_24h: eventos en [t-24h, t), sin incluir el actual.
            esperado = int((previos >= ts[i] - np.timedelta64(24, "h")).sum())
            if esperado != g["n_tx_24h"].iloc[i]:
                fallos += 1

            # mediana_hist: mediana de los montos estrictamente anteriores.
            esperado_med = np.median(montos[:i]) if i else np.nan
            obtenido = g["mediana_hist"].iloc[i]
            if not (np.isnan(esperado_med) and pd.isna(obtenido)):
                if not np.isclose(esperado_med, obtenido, equal_nan=True):
                    fallos += 1

            # fuera_depto_base: comparado con el departamento de la 1a tx.
            if float(deptos[i] != deptos[0]) != g["fuera_depto_base"].iloc[i]:
                fallos += 1

    print(f"  filas verificadas    : {len(sub):,}")
    print(f"  discrepancias        : {fallos}")
    return fallos


def prueba_truncar_futuro(df: pd.DataFrame, proporcion: float = 0.60) -> int:
    """Reconstruye las variables tras eliminar el futuro. Nada debe cambiar."""
    corte = df["timestamp"].quantile(proporcion)
    crudo = df.loc[df["timestamp"] <= corte, COLUMNAS_CRUDAS]
    recortado = ca.construir(crudo)

    comun = (df[df["timestamp"] <= corte]
             .merge(recortado, on="id_transaccion", suffixes=("_full", "_rec")))

    fallos = 0
    for col in A_VERIFICAR:
        a = comun[f"{col}_full"].to_numpy(dtype=float)
        b = comun[f"{col}_rec"].to_numpy(dtype=float)
        distintas = int((~np.isclose(a, b, equal_nan=True)).sum())
        estado = "ok" if distintas == 0 else f"FALLA ({distintas} filas)"
        print(f"  {col:22s} {estado}")
        fallos += distintas

    print(f"  filas comparadas     : {len(comun):,}")
    return fallos


def main() -> int:
    print("Generando datos de verificacion...")
    df = ca.construir(generar(ConfigGenerador(n_tarjetas=120, dias=120)))
    print(f"{len(df):,} transacciones\n")

    print("PRUEBA 1 - Recalculo por fuerza bruta (solo mirando el pasado)")
    f1 = prueba_fuerza_bruta(df)

    print("\nPRUEBA 2 - Truncar el futuro y reconstruir")
    f2 = prueba_truncar_futuro(df)

    print()
    if f1 == 0 and f2 == 0:
        print("RESULTADO: ninguna variable depende de informacion futura.")
        return 0
    print(f"RESULTADO: FUGA DETECTADA (prueba 1: {f1}, prueba 2: {f2})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
