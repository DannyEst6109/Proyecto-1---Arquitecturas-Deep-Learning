"""Pruebas de falsificacion: intentos de refutar nuestra propia conclusion.

Una mejora de AUC-PR no demuestra que el modelo haya usado el orden. Podria
venir de que el modelo secuencial tiene mas parametros, de que ve los eventos
en bruto en vez de resumidos, o del azar. Estas pruebas existen para tratar de
tumbar la afirmacion "el orden aporta".

Prueba 1 (obligatoria) - Permutacion controlada
    Se baraja el orden de los eventos dentro de cada ventana sin cambiar los
    eventos ni sus valores. Las variables agregadas son invariantes a
    permutaciones, asi que siguen siendo identicas. Si el desempeno NO cae, el
    modelo no estaba usando el orden y la afirmacion queda refutada.
    Se repite con varias semillas: una sola permutacion podria ser afortunada.

Prueba 2 (elegida por el equipo) - Desempeno por mecanismo de fraude
    Se eligio esta y no otra porque es la unica de la lista que puede REFUTAR
    la conclusion en lugar de solo matizarla. El generador produce mecanismos
    con dependencia del orden conocida y distinta:

        escalada_prueba   orden fuerte   -> aqui B DEBE ganarle a A
        rafaga_geografica orden parcial  -> ventaja moderada
        vaciado_subito    sin orden      -> aqui B NO deberia ganar
        toma_gradual      orden difuso   -> caso dificil declarado

    Si B superara a A por igual en los cuatro, la ventaja vendria de la mayor
    capacidad del modelo y no del orden, y nuestra conclusion seria falsa.
    El patron esperado es una prediccion arriesgada: puede fallar.

Prueba 3 (apoyo) - Recorte de historia
    Cuanta memoria hace falta. No refuta por si sola, pero calibra: si con una
    sola transaccion se alcanza el mismo desempeno, la secuencia sobra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluacion import auc_pr
from .modelos import predecir


def permutacion_controlada(modelo, datos, y: np.ndarray,
                           n_repeticiones: int = 5,
                           semilla_base: int = 101,
                           fijar_objetivo: bool = True) -> pd.DataFrame:
    """Evalua el modelo con el orden barajado, varias veces.

    `fijar_objetivo=True` (el valor por defecto y el que se reporta) baraja solo
    la HISTORIA y mantiene la transaccion calificada en la ultima posicion. Es
    la version controlada: sin ella, la permutacion tambien le quita al modelo
    el acceso a los atributos del evento que debe puntuar, y la caida medida
    mezcla dos efectos distintos.

    Devuelve una fila por repeticion mas la referencia con el orden original.
    """
    referencia = auc_pr(y, predecir(modelo, datos))
    filas = [{"corrida": "orden original", "auc_pr": referencia,
              "caida_relativa": 0.0}]
    for i in range(n_repeticiones):
        p = predecir(modelo, datos, permutar_historia=True,
                     semilla=semilla_base + i, fijar_objetivo=fijar_objetivo)
        valor = auc_pr(y, p)
        filas.append({
            "corrida": f"barajado {i + 1}",
            "auc_pr": valor,
            "caida_relativa": 1.0 - valor / referencia if referencia else np.nan,
        })
    return pd.DataFrame(filas)


def comparar_variantes_permutacion(modelo, datos, y: np.ndarray,
                                   n_repeticiones: int = 3) -> pd.DataFrame:
    """Contrasta la permutacion controlada con la version sin control.

    Sirve para mostrar cuanto de la caida se debe realmente al orden y cuanto
    seria un artefacto de mover tambien la transaccion calificada.
    """
    filas = []
    for etiqueta, fijar in [("solo la historia (controlada)", True),
                            ("toda la ventana (sin control)", False)]:
        tabla = permutacion_controlada(modelo, datos, y, n_repeticiones,
                                       fijar_objetivo=fijar)
        res = resumen_permutacion(tabla)
        filas.append({
            "variante": etiqueta,
            "auc_pr_original": res["auc_pr_original"],
            "auc_pr_barajado": res["auc_pr_barajado_medio"],
            "caida_relativa": res["caida_relativa_media"],
        })
    return pd.DataFrame(filas)


def resumen_permutacion(tabla: pd.DataFrame) -> dict:
    """Resume la prueba 1 en las cifras que van al informe."""
    barajados = tabla[tabla["corrida"] != "orden original"]
    referencia = float(tabla.loc[tabla["corrida"] == "orden original", "auc_pr"].iloc[0])
    return {
        "auc_pr_original": referencia,
        "auc_pr_barajado_medio": float(barajados["auc_pr"].mean()),
        "auc_pr_barajado_min": float(barajados["auc_pr"].min()),
        "auc_pr_barajado_max": float(barajados["auc_pr"].max()),
        "caida_relativa_media": float(barajados["caida_relativa"].mean()),
    }


def recorte_historia(modelo, datos, y: np.ndarray,
                     recortes: list[int] | None = None) -> pd.DataFrame:
    """Desempeno segun cuantas transaccciones de historia se dejan visibles.

    `recorte=1` deja unicamente la transaccion que se esta calificando: es,
    en la practica, un modelo sin historia y por tanto sin orden.
    """
    recortes = recortes or [1, 2, 3, 5, 10, 15, 20]
    filas = []
    for r in recortes:
        p = predecir(modelo, datos, recorte=r)
        filas.append({"historia_visible": r, "auc_pr": auc_pr(y, p)})
    return pd.DataFrame(filas)


def veredicto_orden(resumen: dict, umbral_caida: float = 0.15) -> str:
    """Conclusion honesta segun la evidencia, sin exagerar.

    `umbral_caida` es la caida relativa minima de AUC-PR que consideramos
    evidencia de uso del orden. Se fija antes de ver el resultado.
    """
    caida = resumen["caida_relativa_media"]
    if caida >= umbral_caida:
        return (f"El orden APORTA: destruirlo reduce AUC-PR en "
                f"{100 * caida:.1f} % (de {resumen['auc_pr_original']:.4f} a "
                f"{resumen['auc_pr_barajado_medio']:.4f}). El modelo no puede "
                f"reproducir su desempeno con los mismos eventos desordenados.")
    return (f"NO se puede afirmar que el orden aporte: barajarlo solo cambia "
            f"AUC-PR en {100 * caida:.1f} % (de {resumen['auc_pr_original']:.4f} "
            f"a {resumen['auc_pr_barajado_medio']:.4f}), por debajo del umbral "
            f"de {100 * umbral_caida:.0f} % fijado de antemano. Con esta "
            f"evidencia, el desempeno del modelo se explica sin la secuencia.")
