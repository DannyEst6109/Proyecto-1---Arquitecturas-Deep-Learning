"""Metricas, umbral por costo y analisis economico.

Por que no se reporta exactitud
-------------------------------
Con 1.1 % de fraude, un modelo que responda "todo es legitimo" alcanza 98.9 %
de exactitud y no detecta nada. La metrica principal es AUC-PR (precision
promedio), que se concentra en la clase minoritaria; en el umbral elegido se
reportan precision, exhaustividad y F1.

Umbral
------
El umbral no se elige maximizando F1 sino MINIMIZANDO COSTO ESPERADO, con los
costos que dio el comite: Q4,200 por fraude no detectado y Q180 por bloquear
una transaccion legitima. El umbral se ajusta en VALIDACION y se aplica sin
cambios al conjunto de prueba.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)

COSTO_FRAUDE_NO_DETECTADO = 4200.0   # quetzales
COSTO_BLOQUEO_LEGITIMO = 180.0       # quetzales


def auc_pr(y: np.ndarray, puntajes: np.ndarray) -> float:
    """Precision promedio. Metrica principal del proyecto."""
    return float(average_precision_score(y, puntajes))


def auc_roc(y: np.ndarray, puntajes: np.ndarray) -> float:
    return float(roc_auc_score(y, puntajes))


def tasa_base(y: np.ndarray) -> float:
    """AUC-PR de un clasificador aleatorio: la prevalencia."""
    return float(np.mean(y))


# --------------------------------------------------------------------------
# Umbral y costo
# --------------------------------------------------------------------------

@dataclass
class ResultadoUmbral:
    umbral: float
    precision: float
    exhaustividad: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    costo_total: float
    costo_sin_modelo: float

    @property
    def ahorro(self) -> float:
        return self.costo_sin_modelo - self.costo_total

    def como_fila(self) -> dict:
        return {
            "umbral": self.umbral,
            "precision": self.precision,
            "exhaustividad": self.exhaustividad,
            "f1": self.f1,
            "TP": self.tp, "FP": self.fp, "FN": self.fn, "TN": self.tn,
            "costo_Q": self.costo_total,
            "ahorro_Q": self.ahorro,
        }


def _metricas(y: np.ndarray, puntajes: np.ndarray, umbral: float) -> ResultadoUmbral:
    pred = (puntajes >= umbral).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    exhaustividad = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * exhaustividad / (precision + exhaustividad)
          if precision + exhaustividad else 0.0)

    costo = fn * COSTO_FRAUDE_NO_DETECTADO + fp * COSTO_BLOQUEO_LEGITIMO
    # Referencia: no hacer nada (el fraude pasa completo, no se bloquea nada).
    costo_sin = float(y.sum()) * COSTO_FRAUDE_NO_DETECTADO

    return ResultadoUmbral(umbral, precision, exhaustividad, f1,
                           tp, fp, fn, tn, costo, costo_sin)


def _candidatos_umbral(puntajes: np.ndarray, n: int) -> np.ndarray:
    """Rejilla de umbrales, incluyendo los dos extremos degenerados.

    Los cuantiles por si solos nunca llegan a "no bloquear nada": el candidato
    mas alto es max(puntajes), y con la regla `puntaje >= umbral` eso siempre
    marca al menos un positivo. Se anade un umbral por encima del maximo para
    que la opcion de no bloquear nada compita de verdad, y uno en cero para el
    otro extremo. Con la asimetria 23:1 es improbable que gane el extremo
    conservador, pero dejarlo fuera truncaria la curva de costo del informe.
    """
    base = np.quantile(puntajes, np.linspace(0.0, 1.0, n))
    extremos = [0.0, float(np.nextafter(puntajes.max(), np.inf))]
    return np.unique(np.concatenate([base, extremos]))


def umbral_por_costo(y: np.ndarray, puntajes: np.ndarray,
                     n_candidatos: int = 400) -> ResultadoUmbral:
    """Umbral que minimiza el costo esperado. AJUSTAR SOLO EN VALIDACION."""
    mejor = None
    for u in _candidatos_umbral(puntajes, n_candidatos):
        r = _metricas(y, puntajes, float(u))
        if mejor is None or r.costo_total < mejor.costo_total:
            mejor = r
    return mejor


def evaluar_en_umbral(y: np.ndarray, puntajes: np.ndarray,
                      umbral: float) -> ResultadoUmbral:
    """Aplica un umbral ya fijado (el de validacion) a otro conjunto."""
    return _metricas(y, puntajes, umbral)


def curva_costo(y: np.ndarray, puntajes: np.ndarray,
                n: int = 200) -> pd.DataFrame:
    """Costo total en funcion del umbral. Para la figura del informe."""
    filas = [_metricas(y, puntajes, float(u)).como_fila()
             for u in _candidatos_umbral(puntajes, n)]
    return pd.DataFrame(filas)


def proyeccion_mensual(resultado: ResultadoUmbral, dias_evaluados: float,
                       tarjetas_simuladas: int, tarjetas_banco: int = 1_400_000
                       ) -> dict:
    """Extrapola el ahorro del conjunto de prueba a la cartera del banco.

    Es una extrapolacion lineal y debe presentarse como tal: supone que la
    cartera completa se comporta como la muestra simulada. Es una cota
    indicativa, no una promesa.
    """
    factor_tiempo = 30.0 / dias_evaluados
    factor_cartera = tarjetas_banco / tarjetas_simuladas
    return {
        "ahorro_mensual_muestra_Q": resultado.ahorro * factor_tiempo,
        "ahorro_mensual_cartera_Q": resultado.ahorro * factor_tiempo * factor_cartera,
        "costo_mensual_con_modelo_Q": resultado.costo_total * factor_tiempo * factor_cartera,
        "costo_mensual_sin_modelo_Q": resultado.costo_sin_modelo * factor_tiempo * factor_cartera,
        "bloqueos_legitimos_por_mes": resultado.fp * factor_tiempo * factor_cartera,
        "fraudes_no_detectados_por_mes": resultado.fn * factor_tiempo * factor_cartera,
        "factor_tiempo": factor_tiempo,
        "factor_cartera": factor_cartera,
    }


# --------------------------------------------------------------------------
# Incertidumbre
# --------------------------------------------------------------------------

class _Remuestreador:
    """Genera remuestras bootstrap, opcionalmente POR BLOQUES.

    Por que por bloques: el fraude no llega de forma independiente, llega en
    episodios. Una tarjeta comprometida aporta 7 transacciones fraudulentas en
    promedio, y son casi el mismo evento repetido. Un bootstrap que remuestrea
    filas i.i.d. trata esas 7 como 7 observaciones independientes e infravalora
    la varianza: el tamano de muestra efectivo de la clase positiva es el numero
    de EPISODIOS (~65 en prueba), no el de filas (~465).

    Remuestrear tarjetas completas respeta esa correlacion y produce intervalos
    honestos, que suelen ser bastante mas anchos.
    """

    def __init__(self, n_obs: int, grupos: np.ndarray | None) -> None:
        self.n_obs = n_obs
        self.por_bloques = grupos is not None
        if self.por_bloques:
            orden = np.argsort(grupos, kind="mergesort")
            g_ord = np.asarray(grupos)[orden]
            cortes = np.flatnonzero(np.r_[True, g_ord[1:] != g_ord[:-1]])
            self.bloques = np.split(orden, cortes[1:])
            self.n_bloques = len(self.bloques)

    def muestra(self, rng: np.random.Generator) -> np.ndarray:
        if not self.por_bloques:
            return rng.integers(0, self.n_obs, self.n_obs)
        elegidos = rng.integers(0, self.n_bloques, self.n_bloques)
        return np.concatenate([self.bloques[j] for j in elegidos])


def bootstrap_auc_pr(y: np.ndarray, puntajes: np.ndarray, n: int = 400,
                     semilla: int = 20853,
                     grupos: np.ndarray | None = None
                     ) -> tuple[float, float, float]:
    """AUC-PR con intervalo percentil 95 %.

    Pasar `grupos` (por ejemplo `id_tarjeta`) activa el bootstrap por bloques,
    que es lo correcto cuando las observaciones estan correlacionadas dentro de
    cada grupo. Ver `_Remuestreador`.

    Sirve para no exagerar: una diferencia entre modelos menor que el ancho del
    intervalo no puede presentarse como una mejora demostrada.
    """
    rng = np.random.default_rng(semilla)
    rem = _Remuestreador(len(y), grupos)
    valores = np.empty(n)
    for i in range(n):
        m = rem.muestra(rng)
        if y[m].sum() == 0:       # remuestreo degenerado
            valores[i] = np.nan
            continue
        valores[i] = average_precision_score(y[m], puntajes[m])
    valores = valores[~np.isnan(valores)]
    return (auc_pr(y, puntajes),
            float(np.percentile(valores, 2.5)),
            float(np.percentile(valores, 97.5)))


def bootstrap_diferencia(y: np.ndarray, puntajes_a: np.ndarray,
                         puntajes_b: np.ndarray, n: int = 400,
                         semilla: int = 20853,
                         grupos: np.ndarray | None = None) -> dict:
    """Intervalo para AUC-PR(B) - AUC-PR(A) sobre las MISMAS remuestras.

    Emparejar las remuestras es lo que permite comparar dos modelos evaluados
    en el mismo conjunto sin inflar la varianza. `grupos` activa el bootstrap
    por bloques (ver `_Remuestreador`).
    """
    rng = np.random.default_rng(semilla)
    rem = _Remuestreador(len(y), grupos)
    dif = np.empty(n)
    for i in range(n):
        m = rem.muestra(rng)
        if y[m].sum() == 0:
            dif[i] = np.nan
            continue
        dif[i] = (average_precision_score(y[m], puntajes_b[m])
                  - average_precision_score(y[m], puntajes_a[m]))
    dif = dif[~np.isnan(dif)]
    observada = auc_pr(y, puntajes_b) - auc_pr(y, puntajes_a)
    return {
        "diferencia": observada,
        "ic_inf": float(np.percentile(dif, 2.5)),
        "ic_sup": float(np.percentile(dif, 97.5)),
        # Fraccion de remuestras en que B no supera a A. NO es un valor p:
        # es una medida descriptiva de la estabilidad del signo.
        "frac_sin_mejora": float(np.mean(dif <= 0)),
        "por_bloques": bool(grupos is not None),
    }


# --------------------------------------------------------------------------
# Desglose por mecanismo
# --------------------------------------------------------------------------

def por_rol(df_eval: pd.DataFrame, puntajes: dict[str, np.ndarray],
            umbrales: dict[str, float]) -> pd.DataFrame:
    """Desglose por ROL del evento dentro del episodio de fraude.

    Es un corte mas fino que `por_mecanismo` y responde a la pregunta exacta
    del comite. En `escalada_prueba`, el evento "golpe" tiene un monto enorme y
    cualquier modelo lo atrapa por magnitud; los eventos "sondeo", en cambio,
    son microcompras indistinguibles de una racha legitima de suscripciones y
    ocurren ANTES del golpe, de modo que ningun modelo causal puede usarlo. Si
    el orden vale algo, tiene que notarse justo ahi.
    """
    return _desglose(df_eval, "rol", puntajes, umbrales)


def por_mecanismo(df_eval: pd.DataFrame, puntajes: dict[str, np.ndarray],
                  umbrales: dict[str, float]) -> pd.DataFrame:
    """Exhaustividad de cada modelo por mecanismo de fraude.

    Comparar los legitimos completos contra el fraude de UN mecanismo a la vez
    mantiene constante la clase negativa, de modo que las filas son
    comparables entre si.
    """
    return _desglose(df_eval, "mecanismo_nom", puntajes, umbrales)


def _desglose(df_eval: pd.DataFrame, columna: str,
              puntajes: dict[str, np.ndarray],
              umbrales: dict[str, float]) -> pd.DataFrame:
    """Metricas por grupo de fraude, manteniendo fija la clase negativa.

    Cada fila enfrenta TODOS los legitimos contra el fraude de un solo grupo.
    Eso mantiene constante la clase negativa, de modo que la comparacion
    ENTRE MODELOS dentro de una misma fila es limpia: A, B y C se evaluan sobre
    exactamente los mismos datos.

    Advertencia sobre comparar ENTRE FILAS
    --------------------------------------
    La prevalencia SI cambia de una fila a otra, porque el numero de fraudes de
    cada grupo es distinto mientras los negativos son siempre los mismos. Y el
    AUC-PR de un clasificador aleatorio es exactamente la prevalencia. Un grupo
    pequeno tiene una linea base mas baja y, por tanto, AUC-PR estructuralmente
    menores para TODOS los modelos.

    Por eso se devuelve la columna `tasa_base` y, para cada modelo, un `lift_*`
    = AUC-PR / tasa_base. El lift si es comparable entre filas; el AUC-PR
    crudo, no. Interpretar la magnitud de `aucpr_B - aucpr_A` de una fila
    frente a otra sin mirar la tasa base es un error.
    """
    y = df_eval["es_fraude"].to_numpy()
    etiquetas = df_eval[columna].astype(str).to_numpy()
    grupos = sorted(set(etiquetas[y == 1]))

    filas = []
    for grupo in grupos:
        sel = (y == 0) | (etiquetas == grupo)
        y_sub = y[sel]
        base = float(y_sub.mean())
        fila = {columna.replace("_nom", ""): grupo,
                "n_fraudes": int(y_sub.sum()),
                "tasa_base": base}
        for nombre, p in puntajes.items():
            p_sub = p[sel]
            valor = auc_pr(y_sub, p_sub)
            fila[f"aucpr_{nombre}"] = valor
            fila[f"lift_{nombre}"] = valor / base if base else np.nan
            fila[f"exhaustividad_{nombre}"] = _metricas(
                y_sub, p_sub, umbrales[nombre]).exhaustividad
        filas.append(fila)
    return pd.DataFrame(filas)
