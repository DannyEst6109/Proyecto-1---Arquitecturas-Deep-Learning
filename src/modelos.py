"""Modelos A (sin orden), B (secuencial) y C (apuesta del equipo).

A  Gradient boosting sobre variables agregadas. Representa el motor actual del
   banco: ve un resumen de la ventana que es invariante a permutaciones.

B  GRU con embeddings sobre la secuencia ordenada de las ultimas K
   transacciones. Ve exactamente los mismos atributos que A resume, pero
   conservando su posicion.

C  Hibrido con atencion: GRU + atencion aditiva sobre los estados ocultos,
   concatenada con las variables agregadas de A. La atencion produce, ademas
   del puntaje, un peso por transaccion de la historia -- es decir, una
   respuesta a "que evento disparo la alerta", que es lo que el area de riesgos
   necesita para investigar un caso.

Los tres devuelven un puntaje continuo de riesgo en [0, 1]. La eleccion de
umbral ocurre despues, en `src/evaluacion.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .caracteristicas import CATEGORICAS_EVENTO, NUMERICAS_EVENTO

DIMS_EMBEDDING = {"categoria": 6, "canal": 4, "departamento": 5}


def fijar_semilla(semilla: int = 20853) -> None:
    """Reproducibilidad del entrenamiento."""
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(False)  # cuDNN/GRU no lo soporta en CPU


# --------------------------------------------------------------------------
# Lotes
# --------------------------------------------------------------------------

class BaseEventos:
    """Arreglos globales de eventos + matriz de indices de secuencia.

    Guardar (N, K, F) explicitamente costaria cientos de MB; en cambio los
    eventos se guardan UNA vez y las secuencias se arman por lote indexando.
    Las tres particiones comparten esta misma base: una secuencia de prueba
    puede mirar hacia atras a eventos anteriores al corte, que es exactamente
    lo que ocurriria en produccion (ver `src/particion.py`).
    """

    def __init__(self, num: np.ndarray, cat: np.ndarray, idx: np.ndarray,
                 agregadas: np.ndarray, y: np.ndarray) -> None:
        self.num = torch.from_numpy(np.ascontiguousarray(num)).float()
        self.cat = torch.from_numpy(np.ascontiguousarray(cat)).long()
        self.idx = torch.from_numpy(np.ascontiguousarray(idx)).long()
        self.agregadas = torch.from_numpy(np.ascontiguousarray(agregadas)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()
        self.pad_cat = torch.tensor([n for _, n in CATEGORICAS_EVENTO], dtype=torch.long)

    def vista(self, filas: np.ndarray) -> "Tensores":
        """Subconjunto evaluable (train, val o test) sobre la misma base."""
        return Tensores(self, np.asarray(filas, dtype=np.int64))


class Tensores:
    """Vista de una particion sobre `BaseEventos`."""

    def __init__(self, base: BaseEventos, filas: np.ndarray) -> None:
        self.base = base
        self.filas = torch.from_numpy(filas).long()

    def __len__(self) -> int:
        return len(self.filas)

    @property
    def objetivo(self) -> torch.Tensor:
        return self.base.y[self.filas]

    def lote(self, posiciones: torch.Tensor, permutar_historia: bool = False,
             generador: torch.Generator | None = None,
             recorte: int | None = None) -> dict:
        """Arma un lote de secuencias.

        permutar_historia : baraja el orden de los eventos DENTRO de cada
            secuencia sin alterar los eventos ni sus valores. Es la prueba de
            permutacion controlada exigida por el enunciado.
        recorte : conserva solo las ultimas `recorte` posiciones de la ventana
            (las anteriores se marcan como relleno). Prueba de historia corta.
        """
        base = self.base
        filas = self.filas[posiciones]
        idx = base.idx[filas]                       # (B, K)
        mask = idx >= 0

        if recorte is not None and recorte < idx.shape[1]:
            corte = idx.shape[1] - recorte
            mask = mask.clone()
            mask[:, :corte] = False

        if permutar_historia:
            idx, mask = _permutar(idx, mask, generador)

        seguro = torch.where(mask, idx, torch.zeros_like(idx))
        x_num = base.num[seguro] * mask.unsqueeze(-1)
        x_cat = torch.where(mask.unsqueeze(-1), base.cat[seguro],
                            base.pad_cat.expand_as(base.cat[seguro]))
        return {
            "num": x_num,
            "cat": x_cat,
            "mask": mask.float(),
            "agregadas": base.agregadas[filas],
            "y": base.y[filas],
        }


def _permutar(idx: torch.Tensor, mask: torch.Tensor,
              generador: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Baraja las posiciones validas de cada secuencia, incluida la actual.

    Se permuta el CONJUNTO de eventos de la ventana. Las variables agregadas no
    cambian (son invariantes a permutaciones) y los eventos son los mismos: lo
    unico que se destruye es la secuencia.
    """
    b, k = idx.shape
    # Relleno con clave -1 para que el orden ascendente lo mantenga a la
    # izquierda; los eventos validos reciben ruido uniforme y quedan barajados
    # entre si, conservando la forma de la secuencia (relleno | eventos).
    ruido = torch.rand(b, k, generator=generador)
    clave = torch.where(mask, ruido, torch.full_like(ruido, -1.0))
    orden = torch.argsort(clave, dim=1)
    return torch.gather(idx, 1, orden), torch.gather(mask, 1, orden)


def iterar_lotes(n: int, tam: int, barajar: bool,
                 generador: torch.Generator | None = None):
    """Indices de fila por lote, sin la sobrecarga de DataLoader."""
    orden = torch.randperm(n, generator=generador) if barajar else torch.arange(n)
    for i in range(0, n, tam):
        yield orden[i:i + tam]


# --------------------------------------------------------------------------
# Arquitecturas
# --------------------------------------------------------------------------

class _CodificadorEventos(nn.Module):
    """Numericas + embeddings categoricos -> vector por evento."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb = 0
        for nombre, n_niveles in CATEGORICAS_EVENTO:
            dim = DIMS_EMBEDDING[nombre]
            # El indice n_niveles es el relleno y aporta el vector cero.
            self.embeddings.append(
                nn.Embedding(n_niveles + 1, dim, padding_idx=n_niveles))
            total_emb += dim
        self.dim_salida = len(NUMERICAS_EVENTO) + total_emb

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        partes = [num]
        for j, emb in enumerate(self.embeddings):
            partes.append(emb(cat[..., j]))
        return torch.cat(partes, dim=-1)


class ModeloSecuencial(nn.Module):
    """B - GRU sobre la secuencia ordenada. Lee el orden, no ve los agregados."""

    def __init__(self, oculto: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.codificador = _CodificadorEventos()
        self.gru = nn.GRU(self.codificador.dim_salida, oculto, batch_first=True)
        self.cabeza = nn.Sequential(
            nn.Linear(oculto, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, lote: dict) -> torch.Tensor:
        x = self.codificador(lote["num"], lote["cat"])
        salida, _ = self.gru(x)
        # El relleno esta a la izquierda y la transaccion a calificar es la
        # ultima posicion, asi que el estado final siempre corresponde a ella.
        return self.cabeza(salida[:, -1, :]).squeeze(-1)


class ModeloHibridoAtencion(nn.Module):
    """C - GRU + atencion aditiva, concatenada con las variables agregadas.

    Hipotesis del equipo (declarada antes de ver el conjunto de prueba):
    combinar el resumen agregado con una lectura atendida de la secuencia
    mejora AUC-PR sobre el mejor de A y B, y ademas entrega al analista el peso
    de cada transaccion de la historia.
    """

    def __init__(self, n_agregadas: int, oculto: int = 64, dropout: float = 0.2,
                 usar_agregadas: bool = True) -> None:
        super().__init__()
        self.usar_agregadas = usar_agregadas
        self.codificador = _CodificadorEventos()
        self.gru = nn.GRU(self.codificador.dim_salida, oculto, batch_first=True)
        self.atencion = nn.Sequential(
            nn.Linear(oculto, 32), nn.Tanh(), nn.Linear(32, 1))

        dim_cabeza = oculto * 2 + (n_agregadas if usar_agregadas else 0)
        self.cabeza = nn.Sequential(
            nn.Linear(dim_cabeza, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.ultimos_pesos: torch.Tensor | None = None

    def forward(self, lote: dict) -> torch.Tensor:
        x = self.codificador(lote["num"], lote["cat"])
        salida, _ = self.gru(x)

        puntajes = self.atencion(salida).squeeze(-1)
        # Las posiciones de relleno no deben recibir atencion.
        puntajes = puntajes.masked_fill(lote["mask"] < 0.5, float("-inf"))
        pesos = torch.softmax(puntajes, dim=1)
        self.ultimos_pesos = pesos.detach()

        contexto = torch.bmm(pesos.unsqueeze(1), salida).squeeze(1)
        partes = [contexto, salida[:, -1, :]]
        if self.usar_agregadas:
            partes.append(lote["agregadas"])
        return self.cabeza(torch.cat(partes, dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------

@dataclass
class ConfigEntrenamiento:
    epocas: int = 25
    tam_lote: int = 512
    tasa_aprendizaje: float = 2e-3
    # El hibrido C tiene mas parametros y sobreajusta antes; se regulariza con
    # decaimiento de pesos ademas del dropout.
    decaimiento_pesos: float = 1e-4
    paciencia: int = 5
    semilla: int = 20853


def entrenar(modelo: nn.Module, datos_train: Tensores, datos_val: Tensores,
             metrica_val, cfg: ConfigEntrenamiento | None = None,
             verboso: bool = True) -> dict:
    """Entrena con parada temprana sobre la metrica de validacion.

    El conjunto de prueba NO participa: ni en la parada temprana, ni en la
    seleccion de arquitectura, ni en el umbral.
    """
    cfg = cfg or ConfigEntrenamiento()
    gen = torch.Generator().manual_seed(cfg.semilla)

    # El desbalance se maneja ponderando la clase positiva en la perdida, no
    # remuestreando: remuestrear romperia la estructura temporal de las series.
    y_train = datos_train.objetivo
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(min(n_neg / max(n_pos, 1.0), 50.0))
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.tasa_aprendizaje,
                                   weight_decay=cfg.decaimiento_pesos)

    mejor = -np.inf
    mejor_estado = None
    sin_mejora = 0
    historial = []

    for epoca in range(cfg.epocas):
        modelo.train()
        perdida_total, vistos = 0.0, 0
        for posiciones in iterar_lotes(len(datos_train), cfg.tam_lote, True, gen):
            lote = datos_train.lote(posiciones)
            optimizador.zero_grad()
            logits = modelo(lote)
            perdida = criterio(logits, lote["y"])
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optimizador.step()
            perdida_total += perdida.detach().item() * len(posiciones)
            vistos += len(posiciones)

        puntajes_val = predecir(modelo, datos_val, cfg.tam_lote)
        valor = metrica_val(datos_val.objetivo.numpy(), puntajes_val)
        historial.append({"epoca": epoca, "perdida": perdida_total / vistos,
                          "metrica_val": valor})
        if verboso:
            print(f"  epoca {epoca:2d}  perdida {perdida_total / vistos:.4f}"
                  f"  AUC-PR val {valor:.4f}")

        if valor > mejor + 1e-5:
            mejor, sin_mejora = valor, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= cfg.paciencia:
                if verboso:
                    print(f"  parada temprana en la epoca {epoca}")
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return {"mejor_metrica_val": mejor, "historial": historial}


def entrenar_linea_base(x_train, y_train, x_val, y_val, cols_categoricas,
                        metrica, semilla: int = 20853, verboso: bool = True):
    """A - Gradient boosting sobre agregados, con busqueda honesta.

    La linea base tiene que ser COMPETITIVA: si se compara un modelo secuencial
    ajustado contra una linea base descuidada, cualquier ventaja del orden es
    un artefacto. Por eso se explora una rejilla pequena y se elige la mejor
    configuracion por AUC-PR de VALIDACION, con el mismo criterio que se usa
    para el modelo secuencial.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    # El desbalance se compensa con pesos, igual que el pos_weight de B.
    peso_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    pesos = np.where(y_train == 1, min(peso_pos, 50.0), 1.0)

    rejilla = [
        {"learning_rate": 0.06, "max_iter": 300, "max_leaf_nodes": 31},
        {"learning_rate": 0.06, "max_iter": 500, "max_leaf_nodes": 63},
        {"learning_rate": 0.12, "max_iter": 250, "max_leaf_nodes": 31},
        {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 31},
    ]

    mejor, mejor_valor, mejor_cfg = None, -np.inf, None
    for params in rejilla:
        modelo = HistGradientBoostingClassifier(
            categorical_features=cols_categoricas,
            early_stopping=False,       # la parada la decide la validacion temporal
            random_state=semilla,
            l2_regularization=1.0,
            **params)
        modelo.fit(x_train, y_train, sample_weight=pesos)
        valor = metrica(y_val, modelo.predict_proba(x_val)[:, 1])
        if verboso:
            print(f"  {params}  AUC-PR val {valor:.4f}")
        if valor > mejor_valor:
            mejor, mejor_valor, mejor_cfg = modelo, valor, params

    if verboso:
        print(f"  elegido: {mejor_cfg}  AUC-PR val {mejor_valor:.4f}")
    return mejor, {"mejor_metrica_val": mejor_valor, "config": mejor_cfg}


@torch.no_grad()
def predecir(modelo: nn.Module, datos: Tensores, tam_lote: int = 1024,
             permutar_historia: bool = False, semilla: int = 7,
             recorte: int | None = None) -> np.ndarray:
    """Puntaje continuo de riesgo en [0, 1]."""
    modelo.eval()
    gen = torch.Generator().manual_seed(semilla)
    salida = np.empty(len(datos), dtype=np.float64)
    for posiciones in iterar_lotes(len(datos), tam_lote, False):
        lote = datos.lote(posiciones, permutar_historia=permutar_historia,
                          generador=gen, recorte=recorte)
        salida[posiciones.numpy()] = torch.sigmoid(modelo(lote)).numpy()
    return salida
