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

class Tensores:
    """Arreglos base + matriz de indices. Las secuencias se arman por lote.

    Guardar (N, K, F) explicitamente costaria cientos de MB; en cambio se
    guardan los eventos una sola vez y se indexan al vuelo.
    """

    def __init__(self, num: np.ndarray, cat: np.ndarray, idx: np.ndarray,
                 agregadas: np.ndarray, y: np.ndarray) -> None:
        self.num = torch.from_numpy(np.ascontiguousarray(num)).float()
        self.cat = torch.from_numpy(np.ascontiguousarray(cat)).long()
        self.idx = torch.from_numpy(np.ascontiguousarray(idx)).long()
        self.agregadas = torch.from_numpy(np.ascontiguousarray(agregadas)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).float()
        self.pad_cat = torch.tensor([n for _, n in CATEGORICAS_EVENTO], dtype=torch.long)

    def lote(self, filas: torch.Tensor, permutar_historia: bool = False,
             generador: torch.Generator | None = None,
             recorte: int | None = None) -> dict:
        """Arma un lote de secuencias.

        permutar_historia : baraja el orden de los eventos DENTRO de cada
            secuencia sin alterar los eventos ni sus valores. Es la prueba de
            permutacion controlada exigida por el enunciado.
        recorte : conserva solo las ultimas `recorte` posiciones de la historia
            (las anteriores se marcan como relleno). Prueba de historia corta.
        """
        idx = self.idx[filas]                       # (B, K)
        mask = idx >= 0

        if recorte is not None and recorte < idx.shape[1]:
            corte = idx.shape[1] - recorte
            mask = mask.clone()
            mask[:, :corte] = False

        if permutar_historia:
            idx, mask = _permutar(idx, mask, generador)

        seguro = torch.where(mask, idx, torch.zeros_like(idx))
        x_num = self.num[seguro] * mask.unsqueeze(-1)
        x_cat = torch.where(mask.unsqueeze(-1), self.cat[seguro],
                            self.pad_cat.expand_as(self.cat[seguro]))
        return {
            "num": x_num,
            "cat": x_cat,
            "mask": mask.float(),
            "agregadas": self.agregadas[filas],
            "y": self.y[filas],
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
    epocas: int = 18
    tam_lote: int = 512
    tasa_aprendizaje: float = 2e-3
    paciencia: int = 4
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
    n_pos = float(datos_train.y.sum())
    n_neg = float(len(datos_train.y) - n_pos)
    pos_weight = torch.tensor(min(n_neg / max(n_pos, 1.0), 50.0))
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.tasa_aprendizaje)

    mejor = -np.inf
    mejor_estado = None
    sin_mejora = 0
    historial = []

    for epoca in range(cfg.epocas):
        modelo.train()
        perdida_total, vistos = 0.0, 0
        for filas in iterar_lotes(len(datos_train.y), cfg.tam_lote, True, gen):
            lote = datos_train.lote(filas)
            optimizador.zero_grad()
            logits = modelo(lote)
            perdida = criterio(logits, lote["y"])
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optimizador.step()
            perdida_total += float(perdida) * len(filas)
            vistos += len(filas)

        puntajes_val = predecir(modelo, datos_val, cfg.tam_lote)
        valor = metrica_val(datos_val.y.numpy(), puntajes_val)
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


@torch.no_grad()
def predecir(modelo: nn.Module, datos: Tensores, tam_lote: int = 1024,
             permutar_historia: bool = False, semilla: int = 7,
             recorte: int | None = None) -> np.ndarray:
    """Puntaje continuo de riesgo en [0, 1]."""
    modelo.eval()
    gen = torch.Generator().manual_seed(semilla)
    salida = np.empty(len(datos.y), dtype=np.float64)
    for filas in iterar_lotes(len(datos.y), tam_lote, False):
        lote = datos.lote(filas, permutar_historia=permutar_historia,
                          generador=gen, recorte=recorte)
        salida[filas.numpy()] = torch.sigmoid(modelo(lote)).numpy()
    return salida
