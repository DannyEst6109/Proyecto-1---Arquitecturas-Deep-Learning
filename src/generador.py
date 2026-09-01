"""Generador sintetico de transacciones con tarjeta - Banco del Altiplano.

El generador es un entregable del proyecto: con la misma semilla debe producir
exactamente el mismo conjunto de datos.

Diseno
------
Cada tarjeta tiene un *perfil estable* (nivel de gasto, preferencia de comercios,
departamento base, ritmo horario). Sobre ese comportamiento legitimo se inyectan
episodios de fraude de cuatro mecanismos, que se diferencian deliberadamente en
cuanta informacion aportan el *orden* de los eventos frente a las *variables
agregadas*:

    F1  escalada_prueba   -> depende FUERTEMENTE del orden
    F2  rafaga_geografica -> depende PARCIALMENTE del orden
    F3  vaciado_subito    -> NO depende del orden
    F4  toma_gradual      -> depende del orden pero de forma difusa (caso dificil)

Ese contraste es lo que permite responder la pregunta del comite: si el modelo
secuencial solo mejora donde el orden importa (F1) y empata donde no (F3),
la mejora es atribuible al orden y no a la mayor capacidad del modelo.

Confusores legitimos
--------------------
Sin ruido realista el problema seria trivial. Por eso se inyecta comportamiento
legitimo que *se parece* a cada fraude pero no lo es:

    - rachas de microcompras online (suscripciones, apps)  -> se parece a F1
    - compras grandes legitimas (electronica, viajes)      -> se parece a F3
    - viajes legitimos con cambio de departamento          -> se parece a F2

Sin estos confusores un umbral sobre el monto maximo resolveria el problema y
la comparacion A vs B no tendria sentido.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Catalogos del dominio
# --------------------------------------------------------------------------

# mu y sigma son parametros de una lognormal sobre el monto en quetzales.
# El nivel de gasto del cliente escala el monto resultante.
CATEGORIAS: dict[str, dict] = {
    "supermercado":  {"mu": np.log(190.0),  "sigma": 0.60, "peso": 0.20},
    "gasolinera":    {"mu": np.log(260.0),  "sigma": 0.45, "peso": 0.13},
    "restaurante":   {"mu": np.log(125.0),  "sigma": 0.70, "peso": 0.16},
    "farmacia":      {"mu": np.log(95.0),   "sigma": 0.60, "peso": 0.09},
    "ropa":          {"mu": np.log(330.0),  "sigma": 0.80, "peso": 0.09},
    "servicios":     {"mu": np.log(420.0),  "sigma": 0.55, "peso": 0.10},
    "suscripciones": {"mu": np.log(65.0),   "sigma": 0.50, "peso": 0.09},
    "atm":           {"mu": np.log(620.0),  "sigma": 0.50, "peso": 0.07},
    "electronica":   {"mu": np.log(1500.0), "sigma": 0.85, "peso": 0.04},
    "viajes":        {"mu": np.log(2700.0), "sigma": 0.70, "peso": 0.02},
    "joyeria":       {"mu": np.log(2100.0), "sigma": 0.80, "peso": 0.01},
}

# Distribucion de canal condicionada a la categoria.
CANALES_POR_CATEGORIA: dict[str, dict[str, float]] = {
    "supermercado":  {"chip": 0.55, "contactless": 0.33, "banda": 0.08, "online": 0.04},
    "gasolinera":    {"chip": 0.62, "contactless": 0.28, "banda": 0.10, "online": 0.00},
    "restaurante":   {"chip": 0.50, "contactless": 0.38, "banda": 0.09, "online": 0.03},
    "farmacia":      {"chip": 0.52, "contactless": 0.34, "banda": 0.08, "online": 0.06},
    "ropa":          {"chip": 0.44, "contactless": 0.24, "banda": 0.06, "online": 0.26},
    "servicios":     {"chip": 0.18, "contactless": 0.06, "banda": 0.04, "online": 0.72},
    "suscripciones": {"chip": 0.01, "contactless": 0.01, "banda": 0.00, "online": 0.98},
    "atm":           {"atm": 1.00},
    "electronica":   {"chip": 0.38, "contactless": 0.10, "banda": 0.04, "online": 0.48},
    "viajes":        {"chip": 0.12, "contactless": 0.03, "banda": 0.02, "online": 0.83},
    "joyeria":       {"chip": 0.55, "contactless": 0.12, "banda": 0.08, "online": 0.25},
}

DEPARTAMENTOS: list[str] = [
    "Guatemala", "Sacatepequez", "Escuintla", "Quetzaltenango", "Peten",
    "Izabal", "Alta Verapaz", "Huehuetenango", "San Marcos", "Zacapa",
]

# Peso de residencia: la mayoria de tarjetas vive en la capital.
PESO_DEPARTAMENTO = np.array([0.42, 0.08, 0.09, 0.12, 0.05, 0.05, 0.06, 0.05, 0.05, 0.03])

CANALES: list[str] = ["chip", "contactless", "banda", "online", "atm"]

MECANISMOS: list[str] = ["ninguno", "escalada_prueba", "rafaga_geografica",
                         "vaciado_subito", "toma_gradual"]


@dataclass
class ConfigGenerador:
    """Parametros del generador. Cambiar la semilla cambia el dataset completo."""

    semilla: int = 20853
    n_tarjetas: int = 4000
    fecha_inicio: str = "2025-01-01"
    dias: int = 243  # ~8 meses, suficiente para una particion temporal 70/15/15

    # Ritmo de transacciones por tarjeta y por dia (Gamma).
    forma_tasa: float = 3.0
    escala_tasa: float = 0.085

    # Proporcion de tarjetas que sufren un episodio de fraude.
    prop_victimas: float = 0.16
    # Pesos relativos de cada mecanismo entre las victimas.
    pesos_mecanismos: dict[str, float] = field(default_factory=lambda: {
        "escalada_prueba": 0.32,
        "rafaga_geografica": 0.26,
        "vaciado_subito": 0.26,
        "toma_gradual": 0.16,
    })

    # Confusores legitimos (probabilidad por tarjeta).
    prop_racha_suscripciones: float = 0.22
    prop_compra_grande: float = 0.30
    prop_viaje: float = 0.24

    def __post_init__(self) -> None:
        total = sum(self.pesos_mecanismos.values())
        if not np.isclose(total, 1.0):
            raise ValueError(f"pesos_mecanismos debe sumar 1.0, suma {total}")


# --------------------------------------------------------------------------
# Perfiles de tarjeta
# --------------------------------------------------------------------------

def _perfiles(rng: np.random.Generator, cfg: ConfigGenerador) -> pd.DataFrame:
    """Crea el perfil estable de cada tarjeta."""
    n = cfg.n_tarjetas
    nombres_cat = list(CATEGORIAS)

    # Preferencia por categoria: Dirichlet alrededor del peso poblacional.
    base = np.array([CATEGORIAS[c]["peso"] for c in nombres_cat])
    # alpha alto => clientes parecidos entre si; alpha bajo => muy idiosincraticos.
    preferencias = rng.dirichlet(base * 22.0, size=n)

    # Preferencia horaria: mezcla de tres franjas (manana, medio dia, noche).
    pesos_franja = rng.dirichlet(np.array([2.5, 3.0, 2.5]), size=n)

    return pd.DataFrame({
        "id_tarjeta": np.arange(n),
        # Nivel de gasto multiplicativo: la mayoria cerca de 1, cola larga a la derecha.
        "nivel_gasto": np.exp(rng.normal(0.0, 0.45, size=n)),
        "tasa_diaria": rng.gamma(cfg.forma_tasa, cfg.escala_tasa, size=n).clip(0.05, 1.2),
        "depto_base": rng.choice(len(DEPARTAMENTOS), size=n, p=PESO_DEPARTAMENTO),
        # Propension a operar en linea (afecta cuan sospechoso es un canal online).
        "afinidad_online": rng.beta(2.0, 3.5, size=n),
        "pref_categoria": list(preferencias),
        "pesos_franja": list(pesos_franja),
    })


def _horas(rng: np.random.Generator, pesos_franja: np.ndarray, n: int) -> np.ndarray:
    """Hora del dia como mezcla de gaussianas (manana / medio dia / noche)."""
    centros = np.array([8.5, 13.0, 19.5])
    anchos = np.array([1.6, 1.4, 2.0])
    franja = rng.choice(3, size=n, p=pesos_franja)
    horas = rng.normal(centros[franja], anchos[franja])
    # 2% de actividad de madrugada, legitima pero rara.
    madrugada = rng.random(n) < 0.02
    horas[madrugada] = rng.uniform(0.0, 5.0, size=madrugada.sum())
    return np.mod(horas, 24.0)


def _montos(rng: np.random.Generator, categorias: np.ndarray,
            nivel_gasto: float) -> np.ndarray:
    """Monto lognormal por categoria, escalado por el nivel de gasto del cliente."""
    nombres_cat = list(CATEGORIAS)
    mu = np.array([CATEGORIAS[nombres_cat[c]]["mu"] for c in categorias])
    sigma = np.array([CATEGORIAS[nombres_cat[c]]["sigma"] for c in categorias])
    return np.round(np.exp(rng.normal(mu, sigma)) * nivel_gasto, 2)


def _canales(rng: np.random.Generator, categorias: np.ndarray,
             afinidad_online: float) -> np.ndarray:
    """Canal condicionado a la categoria, inclinado por la afinidad online del cliente."""
    nombres_cat = list(CATEGORIAS)
    salida = np.empty(len(categorias), dtype=np.int64)
    for i, c in enumerate(categorias):
        dist = CANALES_POR_CATEGORIA[nombres_cat[c]]
        probs = np.array([dist.get(canal, 0.0) for canal in CANALES], dtype=float)
        if probs[CANALES.index("online")] > 0:
            # Un cliente muy digital desplaza masa hacia online.
            ajuste = 0.6 + 0.8 * afinidad_online
            probs[CANALES.index("online")] *= ajuste
        probs /= probs.sum()
        salida[i] = rng.choice(len(CANALES), p=probs)
    return salida


# --------------------------------------------------------------------------
# Comportamiento legitimo
# --------------------------------------------------------------------------

def _transacciones_legitimas(rng: np.random.Generator,
                             perfiles: pd.DataFrame,
                             cfg: ConfigGenerador) -> pd.DataFrame:
    """Genera el flujo legitimo de todas las tarjetas, con confusores incluidos."""
    inicio = pd.Timestamp(cfg.fecha_inicio)
    # Los fines de semana concentran algo mas de consumo.
    dow = (np.arange(cfg.dias) + inicio.dayofweek) % 7
    peso_dia = np.where(dow >= 5, 1.35, 1.0)
    peso_dia = peso_dia / peso_dia.sum()

    bloques: list[pd.DataFrame] = []
    n_cat = len(CATEGORIAS)

    for fila in perfiles.itertuples(index=False):
        n_tx = rng.poisson(fila.tasa_diaria * cfg.dias)
        if n_tx < 12:  # tarjetas con historial demasiado corto no aportan secuencia
            n_tx = 12

        dias_tx = rng.choice(cfg.dias, size=n_tx, p=peso_dia)
        horas_tx = _horas(rng, np.asarray(fila.pesos_franja), n_tx)
        ts = (inicio
              + pd.to_timedelta(dias_tx, unit="D")
              + pd.to_timedelta(horas_tx * 3600.0, unit="s"))

        cats = rng.choice(n_cat, size=n_tx, p=np.asarray(fila.pref_categoria))
        montos = _montos(rng, cats, fila.nivel_gasto)
        canales = _canales(rng, cats, fila.afinidad_online)
        deptos = np.full(n_tx, fila.depto_base, dtype=np.int64)

        # --- Confusor 1: viaje legitimo (se parece a F2 rafaga_geografica) ---
        if rng.random() < cfg.prop_viaje:
            destino = rng.choice([d for d in range(len(DEPARTAMENTOS))
                                  if d != fila.depto_base])
            ini_viaje = rng.integers(0, max(1, cfg.dias - 12))
            dur = rng.integers(3, 11)
            en_viaje = (dias_tx >= ini_viaje) & (dias_tx < ini_viaje + dur)
            deptos[en_viaje] = destino

        # --- Confusor 2: racha de microcompras online (se parece a F1) ---
        extras: list[dict] = []
        if rng.random() < cfg.prop_racha_suscripciones:
            n_rachas = rng.integers(1, 4)
            for _ in range(n_rachas):
                k = rng.integers(3, 7)
                t0 = (inicio
                      + pd.Timedelta(days=int(rng.integers(0, cfg.dias)))
                      + pd.Timedelta(seconds=float(rng.uniform(9, 22) * 3600)))
                for j in range(k):
                    extras.append({
                        "timestamp": t0 + pd.Timedelta(minutes=float(rng.uniform(1, 9)) * j),
                        "monto": round(float(np.exp(rng.normal(np.log(55), 0.45))), 2),
                        "categoria": list(CATEGORIAS).index("suscripciones"),
                        "canal": CANALES.index("online"),
                        "departamento": int(fila.depto_base),
                    })

        # --- Confusor 3: compra grande legitima (se parece a F3) ---
        if rng.random() < cfg.prop_compra_grande:
            for _ in range(int(rng.integers(1, 3))):
                cat_grande = list(CATEGORIAS).index(
                    rng.choice(["electronica", "viajes", "joyeria"], p=[0.6, 0.3, 0.1]))
                extras.append({
                    "timestamp": (inicio
                                  + pd.Timedelta(days=int(rng.integers(0, cfg.dias)))
                                  + pd.Timedelta(seconds=float(rng.uniform(9, 21) * 3600))),
                    "monto": round(float(np.exp(rng.normal(
                        CATEGORIAS[list(CATEGORIAS)[cat_grande]]["mu"] + 0.6, 0.5))
                        * fila.nivel_gasto), 2),
                    "categoria": cat_grande,
                    "canal": int(_canales(rng, np.array([cat_grande]),
                                          fila.afinidad_online)[0]),
                    "departamento": int(fila.depto_base),
                })

        bloque = pd.DataFrame({
            "id_tarjeta": fila.id_tarjeta,
            "timestamp": ts,
            "monto": montos,
            "categoria": cats,
            "canal": canales,
            "departamento": deptos,
        })
        if extras:
            extra_df = pd.DataFrame(extras)
            extra_df["id_tarjeta"] = fila.id_tarjeta
            bloque = pd.concat([bloque, extra_df], ignore_index=True)

        bloques.append(bloque)

    df = pd.concat(bloques, ignore_index=True)
    df["es_fraude"] = 0
    df["mecanismo"] = 0  # indice en MECANISMOS -> "ninguno"
    df["id_episodio"] = -1
    return df


# --------------------------------------------------------------------------
# Mecanismos de fraude
# --------------------------------------------------------------------------

def _episodio_escalada_prueba(rng, perfil, t0) -> list[dict]:
    """F1 - Escalada de prueba. DEPENDE FUERTEMENTE DEL ORDEN.

    El defraudador prueba la tarjeta con varias microcompras online para
    confirmar que esta viva y solo entonces ejecuta el cargo grande.

    Por que el orden importa: los agregados de la ventana (monto promedio,
    conteo, monto maximo, diversidad de comercios) son *identicos* si se baraja
    la secuencia. Lo que distingue al fraude es la progresion
    "varias pruebas pequenas -> golpe grande", no el conjunto de valores.
    Un confusor legitimo (racha de suscripciones) produce las microcompras
    pero nunca el golpe final.
    """
    eventos: list[dict] = []
    n_pruebas = int(rng.integers(4, 9))
    t = t0
    monto_prueba = float(rng.uniform(4.0, 12.0))
    for _ in range(n_pruebas):
        t = t + pd.Timedelta(minutes=float(rng.uniform(0.8, 7.0)))
        # Escalada suave: cada prueba es un poco mayor que la anterior.
        monto_prueba *= float(rng.uniform(1.05, 1.45))
        eventos.append({
            "timestamp": t,
            "monto": round(min(monto_prueba, 60.0), 2),
            "categoria": list(CATEGORIAS).index("suscripciones"),
            "canal": CANALES.index("online"),
            "departamento": int(perfil.depto_base),
        })
    # El golpe: 1 o 2 cargos grandes al final.
    for _ in range(int(rng.integers(1, 3))):
        t = t + pd.Timedelta(minutes=float(rng.uniform(2.0, 25.0)))
        cat = list(CATEGORIAS).index(rng.choice(["electronica", "viajes", "joyeria"]))
        eventos.append({
            "timestamp": t,
            "monto": round(float(np.exp(rng.normal(CATEGORIAS[list(CATEGORIAS)[cat]]["mu"] + 0.35, 0.45))
                                 * perfil.nivel_gasto), 2),
            "categoria": cat,
            "canal": CANALES.index("online"),
            "departamento": int(perfil.depto_base),
        })
    return eventos


def _episodio_rafaga_geografica(rng, perfil, t0) -> list[dict]:
    """F2 - Rafaga geografica. DEPENDE PARCIALMENTE DEL ORDEN.

    Compras presenciales en departamentos distantes en muy poco tiempo:
    fisicamente imposible para una sola persona. Un agregado del tipo
    "numero de departamentos distintos en 1 hora" ya captura buena parte de
    esto, asi que se espera que la linea base A tambien lo detecte. Sirve como
    punto de comparacion intermedio.
    """
    eventos: list[dict] = []
    n = int(rng.integers(3, 7))
    otros = [d for d in range(len(DEPARTAMENTOS)) if d != perfil.depto_base]
    deptos = rng.choice(otros, size=n, replace=False) if n <= len(otros) else \
        rng.choice(otros, size=n, replace=True)
    t = t0
    for d in deptos:
        t = t + pd.Timedelta(minutes=float(rng.uniform(6.0, 28.0)))
        cat = list(CATEGORIAS).index(rng.choice(
            ["ropa", "electronica", "supermercado", "gasolinera"], p=[0.3, 0.3, 0.2, 0.2]))
        eventos.append({
            "timestamp": t,
            "monto": round(float(np.exp(rng.normal(
                CATEGORIAS[list(CATEGORIAS)[cat]]["mu"] + 0.25, 0.55)) * perfil.nivel_gasto), 2),
            "categoria": cat,
            "canal": int(rng.choice([CANALES.index("banda"), CANALES.index("chip")],
                                    p=[0.7, 0.3])),
            "departamento": int(d),
        })
    return eventos


def _episodio_vaciado_subito(rng, perfil, t0) -> list[dict]:
    """F3 - Vaciado subito. NO DEPENDE DEL ORDEN.

    Uno a tres retiros o compras de monto muy superior al habitual del cliente.
    Es detectable con una sola variable agregada (monto / gasto historico), de
    modo que aqui *no* esperamos ventaja del modelo secuencial. Funciona como
    control negativo: si B tambien mejora mucho en F3, la mejora no proviene
    del orden sino de la capacidad del modelo.
    """
    eventos: list[dict] = []
    t = t0
    for _ in range(int(rng.integers(1, 4))):
        t = t + pd.Timedelta(minutes=float(rng.uniform(3.0, 90.0)))
        usa_atm = rng.random() < 0.55
        cat = list(CATEGORIAS).index("atm" if usa_atm else "electronica")
        eventos.append({
            "timestamp": t,
            "monto": round(float(rng.uniform(9.0, 22.0)) * 620.0 * perfil.nivel_gasto, 2),
            "categoria": cat,
            "canal": CANALES.index("atm") if usa_atm else CANALES.index("banda"),
            "departamento": int(perfil.depto_base),
        })
    return eventos


def _episodio_toma_gradual(rng, perfil, t0) -> list[dict]:
    """F4 - Toma gradual de la cuenta. ORDEN DIFUSO -> CASO DIFICIL DECLARADO.

    El atacante toma control y migra el perfil poco a poco: los horarios se
    desplazan a la madrugada, las categorias se alejan de las habituales del
    cliente y los montos suben lentamente. No hay ningun evento abrupto.

    Este es el caso en el que ESPERAMOS QUE EL MODELO FALLE: la senal esta
    repartida en decenas de transacciones y ninguna es sospechosa por si misma.
    Se declara por adelantado para poder contrastarlo en el analisis de error.
    """
    eventos: list[dict] = []
    n = int(rng.integers(14, 26))
    cats_raras = [list(CATEGORIAS).index(c) for c in ["suscripciones", "servicios",
                                                      "electronica", "ropa"]]
    t = t0
    for i in range(n):
        avance = (i + 1) / n  # 0 -> 1, controla cuanto se ha desviado el perfil
        t = t + pd.Timedelta(hours=float(rng.uniform(4.0, 30.0)))
        # El horario se corre progresivamente hacia la madrugada.
        hora_obj = (1.0 + 4.0 * rng.random()) if rng.random() < avance else t.hour
        t = t.normalize() + pd.Timedelta(hours=float(hora_obj))
        cat = int(rng.choice(cats_raras)) if rng.random() < avance else \
            int(rng.choice(len(CATEGORIAS), p=np.asarray(perfil.pref_categoria)))
        eventos.append({
            "timestamp": t,
            "monto": round(float(np.exp(rng.normal(
                CATEGORIAS[list(CATEGORIAS)[cat]]["mu"] + 0.5 * avance, 0.6))
                * perfil.nivel_gasto), 2),
            "categoria": cat,
            "canal": CANALES.index("online") if rng.random() < 0.4 + 0.4 * avance
            else CANALES.index("chip"),
            "departamento": int(perfil.depto_base),
        })
    return eventos


_CONSTRUCTORES = {
    "escalada_prueba": _episodio_escalada_prueba,
    "rafaga_geografica": _episodio_rafaga_geografica,
    "vaciado_subito": _episodio_vaciado_subito,
    "toma_gradual": _episodio_toma_gradual,
}


def _inyectar_fraude(rng: np.random.Generator, perfiles: pd.DataFrame,
                     cfg: ConfigGenerador) -> pd.DataFrame:
    """Crea un episodio de fraude para una muestra de tarjetas victima."""
    inicio = pd.Timestamp(cfg.fecha_inicio)
    n_victimas = int(cfg.n_tarjetas * cfg.prop_victimas)
    victimas = rng.choice(cfg.n_tarjetas, size=n_victimas, replace=False)

    nombres = list(cfg.pesos_mecanismos)
    probs = np.array([cfg.pesos_mecanismos[m] for m in nombres])
    asignado = rng.choice(len(nombres), size=n_victimas, p=probs)

    filas: list[dict] = []
    for episodio, (tarjeta, idx_mec) in enumerate(zip(victimas, asignado)):
        perfil = perfiles.iloc[tarjeta]
        mecanismo = nombres[idx_mec]
        # El inicio del episodio se reparte por todo el periodo para que haya
        # fraude en las tres particiones temporales.
        t0 = (inicio
              + pd.Timedelta(days=float(rng.uniform(6, cfg.dias - 4)))
              + pd.Timedelta(seconds=float(rng.uniform(0, 24 * 3600))))
        for evento in _CONSTRUCTORES[mecanismo](rng, perfil, t0):
            evento.update({
                "id_tarjeta": int(tarjeta),
                "es_fraude": 1,
                "mecanismo": MECANISMOS.index(mecanismo),
                "id_episodio": episodio,
            })
            filas.append(evento)

    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# API publica
# --------------------------------------------------------------------------

def generar(cfg: ConfigGenerador | None = None) -> pd.DataFrame:
    """Genera el dataset completo. Determinista dada `cfg.semilla`."""
    cfg = cfg or ConfigGenerador()
    rng = np.random.default_rng(cfg.semilla)

    perfiles = _perfiles(rng, cfg)
    legitimas = _transacciones_legitimas(rng, perfiles, cfg)
    fraudes = _inyectar_fraude(rng, perfiles, cfg)

    df = pd.concat([legitimas, fraudes], ignore_index=True)

    # Recortar al periodo simulado y ordenar por tarjeta y tiempo: este orden
    # es el objeto de estudio del proyecto.
    fin = pd.Timestamp(cfg.fecha_inicio) + pd.Timedelta(days=cfg.dias)
    df = df[(df["timestamp"] >= pd.Timestamp(cfg.fecha_inicio)) & (df["timestamp"] < fin)]
    df = df.sort_values(["id_tarjeta", "timestamp"], kind="mergesort").reset_index(drop=True)

    # Etiquetas legibles (utiles para el informe; los modelos usan los indices).
    df["categoria_nom"] = pd.Categorical.from_codes(df["categoria"], list(CATEGORIAS))
    df["canal_nom"] = pd.Categorical.from_codes(df["canal"], CANALES)
    df["departamento_nom"] = pd.Categorical.from_codes(df["departamento"], DEPARTAMENTOS)
    df["mecanismo_nom"] = pd.Categorical.from_codes(df["mecanismo"], MECANISMOS)

    df.insert(0, "id_transaccion", np.arange(len(df), dtype=np.int64))
    return df


def resumen(df: pd.DataFrame) -> str:
    """Resumen legible del dataset, para el notebook y el informe."""
    n = len(df)
    n_fraude = int(df["es_fraude"].sum())
    lineas = [
        f"Transacciones      : {n:,}",
        f"Tarjetas           : {df['id_tarjeta'].nunique():,}",
        f"Periodo            : {df['timestamp'].min():%Y-%m-%d} a {df['timestamp'].max():%Y-%m-%d}",
        f"Transacciones fraude: {n_fraude:,}  ({100 * n_fraude / n:.3f} %)",
        f"Episodios de fraude : {df.loc[df['es_fraude'] == 1, 'id_episodio'].nunique():,}",
        f"Tx por tarjeta (med): {df.groupby('id_tarjeta').size().median():.0f}",
        "",
        "Fraude por mecanismo:",
    ]
    por_mec = (df[df["es_fraude"] == 1]
               .groupby("mecanismo_nom", observed=True)
               .agg(transacciones=("id_transaccion", "size"),
                    episodios=("id_episodio", "nunique"),
                    monto_mediano=("monto", "median")))
    lineas.append(por_mec.to_string())
    return "\n".join(lineas)


if __name__ == "__main__":
    import pathlib

    cfg = ConfigGenerador()
    datos = generar(cfg)
    print(resumen(datos))

    destino = pathlib.Path(__file__).resolve().parents[1] / "datos"
    destino.mkdir(exist_ok=True)
    salida = destino / "transacciones.parquet"
    datos.to_parquet(salida, index=False)
    print(f"\nGuardado en {salida}")
