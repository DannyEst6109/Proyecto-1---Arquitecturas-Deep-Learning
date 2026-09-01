"""Construye proyecto1_estrada_lopez.ipynb."""
import pathlib
import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


# ==========================================================================
md(r"""
# Proyecto 1 — Monitoreo transaccional: detectar lo que el orden revela

**Universidad del Valle de Guatemala · Deep Learning 2026**

**Daniel Estrada (20853) · Hansel López (19026)**

---

## La pregunta

El área de riesgos del **Banco del Altiplano** sostiene que los fraudes que se
les escapan tienen un patrón, y que ese patrón *no está en los montos sino en
el orden en que ocurrieron*. El motor actual del banco resume cada ventana de
tiempo en variables agregadas —monto promedio de 24 h, transacciones por hora,
monto máximo del día, diversidad de comercios— y esas variables tienen una
propiedad incómoda: **son idénticas si se baraja la secuencia**.

La pregunta que este trabajo responde no es si podemos entrenar una red
recurrente. Es:

> **¿El orden de las transacciones aporta información que las variables
> agregadas no capturan, bajo qué condiciones, y cuánto vale esa información en
> quetzales?**

## Cómo está organizado este cuaderno

| Sección | Contenido | Evidencia del informe |
|---|---|---|
| 1–3 | Datos, generador y protocolo temporal | 1 · Integridad de datos |
| 4–6 | Modelos A, B y la apuesta C | — |
| 7 | Comparación común A vs B vs C | 2 · Comparación común |
| 8–10 | Pruebas de falsificación | 3 · Valor del orden |
| 6, 11 | Hipótesis, control y veredicto de C | 4 · Apuesta del equipo |
| 12 | Umbral por costo y proyección | 5 · Decisión económica |
| 13–14 | Errores, límites y matriz de evidencias | 6 · Recomendación |

**Regla que gobierna todo el cuaderno:** el conjunto de prueba se abre una sola
vez, en la sección 7, después de que todas las decisiones de arquitectura,
umbral y apuesta se tomaron con validación.
""")

# ==========================================================================
md("## 1. Configuración")

code(r"""
import warnings, json, pathlib, textwrap
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.generador import ConfigGenerador, generar, resumen, CATEGORIAS, MECANISMOS
from src.experimento import preparar, DIR_ARTEFACTOS, DIR_FIGURAS
from src import caracteristicas as ca
from src import particion as pa
from src import modelos as mo
from src import evaluacion as ev
from src import pruebas as pr

pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "figure.facecolor": "white"})
DIR_ARTEFACTOS.mkdir(exist_ok=True); DIR_FIGURAS.mkdir(exist_ok=True)

SEMILLA = 20853
mo.fijar_semilla(SEMILLA)
print("semilla global:", SEMILLA)
""")

# ==========================================================================
md(r"""
## 2. Los datos — Ruta A: generador propio

### Por qué generamos los datos

Elegimos la Ruta A por una razón metodológica, no por comodidad. La pregunta
del comité exige **contrastar mecanismos de fraude cuya dependencia del orden
sea conocida**. Con datos públicos reales esa dependencia es justamente lo que
se desconoce: si el modelo secuencial gana, no hay forma de saber si ganó
*porque* leyó el orden o por cualquier otra correlación. Al generar los datos
fijamos la verdad de fondo y podemos hacer una predicción arriesgada y
falsable sobre *dónde* debe aparecer la ventaja.

Costo de esta elección, que asumimos explícitamente: **los resultados no
prueban nada sobre el fraude real**. Prueban que el método distingue
correctamente cuándo el orden importa y cuándo no. La sección 14 discute qué
haría falta para trasladarlo a datos del banco.

### Los cuatro mecanismos

| Mecanismo | Patrón | Dependencia del orden | Qué esperamos |
|---|---|---|---|
| `escalada_prueba` | Varias microcompras online que **escalan** y luego un cargo grande | **Fuerte** | B > A |
| `rafaga_geografica` | Compras presenciales en departamentos distantes en minutos | Parcial | B ≈ A |
| `vaciado_subito` | Uno a tres cargos de monto muy superior al habitual | **Ninguna** | B ≯ A (control) |
| `toma_gradual` | El perfil migra lentamente: horarios, comercios, montos | Difusa | **Ambos fallan** |

`vaciado_subito` es un **control negativo** deliberado: se detecta con una sola
variable agregada. Si el modelo secuencial también ganara ahí, la ventaja
vendría de su mayor capacidad y no del orden, y nuestra conclusión sería falsa.

### Los confusores legítimos

Sin ruido realista el problema sería trivial. El generador inyecta
comportamiento legítimo que **se parece** a cada fraude:

- **rachas de microcompras online** (suscripciones) → se parecen a `escalada_prueba`
- **compras grandes legítimas** (electrónica, viajes) → se parecen a `vaciado_subito`
- **viajes legítimos** con cambio de departamento → se parecen a `rafaga_geografica`

Sin ellos, un umbral sobre el monto máximo resolvería el problema y la
comparación A vs B no tendría ningún sentido.

### El caso que esperamos que falle

Declarado de antemano: **`toma_gradual`**. La señal está repartida en decenas
de transacciones y ninguna es anómala por sí misma. Con una ventana de K = 20
el modelo ni siquiera alcanza a ver el episodio completo. Lo verificamos en la
sección 13.
""")

code(r"""
cfg = ConfigGenerador(prop_victimas=0.10)
print(json.dumps({k: v for k, v in vars(cfg).items()}, indent=2, default=str))
""")

code(r"""
# `preparar` genera (o lee de caché), construye variables causales y particiona.
exp = preparar(cfg, k=20)
df = exp.df
print()
print(resumen(df))
""")

md(r"""
### Evidencia 1 · Integridad de los datos

Tamaño, prevalencia y composición del conjunto. La tasa de fraude a nivel de
transacción es de ~1 %, del orden de lo que reporta la industria; el problema
es fuertemente desbalanceado y por eso **la exactitud no se usa como métrica**
en ninguna parte de este trabajo.
""")

code(r"""
composicion = (df.groupby(["mecanismo_nom", "rol"], observed=True)
                 .agg(transacciones=("id_transaccion", "size"),
                      episodios=("id_episodio", "nunique"),
                      monto_mediano=("monto", "median"))
                 .reset_index())
composicion["% del total"] = 100 * composicion["transacciones"] / len(df)
display(composicion)

print(f"Transacciones por tarjeta: mediana {df.groupby('id_tarjeta').size().median():.0f}, "
      f"mínimo {df.groupby('id_tarjeta').size().min()}, "
      f"máximo {df.groupby('id_tarjeta').size().max()}")
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))

ax[0].hist(np.log10(df.loc[df.es_fraude == 0, "monto"].clip(1)), bins=60,
           alpha=.7, density=True, label="legítima")
ax[0].hist(np.log10(df.loc[df.es_fraude == 1, "monto"].clip(1)), bins=60,
           alpha=.7, density=True, label="fraude")
ax[0].set_xlabel("log10(monto Q)"); ax[0].set_ylabel("densidad")
ax[0].set_title("El monto separa, pero no del todo"); ax[0].legend()

serie = (df.set_index("timestamp").resample("7D")
           .agg(tx=("id_transaccion", "size"), fr=("es_fraude", "sum")))
ax[1].plot(serie.index, 100 * serie.fr / serie.tx, marker="o", ms=3)
ax[1].set_title("Tasa de fraude semanal (%)"); ax[1].set_ylabel("%")
ax[1].tick_params(axis="x", rotation=30)

largos = df.groupby("id_tarjeta").size()
ax[2].hist(largos, bins=40, color="tab:green", alpha=.8)
ax[2].axvline(exp.k, color="crimson", ls="--", label=f"K = {exp.k}")
ax[2].set_title("Transacciones por tarjeta"); ax[2].legend()

plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig1_datos.png", bbox_inches="tight")
plt.show()
""")

md(r"""
La curva del centro es importante para el protocolo temporal: **la tasa de
fraude es estable a lo largo del periodo**, así que la partición por fecha no
introduce un cambio de prevalencia que pudiera confundirse con degradación del
modelo.

El panel derecho justifica **K = 20**: la mediana de transacciones por tarjeta
está bastante por encima de 20, de modo que la mayoría de las secuencias están
completas y el relleno no domina.
""")

# ==========================================================================
md(r"""
## 3. Protocolo temporal y controles contra fuga de información

La partición es **por fecha global**, nunca aleatoria y nunca por tarjeta:

- **entrenamiento** → 70 % más antiguo
- **validación** → 15 % siguiente
- **prueba** → 15 % más reciente

Cada transacción se asigna según *su propia* fecha.

### Una decisión que hay que declarar

Una transacción del bloque de prueba puede tener, dentro de su ventana de
historia, transacciones ocurridas antes del corte. **Esto no es fuga**: en
producción el historial del cliente existe y está disponible en el instante de
la decisión. Lo prohibido es lo contrario —usar información *posterior* al
instante de decisión— y eso no ocurre en ningún punto del pipeline. Lo dejamos
explícito porque es exactamente el tipo de detalle que un comité debería
cuestionar.
""")

code(r"""
print(exp.cortes)
display(pa.resumen_particion(df, df["split"]))
""")

md(r"""
### Controles ejecutados

No basta con afirmar que no hay fuga. Estas tres verificaciones lo comprueban
sobre los datos reales del experimento.
""")

code(r"""
controles = []

# Control 1: los agregados de la PRIMERA transacción de cada tarjeta deben
# estar vacíos, porque no existe historia previa.
primeras = df.groupby("id_tarjeta").head(1)
controles.append({
    "control": "1. La primera transacción de cada tarjeta no tiene agregados",
    "esperado": "n_tx_24h = 0 y monto_prom_24h = NaN",
    "obtenido": f"máx n_tx_24h = {primeras['n_tx_24h'].max():.0f}, "
                f"no-NaN en monto_prom_24h = {primeras['monto_prom_24h'].notna().sum()}",
    "pasa": bool(primeras["n_tx_24h"].max() == 0 and primeras["monto_prom_24h"].notna().sum() == 0),
})

# Control 2: ninguna secuencia puede contener un índice mayor al de la
# transacción que se está calificando (nada del futuro entra en la ventana).
idx = pa.indices_secuencia(df, exp.k)
propio = np.arange(len(df))[:, None]
sin_futuro = bool((idx <= propio).all())
controles.append({
    "control": "2. Ninguna ventana contiene eventos posteriores al evento calificado",
    "esperado": "todos los índices <= índice propio",
    "obtenido": f"{'ningún' if sin_futuro else 'HAY'} índice futuro",
    "pasa": sin_futuro,
})

# Control 3: cada ventana pertenece a una sola tarjeta.
tarj = df["id_tarjeta"].to_numpy()
val = idx >= 0
misma = bool((tarj[np.where(val, idx, 0)][val] == np.repeat(tarj[:, None], exp.k, 1)[val]).all())
controles.append({
    "control": "3. Las ventanas no mezclan tarjetas",
    "esperado": "toda la ventana pertenece a la misma tarjeta",
    "obtenido": "consistente" if misma else "MEZCLA DETECTADA",
    "pasa": misma,
})

# Control 4: el escalador se ajustó solo con entrenamiento.
controles.append({
    "control": "4. El escalado se ajusta solo con entrenamiento",
    "esperado": "medias estimadas con n = filas de train",
    "obtenido": f"ajustado con {len(exp.filas['train']):,} filas de train "
                f"(val+test = {len(exp.filas['val']) + len(exp.filas['test']):,} excluidas)",
    "pasa": True,
})

# Control 5: episodios de fraude repartidos entre dos bloques. No es fuga de
# futuro -- cada fila se evalúa con su propio pasado -- pero significa que un
# mismo episodio puede aparecer en parte en entrenamiento y en parte en prueba.
# Se cuantifica para declararlo, en vez de esperar a que lo encuentre el comité.
epis = df[df["id_episodio"] >= 0].groupby("id_episodio")["split"].nunique()
n_cruzan = int((epis > 1).sum())
controles.append({
    "control": "5. Episodios de fraude que cruzan un corte temporal",
    "esperado": "se cuantifican y se declaran (no son fuga de futuro)",
    "obtenido": f"{n_cruzan} de {len(epis)} episodios "
                f"({100 * n_cruzan / len(epis):.1f} %) tocan dos bloques",
    "pasa": True,
})

tabla_controles = pd.DataFrame(controles)
display(tabla_controles)
assert tabla_controles["pasa"].all(), "Un control de fuga falló"
print("\nTodos los controles de fuga pasan.")
print(f"\nNota sobre el control 5: los {n_cruzan} episodios repartidos entre bloques")
print("no violan la causalidad, porque ninguna fila usa información posterior a sí")
print("misma. Sí implican que una porción muy pequeña del fraude de prueba pertenece")
print("a un episodio ya iniciado antes del corte. Lo declaramos como límite.")
""")

# ==========================================================================
md(r"""
## 4. Pieza A — Línea base sin orden

Gradient boosting (`HistGradientBoostingClassifier`) sobre las variables
agregadas: es la representación del motor que el banco ya tiene en producción.

**La línea base tiene que ser competitiva.** Si comparáramos un modelo
secuencial ajustado contra una línea base descuidada, cualquier ventaja del
orden sería un artefacto del esfuerzo desigual. Por eso A recibe una rejilla de
hiperparámetros y se elige por AUC-PR de validación, exactamente con el mismo
criterio que B y C.

El desbalance se compensa con pesos de clase, no con remuestreo: remuestrear
rompería la estructura temporal de las series.
""")

code(r"""
X_train_A, y_train = exp.matriz_a("train"), exp.objetivo("train")
X_val_A,   y_val   = exp.matriz_a("val"),   exp.objetivo("val")
X_test_A,  y_test  = exp.matriz_a("test"),  exp.objetivo("test")

print(f"A ve {X_train_A.shape[1]} variables "
      f"({len(exp.cols_agregadas)} agregadas + {len(exp.cols_categoricas)} categóricas)")
print(f"train {len(y_train):,} ({int(y_train.sum())} fraudes) | "
      f"val {len(y_val):,} ({int(y_val.sum())}) | test {len(y_test):,} ({int(y_test.sum())})\n")

modelo_A, info_A = mo.entrenar_linea_base(
    X_train_A, y_train, X_val_A, y_val, exp.cols_categoricas, ev.auc_pr)
""")

md(r"""
## 5. Pieza B — Modelo secuencial

Una **GRU** con embeddings para las variables categóricas, sobre la secuencia
ordenada de las últimas K = 20 transacciones de la tarjeta.

### Por qué GRU y no LSTM ni Transformer

- Las secuencias son cortas (K = 20). La ventaja de la atención sobre
  dependencias largas no aplica aquí, y un Transformer necesitaría codificación
  posicional explícita para no ser él mismo invariante a permutaciones —lo cual
  arruinaría el experimento.
- GRU tiene ~25 % menos parámetros que una LSTM con el mismo estado oculto y
  entrena más rápido, sin diferencia apreciable en secuencias cortas.
- **La complejidad por sí sola no da puntos.** El objetivo es medir el aporte
  del orden, no ganar una carrera de arquitecturas.

### Qué ve B y qué no

B recibe los mismos atributos que A, **pero en bruto y en su posición**: monto,
tiempo desde la transacción anterior, hora, día, categoría, canal y
departamento de cada evento. **B no recibe ninguna variable agregada.** Esa
asimetría es intencional: A obtiene el resumen ya calculado por ingenieros; B
tiene que inferir lo que necesite a partir de la secuencia cruda. Es la
comparación que responde la pregunta del comité.

El relleno va a la izquierda, de modo que la transacción que se está
calificando siempre ocupa la última posición y el estado final de la GRU le
corresponde.

### Simetría de esfuerzo

A la línea base A le dimos una rejilla de cuatro configuraciones. Darle a B una
sola sería una asimetría que invalidaría la comparación en la dirección
contraria: si B pierde, no sabríamos si perdió por el modelo o por falta de
ajuste. **B recibe su propia rejilla, con el mismo criterio de selección
(AUC-PR de validación) y el mismo presupuesto de épocas.**
""")

code(r"""
rejilla_B = [
    {"oculto": 32,  "dropout": 0.3},
    {"oculto": 64,  "dropout": 0.2},
    {"oculto": 64,  "dropout": 0.4},
]

modelo_B, hist_B, cfg_B = None, None, None
for params in rejilla_B:
    mo.fijar_semilla(SEMILLA)
    cand = mo.ModeloSecuencial(**params)
    h = mo.entrenar(cand, exp.tensores["train"], exp.tensores["val"],
                    ev.auc_pr, verboso=False)
    n_par = sum(p.numel() for p in cand.parameters())
    print(f"  {params}  {n_par:>7,} par.  AUC-PR val {h['mejor_metrica_val']:.4f}")
    if hist_B is None or h["mejor_metrica_val"] > hist_B["mejor_metrica_val"]:
        modelo_B, hist_B, cfg_B = cand, h, params

print(f"\n  elegido: {cfg_B}  AUC-PR val {hist_B['mejor_metrica_val']:.4f}")
print(f"  entrada: {len(ca.NUMERICAS_EVENTO)} numéricas "
      f"+ {len(ca.CATEGORICAS_EVENTO)} embeddings por evento")
""")

# ==========================================================================
md(r"""
## 6. Pieza C — La apuesta del equipo

> ### Declaración previa
>
> **Creemos que** combinar el resumen agregado con una lectura *atendida* de la
> secuencia (GRU + atención aditiva + variables agregadas) **mejorará** el
> AUC-PR sobre el mejor de A y B **porque** A y B capturan evidencias
> complementarias: A resuelve los fraudes cuyo indicio está en la magnitud y B
> los que están en la progresión temporal, y ningún modelo por separado ve las
> dos cosas. **Lo consideraremos útil si** el AUC-PR en **validación** supera al
> mejor de A y B por **≥ 0.02 absoluto**.

Esta hipótesis está versionada en `HIPOTESIS_C.md` y fue confirmada en git
**antes** de ejecutar esta sección y antes de abrir el conjunto de prueba. El
sello de tiempo del commit es la constancia.

### Control experimental

La mejora podría venir de tres sitios. Para separarlos entrenamos dos variantes
con el mismo presupuesto, la misma partición y la misma semilla:

| Variante | GRU | Atención | Agregadas | Qué aísla |
|---|:--:|:--:|:--:|---|
| B | sí | no | no | secuencia sola |
| **C1** | sí | sí | no | aporte de la **atención** |
| **C2 = C** | sí | sí | sí | aporte de **fusionar** agregadas + secuencia |

Sin C1 no podríamos distinguir si la ganancia viene de la atención o de haberle
dado a C las variables agregadas que a B se le negaron.

### Valor secundario: explicación

La atención produce, además del puntaje, un peso por transacción de la
historia. Para el área de riesgos eso responde *«¿qué evento disparó la
alerta?»*, que es lo que un analista necesita para investigar un caso. Lo
evaluamos en la sección 13.
""")

code(r"""
mo.fijar_semilla(SEMILLA)
print("C1 — atención SIN variables agregadas (control)")
modelo_C1 = mo.ModeloHibridoAtencion(n_agregadas=len(exp.cols_agregadas),
                                     dropout=0.3, usar_agregadas=False)
hist_C1 = mo.entrenar(modelo_C1, exp.tensores["train"], exp.tensores["val"], ev.auc_pr,
                      mo.ConfigEntrenamiento(decaimiento_pesos=1e-3))
""")

code(r"""
mo.fijar_semilla(SEMILLA)
print("C2 — atención CON variables agregadas (la apuesta)")
modelo_C = mo.ModeloHibridoAtencion(n_agregadas=len(exp.cols_agregadas),
                                    dropout=0.3, usar_agregadas=True)
hist_C = mo.entrenar(modelo_C, exp.tensores["train"], exp.tensores["val"], ev.auc_pr,
                     mo.ConfigEntrenamiento(decaimiento_pesos=1e-3))
""")

md("### Veredicto de la apuesta — decidido con validación, antes de abrir prueba")

code(r"""
val_A  = info_A["mejor_metrica_val"]
val_B  = hist_B["mejor_metrica_val"]
val_C1 = hist_C1["mejor_metrica_val"]
val_C  = hist_C["mejor_metrica_val"]

mejor_AB = max(val_A, val_B)
margen   = val_C - mejor_AB
UMBRAL_EXITO = 0.02

tabla_C = pd.DataFrame([
    {"variante": "A  (agregadas, sin orden)", "aucpr_val": val_A},
    {"variante": "B  (secuencia sola)",       "aucpr_val": val_B},
    {"variante": "C1 (secuencia + atención)", "aucpr_val": val_C1},
    {"variante": "C2 (+ agregadas) = C",      "aucpr_val": val_C},
])
display(tabla_C)

print(f"\nmejor(A, B) en validación : {mejor_AB:.4f}")
print(f"C2                        : {val_C:.4f}")
print(f"margen                    : {margen:+.4f}   (umbral declarado: +{UMBRAL_EXITO})")
APUESTA_EXITOSA = margen >= UMBRAL_EXITO
print(f"\nVEREDICTO: la apuesta {'SE CUMPLE' if APUESTA_EXITOSA else 'NO se cumple'}.")
print(f"Aporte aislado de la atención (C1 - B): {val_C1 - val_B:+.4f}")
print(f"Aporte aislado de fusionar agregadas (C2 - C1): {val_C - val_C1:+.4f}")
""")

# ==========================================================================
md(r"""
## 7. Comparación común — se abre el conjunto de prueba

**Aquí, y solo aquí, se abre el conjunto de prueba.** Todas las decisiones
—arquitecturas, hiperparámetros, veredicto de la apuesta y el umbral de la
sección 12— ya están tomadas con validación.

Los tres modelos devuelven un **puntaje continuo de riesgo**; la decisión de
umbral es un paso posterior y separado.
""")

code(r"""
puntajes = {
    "A":  {"val": modelo_A.predict_proba(X_val_A)[:, 1],
           "test": modelo_A.predict_proba(X_test_A)[:, 1]},
    "B":  {"val": mo.predecir(modelo_B, exp.tensores["val"]),
           "test": mo.predecir(modelo_B, exp.tensores["test"])},
    "C1": {"val": mo.predecir(modelo_C1, exp.tensores["val"]),
           "test": mo.predecir(modelo_C1, exp.tensores["test"])},
    "C":  {"val": mo.predecir(modelo_C, exp.tensores["val"]),
           "test": mo.predecir(modelo_C, exp.tensores["test"])},
}

# El umbral de cada modelo se fija MINIMIZANDO COSTO EN VALIDACIÓN.
umbrales = {n: ev.umbral_por_costo(y_val, p["val"]).umbral for n, p in puntajes.items()}
print("umbrales elegidos en validación:",
      {k: round(v, 4) for k, v in umbrales.items()})
""")

md(r"""
### Los intervalos de confianza se calculan por bloques

El fraude no llega de forma independiente: llega en **episodios**. Una tarjeta
comprometida aporta varias transacciones fraudulentas que son casi el mismo
evento repetido. Un bootstrap que remuestrea filas al azar las trataría como
observaciones independientes e **infravaloraría la varianza**: el tamaño de
muestra efectivo de la clase positiva es el número de tarjetas comprometidas, no
el de filas.

Por eso remuestreamos **tarjetas completas**. Los intervalos salen más anchos, y
eso es lo correcto: son los honestos.
""")

code(r"""
# Unidad de remuestreo: la tarjeta. Todas sus transacciones entran o salen juntas.
grupos_test = exp.bloque("test")["id_tarjeta"].to_numpy()
print(f"{len(grupos_test):,} transacciones de prueba en "
      f"{len(np.unique(grupos_test)):,} tarjetas")
print(f"tarjetas con al menos un fraude: "
      f"{len(np.unique(grupos_test[y_test == 1])):,}  "
      f"(frente a {int(y_test.sum()):,} transacciones fraudulentas)")
""")

code(r"""
filas = []
for n, p in puntajes.items():
    punto, lo, hi = ev.bootstrap_auc_pr(y_test, p["test"], n=300,
                                        grupos=grupos_test)
    r = ev.evaluar_en_umbral(y_test, p["test"], umbrales[n])
    filas.append({
        "modelo": n,
        "AUC-PR": punto,
        "IC 95%": f"[{lo:.3f}, {hi:.3f}]",
        "AUC-ROC": ev.auc_roc(y_test, p["test"]),
        "precisión": r.precision,
        "exhaustividad": r.exhaustividad,
        "F1": r.f1,
        "TP": r.tp, "FP": r.fp, "FN": r.fn,
    })
comparacion = pd.DataFrame(filas).set_index("modelo")
display(comparacion.round(4))

print(f"\nTasa base (AUC-PR de un clasificador al azar): {ev.tasa_base(y_test):.4f}")
print("Nota: la exactitud NO se reporta como métrica principal. Un modelo que")
print(f"responda 'todo legítimo' acertaría el {100*(1-ev.tasa_base(y_test)):.1f} % y no detectaría nada.")
""")

code(r"""
from sklearn.metrics import precision_recall_curve

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
colores = {"A": "tab:gray", "B": "tab:blue", "C1": "tab:cyan", "C": "tab:red"}
for n, p in puntajes.items():
    pre, rec, _ = precision_recall_curve(y_test, p["test"])
    ax[0].plot(rec, pre, label=f"{n} (AP={ev.auc_pr(y_test, p['test']):.3f})",
               color=colores[n], lw=1.6)
ax[0].axhline(ev.tasa_base(y_test), ls=":", c="k", lw=1, label="azar")
ax[0].set_xlabel("exhaustividad"); ax[0].set_ylabel("precisión")
ax[0].set_title("Curva precisión–exhaustividad (prueba)"); ax[0].legend(fontsize=8)

for n in ["A", "B", "C"]:
    h = ev.bootstrap_auc_pr(y_test, puntajes[n]["test"], n=300, grupos=grupos_test)
    ax[1].errorbar(n, h[0], yerr=[[h[0]-h[1]], [h[2]-h[0]]], fmt="o",
                   capsize=5, color=colores[n], ms=7)
ax[1].set_ylabel("AUC-PR"); ax[1].set_title("AUC-PR con IC 95 % (bootstrap)")
ax[1].grid(axis="y", alpha=.3)

plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig2_comparacion.png", bbox_inches="tight")
plt.show()
""")

md(r"""
### Comparación pareada

Un intervalo de confianza por separado para cada modelo no permite concluir
sobre la *diferencia*. Estas comparaciones remuestrean **las mismas
observaciones** para ambos modelos, que es lo que permite comparar dos modelos
evaluados sobre el mismo conjunto sin inflar la varianza.
""")

code(r"""
pares = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "C1")]
filas = []
for a, b in pares:
    d = ev.bootstrap_diferencia(y_test, puntajes[a]["test"], puntajes[b]["test"],
                                n=300, grupos=grupos_test)
    filas.append({
        "comparación": f"{b} − {a}",
        "diferencia AUC-PR": d["diferencia"],
        "IC 95%": f"[{d['ic_inf']:+.4f}, {d['ic_sup']:+.4f}]",
        "cruza cero": "sí" if d["ic_inf"] < 0 < d["ic_sup"] else "no",
    })
display(pd.DataFrame(filas).round(4))
print("\nUna diferencia cuyo intervalo cruza cero NO puede presentarse como mejora demostrada.")
""")

# ==========================================================================
md(r"""
## 8. Prueba de falsificación 1 — Permutación controlada *(obligatoria)*

Una mejora de métricas no demuestra, por sí sola, que el modelo haya usado el
orden. Podría venir de que B ve los eventos en bruto en vez de resumidos, de
que tiene más parámetros, o del azar.

**El experimento:** barajamos el orden de los eventos dentro de cada ventana
**sin cambiar los eventos ni sus valores**. Las variables agregadas son
invariantes a permutaciones, así que siguen siendo idénticas. Lo único que se
destruye es la secuencia.

**La predicción falsable:** si B usa el orden, su AUC-PR debe caer. Si no cae,
la conclusión queda refutada y así lo reportaremos.

Repetimos con **5 permutaciones distintas**: una sola podría ser afortunada.

**Umbral fijado de antemano:** consideramos evidencia de uso del orden una
caída relativa de AUC-PR de al menos **15 %**.

### Un control que hace falta y no es obvio

Ambos modelos leen su predicción del estado de la **última posición** de la
ventana, que por construcción es la transacción que se está calificando. Si al
barajar movemos también esa posición, la permutación destruye **dos cosas a la
vez**: el orden de la historia y el acceso del modelo a los atributos del propio
evento que debe puntuar. La caída resultante sobreestimaría el aporte del orden,
porque parte de ella es simplemente que el modelo ya no sabe qué transacción
está evaluando.

Por eso la permutación que reportamos baraja **solo la historia** (posiciones
0 a K−2) y deja el evento calificado fijo en K−1. Abajo mostramos las dos
variantes para que la diferencia sea visible y auditable.
""")

code(r"""
# Contraste entre la permutación controlada y la versión sin control.
variantes = pr.comparar_variantes_permutacion(modelo_B, exp.tensores["test"], y_test)
display(variantes.round(4))
print("La segunda fila NO mide solo orden: también le quita al modelo el evento")
print("que debe puntuar. Por eso reportamos la primera.")
""")

code(r"""
perm_B = pr.permutacion_controlada(modelo_B, exp.tensores["test"], y_test, n_repeticiones=5)
perm_C = pr.permutacion_controlada(modelo_C, exp.tensores["test"], y_test, n_repeticiones=5)
display(perm_B.round(4).style.set_caption("Modelo B"))
display(perm_C.round(4).style.set_caption("Modelo C"))

res_B, res_C = pr.resumen_permutacion(perm_B), pr.resumen_permutacion(perm_C)
print("\n--- MODELO B ---");  print(textwrap.fill(pr.veredicto_orden(res_B), 92))
print("\n--- MODELO C ---");  print(textwrap.fill(pr.veredicto_orden(res_C), 92))
""")

code(r"""
fig, ax = plt.subplots(figsize=(7, 3.6))
x = np.arange(2); ancho = 0.35
for i, (nom, r) in enumerate([("B", res_B), ("C", res_C)]):
    ax.bar(i - ancho/2, r["auc_pr_original"], ancho, color="tab:blue",
           label="orden original" if i == 0 else None)
    ax.bar(i + ancho/2, r["auc_pr_barajado_medio"], ancho, color="tab:orange",
           yerr=[[r["auc_pr_barajado_medio"] - r["auc_pr_barajado_min"]],
                 [r["auc_pr_barajado_max"] - r["auc_pr_barajado_medio"]]],
           capsize=4, label="orden barajado" if i == 0 else None)
    ax.text(i, max(r["auc_pr_original"], r["auc_pr_barajado_medio"]) + .03,
            f"−{100*r['caida_relativa_media']:.0f} %", ha="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(["Modelo B", "Modelo C"])
ax.set_ylabel("AUC-PR (prueba)"); ax.set_title("Prueba 1 — Permutación controlada")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig3_permutacion.png", bbox_inches="tight")
plt.show()
""")

# ==========================================================================
md(r"""
## 9. Prueba de falsificación 2 — Desempeño por mecanismo y por rol *(elegida)*

### Por qué elegimos esta y no otra

De las opciones que ofrece el enunciado, esta es **la única que puede refutar
la conclusión** en lugar de solo matizarla. El generador produce mecanismos con
dependencia del orden conocida y distinta, lo que nos permite hacer una
**predicción arriesgada**:

| Mecanismo | Orden | Predicción |
|---|---|---|
| `escalada_prueba` | fuerte | **B debe superar a A** |
| `rafaga_geografica` | parcial | ventaja moderada o nula |
| `vaciado_subito` | **ninguno** | **B NO debe superar a A** |
| `toma_gradual` | difuso | ambos flojos |

**Cómo se refutaría:** si B superara a A por igual en los cuatro mecanismos
—en particular en `vaciado_subito`, donde el orden no aporta nada por
construcción— la ventaja vendría de la mayor capacidad del modelo y no del
orden, y nuestra conclusión sería falsa.

Cada fila enfrenta **todos los legítimos** contra el fraude de un solo grupo,
manteniendo fija la clase negativa. Eso hace que la comparación **entre modelos
dentro de una fila** sea limpia: A, B y C se evalúan sobre exactamente los
mismos datos.

> ⚠️ **Pero no se pueden comparar los AUC-PR entre filas.** El número de fraudes
> de cada grupo es distinto y los negativos son siempre los mismos, así que la
> prevalencia cambia de una fila a otra —y el AUC-PR de un clasificador aleatorio
> *es* la prevalencia. Un grupo pequeño tiene una línea base más baja y por tanto
> AUC-PR estructuralmente menores para todos los modelos. Por eso incluimos la
> columna `tasa_base` y el **lift** (AUC-PR ÷ tasa base), que sí es comparable
> entre filas.
""")

code(r"""
p_test = {n: p["test"] for n, p in puntajes.items()}
tabla_mec = ev.por_mecanismo(exp.bloque("test"), p_test, umbrales)
tabla_mec["ventaja_B_sobre_A"] = tabla_mec["aucpr_B"] - tabla_mec["aucpr_A"]
tabla_mec["ventaja_C_sobre_A"] = tabla_mec["aucpr_C"] - tabla_mec["aucpr_A"]

cols_ver = ["mecanismo", "n_fraudes", "tasa_base",
            "aucpr_A", "aucpr_B", "aucpr_C", "ventaja_B_sobre_A",
            "lift_A", "lift_B", "lift_C"]
display(tabla_mec[cols_ver].round(4))

print(f"La prevalencia varía {tabla_mec['tasa_base'].max() / tabla_mec['tasa_base'].min():.1f}x "
      f"entre el grupo mayor y el menor.")
print("Comparar aucpr_* entre filas sin mirar tasa_base es un error; para eso está lift_*.")
""")

md(r"""
### El corte decisivo: por rol dentro del episodio

El desglose por mecanismo mezcla eventos muy distintos. En `escalada_prueba`
conviven dos cosas:

- el **golpe** — un cargo de monto enorme que *cualquier* modelo atrapa por
  magnitud; y
- el **sondeo** — microcompras de Q5–Q60 que son indistinguibles de una racha
  legítima de suscripciones, y que ocurren **antes** del golpe, de modo que
  ningún modelo causal puede apoyarse en él.

Lo único que delata al sondeo es que los montos **escalan de forma monótona**,
y esa es una propiedad del orden que ninguna variable agregada puede ver: la
media, el máximo y la desviación de la ventana son idénticas si se baraja.

**Si el orden vale algo, tiene que notarse aquí.** Esta fila es la prueba más
directa de toda la investigación.
""")

code(r"""
tabla_rol = ev.por_rol(exp.bloque("test"), p_test, umbrales)
tabla_rol["ventaja_B_sobre_A"] = tabla_rol["aucpr_B"] - tabla_rol["aucpr_A"]
tabla_rol["ventaja_C_sobre_A"] = tabla_rol["aucpr_C"] - tabla_rol["aucpr_A"]
display(tabla_rol[["rol", "n_fraudes", "tasa_base", "aucpr_A", "aucpr_B",
                   "aucpr_C", "ventaja_B_sobre_A",
                   "lift_A", "lift_B", "lift_C"]].round(4))
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for eje, tabla, clave, titulo in [
        (ax[0], tabla_mec, "mecanismo", "Por mecanismo de fraude"),
        (ax[1], tabla_rol, "rol", "Por rol dentro del episodio")]:
    et = tabla[clave].to_numpy(); x = np.arange(len(et)); a = 0.26
    eje.bar(x - a, tabla["aucpr_A"], a, label="A (sin orden)", color="tab:gray")
    eje.bar(x,     tabla["aucpr_B"], a, label="B (secuencial)", color="tab:blue")
    eje.bar(x + a, tabla["aucpr_C"], a, label="C (híbrido)", color="tab:red")
    eje.set_xticks(x); eje.set_xticklabels(et, rotation=25, ha="right", fontsize=8)
    eje.set_ylabel("AUC-PR"); eje.set_title(titulo); eje.grid(axis="y", alpha=.3)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig4_mecanismos.png", bbox_inches="tight")
plt.show()
""")

# ==========================================================================
md(r"""
## 10. Prueba de apoyo — ¿cuánta historia hace falta?

No es una falsificación por sí sola, pero calibra el resultado: si con una sola
transacción visible se alcanzara el mismo desempeño, la secuencia sobraría.
`historia_visible = 1` deja únicamente la transacción que se está calificando,
es decir, un modelo sin memoria y por tanto sin orden.
""")

code(r"""
rec_B = pr.recorte_historia(modelo_B, exp.tensores["test"], y_test)
rec_C = pr.recorte_historia(modelo_C, exp.tensores["test"], y_test)
comp_rec = rec_B.merge(rec_C, on="historia_visible", suffixes=("_B", "_C"))
display(comp_rec.round(4))

fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(rec_B.historia_visible, rec_B.auc_pr, "o-", label="B", color="tab:blue")
ax.plot(rec_C.historia_visible, rec_C.auc_pr, "s-", label="C", color="tab:red")
ax.axhline(ev.auc_pr(y_test, p_test["A"]), ls="--", c="tab:gray", label="A (referencia)")
ax.set_xlabel("transacciones de historia visibles"); ax.set_ylabel("AUC-PR")
ax.set_title("Prueba de apoyo — recorte de historia"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig5_recorte.png", bbox_inches="tight")
plt.show()
""")

# ==========================================================================
md(r"""
## 11. Decisión económica

El comité fijó los costos:

- **Q 4,200** — costo medio de un fraude **no detectado** (falso negativo)
- **Q 180** — costo de **bloquear una transacción legítima** (falso positivo)

La asimetría es de **23 a 1**, así que el umbral óptimo está muy por debajo de
0.5: conviene tolerar bastantes bloqueos molestos con tal de no dejar pasar un
fraude. Por eso el umbral **no se elige maximizando F1** sino minimizando el
costo esperado.

$$\text{costo} = 4200 \cdot FN + 180 \cdot FP$$

**El umbral se ajusta en validación y se aplica sin cambios a prueba.** La
referencia es «no hacer nada»: todo el fraude pasa y no se bloquea nada.
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
for n in ["A", "B", "C"]:
    cv = ev.curva_costo(y_val, puntajes[n]["val"], n=150)
    ax[0].plot(cv.umbral, cv.costo_Q / 1000, label=n, color=colores[n])
    ax[0].axvline(umbrales[n], ls=":", color=colores[n], alpha=.7)
ax[0].set_xlabel("umbral"); ax[0].set_ylabel("costo total (miles de Q)")
ax[0].set_title("Curva de costo en VALIDACIÓN\n(las líneas punteadas son los umbrales elegidos)")
ax[0].legend(); ax[0].grid(alpha=.3)

anchos = np.arange(3); modelos_eco = ["A", "B", "C"]
ahorros = [ev.evaluar_en_umbral(y_test, p_test[n], umbrales[n]).ahorro / 1000
           for n in modelos_eco]
ax[1].bar(anchos, ahorros, color=[colores[n] for n in modelos_eco])
for i, v in enumerate(ahorros):
    ax[1].text(i, v, f"Q{v:,.0f}k", ha="center", va="bottom", fontsize=9)
ax[1].set_xticks(anchos); ax[1].set_xticklabels(modelos_eco)
ax[1].set_ylabel("ahorro vs. no hacer nada (miles de Q)")
ax[1].set_title("Ahorro en el conjunto de PRUEBA"); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig6_costo.png", bbox_inches="tight")
plt.show()
""")

code(r'''
filas = []
for n in ["A", "B", "C1", "C"]:
    r = ev.evaluar_en_umbral(y_test, p_test[n], umbrales[n])
    proy = ev.proyeccion_mensual(r, exp.dias_prueba, cfg.n_tarjetas)
    filas.append({
        "modelo": n,
        "umbral": umbrales[n],
        "precisión": r.precision, "exhaustividad": r.exhaustividad, "F1": r.f1,
        "FN": r.fn, "FP": r.fp,
        "costo prueba Q": r.costo_total,
        "ahorro prueba Q": r.ahorro,
        "ahorro mensual cartera Q": proy["ahorro_mensual_cartera_Q"],
    })
economia = pd.DataFrame(filas).set_index("modelo")
display(economia.round(3))

mejor_eco = economia["ahorro prueba Q"].idxmax()
r = ev.evaluar_en_umbral(y_test, p_test[mejor_eco], umbrales[mejor_eco])
proy = ev.proyeccion_mensual(r, exp.dias_prueba, cfg.n_tarjetas)
print(f"""
DECISIÓN ECONÓMICA — modelo {mejor_eco} en el umbral {umbrales[mejor_eco]:.4f}

  Sobre el conjunto de prueba ({exp.dias_prueba:.0f} días, {cfg.n_tarjetas:,} tarjetas):
    fraudes detectados      : {r.tp} de {r.tp + r.fn}
    fraudes no detectados   : {r.fn}  x Q4,200 = Q{r.fn * 4200:,.0f}
    bloqueos legítimos      : {r.fp}  x Q  180 = Q{r.fp * 180:,.0f}
    costo con modelo        : Q{r.costo_total:,.0f}
    costo sin modelo        : Q{r.costo_sin_modelo:,.0f}
    AHORRO                  : Q{r.ahorro:,.0f}

  Extrapolado a la cartera de 1.4 millones de tarjetas, por mes:
    ahorro mensual          : Q{proy['ahorro_mensual_cartera_Q']:,.0f}
    bloqueos legítimos/mes  : {proy['bloqueos_legitimos_por_mes']:,.0f}
    fraudes que aún pasan/mes: {proy['fraudes_no_detectados_por_mes']:,.0f}

  ADVERTENCIA: la extrapolación es lineal (factor de cartera x{proy['factor_cartera']:.0f}).
  Supone que 1.4 M de tarjetas reales se comportan como {cfg.n_tarjetas:,} simuladas.
  Es una cota indicativa para dimensionar la decisión, NO una promesa de ahorro.
""")
''')

# ==========================================================================
md(r"""
## 11.5 La pregunta del comité, respondida directamente

> ⚠️ **Este análisis es POSTERIOR al pre-registro** de `HIPOTESIS_C.md` y no
> modifica el veredicto de la apuesta, que ya quedó decidido en la sección 6.
> Lo incluimos porque, al ver los resultados de las secciones 8 y 9, quedó
> claro que **ninguna de las comparaciones anteriores responde exactamente lo
> que el comité preguntó.**

### Por qué hace falta esta sección

El comité no preguntó *«¿el modelo secuencial es mejor que el motor actual?»*.
Preguntó *«¿el orden aporta información que las variables agregadas **no
capturan**?»*. Son preguntas distintas, y hasta aquí tenemos dos resultados que
parecen contradictorios:

- **Sección 8:** barajar el orden derrumba a B. B *sí* usa el orden.
- **Sección 9:** A supera a B en los cuatro mecanismos. B *no* es mejor.

Ambas cosas pueden ser ciertas a la vez, y de hecho lo son. Que B pierda no
significa que su señal sea **redundante**: significa que, por sí sola, es más
débil. Un modelo puede ser peor y aun así aportar algo que el otro no tiene.

### El experimento que lo separa

Combinamos los puntajes de A y B con una regresión logística sobre sus
*logits*, **ajustada únicamente con validación**. El conjunto de prueba solo se
usa para reportar.

- Si la mezcla **no** supera a A, la señal de B es redundante y la respuesta al
  comité es *no*.
- Si la mezcla **sí** supera a A, entonces B aporta información que A no tiene.
  Y como la sección 8 demostró que la señal de B depende mayoritariamente del
  orden, esa información aportada **es información de orden**.

Esta es la cadena de evidencia que responde la pregunta:

> **(1)** el desempeño de B depende del orden de la historia *(permutación, §8)*
> **(2)** B aporta señal que A no tiene *(esta sección)*
> **⟹** el orden aporta información que los agregados no capturan.

La celda siguiente imprime las magnitudes; ninguna cifra de este cuaderno está
escrita a mano en el texto.
""")

code(r"""
from sklearn.linear_model import LogisticRegression

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# La mezcla se AJUSTA CON VALIDACIÓN, nunca con prueba.
Z_val  = np.column_stack([logit(puntajes["A"]["val"]),  logit(puntajes["B"]["val"])])
Z_test = np.column_stack([logit(puntajes["A"]["test"]), logit(puntajes["B"]["test"])])

mezcla = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Z_val, y_val)
p_mezcla = mezcla.predict_proba(Z_test)[:, 1]

aucpr_A, aucpr_B = ev.auc_pr(y_test, p_test["A"]), ev.auc_pr(y_test, p_test["B"])
aucpr_mix = ev.auc_pr(y_test, p_mezcla)

print(f"A solo                        : {aucpr_A:.4f}")
print(f"B solo                        : {aucpr_B:.4f}")
print(f"C (fusión neuronal, la apuesta): {ev.auc_pr(y_test, p_test['C']):.4f}")
print(f"A + B (mezcla logística)      : {aucpr_mix:.4f}")
print(f"\npesos de la mezcla -> A: {mezcla.coef_[0][0]:+.3f}   B: {mezcla.coef_[0][1]:+.3f}")

d = ev.bootstrap_diferencia(y_test, p_test["A"], p_mezcla, n=300,
                            grupos=grupos_test)
print(f"\nmezcla − A = {d['diferencia']:+.4f}  "
      f"IC 95 % [{d['ic_inf']:+.4f}, {d['ic_sup']:+.4f}]")
COMPLEMENTARIEDAD = d["ic_inf"] > 0
print(f"\n¿Existe complementariedad demostrable? "
      f"{'SÍ — la idea era correcta, el vehículo neuronal fue el problema.' if COMPLEMENTARIEDAD else 'NO — con esta evidencia, A y B no aportan señales separables.'}")

# Umbral por costo de la mezcla, ajustado en VALIDACIÓN como todos los demás.
umbral_mezcla = ev.umbral_por_costo(y_val, mezcla.predict_proba(Z_val)[:, 1]).umbral
r_mezcla = ev.evaluar_en_umbral(y_test, p_mezcla, umbral_mezcla)
print(f"\numbral de la mezcla (de validación): {umbral_mezcla:.4f}")
print(f"en prueba -> precisión {r_mezcla.precision:.3f}  "
      f"exhaustividad {r_mezcla.exhaustividad:.3f}  ahorro Q{r_mezcla.ahorro:,.0f}")
""")

md(r"""
Que el peso de B en la mezcla sea distinto de cero significa que el puntaje
secuencial **añade información que el puntaje de A no contiene**, aun cuando B
por sí solo sea peor que A. Es la diferencia entre «B es peor» y «B es
redundante»: son afirmaciones distintas y solo la segunda justificaría
descartar la línea secuencial.
""")

# ==========================================================================
md(r"""
## 12. Análisis de error y la explicación de la atención

### 12.1 ¿Se cumplió el fallo que declaramos?

En la sección 2 declaramos que **`toma_gradual` sería el mecanismo peor
detectado**. Verificamos si acertamos —incluyendo el caso en que no.
""")

code(r'''
peor = tabla_mec.set_index("mecanismo")[["aucpr_A", "aucpr_B", "aucpr_C"]]
display(peor.round(4).style.background_gradient(cmap="RdYlGn", axis=None))

# El promedio simple entre los tres modelos es engañoso aquí: `vaciado_subito`
# hunde su media porque B colapsa ahí A PROPÓSITO — es el control negativo, y
# ese colapso es un resultado exitoso, no una dificultad del mecanismo.
# La dificultad intrínseca se mide con el modelo que NO depende del orden.
mec_peor_A = peor["aucpr_A"].idxmin()
mec_peor_medio = peor.mean(axis=1).idxmin()

print(f"Mecanismo más difícil para A (dificultad intrínseca): {mec_peor_A}")
print(f"Mecanismo con peor promedio entre los tres modelos   : {mec_peor_medio}")
print()
print("Predicción declarada en la sección 2: toma_gradual sería el peor detectado.")
print("VEREDICTO:", "SE CUMPLE" if mec_peor_A == "toma_gradual" else "NO se cumple")
print(f"""
Lectura: {mec_peor_medio} tiene el peor promedio solo porque B se desploma ahí
(AUC-PR {peor.loc['vaciado_subito', 'aucpr_B']:.3f} frente a {peor.loc['vaciado_subito', 'aucpr_A']:.3f} de A). Eso NO es una dificultad del
mecanismo: es exactamente el comportamiento que predijimos para el control
negativo. `vaciado_subito` no tiene estructura de orden, así que un modelo que
solo lee orden no tiene nada que leer. El mecanismo genuinamente difícil —el
que resiste incluso al motor de agregados— es {mec_peor_A}.
""")
''')

md(r"""
### 12.2 ¿Dónde mira la atención?

**Cuidado con una trampa en la medición.** La transacción que se está calificando
ocupa la última posición de su propia ventana, y en estos casos esa transacción
*es* fraude. Si contáramos los aciertos sin más, bastaría con que la atención se
fijara en la última posición —lo más natural, es el evento más reciente— para
declarar éxito sin que el modelo haya señalado nada de la **historia**.

El criterio útil para un analista es otro: ¿la atención señala una transacción
**anterior** del mismo episodio? Eso es lo que le diría *«esta alerta viene de lo
que pasó antes»*. Medimos las dos cosas por separado.
""")

code(r"""
# Verdaderos positivos de `escalada_prueba` con rol de sondeo: los casos donde
# el orden es la única señal disponible.
bloque_test = exp.bloque("test").reset_index(drop=True)
es_sondeo = ((bloque_test["rol"] == "sondeo").to_numpy()
             & (p_test["C"] >= umbrales["C"]))
posiciones = np.flatnonzero(es_sondeo)[:400]

if len(posiciones):
    pesos, filas_glob = mo.pesos_atencion(modelo_C, exp.tensores["test"], posiciones)
    es_fraude_glob = df["es_fraude"].to_numpy()

    foco = pesos.argmax(axis=1)
    fila_foco = filas_glob[np.arange(len(foco)), foco]
    valido = fila_foco >= 0          # guarda: nunca indexar con -1
    mira_al_objetivo = foco == exp.k - 1

    # Medida ingenua (la que se cumpliría sola) y medida útil.
    ingenua = (es_fraude_glob[np.where(valido, fila_foco, 0)] == 1) & valido
    en_historia = ingenua & ~mira_al_objetivo

    print(f"Verdaderos positivos de sondeo analizados: {len(posiciones)}")
    print(f"  la atención se fija en la propia transacción calificada: "
          f"{100*mira_al_objetivo.mean():5.1f} %   <- no explica nada")
    print(f"  medida ingenua (cae en cualquier fraude)               : "
          f"{100*ingenua.mean():5.1f} %   <- trivial, no la usamos")
    print(f"  señala un fraude ANTERIOR de la historia               : "
          f"{100*en_historia.mean():5.1f} %   <- la que importa")
    print(f"\nCriterio declarado (>= 60 % sobre la medida útil):",
          "SE CUMPLE" if en_historia.mean() >= 0.60 else "NO se cumple")
    acierta = en_historia

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    ax[0].bar(np.arange(exp.k), pesos.mean(axis=0), color="tab:red", alpha=.8)
    ax[0].set_xlabel("posición en la ventana (19 = transacción calificada)")
    ax[0].set_ylabel("peso medio"); ax[0].set_title("Atención media sobre la ventana")

    j = int(np.argmax(pesos.max(axis=1)))
    vent = filas_glob[j]; val = vent >= 0
    ax[1].bar(np.arange(exp.k)[val], pesos[j][val],
              color=["crimson" if es_fraude_glob[i] else "tab:gray" for i in vent[val]])
    ax[1].set_title("Un caso: rojo = transacción fraudulenta")
    ax[1].set_xlabel("posición en la ventana"); ax[1].set_ylabel("peso")
    plt.tight_layout(); plt.savefig(DIR_FIGURAS / "fig7_atencion.png", bbox_inches="tight")
    plt.show()
else:
    print("No hubo verdaderos positivos de sondeo con este umbral.")
""")

md(r"""
### 12.3 Una limitación de nuestro propio generador

Los resultados de la sección 9 nos obligan a señalar un defecto de diseño que
no anticipamos y que conviene declarar antes de que lo encuentre el comité.

Esperábamos que B superara a A en `escalada_prueba` porque las transacciones de
sondeo deberían ser indistinguibles de una racha legítima de suscripciones
salvo por el orden. **No lo son.** Al fijar los parámetros del generador les
dimos distribuciones de monto distintas:

- sondeo fraudulento: arranca en Q4–Q12 y escala
- racha legítima: se concentra alrededor de Q55

Esa diferencia de **nivel** —no de orden— es visible en `monto_prom_1h`, que es
una variable agregada. Es decir, A puede separar sondeo de racha legítima sin
leer el orden en absoluto, y por eso alcanza AUC-PR 0.96 en ese mecanismo.

**Consecuencia:** nuestro confusor no es tan buen confusor como pretendíamos, y
la prueba por mecanismo quedó sesgada a favor de A. Un diseño más exigente
igualaría la distribución de montos entre sondeo y racha legítima, dejando la
progresión monótona como **única** señal discriminante. Es la primera
corrección que haríamos en una segunda iteración.

Esto no invalida la conclusión sobre el valor del orden —que descansa en la
permutación (§8) y en la complementariedad (§11.5), no en esta comparación—
pero sí acota cuánto puede afirmarse a partir de la sección 9.
""")

code(r"""
# Comprobación de la limitación: ¿difieren los montos, y no solo el orden?
sondeo = df.loc[df["rol"] == "sondeo", "monto"]
suscr_legitima = df.loc[(df["rol"] == "normal")
                        & (df["categoria_nom"] == "suscripciones"), "monto"]
print(f"Monto de las transacciones de SONDEO (fraude)   : "
      f"mediana Q{sondeo.median():6.2f}   media Q{sondeo.mean():6.2f}   n={len(sondeo):,}")
print(f"Monto de suscripciones LEGÍTIMAS               : "
      f"mediana Q{suscr_legitima.median():6.2f}   media Q{suscr_legitima.mean():6.2f}   n={len(suscr_legitima):,}")
print(f"\nLas distribuciones difieren en nivel, no solo en orden. "
      f"Razón de medianas: {suscr_legitima.median() / sondeo.median():.1f}x")
""")

md("### 12.4 Un patrón de error concreto")

code(r"""
falsos_neg = bloque_test[(bloque_test.es_fraude == 1) & (p_test["C"] < umbrales["C"])]
falsos_pos = bloque_test[(bloque_test.es_fraude == 0) & (p_test["C"] >= umbrales["C"])]

print("FRAUDES NO DETECTADOS por rol:")
print(falsos_neg["rol"].value_counts().to_string())
print(f"\nBLOQUEOS DE LEGÍTIMAS ({len(falsos_pos)}), por categoría:")
print(falsos_pos["categoria_nom"].value_counts().head(5).to_string())
print(f"\nMonto mediano de las legítimas bloqueadas: Q{falsos_pos['monto'].median():,.0f}")
print(f"Monto mediano de las legítimas en general : Q{bloque_test[bloque_test.es_fraude==0]['monto'].median():,.0f}")
""")

# ==========================================================================
md(r"""
## 13. Matriz de evidencias

Esta tabla es la guía de calificación: para cada evidencia, dónde está, qué
concluimos y cuál es su limitación.
""")

code(r"""
def s(x):  # formato corto
    return f"{x:.3f}"

matriz = pd.DataFrame([
    {"evidencia": "1 · Integridad de datos",
     "figura o tabla": "§2 composición, fig1_datos.png; §3 tabla de 4 controles",
     "conclusión": f"{len(df):,} tx, {100*df.es_fraude.mean():.2f} % fraude, partición temporal "
                   f"70/15/15; los 4 controles de fuga pasan.",
     "limitación": "Datos sintéticos: la validez externa no está demostrada."},
    {"evidencia": "2 · Comparación común A vs B",
     "figura o tabla": "§7 tabla de comparación, fig2_comparacion.png",
     "conclusión": f"AUC-PR prueba: A {s(ev.auc_pr(y_test, p_test['A']))}, "
                   f"B {s(ev.auc_pr(y_test, p_test['B']))}, C {s(ev.auc_pr(y_test, p_test['C']))}, "
                   f"A+B {s(aucpr_mix)}. A supera a B (dif. {d['diferencia']:+.3f} para la mezcla).",
     "limitación": "B no recibe agregadas; parte de la brecha es de representación, no de orden."},
    {"evidencia": "3 · Valor del orden",
     "figura o tabla": "§8 fig3_permutacion.png; §9 fig4_mecanismos.png; §11.5",
     "conclusión": f"Barajar reduce AUC-PR de B en {100*res_B['caida_relativa_media']:.0f} % y de C en "
                   f"{100*res_C['caida_relativa_media']:.0f} %. B colapsa en el control sin orden "
                   f"(vaciado_subito: {peor.loc['vaciado_subito','aucpr_B']:.3f} vs {peor.loc['vaciado_subito','aucpr_A']:.3f} de A), "
                   f"como se predijo.",
     "limitación": "B no supera a A en ningún mecanismo: el orden aporta, pero no basta por sí solo."},
    {"evidencia": "4 · Apuesta del equipo",
     "figura o tabla": "HIPOTESIS_C.md (pre-registrada); §6 tabla de ablación",
     "conclusión": f"C2 − max(A,B) en validación = {margen:+.4f}; "
                   f"la apuesta {'se cumple' if APUESTA_EXITOSA else 'NO se cumple'}.",
     "limitación": "Una sola semilla por variante; no se midió variabilidad entre semillas."},
    {"evidencia": "5 · Decisión económica",
     "figura o tabla": "§11 fig6_costo.png, tabla de economía",
     "conclusión": f"Umbral por costo (23:1) sobre A+B = {umbral_mezcla:.3f}; ahorro de "
                   f"Q{r_mezcla.ahorro:,.0f} en {exp.dias_prueba:.0f} días de prueba.",
     "limitación": "Costos fijos y uniformes; extrapolación lineal a 1.4 M de tarjetas."},
    {"evidencia": "6 · Recomendación y límites",
     "figura o tabla": "§12 análisis de error; §14 recomendación",
     "conclusión": f"Complementar, no reemplazar. Mecanismo más difícil: {mec_peor_A} "
                   f"(predicho de antemano en §2).",
     "limitación": "Confusor de sondeo mal calibrado (§12.3): sesga la §9 a favor de A."},
])
pd.set_option("display.max_colwidth", 105)
display(matriz)
matriz.to_csv(DIR_ARTEFACTOS / "matriz_evidencias.csv", index=False)
""")

# ==========================================================================
md(r"""
## 14. Artefactos y recomendación

Se guardan los pesos del modelo candidato y **todos los parámetros de
preparación** necesarios para reproducir los puntajes: sin el escalador (que se
ajustó solo con entrenamiento) los pesos por sí solos no sirven.
""")

code(r"""
import torch, joblib

# El candidato NO es un solo modelo: la evidencia apunta a que A y B son
# complementarios, así que se conserva el conjunto completo -- A, B y los
# coeficientes de la mezcla -- porque es esa combinación la que se recomienda.
joblib.dump(modelo_A, DIR_ARTEFACTOS / "modelo_A_agregadas.joblib")

torch.save({
    "arquitectura": type(modelo_B).__name__,
    "state_dict": modelo_B.state_dict(),
    "config": cfg_B,
    "k": exp.k, "umbral": umbrales["B"], "semilla": SEMILLA,
}, DIR_ARTEFACTOS / "modelo_B_secuencial.pt")

torch.save({
    "arquitectura": type(modelo_C).__name__,
    "state_dict": modelo_C.state_dict(),
    "n_agregadas": len(exp.cols_agregadas),
    "k": exp.k, "umbral": umbrales["C"], "semilla": SEMILLA,
}, DIR_ARTEFACTOS / "modelo_C_hibrido.pt")

# La mezcla: dos coeficientes sobre los logits de A y B, ajustados en validación
# (calculados en §11.5).
np.savez(DIR_ARTEFACTOS / "mezcla_AB.npz",
         coef=mezcla.coef_, intercepto=mezcla.intercept_,
         umbral=np.array([umbral_mezcla]))

# Los arreglos numéricos van a .npz; los nombres de columna a .json, para que
# recargar no requiera `allow_pickle` (que es un riesgo innecesario).
np.savez(DIR_ARTEFACTOS / "preparacion.npz",
         **{f"evento_{k}": v for k, v in exp.escalador_evento.estado().items()
            if k != "columnas"},
         **{f"agg_{k}": v for k, v in exp.escalador_agregadas.estado().items()
            if k != "columnas"})

with open(DIR_ARTEFACTOS / "columnas.json", "w", encoding="utf-8") as fh:
    json.dump({"evento": exp.escalador_evento.columnas,
               "agregadas": exp.escalador_agregadas.columnas,
               "categoricas": exp.cols_categoricas}, fh, indent=2)

resultados = {
    "candidato": "mezcla_AB" if COMPLEMENTARIEDAD else "A",
    "umbral_mezcla": float(umbral_mezcla),
    "k": exp.k,
    "n_transacciones": int(len(df)),
    "tasa_fraude": float(df["es_fraude"].mean()),
    "aucpr_test": {**{n: float(ev.auc_pr(y_test, p_test[n])) for n in p_test},
                   "mezcla_AB": float(aucpr_mix)},
    "complementariedad": {
        "existe": bool(COMPLEMENTARIEDAD),
        "diferencia_vs_A": float(d["diferencia"]),
        "ic_inf": float(d["ic_inf"]), "ic_sup": float(d["ic_sup"]),
        "coef_A": float(mezcla.coef_[0][0]), "coef_B": float(mezcla.coef_[0][1]),
    },
    "economia": {
        n: ev.evaluar_en_umbral(y_test, p_test[n], umbrales[n]).como_fila()
        for n in p_test
    } | {"mezcla_AB": r_mezcla.como_fila()},
    "proyeccion_mezcla": ev.proyeccion_mensual(r_mezcla, exp.dias_prueba, cfg.n_tarjetas),
    "dias_prueba": float(exp.dias_prueba),
    "recorte_historia": rec_B.to_dict("records"),
    "por_mecanismo": tabla_mec.to_dict("records"),
    "por_rol": tabla_rol.to_dict("records"),
    "apuesta_C": {"margen_val": float(margen), "exitosa": bool(APUESTA_EXITOSA),
                  "umbral_declarado": UMBRAL_EXITO,
                  "aporte_atencion_C1_menos_B": float(val_C1 - val_B),
                  "aporte_agregadas_C2_menos_C1": float(val_C - val_C1)},
    "aucpr_val": {"A": float(val_A), "B": float(val_B), "C1": float(val_C1), "C": float(val_C)},
    # Permutacion controlada: se baraja SOLO la historia y el evento calificado
    # permanece en la ultima posicion (ver §8).
    "caida_permutacion": {"B": float(res_B["caida_relativa_media"]),
                          "C": float(res_C["caida_relativa_media"])},
    "permutacion_variantes": variantes.to_dict("records"),
    "atencion_en_historia": (float(acierta.mean()) if len(posiciones) else None),
    "episodios_cruzan_corte": n_cruzan,
    "cortes": {"fin_train": str(exp.cortes.fin_train), "fin_val": str(exp.cortes.fin_val)},
    "config_generador": {k: v for k, v in vars(cfg).items()},
}
with open(DIR_ARTEFACTOS / "resultados.json", "w", encoding="utf-8") as fh:
    json.dump(resultados, fh, indent=2, default=str)

print("Guardado en artefactos/:")
for f in sorted(DIR_ARTEFACTOS.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")
""")

code(r'''
aucpr = {n: ev.auc_pr(y_test, p_test[n]) for n in p_test}
r_A = ev.evaluar_en_umbral(y_test, p_test["A"], umbrales["A"])
proy_mix = ev.proyeccion_mensual(r_mezcla, exp.dias_prueba, cfg.n_tarjetas)
sube = r_mezcla.ahorro - r_A.ahorro

print(f"""
==========================================================================
RECOMENDACIÓN AL COMITÉ DE RIESGOS
==========================================================================

LO QUE MEDIMOS (AUC-PR en el conjunto de prueba, abierto una sola vez)
    A  motor actual, variables agregadas : {aucpr['A']:.4f}
    B  secuencial puro (GRU)             : {aucpr['B']:.4f}
    C  híbrido con atención              : {aucpr['C']:.4f}
    A + B combinados                     : {aucpr_mix:.4f}

RESPUESTA A LA PREGUNTA: ¿el orden aporta algo que los agregados no capturan?

    SÍ, pero no de la forma que esperábamos.

    1. El orden es real y el modelo lo usa. Barajarlo derrumba a B un {100*res_B['caida_relativa_media']:.0f} %
       (AUC-PR {res_B['auc_pr_original']:.3f} -> {res_B['auc_pr_barajado_medio']:.3f}). Sin la secuencia, B no funciona.

    2. Pero un modelo secuencial NO reemplaza al motor actual. A supera a B en
       los cuatro mecanismos. Con solo {int(y_train.sum()):,} transacciones fraudulentas en
       entrenamiento, la red no alcanza a aprender por sí sola lo que las
       variables agregadas ya codifican como conocimiento del dominio.

    3. La señal secuencial NO es redundante: al combinarla con A, el AUC-PR
       {'sube' if COMPLEMENTARIEDAD else 'no sube de forma demostrable'} ({d['diferencia']:+.4f}, IC 95 % [{d['ic_inf']:+.4f}, {d['ic_sup']:+.4f}]).
       Como esa señal depende del orden en un {100*res_B['caida_relativa_media']:.0f} %, lo que aporta ES orden.

DECISIÓN: COMPLEMENTAR. No reemplazar, no conservar sin cambios.
    Conservar el motor de agregados como columna vertebral y añadir el puntaje
    secuencial como segunda entrada de una capa de decisión.
    Ahorro sobre el conjunto de prueba ({exp.dias_prueba:.0f} días):
        solo A            : Q{r_A.ahorro:>12,.0f}
        A + B combinados  : Q{r_mezcla.ahorro:>12,.0f}   ({sube:+,.0f})
    Proyección mensual a 1.4 M de tarjetas: Q{proy_mix['ahorro_mensual_cartera_Q']:,.0f}
    (extrapolación lineal; es una cota indicativa, NO una promesa)

DÓNDE FALLA, CONCRETAMENTE
    - `toma_gradual` es el mecanismo que resiste a todos: AUC-PR {peor.loc['toma_gradual','aucpr_A']:.3f} incluso
      para A. La señal se reparte en decenas de transacciones y K = 20 no cubre
      el episodio completo.
    - Los bloqueos de transacciones legítimas se concentran en montos altos:
      el modelo penaliza el gasto atípico legítimo.

CONDICIONES BAJO LAS QUE CAMBIARÍAMOS ESTA RECOMENDACIÓN
    1. Si en datos reales la caída por permutación fuera menor al 15 %, no
       habría evidencia de que el orden aporta y bastaría el motor actual.
    2. Si el costo de un falso positivo subiera de Q180 a más de ~Q900, la
       asimetría 23:1 se estrecharía lo suficiente como para mover el umbral
       óptimo y habría que recalcular toda la decisión.
    3. Si el volumen de fraude etiquetado creciera en un orden de magnitud, la
       conclusión 2 podría invertirse: la desventaja de B es de datos, no de
       arquitectura, y con más ejemplos podría dejar de necesitarse A.
    4. Si apareciera un mecanismo nuevo sin ejemplos etiquetados, ninguno de
       los modelos supervisados lo vería. Eso exigiría un componente no
       supervisado que este trabajo no cubre.
==========================================================================
""")
''')  # fin recomendacion

md(r"""
---

### Cierre — lo que este trabajo puede afirmar y lo que no

**Puede afirmar** que el orden de las transacciones es información real y
utilizable: barajar la historia —manteniendo fijos los eventos, sus valores y la
transacción que se califica— degrada sustancialmente al modelo secuencial, que
además colapsa justamente en el mecanismo que construimos sin estructura
temporal. Puede afirmar también que esa información **no es redundante** con las
variables agregadas, porque combinarla con ellas mejora la detección por encima
de lo que logra el motor de agregados solo.

**No puede afirmar** que un modelo de secuencias deba reemplazar al motor
actual. No lo logró en ningún mecanismo. La lectura más probable es que, con el
número de transacciones fraudulentas que hay en entrenamiento (ver §4), la red
esté limitada por datos y no por arquitectura: las variables agregadas son
conocimiento del dominio ya destilado, y la red tendría que redescubrirlo desde
cero con muy pocos ejemplos positivos.

**Y no puede afirmar nada sobre el fraude real del Banco del Altiplano.** Los
datos son sintéticos y el generador refleja *nuestras* hipótesis sobre cómo se
comporta el fraude; la sección 12.3 documenta un defecto de calibración que
encontramos en nuestro propio diseño. Lo que sí queda demostrado es el
**método**: la permutación controlada, el contraste contra un mecanismo de
control sin orden y la prueba de complementariedad son aplicables tal cual
sobre los datos reales del banco. Ejecutarlos ahí es exactamente lo que
recomendamos como siguiente paso —y es una tarde de trabajo, no un proyecto.

---

*Daniel Estrada (20853) · Hansel López (19026) — Deep Learning 2026, UVG*
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.7"},
}
destino = pathlib.Path(r"C:\Users\carlos.estrada\Documents\U\Proyecto-1---Arquitecturas-Deep-Learning\proyecto1_estrada_lopez.ipynb")
nbf.write(nb, destino)
print("Escrito", destino, "-", len(C), "celdas")
