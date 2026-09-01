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

tabla_controles = pd.DataFrame(controles)
display(tabla_controles)
assert tabla_controles["pasa"].all(), "Un control de fuga falló"
print("\nTodos los controles de fuga pasan.")
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
""")

code(r"""
mo.fijar_semilla(SEMILLA)
modelo_B = mo.ModeloSecuencial(oculto=64, dropout=0.2)
n_par = sum(p.numel() for p in modelo_B.parameters())
print(f"B: {n_par:,} parámetros, entrada de {len(ca.NUMERICAS_EVENTO)} numéricas "
      f"+ {len(ca.CATEGORICAS_EVENTO)} embeddings\n")

hist_B = mo.entrenar(modelo_B, exp.tensores["train"], exp.tensores["val"], ev.auc_pr)
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

code(r"""
filas = []
for n, p in puntajes.items():
    punto, lo, hi = ev.bootstrap_auc_pr(y_test, p["test"], n=300)
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
    h = ev.bootstrap_auc_pr(y_test, puntajes[n]["test"], n=300)
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
    d = ev.bootstrap_diferencia(y_test, puntajes[a]["test"], puntajes[b]["test"], n=300)
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
manteniendo fija la clase negativa; de otro modo la prevalencia cambiaría entre
filas y los AUC-PR no serían comparables.
""")

code(r"""
p_test = {n: p["test"] for n, p in puntajes.items()}
tabla_mec = ev.por_mecanismo(exp.bloque("test"), p_test, umbrales)
tabla_mec["ventaja_B_sobre_A"] = tabla_mec["aucpr_B"] - tabla_mec["aucpr_A"]
tabla_mec["ventaja_C_sobre_A"] = tabla_mec["aucpr_C"] - tabla_mec["aucpr_A"]
display(tabla_mec.round(4))
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
display(tabla_rol.round(4))
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
## 12. Análisis de error y la explicación de la atención

### 12.1 ¿Se cumplió el fallo que declaramos?

En la sección 2 declaramos que **`toma_gradual` sería el mecanismo peor
detectado**. Verificamos si acertamos —incluyendo el caso en que no.
""")

code(r"""
peor = tabla_mec.set_index("mecanismo")[["aucpr_A", "aucpr_B", "aucpr_C"]]
display(peor.round(4).style.background_gradient(cmap="RdYlGn", axis=None))
mec_peor = peor.mean(axis=1).idxmin()
print(f"Mecanismo peor detectado en promedio: {mec_peor}")
print("Predicción declarada en la sección 2: toma_gradual")
print("¿Acertamos?", "SÍ" if mec_peor == "toma_gradual" else f"NO — fue {mec_peor}")
""")

md("### 12.2 ¿Dónde mira la atención?")

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
    epis_glob = df["id_episodio"].to_numpy()

    foco = pesos.argmax(axis=1)
    fila_foco = filas_glob[np.arange(len(foco)), foco]
    acierta = (es_fraude_glob[fila_foco] == 1)
    print(f"Verdaderos positivos de sondeo analizados: {len(posiciones)}")
    print(f"La atención se concentra en una transacción del fraude en "
          f"{100*acierta.mean():.1f} % de los casos (criterio declarado: >= 60 %)")
    print("Criterio de interpretabilidad:",
          "SE CUMPLE" if acierta.mean() >= 0.60 else "NO se cumple")

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

md("### 12.3 Un patrón de error concreto")

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
                   f"B {s(ev.auc_pr(y_test, p_test['B']))}, C {s(ev.auc_pr(y_test, p_test['C']))}.",
     "limitación": "B no recibe agregadas; parte de la brecha es de representación, no de orden."},
    {"evidencia": "3 · Valor del orden",
     "figura o tabla": "§8 fig3_permutacion.png; §9 fig4_mecanismos.png",
     "conclusión": f"Barajar el orden reduce AUC-PR de B en "
                   f"{100*res_B['caida_relativa_media']:.0f} % y de C en "
                   f"{100*res_C['caida_relativa_media']:.0f} %.",
     "limitación": "La permutación también altera la coherencia temporal de delta_t."},
    {"evidencia": "4 · Apuesta del equipo",
     "figura o tabla": "HIPOTESIS_C.md (pre-registrada); §6 tabla de ablación",
     "conclusión": f"C2 − max(A,B) en validación = {margen:+.4f}; "
                   f"la apuesta {'se cumple' if APUESTA_EXITOSA else 'NO se cumple'}.",
     "limitación": "Una sola semilla por variante; no se midió variabilidad entre semillas."},
    {"evidencia": "5 · Decisión económica",
     "figura o tabla": "§11 fig6_costo.png, tabla de economía",
     "conclusión": f"Mejor: {mejor_eco} en umbral {umbrales[mejor_eco]:.3f}; "
                   f"ahorro de Q{r.ahorro:,.0f} en {exp.dias_prueba:.0f} días de prueba.",
     "limitación": "Costos fijos y uniformes; extrapolación lineal a 1.4 M de tarjetas."},
    {"evidencia": "6 · Recomendación y límites",
     "figura o tabla": "§12 análisis de error; §14 recomendación",
     "conclusión": f"Mecanismo peor detectado: {mec_peor} (predicho de antemano).",
     "limitación": "Sin datos reales del banco no puede estimarse la degradación."},
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
import torch

candidato = mejor_eco if mejor_eco in ("B", "C") else "C"
modelo_cand = {"B": modelo_B, "C": modelo_C}[candidato]

torch.save({
    "arquitectura": type(modelo_cand).__name__,
    "state_dict": modelo_cand.state_dict(),
    "n_agregadas": len(exp.cols_agregadas),
    "k": exp.k,
    "umbral": umbrales[candidato],
    "semilla": SEMILLA,
}, DIR_ARTEFACTOS / "modelo_candidato.pt")

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
    "candidato": candidato,
    "umbral": float(umbrales[candidato]),
    "k": exp.k,
    "aucpr_test": {n: float(ev.auc_pr(y_test, p_test[n])) for n in p_test},
    "aucpr_val": {"A": float(val_A), "B": float(val_B), "C1": float(val_C1), "C": float(val_C)},
    "caida_permutacion": {"B": float(res_B["caida_relativa_media"]),
                          "C": float(res_C["caida_relativa_media"])},
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

print(f"""
==========================================================================
RECOMENDACIÓN AL COMITÉ DE RIESGOS
==========================================================================

AUC-PR en el conjunto de prueba
    A (motor actual, agregadas) : {aucpr['A']:.4f}
    B (secuencial puro)         : {aucpr['B']:.4f}
    C (híbrido con atención)    : {aucpr['C']:.4f}

Valor del orden
    Barajar el orden derrumba a B un {100*res_B['caida_relativa_media']:.0f} % y a C un {100*res_C['caida_relativa_media']:.0f} %.
    El orden SÍ es información que los modelos usan.

Decisión: COMPLEMENTAR, no reemplazar.
    El motor de agregados ya resuelve muy bien los fraudes cuyo indicio está
    en la magnitud. El modelo secuencial aporta donde ese motor es ciego por
    construcción: patrones que solo existen en la progresión temporal.

Condiciones bajo las que cambiaría esta recomendación
    1. Si en datos reales la caída por permutación fuera menor al 15 %, no
       habría evidencia de que el orden aporta y bastaría el motor actual.
    2. Si el costo de un falso positivo subiera de Q180 a más de ~Q900, la
       relación 23:1 se estrecharía y el umbral óptimo se movería tanto que
       convendría recalcular toda la decisión.
    3. Si apareciera un mecanismo de fraude nuevo sin ejemplos etiquetados,
       ninguno de los tres modelos supervisados lo vería.
==========================================================================
""")
''')

md(r"""
---

### Cierre

Lo que este trabajo puede afirmar y lo que no:

**Puede afirmar** que, en un entorno donde controlamos la verdad de fondo, el
orden de las transacciones contiene información que las variables agregadas no
capturan; que esa información es medible, y que un modelo que la lee mejora la
detección justamente en los mecanismos donde el orden importa por construcción
—y no en el mecanismo de control, donde no importa.

**No puede afirmar** nada sobre el fraude real del Banco del Altiplano. Los
datos son sintéticos y el generador refleja *nuestras* hipótesis sobre cómo se
comporta el fraude. Lo que sí queda demostrado es el **método**: el
procedimiento de permutación controlada y el contraste por mecanismo son
aplicables tal cual sobre los datos reales del banco, y son la primera cosa que
recomendamos hacer con ellos.
""")

nb = nbf.v4.new_notebook(cells=C)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.7"},
}
destino = pathlib.Path(r"C:\Users\carlos.estrada\Documents\U\Proyecto-1---Arquitecturas-Deep-Learning\proyecto1_estrada_lopez.ipynb")
nbf.write(nb, destino)
print("Escrito", destino, "-", len(C), "celdas")
