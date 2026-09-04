# Proyecto 1 — Monitoreo transaccional: detectar lo que el orden revela

**Universidad del Valle de Guatemala · Deep Learning 2026**
**Daniel Estrada (20853) · Daniela Ramírez (23053)**

Investigación sobre si el **orden** de las transacciones de una tarjeta aporta
información que las variables agregadas por ventana no capturan, bajo qué
condiciones, y cuánto vale esa información en quetzales.

---

## 1. Reproducción

### Requisitos

- Python 3.13 (probado en 3.13.7, Windows 11)
- ~2 GB de RAM libres, CPU (no se requiere GPU)
- ~45 minutos para ejecutar el cuaderno completo desde cero

### Pasos

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Generar los datos y ejecutar todo el análisis:

```bash
jupyter nbconvert --to notebook --execute --inplace proyecto1_estrada_ramirez.ipynb
```

O abrir el cuaderno y ejecutarlo de arriba hacia abajo:

```bash
jupyter lab proyecto1_estrada_ramirez.ipynb
```

Solo regenerar el conjunto de datos:

```bash
python -m src.generador
```

### Verificar que no hay fuga de información

Afirmar que no hay fuga es barato; este script lo demuestra y devuelve código de
salida 1 si falla. Recalcula las variables con un bucle ingenuo que solo puede
mirar hacia atrás, y —la prueba decisiva— **elimina el futuro del dataset y las
reconstruye**: si alguna variable cambiara al borrar las transacciones
posteriores, dependería del futuro. Ninguna de las 16 cambia.

```bash
python herramientas/verificar_causalidad.py
```

### Determinismo

Todo el trabajo es reproducible con la **semilla 20853**:

| Componente | Cómo se fija |
|---|---|
| Generación de datos | `np.random.default_rng(cfg.semilla)` — un único generador para todo el proceso |
| Entrenamiento de A | `random_state=20853` en `HistGradientBoostingClassifier` |
| Entrenamiento de B y C | `mo.fijar_semilla(20853)` antes de instanciar cada modelo |
| Barajado de lotes | `torch.Generator().manual_seed(...)` explícito |
| Permutaciones de la prueba 1 | semillas `101..105`, fijas |

Los datos se cachean en `datos/transacciones_s20853_n4000_d243.parquet`. Borrar
ese archivo fuerza la regeneración; el resultado es idéntico byte a byte.

> **Nota sobre la GRU:** `torch.use_deterministic_algorithms(False)` porque las
> implementaciones de GRU no ofrecen kernels deterministas. Las métricas pueden
> variar en el cuarto decimal entre corridas. Ninguna conclusión de este
> trabajo depende de diferencias de esa magnitud —todas las que reportamos
> vienen acompañadas de intervalos de confianza por bootstrap.

### Versiones

```
python            3.13.7
numpy             2.5.2
pandas            3.0.5
scikit-learn      1.9.0
torch             2.13.0+cpu
matplotlib        3.11.1
pyarrow          25.0.1
```

---

## 2. Estructura del repositorio

```
proyecto1_estrada_ramirez.ipynb  Cuaderno ejecutado: A, B, C, las dos pruebas y la economía
informe.pdf                     Informe para el comité de riesgos (7 páginas)
presentacion.pdf                Presentación (8 diapositivas)
README.md                       Este archivo
HIPOTESIS_C.md                  Pre-registro de la apuesta C (commit anterior a los resultados)
requirements.txt                Versiones exactas
src/
  generador.py                  Generador sintético (Ruta A). 4 mecanismos + confusores
  caracteristicas.py            Variables agregadas estrictamente causales
  particion.py                  Partición temporal, escalado sin fuga, secuencias
  modelos.py                    Modelos A, B, C + entrenamiento
  evaluacion.py                 AUC-PR, umbral por costo, bootstrap, desgloses
  pruebas.py                    Pruebas de falsificación
  experimento.py                Ensamblado del pipeline completo
artefactos/
  modelo_A_agregadas.joblib     Línea base sin orden
  modelo_B_secuencial.pt        GRU sobre la secuencia
  modelo_C_hibrido.pt           Híbrido con atención (la apuesta)
  mezcla_AB.npz                 Coeficientes de la combinación A+B y su umbral
  preparacion.npz               Escaladores ajustados solo con entrenamiento
  columnas.json                 Orden de columnas que espera cada escalador
  resultados.json               Todas las métricas de la corrida
  matriz_evidencias.csv         Matriz de evidencias del informe
herramientas/
  verificar_causalidad.py       Verificación independiente de ausencia de fuga
  generar_informe.py            Genera informe.pdf y presentacion.pdf desde resultados.json
datos/                          Caché del dataset (regenerable, no versionado)
figuras/                        Figuras que produce el cuaderno (regenerables, no versionadas)
```

Los PDF se generan **desde `artefactos/resultados.json`** y las figuras se leen
del propio cuaderno ejecutado, de modo que ninguna cifra ni ninguna gráfica del
informe se transcribe o se rehace a mano:

```bash
python herramientas/generar_informe.py
```

> Requiere Chrome o Edge (conversión HTML → PDF) y `pymupdf` para la numeración
> de páginas. Es la única parte del proyecto con dependencias externas al
> análisis; los resultados no dependen de ella.

---

## 3. Declaración de uso de asistentes de IA

Usamos **Claude (Anthropic)** durante el desarrollo. Detalle honesto de para
qué y de qué verificamos nosotros.

### Para qué lo usamos

| Uso | Alcance |
|---|---|
| Andamiaje de código | Estructura de módulos, firmas de funciones, bucle de entrenamiento en PyTorch, plantillas de gráficas |
| Implementación mecánica | Ventanas rodantes con `closed="left"`, construcción vectorizada de la matriz de índices de secuencia, bootstrap pareado |
| Redacción | Borradores de los textos explicativos del cuaderno y del informe |
| Depuración | Diagnóstico de un `UserWarning` de autograd y de un error de alineación en las ventanas rodantes |

### Qué verificamos nosotros

- **La ausencia de fuga de información.** No la dimos por buena: la sección 3
  del cuaderno ejecuta cuatro controles sobre los datos reales (agregados
  vacíos en la primera transacción de cada tarjeta, ningún índice futuro en las
  ventanas, ninguna ventana que mezcle tarjetas, escalador ajustado solo con
  train) y el cuaderno falla con `assert` si alguno no pasa.
- **La lógica de la permutación.** Verificamos manualmente que barajar
  conserva el conjunto de eventos y su relleno, y que las variables agregadas
  quedan intactas —si no, la prueba no probaría lo que decimos.
- **El diseño de los mecanismos de fraude.** Es nuestro. La decisión de que
  `vaciado_subito` funcione como control negativo, y de separar `sondeo` de
  `golpe` dentro de `escalada_prueba`, es el núcleo argumental del trabajo y
  no salió del asistente.
- **Los números del informe.** Todos provienen de la ejecución del cuaderno; no
  se transcribió ninguna cifra a mano.

### Lo que no delegamos

El diseño experimental —qué hace falsable la afirmación «el orden aporta», qué
mecanismo sirve de control negativo y por qué el corte por rol es el decisivo—
es la contribución intelectual del trabajo y la discutimos y decidimos nosotros.

---

## 4. Decisiones técnicas y la evidencia que las inclinó

> El enunciado pide tres. Incluimos una cuarta porque surgió de los resultados y
> es la que sostiene la conclusión final del trabajo.

### Decisión 1 — Ruta A (datos sintéticos) en lugar de datos públicos reales

**Alternativas consideradas:** IEEE-CIS Fraud Detection; el dataset de tarjetas
de crédito de ULB en Kaggle; datos de Sparkov.

**Por qué la sintética:** la pregunta del comité exige contrastar mecanismos
**cuya dependencia del orden sea conocida**. En datos reales esa dependencia es
exactamente lo desconocido: si el modelo secuencial gana, no hay forma de saber
si ganó *porque* leyó el orden. Además, el dataset de ULB no trae identificador
de tarjeta, así que ni siquiera permite formar secuencias; y en IEEE-CIS la
tarjeta debe inferirse con heurísticas (`card1` + `addr1` + …), lo que
introduce ruido precisamente en el eje que queremos medir.

**Evidencia que la inclinó:** con el generador podemos hacer una **predicción
arriesgada y falsable** —que B supere a A en `escalada_prueba` y *no* en
`vaciado_subito`— que con datos reales sería imposible de formular. Esa
predicción es la prueba 2 de la sección 9.

**Costo asumido:** los resultados no dicen nada sobre el fraude real. Lo que
queda demostrado es el *método*, que es directamente aplicable a los datos del
banco.

---

### Decisión 2 — La línea base A recibe variables agregadas; B no

**Alternativas consideradas:** (a) darle a B también los agregados, para
igualar la información; (b) quitarle a A las referencias históricas
(`log_razon_monto`), para igualar por abajo; (c) la asimetría que elegimos.

**Por qué la asimetría:** es la que responde la pregunta del comité. A
representa el motor que el banco **ya tiene**: un resumen de ventana calculado
por ingenieros, invariante a permutaciones por construcción. B representa la
alternativa: leer la secuencia cruda. Igualar la información volvería la
comparación una carrera de arquitecturas en vez de una medición del aporte del
orden.

**Evidencia que la inclinó:** la opción (a) es precisamente el modelo C, y lo
entrenamos como parte de la ablación —así obtenemos ambas lecturas sin
sacrificar la comparación limpia.

**Limitación que esto introduce, y la declaramos:** parte de la brecha entre A
y B es de **representación** (agregados ingenierizados frente a eventos en
bruto) y no de orden. Por eso la conclusión sobre el valor del orden **no se
apoya en la comparación A vs B**, sino en la permutación controlada, que
compara a B contra sí mismo y deja la representación fija.

---

### Decisión 3 — El umbral minimiza costo esperado, no F1

**Alternativas consideradas:** umbral 0.5; el que maximiza F1; el que fija la
precisión en un valor operativo; el que minimiza costo.

**Por qué el costo:** el comité entregó los costos reales —Q4,200 por fraude no
detectado, Q180 por bloqueo indebido— y son **asimétricos en razón 23:1**. F1
trata precisión y exhaustividad como igual de importantes, lo que en este
problema es simplemente falso: dejar pasar un fraude cuesta 23 veces más que
molestar a un cliente.

**Evidencia que la inclinó:** la curva de costo de la sección 11 muestra que el
umbral óptimo está muy por debajo de 0.5, y que el umbral que maximiza F1
produce un costo total mayor. La diferencia se traduce directamente en
quetzales.

**Control:** el umbral se ajusta en **validación** y se aplica sin cambios al
conjunto de prueba, que se abre una sola vez.

---

### Decisión 4 *(añadida después de ver los resultados)* — separar «B es peor» de «B es redundante»

**El problema que apareció:** la permutación mostró que B depende del orden en
un 91 %, pero el desglose por mecanismo mostró que A supera a B en los cuatro.
Los dos resultados parecen contradictorios y ninguno responde la pregunta del
comité, que era si el orden aporta información que los agregados **no capturan**.

**Alternativas consideradas:** (a) concluir «el orden no aporta» a partir de que
B pierde; (b) concluir «el orden aporta» a partir de la permutación sola;
(c) medir directamente la complementariedad.

**Por qué (c):** las dos primeras son inválidas. Que B pierda no implica que su
señal sea redundante —un modelo puede ser peor y aun así aportar algo que el
otro no tiene— y la permutación sola solo prueba que *B* usa el orden, no que
ese orden añada algo sobre los agregados.

**Evidencia que la inclinó:** combinamos los logits de A y B con una regresión
logística ajustada **solo en validación**. Si el coeficiente de B fuera nulo o la
mezcla no superara a A, la respuesta al comité sería «no». La mezcla sí mejora y
el intervalo no cruza cero, así que la señal de orden no es redundante.

**Y una segunda decisión que salió de ahí:** al ver que la mejora era
estadísticamente detectable pero minúscula, añadimos un criterio de
**materialidad** separado del de significancia. No lo inventamos: es la moneda
que el propio comité fijó (Q4,200 / Q180). Esa distinción es la que evita que el
trabajo concluya «las secuencias mejoran la detección» cuando lo que los datos
dicen es «las secuencias reducen los falsos positivos y casi nada más».

**Honestidad:** este análisis es **posterior** al pre-registro de
`HIPOTESIS_C.md` y está etiquetado como tal en el cuaderno (§11.5). No lo
presentamos como una hipótesis confirmada sino como el análisis que hizo falta
cuando los resultados no encajaron.

---

## 5. Candidato al Proyecto Final

### Qué modelo conservaríamos y dónde está

**No es un solo modelo.** El resultado del proyecto es que A y B son
complementarios, así que el candidato es la **combinación de ambos**:

| Artefacto | Qué contiene |
|---|---|
| `artefactos/modelo_A_agregadas.joblib` | Gradient boosting sobre variables agregadas |
| `artefactos/modelo_B_secuencial.pt` | GRU sobre la secuencia ordenada |
| `artefactos/mezcla_AB.npz` | Los dos coeficientes que combinan ambos puntajes, y el umbral |
| `artefactos/preparacion.npz` | Escaladores ajustados **solo con entrenamiento** |
| `artefactos/columnas.json` | Orden exacto de las columnas que espera cada escalador |
| `artefactos/modelo_C_hibrido.pt` | El híbrido de la apuesta (se conserva aunque no ganara) |

Sin `preparacion.npz` y `columnas.json` los pesos no sirven: los puntajes no
serían reproducibles.

**Por qué la combinación y no un modelo solo — con una advertencia importante.**
El motor de agregados (A) es claramente el mejor individualmente. La señal
secuencial no es redundante (el intervalo de confianza de la diferencia no cruza
cero), pero **su aporte a la detección es de milésimas de AUC-PR y en quetzales
es casi nulo**. El único beneficio con magnitud defendible es que la combinación
reduce sustancialmente los bloqueos a clientes legítimos manteniendo la misma
exhaustividad.

Es decir: se conserva la combinación no porque detecte más fraude —no lo hace—
sino porque molesta a menos clientes detectando el mismo. Las cifras exactas
están en `artefactos/resultados.json`, claves `complementariedad` y
`materialidad`.

Se conserva también C, aunque su apuesta no alcanzara el umbral declarado. Su
capa de atención pretendía explicar cada alerta señalando la transacción de la
historia que la disparó, pero **ese criterio también falló** (ver
`atencion_en_historia` en los resultados): la atención rara vez señala un evento
anterior del episodio. Se guarda como punto de partida, no como algo utilizable.

### Quién usaría el puntaje y qué decidiría

| Consumidor | Decisión |
|---|---|
| **Motor de autorización** (en línea, < 100 ms) | Bloquear o autorizar la transacción según el umbral por costo |
| **Analista de riesgos** (diferido) | Priorizar la cola de casos por puntaje y usar los pesos de atención para localizar el evento que disparó la alerta |

### Contrato preliminar de entrada y salida

**Entrada** — las últimas K = 20 transacciones de la tarjeta, en orden
cronológico, terminando en la que se está calificando:

```jsonc
{
  "id_tarjeta": "string",
  "secuencia": [                        // 1..20 elementos, orden ascendente por timestamp
    {
      "timestamp": "2025-08-14T21:03:11Z",
      "monto": 1250.00,                 // quetzales
      "categoria": "electronica",       // 11 niveles (ver src/generador.CATEGORIAS)
      "canal": "online",                // chip | contactless | banda | online | atm
      "departamento": "Guatemala"       // 10 niveles
    }
  ]
}
```

**Salida:**

```jsonc
{
  "puntaje_riesgo": 0.8731,             // continuo en [0, 1] — el de la mezcla A+B
  "decision": "bloquear",               // según el umbral vigente
  "umbral_aplicado": 0.0421,
  "componentes": {                      // trazabilidad: qué aportó cada modelo
    "agregados": 0.8102,                // puntaje de A
    "secuencial": 0.6440                // puntaje de B
  },
  "atencion": [0.01, 0.02, ..., 0.41],  // un peso por posición (del modelo C)
  "version_modelo": "mezcla-AB-20853"
}
```

Exponer los dos componentes por separado no es un lujo: permite al analista
distinguir «esto es raro por el monto» de «esto es raro por la secuencia», que
son investigaciones distintas.

**Reglas del contrato:**

- Si la tarjeta tiene menos de 20 transacciones, la ventana se rellena a la
  izquierda; el servicio **no** debe rechazar tarjetas nuevas.
- El escalado debe aplicarse con los parámetros de `preparacion.npz`.
  Recalcularlos con datos de producción reintroduciría la fuga que este trabajo
  evita.
- El umbral es **configurable en caliente**: depende de los costos, que el
  negocio puede cambiar sin reentrenar.

### Límites, riesgos y datos que faltarían

**Límites conocidos**

1. `toma_gradual` es el mecanismo peor detectado por los tres modelos. La señal
   se reparte en decenas de transacciones y K = 20 no cubre el episodio
   completo. Es un fallo que declaramos antes de medirlo y que se confirmó.
2. Solo se evaluó una semilla por variante; no medimos la variabilidad entre
   semillas de inicialización.
3. La extrapolación económica a 1.4 M de tarjetas es lineal y supone que la
   cartera real se comporta como la muestra simulada.

**Riesgos para producción**

1. **Deriva de concepto.** El fraude se adapta. Un modelo entrenado con datos
   de hace seis meses puede degradarse sin aviso; hará falta reentrenamiento
   periódico y monitoreo de la distribución de puntajes.
2. **Mecanismos nuevos.** Los tres modelos son supervisados: un fraude sin
   ejemplos etiquetados es invisible para todos. Un componente no supervisado
   sería un complemento razonable.
3. **Latencia.** No se midió el tiempo de inferencia bajo carga; la GRU sobre
   20 pasos es barata, pero **construir la ventana** exige consultar el
   historial de la tarjeta, y ese es el cuello de botella real.

**Datos que faltarían**

- Transacciones reales del banco con identificador de tarjeta y sello de tiempo.
- La etiqueta de fraude **con su fecha de confirmación**: en producción la
  etiqueta llega semanas después de la transacción, y eso cambia por completo
  cómo debe construirse el conjunto de entrenamiento.
- Los desenlaces de los casos bloqueados, para estimar el costo real de un
  falso positivo en lugar de usar el Q180 supuesto.
- Variables de dispositivo y sesión (huella del navegador, IP, geolocalización
  del teléfono), que en la industria son de las más predictivas y que este
  generador no simula.
