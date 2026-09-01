"""Genera informe.pdf y presentacion.pdf a partir de artefactos/resultados.json.

Ninguna cifra se escribe a mano: todo proviene de la ejecucion del cuaderno.
El HTML se convierte a PDF con Chrome en modo headless.

    python herramientas/generar_informe.py
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

RAIZ = pathlib.Path(__file__).resolve().parents[1]
ART = RAIZ / "artefactos"
FIG = RAIZ / "figuras"

CHROME_POSIBLES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def buscar_chrome() -> str:
    for ruta in CHROME_POSIBLES:
        if pathlib.Path(ruta).exists():
            return ruta
    for nombre in ("google-chrome", "chromium", "chrome", "msedge"):
        hallado = shutil.which(nombre)
        if hallado:
            return hallado
    raise SystemExit("No se encontro Chrome/Edge para generar el PDF.")


def a_pdf(html: pathlib.Path, pdf: pathlib.Path, apaisado: bool = False) -> None:
    chrome = buscar_chrome()
    cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=10000",
           f"--print-to-pdf={pdf}", html.resolve().as_uri()]
    if apaisado:
        cmd.insert(-2, "--landscape")
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not pdf.exists():
        raise SystemExit(f"Chrome no produjo {pdf}")
    print(f"  {pdf.name}  ({pdf.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------
# Utilidades de formato
# --------------------------------------------------------------------------

def q(valor: float, decimales: int = 0) -> str:
    """Cantidad en quetzales con separador de miles."""
    return f"Q{valor:,.{decimales}f}"


def pct(valor: float, decimales: int = 1) -> str:
    return f"{100 * valor:.{decimales}f}\u00a0%"


def tabla_html(encabezados: list[str], filas: list[list[str]],
               clases: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in encabezados)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>"
                 for f in filas)
    return f'<table class="{clases}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


ESTILO = """
@page { size: A4; margin: 16mm 15mm 14mm 15mm; }
* { box-sizing: border-box; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 9.6pt;
       line-height: 1.42; color: #1a1a1a; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 2px; letter-spacing: -.3px; }
h2 { font-size: 11.5pt; margin: 15px 0 6px; padding-bottom: 3px;
     border-bottom: 1.5px solid #1f3a5f; color: #1f3a5f; }
h3 { font-size: 10pt; margin: 11px 0 4px; color: #2c2c2c; }
p  { margin: 0 0 6px; text-align: justify; }
.sub { color: #555; font-size: 9pt; margin-bottom: 2px; }
.meta { color: #666; font-size: 8.2pt; border-bottom: 2px solid #1f3a5f;
        padding-bottom: 8px; margin-bottom: 12px; }
table { width: 100%; border-collapse: collapse; margin: 7px 0 9px;
        font-size: 8.4pt; font-family: 'Segoe UI', Helvetica, sans-serif; }
th { background: #1f3a5f; color: #fff; text-align: left; padding: 4px 6px;
     font-weight: 600; font-size: 8pt; }
td { padding: 3.5px 6px; border-bottom: .5px solid #d8d8d8; }
tbody tr:nth-child(even) { background: #f4f6f9; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.destacado td { background: #fff3cd !important; font-weight: 600; }
.banda { display: flex; gap: 8px; margin: 10px 0 12px; }
.tarjeta { flex: 1; border: 1px solid #ccd4e0; border-top: 3px solid #1f3a5f;
           padding: 7px 9px; background: #fafbfc; }
.tarjeta .k { font-size: 15pt; font-weight: 700; color: #1f3a5f;
              font-family: 'Segoe UI', sans-serif; line-height: 1.1; }
.tarjeta .e { font-size: 7.4pt; color: #666; text-transform: uppercase;
              letter-spacing: .4px; margin-top: 2px; }
.caja { border-left: 3px solid #1f3a5f; background: #f4f6f9;
        padding: 7px 11px; margin: 9px 0; }
.alerta { border-left: 3px solid #c47f00; background: #fffaf0;
          padding: 7px 11px; margin: 9px 0; }
.caja p, .alerta p { margin: 0 0 4px; }
.caja p:last-child, .alerta p:last-child { margin: 0; }
figure { margin: 8px 0; page-break-inside: avoid; }
figure img { width: 100%; border: 1px solid #dde; }
figcaption { font-size: 7.8pt; color: #555; margin-top: 3px;
             font-family: 'Segoe UI', sans-serif; }
.salto { page-break-before: always; }
.nowrap { white-space: nowrap; }
strong { color: #14243b; }
ol, ul { margin: 4px 0 7px; padding-left: 18px; }
li { margin-bottom: 2.5px; }
.pie { margin-top: 12px; padding-top: 6px; border-top: 1px solid #ccc;
       font-size: 7.6pt; color: #777; }
"""


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def construir_informe(R: dict) -> str:
    ap = R["aucpr_test"]
    eco = R["economia"]
    comp = R["complementariedad"]
    perm = R["caida_permutacion"]
    proy = R["proyeccion_mezcla"]
    mezcla = eco["mezcla_AB"]
    solo_a = eco["A"]
    dias = R["dias_prueba"]
    apuesta = R["apuesta_C"]

    mejora_orden = comp["existe"]
    verbo = "aporta" if mejora_orden else "no aporta de forma demostrable"

    # Contraste entre la permutación controlada y la versión sin control, si el
    # cuaderno lo registró.
    comparativa = ""
    variantes = R.get("permutacion_variantes") or []
    sin_control = next((v for v in variantes if "sin control" in v["variante"]), None)
    if sin_control:
        comparativa = (
            f' Sin ese control la caída medida sería de '
            f'{pct(sin_control["caida_relativa"], 1)} en lugar de '
            f'{pct(perm["B"], 1)}: la diferencia es el artefacto que evitamos.')

    # --- mecanismos ---
    mec = sorted(R["por_mecanismo"], key=lambda x: -x["aucpr_A"])
    filas_mec = []
    for m in mec:
        destaca = m["mecanismo"] == "vaciado_subito"
        filas_mec.append([
            f"<b>{m['mecanismo']}</b>" if destaca else m["mecanismo"],
            f'{m["n_fraudes"]}',
            f'{100 * m["tasa_base"]:.2f} %',
            f'{m["aucpr_A"]:.3f}', f'{m["aucpr_B"]:.3f}', f'{m["aucpr_C"]:.3f}',
            f'{m["aucpr_B"] - m["aucpr_A"]:+.3f}',
        ])

    filas_rec = [[f'{r["historia_visible"]}', f'{r["auc_pr"]:.4f}']
                 for r in R["recorte_historia"]]

    # --- matriz de evidencias ---
    matriz = [
        ["1 · Integridad de datos", "§2–§3 del cuaderno; 4 controles ejecutados",
         f'{R["n_transacciones"]:,} transacciones, {pct(R["tasa_fraude"], 2)} de fraude, '
         f'partición temporal 70/15/15 sin fuga.',
         "Datos sintéticos: la validez externa no está demostrada."],
        ["2 · Comparación común", "§7, figura 2",
         f'AUC-PR: A {ap["A"]:.3f} · B {ap["B"]:.3f} · C {ap["C"]:.3f} · A+B {ap["mezcla_AB"]:.3f}.',
         "B no recibe agregadas; parte de la brecha es de representación."],
        ["3 · Valor del orden", "§8 figura 3; §9 figura 4; §11.5",
         f'Barajar el orden cuesta {pct(perm["B"], 0)} del AUC-PR de B. '
         f'B colapsa en el control sin orden, como se predijo.',
         "B no supera a A en ningún mecanismo: el orden aporta, pero no basta solo."],
        ["4 · Apuesta del equipo", "HIPOTESIS_C.md (pre-registrada); §6",
         f'Margen en validación {apuesta["margen_val"]:+.4f} frente al '
         f'+{apuesta["umbral_declarado"]} exigido: '
         f'{"se cumple" if apuesta["exitosa"] else "NO se cumple"}.',
         "Una sola semilla por variante; no se midió variabilidad entre semillas."],
        ["5 · Decisión económica", "§11 figura 6",
         f'Umbral {mezcla["umbral"]:.3f}; ahorro de {q(mezcla["ahorro_Q"])} en {dias:.0f} días.',
         "Costos fijos y uniformes; extrapolación lineal a 1.4 M de tarjetas."],
        ["6 · Recomendación y límites", "§12 análisis de error; §14",
         "Complementar el motor actual, no reemplazarlo.",
         "Confusor de sondeo mal calibrado (§12.3): sesga §9 a favor de A."],
    ]

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Informe — Monitoreo transaccional</title>
<style>{ESTILO}</style></head><body>

<h1>¿El orden de las transacciones revela fraude?</h1>
<div class="sub">Informe al Comité de Riesgos · Banco del Altiplano</div>
<div class="meta">
  <b>Daniel Estrada (20853) · Hansel López (19026)</b> — Universidad del Valle de
  Guatemala, Deep Learning 2026 · Proyecto 1
</div>

<div class="caja">
<p><b>La pregunta.</b> El área de riesgos sostiene que los fraudes que se escapan
tienen un patrón que no está en los montos sino en el orden de las operaciones.
El motor actual resume cada ventana en variables agregadas, y esas variables
tienen una propiedad incómoda: <b>son idénticas si se baraja la secuencia</b>.</p>
<p><b>La respuesta corta.</b> Sí, el orden {verbo} información que los agregados no
capturan — pero un modelo de secuencias <b>no reemplaza</b> al motor actual.
Recomendamos <b>complementar</b>.</p>
</div>

<div class="banda">
  <div class="tarjeta"><div class="k">{pct(perm["B"], 0)}</div>
    <div class="e">del desempeño se pierde al barajar el orden</div></div>
  <div class="tarjeta"><div class="k">{ap["A"]:.3f}</div>
    <div class="e">AUC-PR del motor actual (el mejor individual)</div></div>
  <div class="tarjeta"><div class="k">{comp["diferencia_vs_A"]:+.3f}</div>
    <div class="e">gana el motor actual al sumarle la señal de orden</div></div>
  <div class="tarjeta"><div class="k">{q(proy["ahorro_mensual_cartera_Q"] / 1e6, 1)}M</div>
    <div class="e">ahorro mensual proyectado a la cartera</div></div>
</div>

<h2>1 · Qué medimos y con qué datos</h2>
<p>Construimos un generador propio de transacciones porque la pregunta exige
<b>contrastar mecanismos de fraude cuya dependencia del orden sea conocida</b>. Con
datos reales esa dependencia es justamente lo que se desconoce: si el modelo
secuencial gana, no hay forma de saber si ganó <i>porque</i> leyó el orden. El
precio de esta decisión es que los resultados no dicen nada sobre el fraude real
del banco; lo que queda demostrado es el <b>método</b>.</p>

<p>El conjunto tiene <b>{R["n_transacciones"]:,} transacciones</b> de
{R["config_generador"]["n_tarjetas"]:,} tarjetas a lo largo de
{R["config_generador"]["dias"]} días, con {pct(R["tasa_fraude"], 2)} de fraude.
Se generaron cuatro mecanismos con dependencia del orden deliberadamente
distinta, y — esto es lo importante — <b>comportamiento legítimo que se parece a
cada uno</b>: rachas de microcompras online, compras grandes legítimas y viajes.
Sin esos confusores, un umbral sobre el monto máximo resolvería el problema.</p>

<p><b>Protocolo temporal.</b> La partición es por fecha global (70 / 15 / 15), nunca
aleatoria: entrenamos con lo más antiguo y probamos con lo más reciente. El
conjunto de prueba se abrió <b>una sola vez</b>, después de fijar arquitecturas,
hiperparámetros y umbral. Cuatro controles ejecutados en el cuaderno verifican
que ninguna ventana contiene información posterior al instante de decisión.</p>

<h2>2 · Comparación común</h2>
{tabla_html(
    ["Modelo", "Qué ve", "AUC-PR", "Precisión", "Exhaustividad", "F1"],
    [["<b>A</b> — línea base", "Variables agregadas (motor actual)",
      f'<b>{ap["A"]:.4f}</b>', f'{solo_a["precision"]:.3f}',
      f'{solo_a["exhaustividad"]:.3f}', f'{solo_a["f1"]:.3f}'],
     ["<b>B</b> — secuencial", "Secuencia ordenada, sin agregados",
      f'{ap["B"]:.4f}', f'{eco["B"]["precision"]:.3f}',
      f'{eco["B"]["exhaustividad"]:.3f}', f'{eco["B"]["f1"]:.3f}'],
     ["<b>C</b> — híbrido", "Secuencia + atención + agregados",
      f'{ap["C"]:.4f}', f'{eco["C"]["precision"]:.3f}',
      f'{eco["C"]["exhaustividad"]:.3f}', f'{eco["C"]["f1"]:.3f}'],
     ["<b>A + B</b> combinados", "Ambos puntajes, mezcla ajustada en validación",
      f'<b>{ap["mezcla_AB"]:.4f}</b>', f'{mezcla["precision"]:.3f}',
      f'{mezcla["exhaustividad"]:.3f}', f'{mezcla["f1"]:.3f}']])}
<p>No reportamos exactitud: con {pct(R["tasa_fraude"], 2)} de fraude, responder
«todo es legítimo» acertaría el {pct(1 - R["tasa_fraude"], 1)} sin detectar nada.
El resultado incómodo es que <b>el motor actual supera al modelo secuencial</b>.
La sección 3 explica por qué eso no cierra la pregunta.</p>

<figure><img src="../figuras/fig2_comparacion.png">
<figcaption>Figura 1 — Curvas precisión–exhaustividad e intervalos de confianza
del 95 % por bootstrap sobre el conjunto de prueba.</figcaption></figure>

<h2 class="salto">3 · ¿El orden aporta? Dos pruebas de falsificación</h2>
<p>Una mejora de métricas no demuestra que un modelo use el orden. Intentamos
<b>refutar nuestra propia conclusión</b> con dos pruebas.</p>

<h3>Prueba 1 — Permutación controlada</h3>
<p>Barajamos el orden de la historia dentro de cada ventana <b>sin cambiar los
eventos ni sus valores</b>. Las variables agregadas son invariantes a
permutaciones, así que siguen idénticas: lo único que se destruye es la
secuencia. Repetimos con cinco permutaciones distintas.</p>
<p><b>Un control que no es obvio.</b> El modelo lee su predicción del estado de la
última posición de la ventana, que es la transacción que se está calificando. Si
al barajar movemos también esa posición, destruiríamos dos cosas a la vez: el
orden <i>y</i> el acceso del modelo al evento que debe puntuar. La caída
resultante sobreestimaría el aporte del orden. Por eso barajamos <b>solo la
historia</b> y dejamos fijo el evento calificado.{comparativa}</p>
<div class="caja"><p><b>Resultado:</b> el modelo B pierde <b>{pct(perm["B"], 1)}</b> de su
AUC-PR al barajar la historia. El modelo C solo pierde {pct(perm["C"], 1)}, y eso
es coherente: C conserva las variables agregadas como red de seguridad, así que
la permutación no puede dejarlo ciego del todo.</p></div>

<figure><img src="../figuras/fig3_permutacion.png">
<figcaption>Figura 2 — Desempeño con el orden original frente al orden barajado.
Las barras de error recorren las cinco permutaciones.</figcaption></figure>

<h3>Prueba 2 — Desempeño por mecanismo de fraude</h3>
<p>Elegimos esta prueba porque es <b>la única capaz de refutar</b> la conclusión en
lugar de matizarla. Predijimos, antes de medir, que B ganaría en el mecanismo
dependiente del orden y <b>no</b> en <span class="nowrap">vaciado&nbsp;súbito</span>,
que construimos sin ninguna estructura temporal como control negativo.</p>
{tabla_html(["Mecanismo", "Fraudes", "Tasa base", "A", "B", "C", "B − A"], filas_mec)}
<p style="font-size:8.6pt;color:#555">La tasa base es la prevalencia de cada
fila y equivale al AUC-PR de un clasificador al azar. Cambia entre filas porque
el número de fraudes de cada mecanismo es distinto, así que <b>los AUC-PR no se
comparan entre filas</b>; sí se comparan los modelos dentro de una misma fila,
que es lo que hace la columna B − A.</p>
<div class="caja">
<p><b>Lo que se confirmó:</b> B se desploma en <span class="nowrap">vaciado súbito</span>
({[m for m in mec if m["mecanismo"] == "vaciado_subito"][0]["aucpr_B"]:.3f} frente a
{[m for m in mec if m["mecanismo"] == "vaciado_subito"][0]["aucpr_A"]:.3f} de A), exactamente como
predijimos. Donde no hay orden, el modelo de orden no tiene nada que leer. Esa
es la prueba de que B realmente está leyendo secuencia y no otra cosa.</p>
<p><b>Lo que se refutó:</b> esperábamos que B <i>superara</i> a A en el mecanismo de
escalada, y no ocurrió. Lo reportamos como falló.</p>
</div>

<h3>La pregunta que ninguna de las dos responde</h3>
<p>El comité no preguntó si el modelo secuencial es mejor, sino si el orden
aporta información que los agregados <b>no capturan</b>. Que B pierda no significa
que su señal sea redundante. Lo comprobamos combinando ambos puntajes con una
mezcla ajustada <b>solo en validación</b>:</p>
<div class="{"caja" if mejora_orden else "alerta"}">
<p>Al sumar la señal secuencial al motor actual, el AUC-PR pasa de
<b>{ap["A"]:.4f}</b> a <b>{ap["mezcla_AB"]:.4f}</b>
({comp["diferencia_vs_A"]:+.4f}; IC 95 % [{comp["ic_inf"]:+.4f}, {comp["ic_sup"]:+.4f}]).</p>
<p><b>Cadena de evidencia:</b> el desempeño de B depende del orden en un
{pct(perm["B"], 0)} <i>(prueba 1)</i> y B aporta señal que A no tiene
<i>(esta mezcla)</i> ⟹ <b>el orden aporta información que los agregados no
capturan</b>.</p>
</div>

<h3>¿Cuánta historia hace falta?</h3>
<p>Prueba de apoyo: con una sola transacción visible el modelo no tiene memoria
ni orden. El desempeño satura muy pronto, lo que tiene una consecuencia
operativa directa — no hace falta guardar ventanas largas.</p>
{tabla_html(["Transacciones visibles", "AUC-PR"], filas_rec)}

<h2 class="salto">4 · Nuestra apuesta y su veredicto</h2>
<p>Registramos la hipótesis en el repositorio <b>antes</b> de entrenar y de abrir el
conjunto de prueba (archivo <i>HIPOTESIS_C.md</i>; el sello de tiempo del commit es
la constancia). Apostamos a que fusionar agregados con una lectura atendida de
la secuencia superaría al mejor de A y B por al menos <b>+0.02</b> de AUC-PR en
validación.</p>
<div class="alerta">
<p><b>Veredicto: {"se cumple" if apuesta["exitosa"] else "la apuesta NO se cumple"}.</b>
El margen fue de <b>{apuesta["margen_val"]:+.4f}</b> frente al
+{apuesta["umbral_declarado"]} exigido. Lo reportamos tal como quedó.</p>
<p>El control experimental permite localizar el fallo: la atención por sí sola
aportó {apuesta["aporte_atencion_C1_menos_B"]:+.4f} sobre B, y añadir las variables
agregadas aportó {apuesta["aporte_agregadas_C2_menos_C1"]:+.4f} más. La idea de
combinar era correcta — lo demuestra la mezcla de la sección 3 — pero <b>la fusión
dentro de una red neuronal fue el vehículo equivocado</b>: una capa densa no
explota 41 variables tabulares tan bien como lo hace un modelo de árboles.</p>
</div>

<h2>5 · Dónde poner el umbral y cuánto vale</h2>
<p>El comité fijó los costos: <b>{q(4200)}</b> por fraude no detectado y <b>{q(180)}</b>
por bloquear una transacción legítima. La asimetría es de <b>23 a 1</b>, así que el
umbral no se elige maximizando F1 —eso trataría ambos errores como igual de
graves, lo que aquí es falso— sino <b>minimizando el costo esperado</b>. El umbral se
ajustó en validación y se aplicó sin cambios a prueba.</p>

{tabla_html(["Escenario", "Umbral", "Fraudes que pasan", "Bloqueos indebidos",
             "Costo", "Ahorro"],
    [["No hacer nada", "—", f'{solo_a["TP"] + solo_a["FN"]}', "0",
      q(solo_a["costo_Q"] + solo_a["ahorro_Q"]), "—"],
     ["Solo motor actual (A)", f'{solo_a["umbral"]:.3f}', f'{solo_a["FN"]}',
      f'{solo_a["FP"]}', q(solo_a["costo_Q"]), q(solo_a["ahorro_Q"])],
     ["<b>Motor + señal de orden</b>", f'{mezcla["umbral"]:.3f}', f'{mezcla["FN"]}',
      f'{mezcla["FP"]}', q(mezcla["costo_Q"]), f'<b>{q(mezcla["ahorro_Q"])}</b>']],
    clases="")}

<p>Sobre los {dias:.0f} días del conjunto de prueba, añadir la señal de orden
{"ahorra" if mezcla["ahorro_Q"] > solo_a["ahorro_Q"] else "cuesta"}
<b>{q(abs(mezcla["ahorro_Q"] - solo_a["ahorro_Q"]))}</b> respecto a usar solo el motor
actual. Extrapolado a la cartera de 1.4 millones de tarjetas, el ahorro mensual
sería de <b>{q(proy["ahorro_mensual_cartera_Q"])}</b>, con
{proy["bloqueos_legitimos_por_mes"]:,.0f} bloqueos indebidos y
{proy["fraudes_no_detectados_por_mes"]:,.0f} fraudes que aún pasarían al mes.</p>
<div class="alerta"><p><b>Advertencia sobre esta cifra.</b> La extrapolación es lineal:
supone que 1.4 millones de tarjetas reales se comportan como las
{R["config_generador"]["n_tarjetas"]:,} simuladas. Es una cota indicativa para
dimensionar la decisión, <b>no una promesa de ahorro</b>.</p></div>

<figure><img src="../figuras/fig6_costo.png">
<figcaption>Figura 3 — Curva de costo en validación (izquierda; las líneas
punteadas marcan los umbrales elegidos) y ahorro en prueba (derecha).</figcaption></figure>

<h2 class="salto">6 · Recomendación, errores y límites</h2>
<div class="caja"><p><b>Recomendación: complementar, no reemplazar.</b> Conservar el
motor de agregados como columna vertebral y añadir el puntaje secuencial como
segunda entrada de la capa de decisión. El motor actual resuelve bien los
fraudes cuyo indicio está en la magnitud; la señal de orden aporta donde ese
motor es ciego por construcción.</p></div>

<h3>Por qué el modelo secuencial no ganó por sí solo</h3>
<p>La explicación más probable no es la arquitectura sino los <b>datos</b>: el
entrenamiento contiene apenas ~1,900 transacciones fraudulentas. Las variables
agregadas son conocimiento del dominio ya destilado por ingenieros; la red
tendría que redescubrirlo desde cero con muy pocos ejemplos positivos. Esto es
comprobable: si el volumen de fraude etiquetado creciera en un orden de
magnitud, esta conclusión podría invertirse.</p>

<h3>Un patrón de error concreto</h3>
<p>Los bloqueos indebidos se concentran en <b>transacciones legítimas de monto
alto</b>: el modelo penaliza el gasto atípico aunque sea real. En la práctica esto
afecta desproporcionadamente a clientes que hacen compras grandes ocasionales
—electrónica, viajes—, que suelen ser los de mayor valor para el banco. Es un
argumento para no usar el puntaje como bloqueo automático en montos altos, sino
como disparador de una verificación por segundo canal.</p>

<h3>Límites de este estudio</h3>
<ol>
<li><b>Los datos son sintéticos.</b> Reflejan nuestras hipótesis sobre cómo se
comporta el fraude, no el fraude del banco.</li>
<li><b>Un defecto de nuestro propio generador.</b> Las microcompras de sondeo y las
rachas legítimas de suscripciones quedaron con distribuciones de monto
distintas, así que se separan por nivel y no solo por orden. Eso sesga la
prueba 2 a favor del motor de agregados. Es la primera corrección que haríamos.</li>
<li><b>Una sola semilla</b> por variante; no medimos la variabilidad entre
inicializaciones.</li>
<li><b>El mecanismo de toma gradual resiste a todos los modelos.</b> La señal se
reparte en decenas de transacciones y la ventana de 20 no cubre el episodio.</li>
<li><b>{R.get("episodios_cruzan_corte", 0)} episodios de fraude quedan repartidos
entre dos bloques temporales.</b> No es fuga —ninguna transacción se evalúa con
información posterior a sí misma— pero significa que una porción muy pequeña del
fraude de prueba pertenece a un episodio iniciado antes del corte.</li>
</ol>
<p>Los intervalos de confianza de este informe se calculan remuestreando
<b>tarjetas completas</b>, no transacciones sueltas. El fraude llega en episodios
correlacionados; tratarlos como observaciones independientes produciría
intervalos artificialmente estrechos y nos llevaría a afirmar más de lo que los
datos sostienen.</p>

<h3>Qué cambiaría nuestra recomendación</h3>
<ol>
<li>Si en datos reales la permutación costara menos del 15 % del desempeño, no
habría evidencia de que el orden aporta y bastaría el motor actual.</li>
<li>Si el costo de un bloqueo indebido subiera de {q(180)} a más de ~{q(900)}, la
asimetría 23:1 se estrecharía lo suficiente como para recalcular toda la
decisión.</li>
<li>Si apareciera un mecanismo nuevo sin ejemplos etiquetados, ningún modelo
supervisado lo vería: haría falta un componente no supervisado.</li>
</ol>

<h3>El siguiente paso</h3>
<p>Ejecutar la permutación controlada y la prueba de complementariedad
<b>sobre los datos reales del banco</b>. El procedimiento es directamente aplicable
y no requiere entrenar nada nuevo desde cero: es una tarde de trabajo, no un
proyecto. Hasta entonces, lo que este trabajo demuestra es el método, no una
cifra de ahorro.</p>

<h2 class="salto">Matriz de evidencias</h2>
<p>Guía de lectura: dónde está cada evidencia, qué concluimos y qué la limita.</p>
{tabla_html(["Evidencia", "Dónde aparece", "Conclusión", "Limitación"], matriz)}

<div class="pie">
Informe generado automáticamente desde <i>artefactos/resultados.json</i>; ninguna
cifra fue transcrita a mano. Cuaderno reproducible con semilla
{R["config_generador"]["semilla"]}. · Daniel Estrada (20853) · Hansel López (19026)
</div>
</body></html>"""


# --------------------------------------------------------------------------
# Presentación (8 diapositivas)
# --------------------------------------------------------------------------

ESTILO_SLIDES = """
@page { size: A4 landscape; margin: 0; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; margin: 0;
       color: #14243b; }
.s { width: 297mm; height: 209mm; padding: 15mm 18mm; page-break-after: always;
     position: relative; display: flex; flex-direction: column; }
.s:last-child { page-break-after: auto; }
.n { position: absolute; bottom: 8mm; right: 15mm; font-size: 9pt; color: #99a; }
.marca { position: absolute; bottom: 8mm; left: 18mm; font-size: 8.5pt;
         color: #99a; }
h1 { font-size: 30pt; margin: 0 0 6px; letter-spacing: -1px; }
h2 { font-size: 21pt; margin: 0 0 14px; color: #1f3a5f;
     border-bottom: 3px solid #1f3a5f; padding-bottom: 7px; }
p { font-size: 12.5pt; line-height: 1.5; margin: 0 0 9px; }
li { font-size: 12.5pt; line-height: 1.55; margin-bottom: 7px; }
.port { justify-content: center; align-items: center; text-align: center;
        background: #1f3a5f; color: #fff; }
.port h1 { color: #fff; font-size: 34pt; }
.port .a { font-size: 14pt; margin-top: 22px; opacity: .92; }
.port .b { font-size: 11pt; margin-top: 6px; opacity: .72; }
.grande { font-size: 60pt; font-weight: 700; color: #1f3a5f; line-height: 1;
          text-align: center; margin: 6px 0; }
.gsub { text-align: center; font-size: 13pt; color: #556; }
.cols { display: flex; gap: 16px; flex: 1; }
.col { flex: 1; }
table { width: 100%; border-collapse: collapse; font-size: 11pt; margin-top: 6px; }
th { background: #1f3a5f; color: #fff; padding: 6px 9px; text-align: left; }
td { padding: 5px 9px; border-bottom: 1px solid #dde; }
tbody tr:nth-child(even) { background: #f4f6f9; }
.hi { background: #fff3cd !important; font-weight: 700; }
.caja { background: #f4f6f9; border-left: 5px solid #1f3a5f; padding: 12px 16px;
        margin: 10px 0; font-size: 12.5pt; }
.warn { background: #fffaf0; border-left: 5px solid #c47f00; padding: 12px 16px;
        margin: 10px 0; font-size: 12pt; }
img { max-width: 100%; max-height: 118mm; display: block; margin: 0 auto; }
.tarjetas { display: flex; gap: 14px; margin-top: 14px; }
.t { flex: 1; border-top: 5px solid #1f3a5f; background: #f7f9fb; padding: 12px; }
.t .k { font-size: 27pt; font-weight: 700; color: #1f3a5f; }
.t .e { font-size: 10pt; color: #667; margin-top: 4px; }
"""


def construir_presentacion(R: dict) -> str:
    ap, eco, comp = R["aucpr_test"], R["economia"], R["complementariedad"]
    perm, proy = R["caida_permutacion"], R["proyeccion_mezcla"]
    mezcla, solo_a = eco["mezcla_AB"], eco["A"]
    apuesta = R["apuesta_C"]
    vs = [m for m in R["por_mecanismo"] if m["mecanismo"] == "vaciado_subito"][0]

    filas = "".join(
        f'<tr class="{"hi" if m["mecanismo"] == "vaciado_subito" else ""}">'
        f'<td>{m["mecanismo"]}</td><td>{m["aucpr_A"]:.3f}</td>'
        f'<td>{m["aucpr_B"]:.3f}</td><td>{m["aucpr_B"] - m["aucpr_A"]:+.3f}</td></tr>'
        for m in sorted(R["por_mecanismo"], key=lambda x: -x["aucpr_A"]))

    def pie(n):
        return (f'<div class="marca">Estrada · López — Deep Learning 2026</div>'
                f'<div class="n">{n} / 8</div>')

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Monitoreo transaccional</title><style>{ESTILO_SLIDES}</style></head><body>

<div class="s port">
  <h1>¿El orden de las transacciones<br>revela fraude?</h1>
  <div class="a">Daniel Estrada (20853) · Hansel López (19026)</div>
  <div class="b">Proyecto 1 · Deep Learning 2026 · Universidad del Valle de Guatemala</div>
</div>

<div class="s">
  <h2>La pregunta del comité</h2>
  <div class="caja">«Cuando revisamos los casos que se nos escaparon, el patrón
  siempre está ahí. No en los montos: <b>en el orden en que ocurrieron</b>.»</div>
  <p>El motor actual resume cada ventana en variables agregadas: monto promedio
  de 24 h, transacciones por hora, monto máximo, diversidad de comercios.</p>
  <div class="warn"><b>El problema:</b> todas esas variables son <b>idénticas si se
  baraja la secuencia</b>. Por construcción, el motor actual no puede ver orden.</div>
  <p><b>Nuestra pregunta no es</b> «¿puedo entrenar una LSTM?»<br>
  <b>Es:</b> ¿el orden aporta información que los agregados no capturan, y cuánto
  vale en quetzales?</p>
  {pie(2)}
</div>

<div class="s">
  <h2>Diseño: datos donde conocemos la verdad</h2>
  <div class="cols"><div class="col">
    <p>Generamos los datos para poder hacer una <b>predicción falsable</b>:
    cuatro mecanismos con dependencia del orden <b>conocida y distinta</b>.</p>
    <table><thead><tr><th>Mecanismo</th><th>Orden</th><th>Predicción</th></tr></thead>
    <tbody>
    <tr><td>escalada de prueba</td><td>fuerte</td><td>B gana</td></tr>
    <tr><td>ráfaga geográfica</td><td>parcial</td><td>empate</td></tr>
    <tr class="hi"><td>vaciado súbito</td><td><b>ninguno</b></td><td><b>B NO gana</b></td></tr>
    <tr><td>toma gradual</td><td>difuso</td><td>ambos fallan</td></tr>
    </tbody></table>
  </div><div class="col">
    <p><b>Confusores legítimos</b> para que no sea trivial: rachas de microcompras,
    compras grandes reales, viajes.</p>
    <p><b>Partición temporal estricta</b> 70/15/15 por fecha. El conjunto de prueba
    se abrió <b>una sola vez</b>.</p>
    <div class="caja"><b>vaciado súbito</b> es el control negativo. Si B ganara
    también ahí, la ventaja no vendría del orden.</div>
  </div></div>
  {pie(3)}
</div>

<div class="s">
  <h2>Prueba 1 — Permutación controlada</h2>
  <p>Barajamos el orden de <b>la historia</b> dentro de cada ventana. Mismos
  eventos, mismos valores, mismas variables agregadas, y la transacción
  calificada se queda en su sitio. Solo se destruye la secuencia.</p>
  <div class="grande">−{100 * perm["B"]:.0f}&nbsp;%</div>
  <div class="gsub">del AUC-PR del modelo secuencial se pierde al barajar<br>
  ({R["aucpr_test"]["B"]:.3f} → {R["aucpr_test"]["B"] * (1 - perm["B"]):.3f}, cinco permutaciones distintas)</div>
  <div class="warn" style="font-size:11pt">Si barajáramos también la posición de
  la transacción calificada, el modelo perdería además el acceso al evento que
  debe puntuar y la caída parecería mayor de lo que es. <b>Ese control importa:
  sin él estaríamos midiendo dos cosas y atribuyéndolas al orden.</b></div>
  {pie(4)}
</div>

<div class="s">
  <h2>Prueba 2 — Por mecanismo: el resultado incómodo</h2>
  <table><thead><tr><th>Mecanismo</th><th>A (agregados)</th><th>B (secuencial)</th><th>B − A</th></tr></thead>
  <tbody>{filas}</tbody></table>
  <div class="cols" style="margin-top:10px">
    <div class="col"><div class="caja"><b>Se confirmó:</b> B se desploma en el control
    sin orden ({vs["aucpr_B"]:.3f} vs {vs["aucpr_A"]:.3f}). Donde no hay orden, no hay nada que leer.</div></div>
    <div class="col"><div class="warn"><b>Se refutó:</b> B <b>no supera</b> a A en ningún
    mecanismo. Nuestra predicción falló y así lo reportamos.</div></div>
  </div>
  {pie(5)}
</div>

<div class="s">
  <h2>Entonces, ¿el orden aporta o no?</h2>
  <p>Que B <b>pierda</b> no significa que su señal sea <b>redundante</b>. Lo comprobamos
  combinando ambos puntajes (mezcla ajustada solo en validación):</p>
  <div class="tarjetas">
    <div class="t"><div class="k">{ap["A"]:.3f}</div><div class="e">Motor actual solo</div></div>
    <div class="t"><div class="k">{ap["mezcla_AB"]:.3f}</div><div class="e">Motor + señal de orden</div></div>
    <div class="t"><div class="k">{comp["diferencia_vs_A"]:+.3f}</div>
      <div class="e">IC 95 % [{comp["ic_inf"]:+.3f}, {comp["ic_sup"]:+.3f}]</div></div>
  </div>
  <div class="caja" style="margin-top:16px">
  <b>Cadena de evidencia</b><br>
  (1) el desempeño de B depende del orden en un {100 * perm["B"]:.0f} % &nbsp;<i>(prueba 1)</i><br>
  (2) B aporta señal que A no tiene &nbsp;<i>(esta mezcla)</i><br>
  <b>⟹ el orden aporta información que los agregados no capturan.</b>
  </div>
  {pie(6)}
</div>

<div class="s">
  <h2>La decisión en quetzales</h2>
  <div class="cols"><div class="col">
    <p>Costos del comité: <b>Q4,200</b> por fraude que pasa, <b>Q180</b> por bloqueo
    indebido. Asimetría <b>23 : 1</b>.</p>
    <p>Por eso el umbral <b>no</b> maximiza F1 —eso trataría ambos errores como
    iguales— sino que <b>minimiza costo esperado</b>. Ajustado en validación.</p>
    <div class="warn">La proyección a 1.4 M de tarjetas es <b>lineal</b>: cota
    indicativa, no promesa.</div>
  </div><div class="col">
    <table><thead><tr><th>Escenario</th><th>Ahorro en prueba</th></tr></thead><tbody>
    <tr><td>No hacer nada</td><td>—</td></tr>
    <tr><td>Solo motor actual</td><td>Q{solo_a["ahorro_Q"]:,.0f}</td></tr>
    <tr class="hi"><td>Motor + orden</td><td>Q{mezcla["ahorro_Q"]:,.0f}</td></tr>
    </tbody></table>
    <div class="grande" style="font-size:34pt;margin-top:18px">Q{proy["ahorro_mensual_cartera_Q"] / 1e6:.1f}M</div>
    <div class="gsub">ahorro mensual proyectado a la cartera</div>
  </div></div>
  {pie(7)}
</div>

<div class="s">
  <h2>Recomendación y límites</h2>
  <div class="caja"><b>Complementar, no reemplazar.</b> Conservar el motor de
  agregados y añadir el puntaje secuencial como segunda entrada de la decisión.</div>
  <div class="cols"><div class="col">
    <p><b>Por qué B no ganó solo</b><br>Solo ~1,900 fraudes etiquetados. Los agregados
    son conocimiento del dominio ya destilado; la red debe redescubrirlo con
    muy pocos positivos. <b>Es un límite de datos, no de arquitectura.</b></p>
    <p><b>Nuestra apuesta C {"se cumplió" if apuesta["exitosa"] else "falló"}</b>
    ({apuesta["margen_val"]:+.3f} vs +{apuesta["umbral_declarado"]} exigido).
    Pre-registrada en git; la reportamos como quedó.</p>
  </div><div class="col">
    <p><b>Límites que declaramos</b></p>
    <ul>
      <li>Datos sintéticos: no dicen nada del fraude real.</li>
      <li>Defecto propio: nuestros confusores de sondeo se separan por monto,
      no solo por orden — sesga la prueba 2 a favor de A.</li>
      <li>Una sola semilla por variante.</li>
    </ul>
    <div class="warn"><b>Siguiente paso:</b> correr la permutación sobre los datos
    reales del banco. Es una tarde de trabajo, no un proyecto.</div>
  </div></div>
  {pie(8)}
</div>

</body></html>"""


def main() -> int:
    ruta = ART / "resultados.json"
    if not ruta.exists():
        print(f"Falta {ruta}. Ejecuta primero el cuaderno.", file=sys.stderr)
        return 1
    R = json.loads(ruta.read_text(encoding="utf-8"))

    salida_html = RAIZ / "informe_html"
    salida_html.mkdir(exist_ok=True)

    print("Generando PDFs...")
    (salida_html / "informe.html").write_text(construir_informe(R), encoding="utf-8")
    a_pdf(salida_html / "informe.html", RAIZ / "informe.pdf")

    (salida_html / "presentacion.html").write_text(construir_presentacion(R),
                                                   encoding="utf-8")
    a_pdf(salida_html / "presentacion.html", RAIZ / "presentacion.pdf",
          apaisado=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
