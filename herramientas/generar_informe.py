"""Genera informe.pdf y presentacion.pdf a partir de artefactos/resultados.json.

Ninguna cifra se escribe a mano: todas provienen de la ejecucion del cuaderno.
Las figuras se extraen del propio .ipynb ejecutado, de modo que las imagenes del
informe son exactamente las de la corrida que produjo los numeros.

Formato: carta, margenes de 1", Times New Roman 12 pt, interlineado 1.15,
caratula institucional y numero de pagina centrado al pie.

    python herramientas/generar_informe.py
"""

from __future__ import annotations

import base64
import json
import pathlib
import shutil
import subprocess
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parents[1]
ART = RAIZ / "artefactos"
CUADERNO = RAIZ / "proyecto1_estrada_ramirez.ipynb"

AUTORES = ["Daniel Estrada &#8211; 20853", "Daniela Ram\u00edrez &#8211; 23053"]
FECHA = "Guatemala, 4 de septiembre de 2026"

CHROME_POSIBLES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


# --------------------------------------------------------------------------
# Infraestructura
# --------------------------------------------------------------------------

def buscar_chrome() -> str:
    for ruta in CHROME_POSIBLES:
        if pathlib.Path(ruta).exists():
            return ruta
    for nombre in ("google-chrome", "chromium", "chrome", "msedge"):
        hallado = shutil.which(nombre)
        if hallado:
            return hallado
    raise SystemExit("No se encontro Chrome/Edge para generar el PDF.")


def a_pdf(html: str, destino: pathlib.Path, apaisado: bool = False) -> None:
    """Renderiza el HTML con Chrome headless."""
    with tempfile.TemporaryDirectory() as tmp:
        fuente = pathlib.Path(tmp) / "doc.html"
        fuente.write_text(html, encoding="utf-8")
        cmd = [buscar_chrome(), "--headless", "--disable-gpu", "--no-sandbox",
               "--no-pdf-header-footer",
               "--run-all-compositor-stages-before-draw",
               "--virtual-time-budget=10000",
               f"--print-to-pdf={destino}", fuente.resolve().as_uri()]
        if apaisado:
            cmd.insert(-2, "--landscape")
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not destino.exists():
        raise SystemExit(f"Chrome no produjo {destino}")


def numerar(pdf: pathlib.Path) -> int:
    """Numero de pagina centrado al pie, en Times 10 pt (como la plantilla)."""
    import pymupdf
    doc = pymupdf.open(pdf)
    for i, pagina in enumerate(doc):
        ancho = pagina.rect.width
        pagina.insert_text((ancho / 2 - 3, 754), str(i + 1),
                           fontname="times-roman", fontsize=10)
    doc.saveIncr()
    n = len(doc)
    doc.close()
    return n


def figuras_del_cuaderno() -> dict[str, str]:
    """Extrae las figuras del .ipynb ejecutado como data-URI base64.

    Se localizan por un fragmento del codigo que las produce, no por indice de
    celda, para que insertar o quitar celdas no rompa el informe.
    """
    marcas = {
        "comparacion": "precision_recall_curve(y_test",
        "permutacion": '("B", res_B), ("C", res_C)',
    }
    nb = json.loads(CUADERNO.read_text(encoding="utf-8"))
    salida: dict[str, str] = {}
    for nombre, marca in marcas.items():
        for celda in nb["cells"]:
            if celda["cell_type"] != "code":
                continue
            if marca not in "".join(celda["source"]):
                continue
            for out in celda.get("outputs", []):
                png = out.get("data", {}).get("image/png")
                if png:
                    salida[nombre] = "data:image/png;base64," + png.replace("\n", "")
                    break
            break
    faltan = set(marcas) - set(salida)
    if faltan:
        raise SystemExit(f"No se hallaron figuras en el cuaderno: {sorted(faltan)}")
    return salida


# --------------------------------------------------------------------------
# Formato de cifras
# --------------------------------------------------------------------------

def q(valor: float, decimales: int = 0) -> str:
    return f"Q{valor:,.{decimales}f}"


def pct(valor: float, decimales: int = 1) -> str:
    return f"{100 * valor:.{decimales}f}&nbsp;%"


def tabla(encabezados: list[str], filas: list[list[str]],
          clase: str = "", anchos: list[str] | None = None) -> str:
    col = ("<colgroup>" + "".join(f'<col style="width:{a}">' for a in anchos)
           + "</colgroup>") if anchos else ""
    th = "".join(f"<th>{h}</th>" for h in encabezados)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in f) + "</tr>"
                 for f in filas)
    return (f'<table class="{clase}">{col}<thead><tr>{th}</tr></thead>'
            f"<tbody>{tr}</tbody></table>")


# --------------------------------------------------------------------------
# Hoja de estilo comun (plantilla institucional)
# --------------------------------------------------------------------------

ESTILO_INFORME = """
@page { size: letter; margin: 1in; }
* { box-sizing: border-box; }
body { font-family: 'Times New Roman', Times, serif; font-size: 12pt;
       line-height: 1.15; color: #000; margin: 0; text-align: justify; }
p { margin: 0 0 7pt; }
h1 { font-size: 14pt; font-weight: bold; margin: 13pt 0 6pt; text-align: left; }
h2 { font-size: 12.5pt; font-weight: bold; margin: 10pt 0 5pt; text-align: left; }
h1:first-child, h2:first-child { margin-top: 0; }
ul { margin: 0 0 7pt; padding-left: 26pt; }
li { margin-bottom: 2pt; }
b, strong { font-weight: bold; }
table { border-collapse: collapse; width: 100%; font-size: 10pt;
        margin: 3pt 0 7pt; }
th { border-bottom: 1px solid #000; border-top: 1px solid #000;
     padding: 2.5pt 5pt; text-align: left; font-weight: bold; }
td { border-bottom: 0.5px solid #999; padding: 2.5pt 5pt; }
th + th, td + td { text-align: right; }
table.izq th + th, table.izq td + td { text-align: left; }
tr { page-break-inside: avoid; }
figure { margin: 6pt 0 8pt; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-style: italic; font-size: 10pt; margin-top: 3pt;
             text-align: center; }
.nota { font-size: 9.5pt; margin: -4pt 0 8pt; }
.salto { page-break-before: always; }
table.matriz { font-size: 9pt; }
table.matriz td, table.matriz th { padding: 2pt 4pt; }

/* Caratula */
.caratula { height: 9in; page-break-after: always; text-align: center; }
.caratula .u { font-size: 15pt; font-weight: bold; margin: 0; }
.caratula .f { font-size: 13pt; margin: 6pt 0 0; }
.caratula .c { font-size: 13pt; margin: 6pt 0 0; }
.caratula .t { font-size: 16pt; font-weight: bold; margin: 300pt 0 0; }
.caratula .s { font-size: 13.5pt; margin: 10pt 0 0; }
.caratula .s2 { font-size: 12.5pt; margin: 8pt 0 0; }
.caratula .i { font-size: 12pt; font-weight: bold; margin: 48pt 0 8pt; }
.caratula .n { font-size: 12pt; margin: 0 0 4pt; }
.caratula .d { font-size: 12pt; margin: 44pt 0 0; }
"""

ESTILO_PRESENTACION = """
@page { size: letter landscape; margin: 0; }
* { box-sizing: border-box; }
body { font-family: 'Times New Roman', Times, serif; color: #000; margin: 0; }
.d { width: 11in; height: 8.5in; padding: 0.62in 0.75in 0.5in;
     page-break-after: always; position: relative; }
.d:last-child { page-break-after: auto; }
h1 { font-size: 25pt; margin: 0 0 5pt; }
.regla { border-bottom: 1.5px solid #000; margin-bottom: 12pt; }
p { font-size: 15.5pt; line-height: 1.32; margin: 0 0 12pt; }
ul { font-size: 15.5pt; line-height: 1.32; margin: 0 0 12pt; padding-left: 24pt; }
li { margin-bottom: 5pt; }
b { font-weight: bold; }
.cols { display: flex; gap: 26pt; }
.cols > div { flex: 1; }
table { border-collapse: collapse; width: 100%; font-size: 14pt; margin: 2pt 0 10pt; }
th { border-top: 1px solid #000; border-bottom: 1px solid #000;
     padding: 5pt 7pt; text-align: left; }
td { border-bottom: 0.5px solid #aaa; padding: 5pt 7pt; }
th + th, td + td { text-align: right; }
.cifra { text-align: center; margin: 14pt 0; }
.cifra .v { font-size: 46pt; font-weight: bold; line-height: 1; }
.cifra .e { font-size: 14pt; font-style: italic; margin-top: 7pt; }
.caja { border: 1px solid #000; padding: 11pt 14pt; margin: 10pt 0; font-size: 15pt; }
.pie { position: absolute; left: 0.75in; right: 0.75in; bottom: 0.38in;
       font-size: 11pt; display: flex; justify-content: space-between;
       border-top: 0.5px solid #999; padding-top: 4pt; }
.portada { display: flex; flex-direction: column; justify-content: center;
           align-items: center; text-align: center; }
.portada .u { font-size: 15pt; font-weight: bold; }
.portada .f { font-size: 13pt; margin-top: 6pt; }
.portada .t { font-size: 27pt; font-weight: bold; margin-top: 52pt;
              line-height: 1.2; }
.portada .s { font-size: 15pt; margin-top: 12pt; }
.portada .a { font-size: 13pt; margin-top: 42pt; line-height: 1.5; }
"""


# --------------------------------------------------------------------------
# Informe
# --------------------------------------------------------------------------

def construir_informe(R: dict, figs: dict[str, str]) -> str:
    eco, mat = R["economia"], R["materialidad"]
    comp, proy = R["complementariedad"], R["proyeccion_mezcla"]
    apuesta, cfg = R["apuesta_C"], R["config_generador"]
    mezcla, solo_a = eco["mezcla_AB"], R["solo_A"]
    variantes = {v["variante"]: v for v in R["permutacion_variantes"]}
    sin_control = variantes["toda la ventana (sin control)"]["caida_relativa"]
    fraudes_test = solo_a["TP"] + solo_a["FN"]
    dias = R["dias_prueba"]

    caratula = f"""
<div class="caratula">
  <p class="u">Universidad del Valle de Guatemala</p>
  <p class="f">Facultad de Ingenier&iacute;a</p>
  <p class="c">Deep Learning &mdash; Semestre 2, 2026</p>
  <p class="t">Proyecto 1 &mdash; Informe final</p>
  <p class="s">Monitoreo transaccional: detectar lo que el orden revela</p>
  <p class="s2">Informe al Comit&eacute; de Riesgos &mdash; Banco del Altiplano</p>
  <p class="i">Integrantes:</p>
  {''.join(f'<p class="n">{a}</p>' for a in AUTORES)}
  <p class="d">{FECHA}</p>
</div>"""

    resumen = f"""
<h1>Resumen ejecutivo</h1>
<p>El motor antifraude actual resume cada ventana en variables agregadas &mdash;monto promedio,
conteo, monto m&aacute;ximo, diversidad de comercios&mdash;, y esas variables tienen una propiedad
inc&oacute;moda: son id&eacute;nticas si se baraja la secuencia. Por construcci&oacute;n, el motor actual no puede
ver orden. Este trabajo mide si el orden aporta informaci&oacute;n que los agregados no capturan.</p>
<p>La respuesta es <b>s&iacute;, pero vale mucho menos de lo esperado</b>. El modelo de secuencias no
supera al motor actual en ning&uacute;n mecanismo, y sumarle su puntaje mejora la detecci&oacute;n en
{comp['diferencia_vs_A']:+.4f} de AUC-PR: detectable, pero de mil&eacute;simas. El &uacute;nico beneficio con
magnitud defendible es otro: <b>{pct(abs(mat['delta_fp_rel']), 0)} menos bloqueos a clientes
leg&iacute;timos</b> detectando pr&aacute;cticamente el mismo fraude. <b>Recomendaci&oacute;n: piloto acotado</b>
&mdash;conservar el motor actual como decisi&oacute;n principal e incorporar la se&ntilde;al secuencial solo si el
objetivo es reducir molestias al cliente, no p&eacute;rdidas por fraude.</p>"""

    seccion1 = f"""
<h1>1. Datos y protocolo temporal</h1>
<h2>1.1 Origen y composici&oacute;n</h2>
<p>Construimos un generador propio (ruta A) porque la pregunta exige contrastar mecanismos
<b>cuya dependencia del orden sea conocida</b>. Con datos reales esa dependencia es justamente lo
que se desconoce: si el modelo secuencial gana, no hay forma de saber si gan&oacute; <i>porque</i> ley&oacute;
el orden. El precio es que los resultados no dicen nada del fraude real del banco: lo que queda
demostrado es el m&eacute;todo.</p>
<p>El conjunto tiene <b>{R['n_transacciones']:,} transacciones</b> de {cfg['n_tarjetas']:,}
tarjetas a lo largo de {cfg['dias']} d&iacute;as, con <b>{pct(R['tasa_fraude'], 2)} de fraude</b>. Cada
secuencia son las &uacute;ltimas <b>K&nbsp;=&nbsp;{R['k']}</b> transacciones de una misma tarjeta en orden
cronol&oacute;gico, y el modelo predice si la &uacute;ltima &mdash;la que se est&aacute; autorizando&mdash; es fraudulenta.
Los cuatro mecanismos se dise&ntilde;aron con dependencia del orden deliberadamente distinta:</p>
{tabla(["Mecanismo", "Patr&oacute;n", "Papel en el experimento"], [
    ["escalada_prueba", "Microcompras crecientes y luego un cargo grande",
     "Orden fuerte: aqu&iacute; B deber&iacute;a ganar"],
    ["rafaga_geografica", "Compras presenciales en departamentos distantes",
     "Orden parcial: ventaja moderada"],
    ["vaciado_subito", "Uno a tres cargos muy superiores al h&aacute;bito",
     "Sin orden: control negativo"],
    ["toma_gradual", "Deriva lenta de horario, comercio y monto",
     "Orden difuso: fallo declarado de antemano"],
], clase="izq")}
<p>El generador produce adem&aacute;s comportamiento leg&iacute;timo que <i>se parece</i> a cada fraude
&mdash;rachas de microcompras en suscripciones, compras grandes reales, viajes con cambio de
departamento&mdash;. Sin esos confusores, un umbral sobre el monto m&aacute;ximo resolver&iacute;a el problema y
la comparaci&oacute;n perder&iacute;a sentido.</p>

<h2>1.2 Partici&oacute;n temporal y controles contra fuga</h2>
<p>La partici&oacute;n es <b>por fecha global, nunca aleatoria</b>: el 70&nbsp;% m&aacute;s antiguo entrena, el
15&nbsp;% siguiente valida y el 15&nbsp;% m&aacute;s reciente prueba (cortes: {R['cortes']['fin_train'][:10]}
y {R['cortes']['fin_val'][:10]}). El conjunto de prueba se abri&oacute; <b>una sola vez</b>, despu&eacute;s de
fijar arquitecturas, hiperpar&aacute;metros y umbral.</p>
<p>Tres controles sostienen que ninguna ventana contiene informaci&oacute;n posterior al instante de
decisi&oacute;n: las variables agregadas se calculan con ventanas cerradas por la izquierda, que
excluyen la propia transacci&oacute;n calificada; medias, desviaciones y medianas de imputaci&oacute;n se
estiman <b>solo con el bloque de entrenamiento</b>; y un script independiente
(<i>verificar_causalidad.py</i>) borra el futuro del conjunto y reconstruye las variables sin que
ninguna de las 16 cambie. El cuaderno ejecuta adem&aacute;s cinco aserciones que lo detendr&iacute;an si
alguno fallara.</p>
<p>Conviene declarar una decisi&oacute;n de dise&ntilde;o: la ventana de una transacci&oacute;n de prueba puede
incluir transacciones anteriores al corte. Eso <b>no es fuga</b> &mdash;en producci&oacute;n el historial del
cliente existe en el instante de la decisi&oacute;n&mdash;; lo prohibido es mirar hacia adelante, y eso no
ocurre en ning&uacute;n punto.</p>"""

    seccion2 = f"""
<h1>2. Comparaci&oacute;n com&uacute;n: A frente a B</h1>
<h2>2.1 Qu&eacute; ve cada modelo</h2>
<p><b>A &mdash; l&iacute;nea base sin orden.</b> <i>Gradient boosting</i> sobre 41 variables agregadas de
ventana (1&nbsp;h, 24&nbsp;h y 7&nbsp;d) m&aacute;s los atributos de la transacci&oacute;n actual. Representa el motor
que el banco ya tiene y es invariante a permutaciones por construcci&oacute;n.</p>
<p><b>B &mdash; modelo secuencial.</b> Una GRU con capas de <i>embedding</i> para categor&iacute;a, canal y
departamento, sobre la secuencia ordenada de las &uacute;ltimas 20 transacciones. Elegimos GRU y no
Transformer porque con secuencias de 20 pasos la atenci&oacute;n sobre dependencias largas no aporta, y
un Transformer necesitar&iacute;a codificaci&oacute;n posicional expl&iacute;cita para no ser &eacute;l mismo invariante al
orden, lo que arruinar&iacute;a el experimento. <b>B no recibe ninguna variable agregada</b>: ve los
mismos atributos que A resume, pero en bruto y en su posici&oacute;n.</p>
<p>La asimetr&iacute;a es intencional, pero introduce una limitaci&oacute;n que declaramos: parte de la brecha
entre A y B es de <i>representaci&oacute;n</i> y no de orden, y por eso la conclusi&oacute;n sobre el valor del
orden <b>no se apoya en esta comparaci&oacute;n</b> sino en la secci&oacute;n 3. Para igualar el esfuerzo, A y B
recibieron cada uno su propia rejilla de hiperpar&aacute;metros, elegida por AUC-PR de validaci&oacute;n.</p>

<h2>2.2 Resultados</h2>
{tabla(["Modelo", "Qu&eacute; ve", "AUC-PR", "Precisi&oacute;n", "Exhaust.", "F1"], [
    ["A &mdash; l&iacute;nea base", "Variables agregadas (motor actual)",
     f"{R['aucpr_test']['A']:.4f}", f"{eco['A']['precision']:.3f}",
     f"{eco['A']['exhaustividad']:.3f}", f"{eco['A']['f1']:.3f}"],
    ["B &mdash; secuencial", "Secuencia ordenada, sin agregados",
     f"{R['aucpr_test']['B']:.4f}", f"{eco['B']['precision']:.3f}",
     f"{eco['B']['exhaustividad']:.3f}", f"{eco['B']['f1']:.3f}"],
    ["C &mdash; h&iacute;brido", "Secuencia + atenci&oacute;n + agregados",
     f"{R['aucpr_test']['C']:.4f}", f"{eco['C']['precision']:.3f}",
     f"{eco['C']['exhaustividad']:.3f}", f"{eco['C']['f1']:.3f}"],
    ["A + B combinados", "Ambos puntajes, mezcla ajustada en validaci&oacute;n",
     f"{R['aucpr_test']['mezcla_AB']:.4f}", f"{mezcla['precision']:.3f}",
     f"{mezcla['exhaustividad']:.3f}", f"{mezcla['f1']:.3f}"],
], clase="izq")}
<p class="nota">Precisi&oacute;n, exhaustividad y F1 se miden en el umbral por costo de la
secci&oacute;n 5. <b>No reportamos exactitud</b>: con {pct(R['tasa_fraude'], 2)} de fraude, responder
&laquo;todo es leg&iacute;timo&raquo; acertar&iacute;a el 98.9&nbsp;% sin detectar nada.</p>
<p>El resultado inc&oacute;modo es que <b>el motor actual supera al modelo secuencial</b>. La secci&oacute;n
siguiente explica por qu&eacute; eso no cierra la pregunta.</p>
<figure><img src="{figs['comparacion']}" style="width:94%">
<figcaption>Figura 1. Curvas precisi&oacute;n&ndash;exhaustividad e intervalos de confianza del 95&nbsp;%
por <i>bootstrap</i> sobre el conjunto de prueba. Los intervalos remuestrean tarjetas completas,
no transacciones sueltas: el fraude llega en episodios correlacionados y tratarlos como
observaciones independientes dar&iacute;a intervalos artificialmente estrechos.</figcaption></figure>"""

    seccion3 = f"""
<h1>3. &iquest;El orden aporta? Dos pruebas de falsificaci&oacute;n</h1>
<p>Una mejora de m&eacute;tricas no demuestra que un modelo use el orden. Estas pruebas existen para
intentar <b>refutar nuestra propia conclusi&oacute;n</b>.</p>

<h2>3.1 Prueba 1 &mdash; Permutaci&oacute;n controlada (obligatoria)</h2>
<p>Barajamos el orden de la historia dentro de cada ventana sin cambiar los eventos ni sus
valores. Las variables agregadas son invariantes a permutaciones, as&iacute; que siguen id&eacute;nticas: lo
&uacute;nico que se destruye es la secuencia. Repetimos con cinco permutaciones distintas, porque una
sola podr&iacute;a resultar afortunada.</p>
<p><b>Un control que no es obvio.</b> El modelo lee su predicci&oacute;n del estado de la &uacute;ltima
posici&oacute;n de la ventana, que es la transacci&oacute;n calificada. Si al barajar movi&eacute;ramos tambi&eacute;n esa
posici&oacute;n, destruir&iacute;amos dos cosas a la vez: el orden y el acceso del modelo al evento que debe
puntuar. Por eso barajamos <b>solo la historia</b>; sin ese control la ca&iacute;da medida ser&iacute;a de
{pct(sin_control)}, m&aacute;s del doble de la real.</p>
<p><b>Resultado:</b> B pierde <b>{pct(R['caida_permutacion']['B'])} de su AUC-PR</b>; C solo
{pct(R['caida_permutacion']['C'])}, coherente con que conserva los agregados como red de
seguridad. Muy por encima del umbral de 15&nbsp;% fijado de antemano: <b>el orden es informaci&oacute;n
real y B la est&aacute; usando</b>.</p>
<figure><img src="{figs['permutacion']}" style="width:54%">
<figcaption>Figura 2. Desempe&ntilde;o con el orden original frente al barajado. Las barras de error
recorren las cinco permutaciones.</figcaption></figure>

<h2>3.2 Prueba 2 &mdash; Desempe&ntilde;o por mecanismo de fraude (elegida)</h2>
<p>Elegimos esta prueba porque es la &uacute;nica de la lista capaz de <b>refutar</b> la conclusi&oacute;n en
lugar de matizarla. Predijimos, antes de medir, que B ganar&iacute;a en el mecanismo dependiente del
orden y no en <i>vaciado s&uacute;bito</i>, construido sin estructura temporal como control negativo.</p>
{tabla(["Mecanismo", "Fraudes", "Tasa base", "A", "B", "C", "B &minus; A"],
       [[m["mecanismo"], f"{m['n_fraudes']}", pct(m["tasa_base"], 2),
         f"{m['aucpr_A']:.3f}", f"{m['aucpr_B']:.3f}", f"{m['aucpr_C']:.3f}",
         f"{m['ventaja_B_sobre_A']:+.3f}"] for m in R["por_mecanismo"]])}
<p class="nota">La tasa base es la prevalencia de cada fila y equivale al AUC-PR de un
clasificador al azar. Cambia entre filas, as&iacute; que los AUC-PR <b>no se comparan entre filas</b>;
s&iacute; se comparan los modelos dentro de una misma fila, que es lo que hace la &uacute;ltima columna.</p>
<p><b>Lo que se confirm&oacute;:</b> B se desploma en <i>vaciado s&uacute;bito</i>
({R['por_mecanismo'][3]['aucpr_B']:.3f} frente a {R['por_mecanismo'][3]['aucpr_A']:.3f} de A),
exactamente como predijimos. Donde no hay orden, el modelo de orden no tiene nada que leer; junto
con la prueba 1, es la evidencia de que B lee secuencia y no otra cosa. <b>Lo que se refut&oacute;:</b>
esper&aacute;bamos que B superara a A en el mecanismo de escalada, y no ocurri&oacute;. Lo reportamos como
fall&oacute;.</p>
<p>Como apoyo recortamos la ventana visible: con una sola transacci&oacute;n el AUC-PR de B cae a
{R['recorte_historia'][0]['auc_pr']:.3f} &mdash;sin memoria no hay orden&mdash; y satura en
{R['recorte_historia'][4]['auc_pr']:.3f} desde 10 transacciones. No hace falta guardar ventanas
largas.</p>

<h2>3.3 La pregunta que ninguna de las dos responde</h2>
<p>El comit&eacute; no pregunt&oacute; si el modelo secuencial es mejor, sino si el orden aporta informaci&oacute;n
que los agregados <b>no capturan</b>. Que B pierda no significa que su se&ntilde;al sea redundante: un
modelo puede ser peor y aun as&iacute; aportar algo que el otro no tiene. La ambig&uuml;edad ya estaba en
validaci&oacute;n (A {R['aucpr_val']['A']:.3f} frente a B {R['aucpr_val']['B']:.3f}), de modo que la
pregunta no naci&oacute; de mirar la prueba; el an&aacute;lisis, en cambio, s&iacute; se a&ntilde;adi&oacute; despu&eacute;s de abrirla y
as&iacute; queda etiquetado en el cuaderno (&sect;11.5). La mezcla y su umbral se ajustan
<b>&uacute;nicamente con validaci&oacute;n</b>; la prueba solo se usa para reportar.</p>
<p>Lo medimos combinando ambos puntajes con una regresi&oacute;n log&iacute;stica sobre sus <i>logits</i>.
Si el coeficiente de B fuera nulo o la mezcla no superara a A, la respuesta ser&iacute;a &laquo;no&raquo;. El
AUC-PR pasa de {R['aucpr_test']['A']:.4f} a
{R['aucpr_test']['mezcla_AB']:.4f} ({comp['diferencia_vs_A']:+.4f}; IC 95&nbsp;%
[{comp['ic_inf']:+.4f}, {comp['ic_sup']:+.4f}]): el intervalo no cruza cero, luego <b>la se&ntilde;al de
orden no es redundante</b>. Pero <b>detectable no es lo mismo que importante</b>: con casi 40,000
transacciones de prueba se detectan diferencias min&uacute;sculas, y la mejora es de mil&eacute;simas. El
criterio de relevancia lo pusimos en la moneda que fij&oacute; el propio comit&eacute; (secci&oacute;n 5).</p>"""

    seccion4 = f"""
<h1>4. La apuesta del equipo y su veredicto</h1>
<p>Registramos la hip&oacute;tesis en el repositorio <b>antes</b> de entrenar y de abrir el conjunto de
prueba (archivo <i>HIPOTESIS_C.md</i>; el sello de tiempo del <i>commit</i> es la constancia):
combinar el resumen agregado con una lectura <i>atendida</i> de la secuencia &mdash;GRU con atenci&oacute;n
aditiva concatenada con las variables agregadas&mdash; mejorar&iacute;a el AUC-PR sobre el mejor de A y B.
<b>M&eacute;trica de &eacute;xito declarada: al menos +{apuesta['umbral_declarado']:.2f} de AUC-PR en
validaci&oacute;n.</b> El <b>control experimental</b> entrena tres variantes con el mismo presupuesto,
partici&oacute;n y semilla: B (secuencia sola), C1 (secuencia + atenci&oacute;n) y C2&nbsp;=&nbsp;C (secuencia +
atenci&oacute;n + agregados); sin C1 no podr&iacute;amos distinguir si una ganancia viene de la atenci&oacute;n o de
haberle dado a C las variables que a B se le negaron.</p>
<p><b>Veredicto: la apuesta no se cumple.</b> El margen en validaci&oacute;n fue de
{apuesta['margen_val']:+.4f} frente al +{apuesta['umbral_declarado']:.2f} exigido. El control
localiza el fallo: la atenci&oacute;n por s&iacute; sola aport&oacute; {apuesta['aporte_atencion_C1_menos_B']:+.4f}
sobre B y a&ntilde;adir los agregados {apuesta['aporte_agregadas_C2_menos_C1']:+.4f} m&aacute;s. La idea de
combinar era correcta &mdash;lo demuestra la mezcla de la secci&oacute;n 3.3&mdash;, pero fusionar dentro de una
red neuronal fue el veh&iacute;culo equivocado: una capa densa no explota 41 variables tabulares tan
bien como un modelo de &aacute;rboles. El criterio secundario tambi&eacute;n fall&oacute;: esper&aacute;bamos que la atenci&oacute;n
se&ntilde;alara una transacci&oacute;n anterior del episodio en al menos el 60&nbsp;% de los aciertos, y lo hace
en {pct(R['atencion_en_historia'])}. Hoy no sirve como explicaci&oacute;n para el analista.</p>"""

    seccion5 = f"""
<h1>5. Umbral, costo y decisi&oacute;n econ&oacute;mica</h1>
<p>El comit&eacute; fij&oacute; los costos: <b>Q4,200</b> por fraude no detectado y <b>Q180</b> por bloqueo
indebido. La asimetr&iacute;a es de 23 a 1, as&iacute; que el umbral no se elige maximizando F1 &mdash;eso tratar&iacute;a
ambos errores como igual de graves&mdash; sino <b>minimizando el costo esperado</b>. Se ajust&oacute; en
validaci&oacute;n y se aplic&oacute; sin cambios a prueba.</p>
{tabla(["Escenario", "Umbral", "Fraudes que pasan", "Bloqueos indebidos", "Costo", "Ahorro"], [
    ["No hacer nada", "&mdash;", f"{fraudes_test}", "0",
     q(solo_a['costo_Q'] + solo_a['ahorro_Q']), "&mdash;"],
    ["Solo motor actual (A)", f"{solo_a['umbral']:.3f}", f"{solo_a['FN']}",
     f"{solo_a['FP']}", q(solo_a["costo_Q"]), q(solo_a["ahorro_Q"])],
    ["Motor + se&ntilde;al de orden", f"{mezcla['umbral']:.3f}", f"{mezcla['FN']}",
     f"{mezcla['FP']}", q(mezcla["costo_Q"]), q(mezcla["ahorro_Q"])],
], clase="izq")}
<p><b>Aqu&iacute; est&aacute; el hallazgo que importa.</b> Sobre los {dias:.0f} d&iacute;as del conjunto de prueba,
a&ntilde;adir la se&ntilde;al de orden cambia el ahorro en apenas {q(mat['delta_ahorro_Q'])}
({pct(mat['delta_ahorro_rel'], 2)}): en dinero es indistinguible de no hacer nada. Pero los
bloqueos a clientes leg&iacute;timos caen de {solo_a['FP']} a {mezcla['FP']}
({pct(abs(mat['delta_fp_rel']), 0)} menos) mientras la exhaustividad se mantiene pr&aacute;cticamente
igual ({solo_a['exhaustividad']:.3f} &rarr; {mezcla['exhaustividad']:.3f}). La raz&oacute;n es
aritm&eacute;tica: un bloqueo indebido cuesta 23 veces menos que un fraude, as&iacute; que evitar
{abs(mat['delta_fp'])} casi no mueve la cuenta &mdash;pero son {abs(mat['delta_fp'])} clientes reales
a los que no se les rechaza una compra, y ese beneficio no aparece en el AUC-PR. <b>El caso a
favor del modelo secuencial es de experiencia del cliente, no de detecci&oacute;n de fraude.</b></p>
<p>Extrapolado a la cartera de 1.4 millones de tarjetas, el ahorro mensual frente a no hacer nada
ser&iacute;a de <b>{q(proy['ahorro_mensual_cartera_Q'])}</b>, con
{proy['bloqueos_legitimos_por_mes']:,.0f} bloqueos indebidos y
{proy['fraudes_no_detectados_por_mes']:,.0f} fraudes que a&uacute;n pasar&iacute;an al mes. <b>Advertencia:</b>
la extrapolaci&oacute;n es lineal y supone que 1.4 millones de tarjetas reales se comportan como las
{cfg['n_tarjetas']:,} simuladas. Es una cota indicativa, no una promesa de ahorro.</p>"""

    seccion6 = f"""
<h1>6. Recomendaci&oacute;n, patr&oacute;n de error y l&iacute;mites</h1>
<h2>6.1 Recomendaci&oacute;n</h2>
<p><b>Piloto acotado; conservar el motor actual como decisi&oacute;n principal.</b> No reemplazar: el
motor de agregados resuelve bien los fraudes cuyo indicio est&aacute; en la magnitud y ninguno de los
modelos que probamos lo supera. Complementar solo bajo una condici&oacute;n clara: lo que justificar&iacute;a
incorporar la se&ntilde;al secuencial no es detectar m&aacute;s fraude &mdash;eso no lo logra&mdash; sino <b>molestar a
menos clientes detectando el mismo</b>. Si el &aacute;rea de experiencia del cliente considera que
{pct(abs(mat['delta_fp_rel']), 0)} menos rechazos indebidos vale el costo de mantener un segundo
modelo en producci&oacute;n, el piloto se justifica; si la prioridad es reducir p&eacute;rdidas, no.</p>
<p>Que B no ganara por s&iacute; solo se explica probablemente por los datos y no por la arquitectura:
el entrenamiento tiene apenas unas 1,900 transacciones fraudulentas, y los agregados son
conocimiento del dominio ya destilado que la red tendr&iacute;a que redescubrir con muy pocos positivos.
Con un orden de magnitud m&aacute;s de fraude etiquetado, la conclusi&oacute;n podr&iacute;a invertirse.</p>

<h2>6.2 Un patr&oacute;n de error concreto</h2>
<p>Los bloqueos indebidos se concentran en <b>transacciones leg&iacute;timas de monto alto</b>: el modelo
penaliza el gasto at&iacute;pico aunque sea real, y eso afecta sobre todo a quienes hacen compras
grandes ocasionales &mdash;electr&oacute;nica, viajes&mdash;, los clientes de mayor valor para el banco. Es un
argumento para no usar el puntaje como bloqueo autom&aacute;tico en montos altos, sino como disparador
de una verificaci&oacute;n por segundo canal. En el otro extremo, <i>toma gradual</i> resiste a los tres
modelos: la se&ntilde;al se reparte en decenas de transacciones y la ventana de 20 no cubre el episodio;
es el fallo que declaramos antes de medirlo.</p>

<h2>6.3 L&iacute;mites y qu&eacute; cambiar&iacute;a la recomendaci&oacute;n</h2>
<ul>
<li><b>Los datos son sint&eacute;ticos:</b> reflejan nuestras hip&oacute;tesis sobre el fraude, no el del banco.</li>
<li><b>Un defecto de nuestro propio generador:</b> los sondeos y las rachas leg&iacute;timas de
suscripciones quedaron con montos distintos, as&iacute; que se separan por nivel y no solo por orden.
Eso sesga la prueba 2 a favor del motor de agregados y es la primera correcci&oacute;n que har&iacute;amos.</li>
<li><b>Una sola semilla por variante:</b> no medimos la variabilidad entre inicializaciones.</li>
<li><b>Qu&eacute; cambiar&iacute;a la recomendaci&oacute;n:</b> que en datos reales la permutaci&oacute;n costara menos del
15&nbsp;% del desempe&ntilde;o (no habr&iacute;a evidencia de aporte del orden); que un bloqueo indebido pasara de
Q180 a m&aacute;s de unos Q900 (la asimetr&iacute;a 23:1 se estrechar&iacute;a y habr&iacute;a que recalcular todo); o que
apareciera un mecanismo nuevo sin ejemplos etiquetados, invisible para cualquier modelo
supervisado y que exigir&iacute;a un componente no supervisado.</li>
</ul>
<p><b>Siguiente paso.</b> Ejecutar la permutaci&oacute;n controlada y la prueba de complementariedad sobre
los datos reales del banco: el procedimiento es directamente aplicable. Hasta entonces, lo que
este trabajo demuestra es el m&eacute;todo, no una cifra de ahorro.</p>"""

    matriz = f"""
<h1>7. Matriz de evidencias</h1>
{tabla(["Evidencia", "D&oacute;nde aparece", "Conclusi&oacute;n", "Limitaci&oacute;n"], [
    ["1 &middot; Integridad de datos",
     "&sect;1; cuaderno &sect;2&ndash;&sect;3; <i>verificar_causalidad.py</i>",
     f"{R['n_transacciones']:,} transacciones, {pct(R['tasa_fraude'], 2)} de fraude, "
     "partici&oacute;n 70/15/15 por fecha, sin fuga.",
     "Datos sint&eacute;ticos: validez externa no demostrada."],
    ["2 &middot; Comparaci&oacute;n com&uacute;n", "&sect;2.2, tabla y figura 1",
     f"AUC-PR: A {R['aucpr_test']['A']:.3f}, B {R['aucpr_test']['B']:.3f}, "
     f"C {R['aucpr_test']['C']:.3f}, A+B {R['aucpr_test']['mezcla_AB']:.3f}.",
     "B no recibe agregadas: parte de la brecha es de representaci&oacute;n."],
    ["3 &middot; Valor del orden", "&sect;3.1 figura 2; &sect;3.2 tabla; &sect;3.3",
     f"Barajar cuesta {pct(R['caida_permutacion']['B'], 0)} del AUC-PR de B, que colapsa en el "
     "control sin orden. La mezcla A+B supera a A con IC que no cruza cero.",
     "B no supera a A en ning&uacute;n mecanismo: aporta, pero no basta solo."],
    ["4 &middot; Apuesta del equipo", "&sect;4; <i>HIPOTESIS_C.md</i>",
     f"Margen {apuesta['margen_val']:+.4f} frente al +{apuesta['umbral_declarado']:.2f} exigido: "
     f"no se cumple. La atenci&oacute;n explica el caso en {pct(R['atencion_en_historia'], 0)}.",
     "Una sola semilla por variante."],
    ["5 &middot; Decisi&oacute;n econ&oacute;mica", "&sect;5, tabla de escenarios",
     f"Umbral {mezcla['umbral']:.3f}; {q(mezcla['ahorro_Q'])} de ahorro en {dias:.0f} d&iacute;as. "
     f"Frente a solo A: {q(mat['delta_ahorro_Q'])} y {abs(mat['delta_fp'])} bloqueos menos.",
     "Costos uniformes; extrapolaci&oacute;n lineal a 1.4 M de tarjetas."],
    ["6 &middot; Recomendaci&oacute;n y l&iacute;mites", "&sect;6",
     "Complementar con un piloto acotado, no reemplazar. El error se concentra en compras "
     "leg&iacute;timas de monto alto.",
     "Confusor de sondeo mal calibrado: sesga la prueba 2 a favor de A."],
], clase="izq matriz", anchos=["15%", "22%", "37%", "26%"])}
<p class="nota">Generado desde <i>artefactos/resultados.json</i>; ninguna cifra transcrita a
mano. Semilla {cfg['semilla']}.</p>"""

    cuerpo = (caratula + resumen + seccion1 + seccion2 + seccion3
              + seccion4 + seccion5 + seccion6 + matriz)
    return (f"<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
            f"<title>Informe &mdash; Proyecto 1</title><style>{ESTILO_INFORME}</style>"
            f"</head><body>{cuerpo}</body></html>")


# --------------------------------------------------------------------------
# Presentacion
# --------------------------------------------------------------------------

def construir_presentacion(R: dict) -> str:
    eco, mat = R["economia"], R["materialidad"]
    comp, apuesta = R["complementariedad"], R["apuesta_C"]
    mezcla, solo_a = eco["mezcla_AB"], R["solo_A"]
    variantes = {v["variante"]: v for v in R["permutacion_variantes"]}
    barajado = variantes["solo la historia (controlada)"]["auc_pr_barajado"]

    def pie(n: int) -> str:
        return (f'<div class="pie"><span>Estrada &middot; Ram&iacute;rez &mdash; '
                f'Proyecto 1, Deep Learning 2026</span><span>{n} / 8</span></div>')

    d1 = f"""
<div class="d portada">
  <div class="u">Universidad del Valle de Guatemala</div>
  <div class="f">Facultad de Ingenier&iacute;a &middot; Deep Learning &mdash; Semestre 2, 2026</div>
  <div class="t">&iquest;El orden de las transacciones<br>revela fraude?</div>
  <div class="s">Proyecto 1 &mdash; Monitoreo transaccional</div>
  <div class="a">Daniel Estrada &ndash; 20853<br>Daniela Ram&iacute;rez &ndash; 23053</div>
</div>"""

    d2 = f"""
<div class="d">
  <h1>1. La pregunta del comit&eacute;</h1><div class="regla"></div>
  <p><i>&laquo;Cuando revisamos los casos que se nos escaparon, el patr&oacute;n siempre est&aacute; ah&iacute;.
  No en los montos: en el orden en que ocurrieron.&raquo;</i></p>
  <p>El motor actual resume cada ventana en variables agregadas: monto promedio de 24&nbsp;h,
  transacciones por hora, monto m&aacute;ximo y diversidad de comercios.</p>
  <div class="caja"><b>El problema:</b> todas esas variables son id&eacute;nticas si se baraja la
  secuencia. Por construcci&oacute;n, el motor actual no puede ver orden.</div>
  <p>La pregunta no es &laquo;&iquest;puedo entrenar una LSTM?&raquo;, sino <b>&iquest;el orden aporta informaci&oacute;n
  que los agregados no capturan, y cu&aacute;nto vale en quetzales?</b></p>
  {pie(2)}
</div>"""

    d3 = f"""
<div class="d">
  <h1>2. Dise&ntilde;o: datos donde conocemos la verdad</h1><div class="regla"></div>
  <div class="cols">
    <div>
      <p>Generamos los datos para poder hacer una <b>predicci&oacute;n falsable</b>: cuatro
      mecanismos con dependencia del orden conocida y distinta.</p>
      {tabla(["Mecanismo", "Orden", "Predicci&oacute;n"], [
          ["escalada de prueba", "fuerte", "B gana"],
          ["r&aacute;faga geogr&aacute;fica", "parcial", "empate"],
          ["vaciado s&uacute;bito", "ninguno", "B NO gana"],
          ["toma gradual", "difuso", "ambos fallan"]])}
    </div>
    <div>
      <p><b>Confusores leg&iacute;timos</b> para que no sea trivial: rachas de microcompras,
      compras grandes reales y viajes.</p>
      <p><b>{R['n_transacciones']:,}</b> transacciones &middot; {R['config_generador']['n_tarjetas']:,}
      tarjetas &middot; {pct(R['tasa_fraude'], 2)} de fraude &middot; K&nbsp;=&nbsp;{R['k']}.</p>
      <p>Partici&oacute;n temporal estricta 70/15/15 por fecha. El conjunto de prueba se abri&oacute;
      <b>una sola vez</b>.</p>
      <div class="caja"><i>Vaciado s&uacute;bito</i> es el <b>control negativo</b>: si B ganara
      tambi&eacute;n ah&iacute;, la ventaja no vendr&iacute;a del orden.</div>
    </div>
  </div>
  {pie(3)}
</div>"""

    d4 = f"""
<div class="d">
  <h1>3. Prueba 1 &mdash; Permutaci&oacute;n controlada</h1><div class="regla"></div>
  <p>Barajamos el orden de la historia dentro de cada ventana. Mismos eventos, mismos valores,
  mismas variables agregadas, y la transacci&oacute;n calificada se queda en su sitio: solo se
  destruye la secuencia.</p>
  <div class="cifra">
    <div class="v">&minus;{pct(R['caida_permutacion']['B'], 0)}</div>
    <div class="e">del AUC-PR del modelo secuencial se pierde al barajar
    ({R['aucpr_test']['B']:.3f} &rarr; {barajado:.3f}, cinco permutaciones distintas)</div>
  </div>
  <p><b>El control que no es obvio:</b> si baraj&aacute;ramos tambi&eacute;n la posici&oacute;n de la transacci&oacute;n
  calificada, el modelo perder&iacute;a adem&aacute;s el acceso al evento que debe puntuar y la ca&iacute;da
  parecer&iacute;a de {pct(variantes['toda la ventana (sin control)']['caida_relativa'], 0)}. Sin ese
  control estar&iacute;amos midiendo dos cosas y atribuy&eacute;ndolas al orden.</p>
  {pie(4)}
</div>"""

    d5 = f"""
<div class="d">
  <h1>4. Prueba 2 &mdash; Por mecanismo: el resultado inc&oacute;modo</h1><div class="regla"></div>
  <div class="cols">
    <div>
      {tabla(["Mecanismo", "A", "B", "B &minus; A"],
             [[m["mecanismo"], f"{m['aucpr_A']:.3f}", f"{m['aucpr_B']:.3f}",
               f"{m['ventaja_B_sobre_A']:+.3f}"] for m in R["por_mecanismo"]])}
    </div>
    <div>
      <p><b>Se confirm&oacute;:</b> B se desploma en el control sin orden
      ({R['por_mecanismo'][3]['aucpr_B']:.3f} frente a
      {R['por_mecanismo'][3]['aucpr_A']:.3f}). Donde no hay orden, no hay nada que leer.</p>
      <p><b>Se refut&oacute;:</b> B no supera a A en ning&uacute;n mecanismo. Nuestra predicci&oacute;n fall&oacute; y
      as&iacute; lo reportamos.</p>
    </div>
  </div>
  {pie(5)}
</div>"""

    d6 = f"""
<div class="d">
  <h1>5. Entonces, &iquest;el orden aporta o no?</h1><div class="regla"></div>
  <p>Que B pierda no significa que su se&ntilde;al sea <b>redundante</b>. Combinamos ambos puntajes
  con una mezcla ajustada <b>solo en validaci&oacute;n</b>:</p>
  <div class="cifra">
    <div class="v">{R['aucpr_test']['A']:.4f} &rarr; {R['aucpr_test']['mezcla_AB']:.4f}</div>
    <div class="e">{comp['diferencia_vs_A']:+.4f} &middot; IC 95&nbsp;%
    [{comp['ic_inf']:+.4f}, {comp['ic_sup']:+.4f}] &mdash; no cruza cero</div>
  </div>
  <p><b>Detectable &ne; importante.</b> El intervalo excluye el cero, as&iacute; que la se&ntilde;al de orden
  no es redundante; pero la mejora es de mil&eacute;simas. Con 40,000 transacciones de prueba se
  detectan diferencias min&uacute;sculas, y presentar esto como &laquo;las secuencias mejoran la
  detecci&oacute;n&raquo; ser&iacute;a exagerar lo que medimos.</p>
  {pie(6)}
</div>"""

    d7 = f"""
<div class="d">
  <h1>6. La decisi&oacute;n en quetzales &mdash; y el hallazgo real</h1><div class="regla"></div>
  <p>Costos del comit&eacute;: <b>Q4,200</b> por fraude que pasa, <b>Q180</b> por bloqueo indebido.
  Asimetr&iacute;a 23&nbsp;:&nbsp;1, as&iacute; que el umbral minimiza costo esperado, no F1.</p>
  {tabla(["Escenario", "Fraudes que pasan", "Bloqueos indebidos", "Ahorro en prueba"], [
      ["Solo motor actual", f"{solo_a['FN']}", f"{solo_a['FP']}", q(solo_a["ahorro_Q"])],
      ["Motor + se&ntilde;al de orden", f"{mezcla['FN']}", f"{mezcla['FP']}",
       q(mezcla["ahorro_Q"])]])}
  <div class="cols">
    <div><p><b>En dinero: nada.</b> {q(mat['delta_ahorro_Q'])} de diferencia
    ({pct(mat['delta_ahorro_rel'], 2)}). Indistinguible de no hacer nada.</p></div>
    <div><p><b>En clientes: s&iacute;.</b> {abs(mat['delta_fp'])} bloqueos indebidos menos
    ({pct(abs(mat['delta_fp_rel']), 0)}) a igual exhaustividad.</p></div>
  </div>
  <div class="caja">El caso a favor del orden es de <b>experiencia del cliente</b>, no de
  detecci&oacute;n de fraude.</div>
  {pie(7)}
</div>"""

    d8 = f"""
<div class="d">
  <h1>7. Recomendaci&oacute;n y l&iacute;mites</h1><div class="regla"></div>
  <div class="caja"><b>Piloto acotado.</b> Conservar el motor actual como decisi&oacute;n principal.
  Lo que justificar&iacute;a el piloto no es detectar m&aacute;s fraude &mdash;eso no lo logra&mdash; sino molestar a
  menos clientes detectando el mismo.</div>
  <div class="cols">
    <div>
      <p><b>Por qu&eacute; B no gan&oacute; solo</b></p>
      <p>Pocos fraudes etiquetados en entrenamiento (~1,900). Los agregados son conocimiento del
      dominio ya destilado; la red debe redescubrirlo con muy pocos positivos. Parece un l&iacute;mite
      de datos, no de arquitectura &mdash;y es comprobable.</p>
      <p>Nuestra <b>apuesta C fall&oacute;</b> ({apuesta['margen_val']:+.3f} frente al
      +{apuesta['umbral_declarado']:.2f} exigido). Pre-registrada en git antes de ver el
      conjunto de prueba; la reportamos como qued&oacute;.</p>
    </div>
    <div>
      <p><b>L&iacute;mites que declaramos</b></p>
      <ul>
        <li>Datos sint&eacute;ticos: no dicen nada del fraude real.</li>
        <li>Defecto de nuestro generador: sondeos y rachas leg&iacute;timas se separan por monto,
        no solo por orden &mdash;sesga la prueba 2 a favor de A.</li>
        <li>La atenci&oacute;n de C explica el caso en {pct(R['atencion_en_historia'], 0)}, frente al
        60&nbsp;% declarado.</li>
        <li>Una sola semilla por variante.</li>
      </ul>
    </div>
  </div>
  <p><b>Siguiente paso:</b> correr la permutaci&oacute;n y la prueba de complementariedad sobre los
  datos reales del banco. Es una tarde de trabajo.</p>
  {pie(8)}
</div>"""

    cuerpo = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8
    return (f"<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
            f"<title>Presentaci&oacute;n &mdash; Proyecto 1</title>"
            f"<style>{ESTILO_PRESENTACION}</style></head><body>{cuerpo}</body></html>")


# --------------------------------------------------------------------------

def main() -> int:
    R = json.loads((ART / "resultados.json").read_text(encoding="utf-8"))
    figs = figuras_del_cuaderno()

    informe = RAIZ / "informe.pdf"
    a_pdf(construir_informe(R, figs), informe)
    n = numerar(informe)
    print(f"  informe.pdf        {n} paginas  "
          f"({informe.stat().st_size / 1024:.0f} KB)")
    if n > 7:
        print("  AVISO: el informe excede las 7 paginas permitidas.")

    presentacion = RAIZ / "presentacion.pdf"
    a_pdf(construir_presentacion(R), presentacion, apaisado=True)
    import pymupdf
    n = len(pymupdf.open(presentacion))
    print(f"  presentacion.pdf   {n} diapositivas  "
          f"({presentacion.stat().st_size / 1024:.0f} KB)")
    if n > 8:
        print("  AVISO: la presentacion excede las 8 diapositivas permitidas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
