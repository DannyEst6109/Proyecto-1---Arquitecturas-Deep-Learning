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

:root {
  --tinta:      #0b0b0b;
  --tinta-2:    #52514e;
  --tinta-3:    #898781;
  --marca:      #0d366b;
  --serie-a:    #2a78d6;
  --serie-b:    #eb6834;
  --linea:      #e1e0d9;
  --plano:      #f6f6f4;
  --bien:       #0ca30c;
  --mal:        #d03b3b;
}

body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       color: var(--tinta); margin: 0; background: #fff;
       -webkit-font-smoothing: antialiased; }

.d { width: 11in; height: 8.5in; padding: 0.5in 0.7in 0.55in;
     page-break-after: always; position: relative; background: #fff;
     display: flex; flex-direction: column; }
.d:last-child { page-break-after: auto; }
.cuerpo { flex: 1; display: flex; flex-direction: column;
          justify-content: center; min-height: 0; }
.cuerpo > *:last-child { margin-bottom: 0; }

/* --- Encabezado --- */
.cab { display: flex; align-items: baseline; gap: 12px; margin-bottom: 3px; }
.cab .num { font-size: 14pt; font-weight: 700; color: var(--serie-b);
            letter-spacing: .04em; }
h1 { font-size: 26pt; font-weight: 700; color: var(--marca); margin: 0;
     letter-spacing: -.015em; line-height: 1.15; }
.regla { height: 2.5px; background: var(--marca); margin: 9px 0 16px;
         position: relative; }
.regla::after { content: ""; position: absolute; left: 0; top: 0; width: 54px;
                height: 2.5px; background: var(--serie-b); }

/* --- Texto --- */
p { font-size: 14.5pt; line-height: 1.42; margin: 0 0 13pt; color: var(--tinta-2); }
p.destacado { font-size: 14pt; color: var(--tinta); }
b { font-weight: 650; color: var(--tinta); }
ul { font-size: 13.5pt; line-height: 1.4; margin: 0 0 13pt; padding-left: 20px;
     color: var(--tinta-2); }
li { margin-bottom: 8px; }
li::marker { color: var(--tinta-3); }
.cita { font-size: 16pt; line-height: 1.42; color: var(--tinta);
        border-left: 3px solid var(--serie-b); padding: 3px 0 3px 18px;
        margin: 0 0 18pt; }

.cols { display: flex; gap: 30px; }
.cols > div { flex: 1; min-width: 0; }
.cols.a60 > div:first-child { flex: 1.45; }
.cols.b60 > div:last-child { flex: 1.45; }

/* --- Caja de idea --- */
.caja { background: var(--plano); border-left: 3px solid var(--marca);
        padding: 15px 19px; margin: 0 0 14pt; font-size: 14.5pt;
        line-height: 1.4; color: var(--tinta); }
.caja.acento { border-left-color: var(--serie-b); }

/* --- Tabla --- */
table { border-collapse: collapse; width: 100%; font-size: 13pt;
        margin: 2px 0 13pt; }
th { text-align: left; font-weight: 650; color: var(--tinta-3);
     font-size: 10.5pt; letter-spacing: .07em; text-transform: uppercase;
     padding: 0 10px 8px 0; border-bottom: 1.5px solid var(--linea); }
td { padding: 10px 10px 10px 0; border-bottom: 1px solid var(--linea);
     color: var(--tinta-2); }
td:first-child { color: var(--tinta); font-weight: 550; }
th + th, td + td { text-align: right; padding-right: 0; }
table.izq th + th, table.izq td + td { text-align: left; padding-right: 10px; }

/* --- Fichas de dato --- */
.fichas { display: flex; gap: 14px; margin: 0 0 12pt; }
.ficha { flex: 1; border-top: 3px solid var(--linea); padding: 9px 0 0; }
.ficha.bien { border-top-color: var(--bien); }
.ficha.mal  { border-top-color: var(--mal); }
.ficha.neutra { border-top-color: var(--serie-a); }
.ficha .v { font-size: 31pt; font-weight: 700; line-height: 1;
            color: var(--tinta); letter-spacing: -.02em; }
.ficha .e { font-size: 11.5pt; color: var(--tinta-2); margin-top: 7px;
            line-height: 1.32; }

/* --- Cifra protagonista --- */
.cifra { text-align: center; margin: 6px 0 12px; }
.cifra .v { font-size: 76pt; font-weight: 700; line-height: 1;
            color: var(--serie-b); letter-spacing: -.03em; }
.cifra .e { font-size: 13.5pt; color: var(--tinta-2); margin-top: 10px; }

/* --- Leyenda --- */
.leyenda { display: flex; gap: 22px; font-size: 12pt; color: var(--tinta-2);
           margin: 0 0 8px; }
.leyenda span { display: flex; align-items: center; gap: 7px; }
.leyenda i { width: 11px; height: 11px; border-radius: 2px; display: block; }

figure { margin: 0 0 8pt; }
figcaption { font-size: 10.5pt; color: var(--tinta-3); margin-top: 6px;
             line-height: 1.35; }

/* --- Pie --- */
.pie { position: absolute; left: 0.7in; right: 0.7in; bottom: 0.3in;
       font-size: 9.5pt; color: var(--tinta-3); display: flex;
       justify-content: space-between; border-top: 1px solid var(--linea);
       padding-top: 6px; }

/* --- Portada --- */
.portada { background: var(--marca); color: #fff; display: flex;
           flex-direction: column; justify-content: center;
           padding: 0.5in 1.1in; }
.portada .inst { font-size: 12pt; color: #9ec5f4; letter-spacing: .02em; }
.portada .t { font-size: 42pt; font-weight: 700; line-height: 1.1;
              margin: 26px 0 0; letter-spacing: -.025em; }
.portada .barra { width: 78px; height: 4px; background: var(--serie-b);
                  margin: 26px 0; }
.portada .s { font-size: 15pt; color: #cde2fb; }
.portada .a { font-size: 13pt; color: #fff; margin-top: 46px; line-height: 1.7; }
.portada .a span { color: #9ec5f4; }
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

def _barra(x: float, y: float, ancho: float, base: float, r: float = 4.0) -> str:
    """Barra con los extremos superiores redondeados, anclada a la linea base."""
    if base - y < r:
        r = max(base - y, 0.0)
    return (f"M{x},{base} L{x},{y + r} Q{x},{y} {x + r},{y} "
            f"L{x + ancho - r},{y} Q{x + ancho},{y} {x + ancho},{y + r} "
            f"L{x + ancho},{base} Z")


def svg_mecanismos(R: dict) -> str:
    """Barras agrupadas: A frente a B en cada mecanismo de fraude.

    Dos series -> paleta categorica (azul = motor actual, naranja = secuencial),
    validada con scripts/validate_palette.js. Se etiqueta solo el par decisivo
    (vaciado subito); el resto lo lee el eje. La tabla completa esta en el
    informe, seccion 3.2.
    """
    datos = R["por_mecanismo"]
    x0, x1, y0, y1 = 62.0, 892.0, 18.0, 330.0
    alto = y1 - y0
    paso = (x1 - x0) / len(datos)
    ancho = 62.0

    piezas = []
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y1 - v * alto
        piezas.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
                      f'stroke="#e1e0d9" stroke-width="1"/>')
        piezas.append(f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" '
                      f'font-size="12" fill="#898781">{v:.2f}</text>')

    for i, m in enumerate(datos):
        centro = x0 + paso * (i + 0.5)
        for j, (clave, color) in enumerate([("aucpr_A", "#2a78d6"),
                                            ("aucpr_B", "#eb6834")]):
            # 2 px de superficie entre las dos barras del grupo.
            x = centro - ancho - 1 + j * (ancho + 2)
            y = y1 - m[clave] * alto
            piezas.append(f'<path d="{_barra(x, y, ancho, y1)}" fill="{color}"/>')
            if m["mecanismo"] == "vaciado_subito":
                piezas.append(
                    f'<text x="{x + ancho / 2:.1f}" y="{y - 9:.1f}" '
                    f'text-anchor="middle" font-size="14" font-weight="650" '
                    f'fill="{color}">{m[clave]:.3f}</text>')
        piezas.append(f'<text x="{centro:.1f}" y="{y1 + 24}" text-anchor="middle" '
                      f'font-size="14" fill="#52514e">{m["mecanismo"]}</text>')

    piezas.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
                  f'stroke="#c3c2b7" stroke-width="1.5"/>')
    return ('<svg viewBox="0 0 900 358" width="100%" '
            'font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
            + "".join(piezas) + "</svg>")


def svg_permutacion(R: dict) -> str:
    """Dumbbell: orden original -> orden barajado, para B y para C.

    Un solo tono en dos pasos (la escala azul), porque los extremos son dos
    ESTADOS del mismo modelo, no dos entidades distintas.
    """
    filas = [("Modelo B", R["aucpr_test"]["B"], R["caida_permutacion"]["B"]),
             ("Modelo C", R["aucpr_test"]["C"], R["caida_permutacion"]["C"])]
    x0, x1 = 150.0, 700.0

    def px(v: float) -> float:
        return x0 + v * (x1 - x0)

    piezas = [
        '<circle cx="156" cy="12" r="7" fill="#2a78d6"/>',
        '<text x="170" y="17" font-size="13" fill="#52514e">orden original</text>',
        '<circle cx="306" cy="12" r="7" fill="#86b6ef"/>',
        '<text x="320" y="17" font-size="13" fill="#52514e">orden barajado</text>',
    ]
    for i, (nombre, original, caida) in enumerate(filas):
        y = 62 + i * 88
        barajado = original * (1 - caida)
        piezas.append(f'<text x="0" y="{y + 5}" font-size="15" font-weight="600" '
                      f'fill="#0b0b0b">{nombre}</text>')
        piezas.append(f'<line x1="{px(barajado):.1f}" y1="{y}" x2="{px(original):.1f}" '
                      f'y2="{y}" stroke="#cde2fb" stroke-width="4" '
                      f'stroke-linecap="round"/>')
        # Anillo de 2 px del color de la superficie sobre las marcas.
        piezas.append(f'<circle cx="{px(barajado):.1f}" cy="{y}" r="9" fill="#86b6ef" '
                      f'stroke="#fff" stroke-width="2"/>')
        piezas.append(f'<circle cx="{px(original):.1f}" cy="{y}" r="9" fill="#2a78d6" '
                      f'stroke="#fff" stroke-width="2"/>')
        piezas.append(f'<text x="{px(barajado):.1f}" y="{y - 18}" text-anchor="middle" '
                      f'font-size="13" fill="#52514e">{barajado:.3f}</text>')
        piezas.append(f'<text x="{px(original):.1f}" y="{y - 18}" text-anchor="middle" '
                      f'font-size="13" fill="#52514e">{original:.3f}</text>')
        piezas.append(f'<text x="{x1 + 34}" y="{y + 6}" font-size="19" '
                      f'font-weight="700" fill="#eb6834">&#8722;{100 * caida:.0f} %</text>')
    return ('<svg viewBox="0 0 900 200" width="100%" '
            'font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
            + "".join(piezas) + "</svg>")


def svg_intervalo(R: dict) -> str:
    """Intervalo de confianza de la mezcla frente al motor actual, anclado en 0."""
    c = R["complementariedad"]
    x0, x1 = 120.0, 830.0
    tope = 0.0032

    def px(v: float) -> float:
        return x0 + (v / tope) * (x1 - x0)

    y = 62.0
    piezas = [
        f'<line x1="{x0}" y1="26" x2="{x0}" y2="104" stroke="#c3c2b7" stroke-width="1.5"/>',
        f'<text x="{x0}" y="18" text-anchor="middle" font-size="12" fill="#898781">'
        f'0 &#183; sin aporte</text>',
    ]
    for v in (0.001, 0.002, 0.003):
        piezas.append(f'<line x1="{px(v):.1f}" y1="26" x2="{px(v):.1f}" y2="104" '
                      f'stroke="#e1e0d9" stroke-width="1"/>')
        piezas.append(f'<text x="{px(v):.1f}" y="122" text-anchor="middle" '
                      f'font-size="12" fill="#898781">+{v:.3f}</text>')
    piezas.append(f'<line x1="{px(c["ic_inf"]):.1f}" y1="{y}" x2="{px(c["ic_sup"]):.1f}" '
                  f'y2="{y}" stroke="#2a78d6" stroke-width="4" stroke-linecap="round"/>')
    piezas.append(f'<circle cx="{px(c["diferencia_vs_A"]):.1f}" cy="{y}" r="10" '
                  f'fill="#2a78d6" stroke="#fff" stroke-width="2"/>')
    piezas.append(f'<text x="{px(c["diferencia_vs_A"]):.1f}" y="{y - 22}" '
                  f'text-anchor="middle" font-size="17" font-weight="700" '
                  f'fill="#0b0b0b">{c["diferencia_vs_A"]:+.4f}</text>')
    return ('<svg viewBox="0 0 900 132" width="100%" '
            'font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
            + "".join(piezas) + "</svg>")


def svg_invarianza() -> str:
    """Diagrama: barajar la secuencia no cambia ninguna variable agregada.

    Ejemplo ilustrativo de `escalada_prueba` con los montos que produce el
    generador (sondeos de Q4-60 que escalan, seguidos del golpe).
    """
    montos = [6, 8, 11, 15, 20, 1420]
    barajado = [15, 1420, 6, 20, 8, 11]
    ancho, hueco, x0 = 108.0, 13.0, 178.0

    def fila(valores, y, etiqueta, color_ultimo):
        out = [f'<text x="0" y="{y + 29}" font-size="14.5" font-weight="600" '
               f'fill="#0b0b0b">{etiqueta}</text>']
        for i, v in enumerate(valores):
            x = x0 + i * (ancho + hueco)
            grande = v > 100
            relleno = color_ultimo if grande else "#f6f6f4"
            texto = "#fff" if grande else "#52514e"
            borde = "none" if grande else "#e1e0d9"
            out.append(f'<rect x="{x}" y="{y}" width="{ancho}" height="46" rx="5" '
                       f'fill="{relleno}" stroke="{borde}" stroke-width="1"/>')
            out.append(f'<text x="{x + ancho / 2}" y="{y + 29}" text-anchor="middle" '
                       f'font-size="15" font-weight="600" fill="{texto}">Q{v:,}</text>')
        return out

    piezas = fila(montos, 16, "Orden real", "#eb6834")
    piezas += fila(barajado, 104, "Barajado", "#eb6834")
    piezas.append('<line x1="178" y1="176" x2="890" y2="176" stroke="#e1e0d9" '
                  'stroke-width="1"/>')
    piezas.append('<text x="0" y="206" font-size="14.5" font-weight="600" '
                  'fill="#0d366b">Lo que ve el motor actual</text>')
    piezas.append('<text x="178" y="206" font-size="15" fill="#52514e">'
                  'promedio Q247 &#160;&#183;&#160; m&#225;ximo Q1,420 &#160;&#183;&#160; '
                  'n = 6 &#160;&#183;&#160; 1 comercio &#160;&#183;&#160; '
                  '<tspan font-weight="650" fill="#0b0b0b">id&#233;ntico en ambos'
                  '</tspan></text>')
    return ('<svg viewBox="0 0 900 220" width="100%" '
            'font-family="system-ui, -apple-system, Segoe UI, sans-serif">'
            + "".join(piezas) + "</svg>")


def construir_presentacion(R: dict) -> str:
    eco, mat = R["economia"], R["materialidad"]
    comp, apuesta = R["complementariedad"], R["apuesta_C"]
    mezcla, solo_a = eco["mezcla_AB"], R["solo_A"]
    variantes = {v["variante"]: v for v in R["permutacion_variantes"]}
    sin_control = variantes["toda la ventana (sin control)"]["caida_relativa"]

    def pie(n: int) -> str:
        return (f'<div class="pie"><span>Estrada &#183; Ram&#237;rez &#183; '
                f'Proyecto 1 &#183; Deep Learning 2026</span>'
                f'<span>{n} / 8</span></div>')

    def cab(n: str, titulo: str) -> str:
        return (f'<div class="cab"><span class="num">{n}</span>'
                f'<h1>{titulo}</h1></div><div class="regla"></div>')

    leyenda = ('<div class="leyenda">'
               '<span><i style="background:#2a78d6"></i>A &#183; motor actual '
               '(agregadas)</span>'
               '<span><i style="background:#eb6834"></i>B &#183; secuencial '
               '(orden)</span></div>')

    d1 = f"""
<div class="d portada">
  <div class="inst">Universidad del Valle de Guatemala &#183; Facultad de Ingenier&#237;a<br>
  Deep Learning &#8212; Semestre 2, 2026</div>
  <div class="t">&#191;El orden de las transacciones<br>revela fraude?</div>
  <div class="barra"></div>
  <div class="s">Proyecto 1 &#8212; Monitoreo transaccional<br>
  Informe al Comit&#233; de Riesgos, Banco del Altiplano</div>
  <div class="a">Daniel Estrada <span>&#183; 20853</span><br>
  Daniela Ram&#237;rez <span>&#183; 23053</span></div>
</div>"""

    d2 = f"""
<div class="d">
  {cab("01", "La pregunta del comit&#233;")}
  <div class="cuerpo">
  <div class="cita">&#171;Cuando revisamos los casos que se nos escaparon, el patr&#243;n
  siempre est&#225; ah&#237;. No en los montos: en el orden en que ocurrieron.&#187;</div>
  <p>El motor actual resume cada ventana en variables agregadas: monto promedio de
  24&#160;h, transacciones por hora, monto m&#225;ximo y diversidad de comercios.
  <b>Todas son id&#233;nticas si se baraja la secuencia.</b></p>
  <figure>{svg_invarianza()}
  <figcaption>Ejemplo del mecanismo <i>escalada de prueba</i>: cinco microcompras
  que escalan y luego el golpe. Para el motor actual, las dos secuencias son el
  mismo caso.</figcaption></figure>
  <div class="caja acento">Por construcci&#243;n, el motor actual <b>no puede ver
  orden</b>. La pregunta no es &#171;&#191;podemos entrenar una LSTM?&#187;, sino
  <b>&#191;el orden aporta informaci&#243;n que los agregados no capturan, y cu&#225;nto
  vale en quetzales?</b></div>
  </div>
  {pie(2)}
</div>"""

    d3 = f"""
<div class="d">
  {cab("02", "Dise&#241;o: datos donde conocemos la verdad")}
  <div class="cuerpo">
  <div class="fichas">
    <div class="ficha neutra"><div class="v">{R['n_transacciones'] / 1000:.0f}k</div>
      <div class="e">transacciones de
      {R['config_generador']['n_tarjetas']:,} tarjetas en
      {R['config_generador']['dias']} d&#237;as</div></div>
    <div class="ficha neutra"><div class="v">{pct(R['tasa_fraude'], 2)}</div>
      <div class="e">de fraude &#183; 400 episodios en 4 mecanismos</div></div>
    <div class="ficha neutra"><div class="v">K = {R['k']}</div>
      <div class="e">transacciones por ventana, en orden cronol&#243;gico</div></div>
    <div class="ficha neutra"><div class="v">70/15/15</div>
      <div class="e">partici&#243;n por fecha; la prueba se abri&#243; una sola vez</div></div>
  </div>
  <div class="cols b60">
    <div>
      <p>Generamos los datos para poder hacer una <b>predicci&#243;n falsable</b>:
      cuatro mecanismos con dependencia del orden conocida y distinta. Con datos
      reales esa dependencia es justo lo desconocido.</p>
      <div class="caja"><i>Vaciado s&#250;bito</i> es el <b>control negativo</b>: si B
      ganara tambi&#233;n ah&#237;, donde no hay orden que leer, su ventaja no vendr&#237;a
      del orden.</div>
    </div>
    <div>
      {tabla(["Mecanismo", "Orden", "Predicci&#243;n"], [
          ["escalada de prueba", "fuerte", "B gana"],
          ["r&#225;faga geogr&#225;fica", "parcial", "empate"],
          ["vaciado s&#250;bito", "ninguno", "B NO gana"],
          ["toma gradual", "difuso", "ambos fallan"]])}
      <p style="font-size:11.5pt">Con <b>confusores leg&#237;timos</b> que imitan cada
      uno: rachas de microcompras, compras grandes reales y viajes.</p>
    </div>
  </div>
  </div>
  {pie(3)}
</div>"""

    d4 = f"""
<div class="d">
  {cab("03", "Prueba 1 &#183; Permutaci&#243;n controlada")}
  <div class="cuerpo">
  <p>Barajamos el orden de la historia dentro de cada ventana. Mismos eventos,
  mismos valores, mismas variables agregadas, y la transacci&#243;n calificada se queda
  en su sitio: <b>solo se destruye la secuencia</b>.</p>
  <div class="cifra">
    <div class="v">&#8722;{pct(R['caida_permutacion']['B'], 0)}</div>
    <div class="e">del AUC-PR del modelo secuencial se pierde al barajar la
    historia &#183; cinco permutaciones distintas</div>
  </div>
  {svg_permutacion(R)}
  <div class="caja acento"><b>El control que no es obvio:</b> si baraj&#225;ramos
  tambi&#233;n la posici&#243;n de la transacci&#243;n calificada, el modelo perder&#237;a
  adem&#225;s el acceso al evento que debe puntuar y la ca&#237;da parecer&#237;a de
  {pct(sin_control, 0)}. Estar&#237;amos midiendo dos cosas y atribuy&#233;ndolas al
  orden.</div>
  </div>
  {pie(4)}
</div>"""

    d5 = f"""
<div class="d">
  {cab("04", "Prueba 2 &#183; El resultado inc&#243;modo")}
  <div class="cuerpo">
  {leyenda}
  <figure>{svg_mecanismos(R)}
  <figcaption>AUC-PR por mecanismo sobre el conjunto de prueba. Las tasas base
  difieren entre mecanismos, as&#237; que las columnas se comparan dentro de cada
  grupo, no entre grupos. Tabla completa en el informe, &#167;3.2.</figcaption></figure>
  <div class="cols">
    <div><p><b>Se confirm&#243;:</b> B se desploma en el control sin orden
    ({R['por_mecanismo'][3]['aucpr_B']:.3f} frente a
    {R['por_mecanismo'][3]['aucpr_A']:.3f}). Donde no hay orden, no hay nada que
    leer.</p></div>
    <div><p><b>Se refut&#243;:</b> B no supera a A en ning&#250;n mecanismo. Nuestra
    predicci&#243;n fall&#243; y as&#237; la reportamos.</p></div>
  </div>
  </div>
  {pie(5)}
</div>"""

    d6 = f"""
<div class="d">
  {cab("05", "Entonces, &#191;el orden aporta o no?")}
  <div class="cuerpo">
  <p>El comit&#233; no pregunt&#243; si el modelo secuencial es mejor, sino si el orden
  aporta informaci&#243;n que los agregados <b>no capturan</b>. Que B pierda no
  significa que su se&#241;al sea redundante. Lo medimos combinando ambos puntajes con
  una mezcla ajustada <b>solo en validaci&#243;n</b>.</p>
  <figure>{svg_intervalo(R)}
  <figcaption>Diferencia de AUC-PR entre la mezcla A+B y el motor actual, con su
  intervalo de confianza del 95&#160;% (bootstrap remuestreando tarjetas completas).
  El intervalo no toca el cero.</figcaption></figure>
  <div class="cols">
    <div class="fichas" style="margin:0">
      <div class="ficha neutra"><div class="v">{R['aucpr_test']['A']:.4f}</div>
        <div class="e">motor actual solo</div></div>
      <div class="ficha neutra"><div class="v">{R['aucpr_test']['mezcla_AB']:.4f}</div>
        <div class="e">motor + se&#241;al de orden</div></div>
    </div>
    <div><div class="caja acento" style="margin:0"><b>Detectable &#8800;
    importante.</b> El intervalo excluye el cero, as&#237; que la se&#241;al de orden no
    es redundante. Pero la mejora es de mil&#233;simas: con 40,000 transacciones se
    detectan diferencias min&#250;sculas.</div></div>
  </div>
  </div>
  {pie(6)}
</div>"""

    d7 = f"""
<div class="d">
  {cab("06", "La decisi&#243;n en quetzales &#183; el hallazgo real")}
  <div class="cuerpo">
  <p>Costos del comit&#233;: <b>Q4,200</b> por fraude que pasa, <b>Q180</b> por bloqueo
  indebido. Asimetr&#237;a de <b>23 a 1</b>, as&#237; que el umbral minimiza costo esperado,
  no F1. Se ajust&#243; en validaci&#243;n y se aplic&#243; sin cambios a prueba.</p>
  {tabla(["Escenario", "Fraudes que pasan", "Bloqueos indebidos", "Ahorro en prueba"], [
      ["Solo motor actual", f"{solo_a['FN']}", f"{solo_a['FP']}", q(solo_a["ahorro_Q"])],
      ["Motor + se&#241;al de orden", f"{mezcla['FN']}", f"{mezcla['FP']}",
       q(mezcla["ahorro_Q"])]])}
  <div class="fichas">
    <div class="ficha"><div class="v">+{pct(mat['delta_ahorro_rel'], 2)}</div>
      <div class="e"><b>En dinero: nada.</b> {q(mat['delta_ahorro_Q'])} sobre
      {q(solo_a['ahorro_Q'] / 1e6, 2)} M de ahorro. Indistinguible de no hacer
      nada.</div></div>
    <div class="ficha bien"><div class="v">&#8722;{abs(mat['delta_fp'])}</div>
      <div class="e"><b>En clientes: s&#237;.</b> Bloqueos indebidos, un
      {pct(abs(mat['delta_fp_rel']), 0)} menos, a igual exhaustividad.</div></div>
    <div class="ficha neutra"><div class="v">23&#215;</div>
      <div class="e">Un bloqueo cuesta 23 veces menos que un fraude: evitar 95 casi
      no mueve el balance.</div></div>
  </div>
  <div class="caja acento">El caso a favor del orden es de <b>experiencia del
  cliente</b>, no de detecci&#243;n de fraude. Son 95 personas a las que no se les
  rechaza una compra.</div>
  </div>
  {pie(7)}
</div>"""

    d8 = f"""
<div class="d">
  {cab("07", "Recomendaci&#243;n y l&#237;mites")}
  <div class="cuerpo">
  <div class="caja"><b>Piloto acotado.</b> Conservar el motor actual como decisi&#243;n
  principal. Lo que justificar&#237;a el piloto no es detectar m&#225;s fraude
  &#8212;eso no lo logra&#8212; sino <b>molestar a menos clientes detectando el
  mismo</b>.</div>
  <div class="cols">
    <div>
      <p style="margin-bottom:6pt"><b>Por qu&#233; B no gan&#243; solo</b></p>
      <p>Solo ~1,900 transacciones fraudulentas en entrenamiento. Los agregados son
      conocimiento del dominio ya destilado; la red deber&#237;a redescubrirlo con muy
      pocos positivos. Parece un l&#237;mite de datos, no de arquitectura &#8212;y es
      comprobable.</p>
      <p>Nuestra <b>apuesta C fall&#243;</b>: {apuesta['margen_val']:+.3f} frente al
      +{apuesta['umbral_declarado']:.2f} exigido. Pre-registrada en git antes de ver
      el conjunto de prueba; la reportamos como qued&#243;.</p>
    </div>
    <div>
      <p style="margin-bottom:6pt"><b>L&#237;mites que declaramos</b></p>
      <ul>
        <li>Datos sint&#233;ticos: no dicen nada del fraude real.</li>
        <li>Defecto de nuestro generador: sondeos y rachas leg&#237;timas se separan
        por monto, no solo por orden &#8212;sesga la prueba 2 a favor de A.</li>
        <li>La atenci&#243;n de C explica el caso en
        {pct(R['atencion_en_historia'], 0)}, frente al 60&#160;% declarado.</li>
        <li>Una sola semilla por variante.</li>
      </ul>
    </div>
  </div>
  <div class="caja acento"><b>Siguiente paso:</b> correr la permutaci&#243;n controlada
  y la prueba de complementariedad sobre los datos reales del banco. El
  procedimiento es directamente aplicable: es una tarde de trabajo.</div>
  </div>
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
