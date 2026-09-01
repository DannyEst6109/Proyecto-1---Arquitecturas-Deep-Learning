# Apuesta del equipo (Pieza C) — declaración previa

**Daniel Estrada (20853) · Hansel López (19026)**

Este archivo se versiona **antes** de entrenar el modelo C a escala completa y
**antes** de mirar el conjunto de prueba. El commit que lo introduce es la
constancia de que la hipótesis y el criterio de éxito no se ajustaron después
de ver los resultados.

## La frase

> **Creemos que** combinar el resumen agregado de la ventana con una lectura
> *atendida* de la secuencia (GRU + atención aditiva, concatenada con las
> variables agregadas) **mejorará** el AUC-PR sobre el mejor de A y B
> **porque** A y B capturan evidencias complementarias: A resuelve bien los
> fraudes cuyo indicio está en la magnitud (`vaciado_subito`) y B los que están
> en la progresión temporal (`escalada_prueba`); ningún modelo por separado ve
> las dos cosas. **Lo consideraremos útil si** el AUC-PR en **validación**
> supera al mejor de A y B por **≥ 0.02 absoluto**, y además la atención asigna
> su peso máximo a una transacción del episodio de fraude en la mayoría de los
> verdaderos positivos de `escalada_prueba`.

## Control experimental

La mejora podría venir de tres sitios distintos. Para separarlos se entrenan
tres variantes con **el mismo presupuesto de entrenamiento, la misma partición
y la misma semilla**:

| Variante | GRU | Atención | Agregadas | Qué aísla |
|---|:--:|:--:|:--:|---|
| B (referencia) | sí | no | no | secuencia sola |
| C1 | sí | sí | no | aporte de la **atención** |
| C2 = C | sí | sí | sí | aporte de **combinar** agregadas + secuencia |

- Si `C1 ≈ B` y `C2 > B`, la ganancia viene de la fusión con los agregados.
- Si `C1 > B`, la atención aporta por sí sola.
- Si `C2 ≈ max(A, B)`, la apuesta **falla** y así se reportará.

## Métrica de éxito y regla de decisión

- **Métrica:** AUC-PR (precisión promedio) en **validación**.
- **Umbral de éxito:** `AUC-PR(C2) ≥ max(AUC-PR(A), AUC-PR(B)) + 0.02`.
- **Criterio de interpretabilidad:** en ≥ 60 % de los verdaderos positivos de
  `escalada_prueba`, el peso máximo de atención cae sobre una transacción
  etiquetada como fraude del mismo episodio.
- La decisión de éxito/fracaso se toma **con validación**. El conjunto de
  prueba se mira una sola vez, después, y solo para reportar.

## Qué esperamos que falle

Declarado por adelantado, para poder contrastarlo en el análisis de error:

1. **`toma_gradual` seguirá siendo el mecanismo peor detectado** por los tres
   modelos. La señal está repartida en decenas de transacciones y ninguna es
   anómala por sí misma; con K = 20 la ventana ni siquiera cubre el episodio
   completo. Es el caso de falla que el generador incluye a propósito.
2. La atención será **más útil como explicación que como mejora numérica**: es
   plausible que aporte poco AUC-PR y aun así sea valiosa para el analista.

## Nota de honestidad

Durante el desarrollo se ejecutó una corrida piloto a escala reducida
(900 tarjetas, 6 épocas) para depurar el código, cuyos números vimos. El umbral
de +0.02 se fija sobre **validación a escala completa**, que no se había
ejecutado al escribir este archivo. Se deja constancia en lugar de omitirlo.
