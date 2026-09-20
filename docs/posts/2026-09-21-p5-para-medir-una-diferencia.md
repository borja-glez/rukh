# Para medir una diferencia, mide la diferencia

*Borrador del post 3 de la serie de Rukh. Escrito el 2026-09-21 al cerrar P5. Todos los números
salen de `docs/benchmarks.md`, de los informes de `artifacts/eval/` y de
`docs/decisiones-de-ejecucion.md`; ninguno está redondeado a favor.*

---

Rukh es un modelo de lenguaje de ajedrez que estoy construyendo desde cero para aprender cómo
funcionan por dentro. En los cuatro hitos anteriores aprendió a jugar y aprendió a jugar **de otra
manera**. El cuarto terminó con una frase incómoda: afinar cambia el comportamiento de forma total
y barata, y **no cambia la competencia**. Un adaptador de un megabyte y medio consigue que abra
siempre con `1. e4`, y ningún afinado le hizo jugar mejor.

El quinto hito iba de cambiar la competencia. Y lo primero que tuve que hacer no fue entrenar nada.

## El instrumento no daba para la pregunta

El criterio del hito era «+50 puntos de Elo con intervalo de confianza sobre su base». La manera
obvia de comprobarlo es medir el Elo del modelo base, medir el del modelo alineado y restar.

Con los números que ya tenía:

| | Elo | intervalo 95 % |
|---|---|---|
| `medium-v4` | 1504 | 1446 a 1558 |
| `medium-v4` + DPO | 1529 | 1470 a 1583 |

La resta da **+25**. ¿Y su intervalo? Cada medida tiene un semiancho de unos 56 puntos, y los
errores independientes se suman en cuadratura: `√(56² + 56,5²) = 80`. Así que la diferencia va de
**−55 a +105**.

Atraviesa el cero de lado a lado. Y es **más ancha que cualquiera de las dos medidas que entraron**,
que es la parte que nadie espera la primera vez que la ve. Restar dos números medidos no cancela sus
ruidos: los suma.

La razón de que esos intervalos sean tan anchos ya la había medido en el hito anterior. Mi escalera
de Elo hace jugar al modelo contra ocho Stockfish de fuerza limitada, y ese rival **juega con reloj
y aleatoriza a propósito** para acertar la fuerza que se le pide. Corrí la misma medición dos veces
con la misma semilla y salió 1498 y 1558. El suelo de reproducibilidad son unos 40 puntos.

Así que no era cuestión de jugar más partidas. Era el instrumento equivocado.

## El instrumento que sí

Si lo que quieres saber es una diferencia, **mide la diferencia**: que los dos modelos jueguen entre
sí. Desaparece el tercero cuyo humor hay que promediar, porque los dos juegan la misma partida.

La aritmética es favorable y se puede calcular antes de pagarla. Detectar +50 Elo restando dos
proporciones pide unas 750 partidas **por lado**; detectarlo en un enfrentamiento directo pide 185
**en total**. Ocho veces más barato.

Los mismos dos modelos de la tabla de arriba, enfrentados: **+40 Elo, de 11 a 69**. Cuatro minutos
de máquina. Misma conclusión cualitativa, y un instrumento que la puede afirmar mientras el otro no.

## Lo primero que hice con el instrumento nuevo fue romperlo

Un modelo contra una copia exacta de sí mismo tiene que dar 0,5000 clavado. Es la comprobación más
barata que existe.

Salió **0,975**, con 999 jugadas ilegales.

Dos errores, ninguno sutil una vez que hay un número imposible señalándolos. La partida empezaba
desde un tablero con las jugadas del libro de aperturas ya puestas, y **nunca se las reproducía al
jugador**: el modelo veía un tablero con seis jugadas hechas y un prompt vacío. Y al rival no se le
contaba ninguna jugada, porque el protocolo no tenía manera de decírselo y nadie preguntaba si la
tenía.

Arreglados: 0,5000 exacto, 11 ilegales por bando.

Eso costó un minuto de máquina y evitó publicar un «+40» que habría sido un artefacto de fontanería.

## Los modelos, por fin

Con el instrumento en pie, tres maneras de cambiar lo que el modelo **prefiere**:

**DPO sobre pares de fuera de política.** Trece mil pares de jugadas que Stockfish ya había
ordenado, de evaluaciones de Lichess. Dicen «aquí `e4` es mejor que `h4`» tanto si el modelo iba a
considerar alguna como si no.

**DPO sobre pares propios.** Las mismas posiciones, pero las dos jugadas salen del **propio
modelo**: se le pide que proponga cuatro, se puntúan con el motor, y se guarda la mejor y la peor.
El argumento cabe en una línea: un modelo solo pierde partidas con las jugadas que juega.

**GRPO contra una recompensa verificable.** Sin pares y sin modelo de recompensa: el modelo propone
un grupo de ocho jugadas, una **función** las puntúa —legalidad, calidad en centipeones, mate,
repetición— y cada una se empuja arriba o abajo según cómo le fue frente a la media de su grupo.

Los tres baten a la base, y los tres con el intervalo separado del cero:

| | Elo sobre la base | intervalo |
|---|---|---|
| DPO, pares de fuera | **+67** | 45 a 89 |
| DPO, pares propios | **+57** | 36 a 79 |
| GRPO | **+44** | 23 a 66 |

Es la primera vez en el proyecto que este criterio se mide separado del cero. Lo que cambió no fue
el modelo: fue cómo se mira.

## Y entonces el triángulo no cerró

Si el de fuera está 67 por encima de la base y el de dentro está 57, el de dentro debería estar unos
diez **por debajo** del de fuera.

Los enfrenté. El de dentro gana por **+37**, con 800 partidas y el intervalo separado del cero.

El triángulo deja un residuo de **47 puntos de Elo**, a 2,4 sigmas. No es que los intervalos sean
generosos: es que un solo número de fuerza por modelo supone que la fuerza es un **orden total**, y
un emparejamiento no está obligado a respetarlo. El modelo entrenado con pares de fuera juega más
afilado —lo dice su tasa de jugadas ilegales— y eso le va bien contra la base y mal contra un modelo
más limpio.

La lección práctica es la misma de antes, un piso más arriba: si te importa cuál de dos modelos
gana, **enfréntalos**. Restar dos enfrentamientos contra un tercero es el mismo error que restar dos
escaleras.

## El peaje que no estaba buscando

El instrumento cuenta las jugadas ilegales de cada bando porque el control lo necesitaba. Y salió
una métrica que nadie pidió:

| | ilegales sobre sus propias jugadas |
|---|---|
| base | 1,3 – 1,5 % |
| GRPO | 1,9 % |
| DPO, pares propios | 2,4 – 2,6 % |
| DPO, pares de fuera | 2,9 – 3,8 % |

Alinear **dobla** la tasa de propuestas ilegales, y el anclaje que existe precisamente para evitarlo
no lo evitó. No rompe las partidas —la propuesta ilegal se rescata— pero es distribución que se ha
ido a otro sitio.

Y la diferencia entre los dos DPO tiene mecanismo, no solo signo. Los pares de fuera contienen mates
que el modelo **nunca iba a proponer**, así que el entrenamiento empuja probabilidad hacia tokens
donde no había casi nada. Los pares propios solo mueven probabilidad **entre jugadas que ya estaban
sobre la mesa**. El brazo que más deforma es el que más ilegales propone, y son el mismo.

Si el instrumento no las hubiera contado, habría publicado el modelo diciendo la mitad de lo que sé
de él.

## La recompensa que se maximiza no jugando

Esta es la parte que más me gustó, y la que peor me dejó.

Había escrito un laboratorio para enseñar *reward hacking*: seis funciones de recompensa, cinco de
ellas rotas a propósito de maneras que parecen razonables sobre el papel, midiendo cuál corona cada
una. La hipótesis era que las rotas coronarían otra jugada.

**Las seis coronan la misma.** En retrospectiva es obvio: todas son monótonas en la evaluación del
motor, así que el máximo no se puede mover. El hackeo de recompensa no vive en la cima de la
ordenación.

Vive en lo que el optimizador consume de verdad, que no es la recompensa sino la recompensa **menos
la media de su grupo**. Y ahí sí se ve: sin tope por abajo en el término de calidad, una sola
candidata catastrófica se lleva el 46 % de toda la señal del grupo, y las jugadas **ilegales** salen
puntuando por encima de una legal que está a 355 centipeones del óptimo. La puerta de legalidad no
se había tocado; lo que cambió fue el rango de lo que hay debajo, y un suelo solo es un suelo en
relación con lo que tiene que estar por debajo.

Arreglé la galería. Y entonces me hice la misma pregunta sobre **mi propia** recompensa, la que no
estaba rota: *¿qué política maximizaría exactamente este número?*

Una que proponga la misma jugada ocho veces de ocho. Porque la calidad se mide contra la mejor
candidata **del grupo**, y si las ocho son la misma, cada una es la mejor.

Es decir: mi métrica de «la recompensa mejoró» sube cuando el modelo **deja de ser diverso**. Lo
comprobé con tres corridas que solo cambiaban la tasa de aprendizaje:

| tasa | recompensa | grupos donde las ocho eran la misma jugada |
|---|---|---|
| 1e-6 | +0,002 | 40 % |
| 5e-6 | +0,021 | 46 % |
| 2e-5 | +0,027 | **53 %** |

Sube la recompensa y sube el colapso, monótono. Con dosis-respuesta y todo.

El arreglo es separar dos números: el entrenamiento sigue usando la referencia del grupo —quitarla
dejaría sin gradiente justo los grupos donde más hay que aprender— y la **métrica publicada** usa el
mejor movimiento que había realmente disponible, una llamada al motor por posición. Con esa, GRPO
mejora de verdad: **+0,028**. Y la mejora tiene forma de U invertida, porque a la tasa más alta
vuelve a caer mientras el colapso sigue subiendo.

## Lo que me llevo

Tres cosas, y ninguna es sobre alineamiento.

**Antes de explicar una diferencia pequeña, mide cuánto se mueve tu montaje cuando no cambias nada.**
Me pasó dos veces en este hito. La escalera daba 1498 y 1558. Y el modelo de recompensa, reentrenado
con la misma configuración y la misma semilla para añadirle un campo al informe, dio 74,21 % donde
antes había dado 72,91 %: 1,3 puntos de kernels de GPU que no son deterministas. El criterio del
hito pedía 75 %, que está por debajo del ruido del instrumento que lo mide.

**Cuando un experimento contesta «no hay efecto», la primera sospecha es la pregunta.** La galería
preguntaba por la cima y el daño estaba en la forma.

**Y antes de publicar una métrica, describe la política que la maximizaría.** Si esa política no es
la que quieres, la métrica no es la que quieres. Esa pregunta encontró un fallo en mi propio bucle
que ninguna de las cinco recompensas rotas a propósito habría enseñado.

El curso entero está en [lab.rukh.borjaglez.com](https://lab.rukh.borjaglez.com), y contra el modelo
se puede jugar en [rukh.borjaglez.com](https://rukh.borjaglez.com).
