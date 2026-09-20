# Lo que un afinado cambia, y lo que no

*Borrador del post 2 de la serie de Rukh. Escrito el 2026-09-20 al cerrar P4. Todos los números
salen de `docs/benchmarks.md` y de `docs/decisiones-de-ejecucion.md`; ninguno está redondeado a
favor.*

---

Rukh es un modelo de lenguaje de ajedrez que estoy construyendo desde cero para aprender cómo
funcionan por dentro. No juega con búsqueda: lee una partida como una secuencia de tokens y predice
el siguiente, igual que un modelo de texto predice la siguiente palabra. En los tres hitos
anteriores aprendió a jugar. En el cuarto tocaba **afinarlo**: enseñarle a jugar de otra manera sin
volver a entrenarlo desde cero.

Salió que sí, pero no como esperaba. Y lo que salió es mejor que lo que buscaba.

## El plan era un dial

La idea era sencilla y sigue siéndolo. Cada partida de entrenamiento empieza con tres tokens:

```
<bos> <w1800> <b1800> e2e4 e7e5 g1f3 ...
```

Los dos del medio son el Elo de cada jugador. Si el modelo ve millones de partidas de gente de 1500
precedidas de `<w1500>` y millones de gente de 2400 precedidas de `<w2400>`, debería aprender dos
distribuciones distintas y elegir según el prefijo. Es *instruction tuning* en su forma más pequeña
posible: la instrucción no es una capa ni una plantilla, es un token.

Y en la demo se convierte en un selector: **«juega como un 1500»**. Ese era el objetivo del hito.

## El fallo llevaba tres hitos escondido en un YAML

Antes de afinar nada, se me ocurrió contar. El segundo token de cada partida empaquetada *es* la
cabecera de las blancas, así que un histograma de tres líneas dice cuántas veces ha visto el modelo
cada condición. Sobre los mil seiscientos ochenta millones de tokens con los que se entrenó:

```
headers below <w1800>: 0 of 12 ever seen in this corpus
```

Cero de doce. Los tokens por debajo de 1800 existían en el vocabulario, tenían su fila en la tabla
de embeddings, y **nunca habían recibido un gradiente**. Su vector seguía donde lo dejó la
inicialización.

La causa estaba en un fichero de configuración escrito en el primer hito:

```yaml
min_elo: 1800
```

Ese filtro se aplica a los dos jugadores, y con él se descargó todo: el corpus base, la base de
élite, todo. Así que pedirle al modelo «juega como un 1500» no era pedirle nada. Era ponerle ruido
en la entrada.

Nada falló. No hubo excepción, ni aviso, ni test en rojo: el token existe, el tokenizador lo emite,
el modelo lo lee y produce una distribución perfectamente normal sobre jugadas legales. Solo que la
distribución no cambiaba como debía, y eso se confunde con facilidad con «el condicionamiento no
funciona».

**La regla que deja: antes de concluir que un mecanismo no funciona, comprueba que ha recibido
datos.**

## El arreglo funcionó, y se ve sin jugar una partida

Descargué el tramo que faltaba, construí un corpus plano en Elo —igual número de partidas por cada
banda de 100 puntos, de 1000 a 2400— y afiné el modelo sobre él.

La primera medida no necesita ni una partida: es la distribución del propio modelo sobre las veinte
primeras jugadas legales, leída directamente de la softmax, sin muestreo ni semilla. Su entropía en
bits dice cuánto varía su repertorio.

| Cabecera  | antes       | después      |
| --------- | ----------: | -----------: |
| `<w1200>` | 2,8556 bits | **1,5955**   |
| `<w1500>` | 2,9369      | **1,6470**   |
| `<w1800>` | 1,7695      | 1,7408       |
| `<w2000>` | 1,8694      | 1,8406       |
| `<w2100>` | 1,9033      | 1,8809       |
| `<w2400>` | 2,0104      | 1,9916       |

Dos cosas ahí. La primera: en la columna «antes» hay un escalón de **1,17 bits** justo entre 1500 y
1800, exactamente donde cortaban los datos, y con el signo del ruido — por debajo del suelo el
modelo estaba *menos* decidido, no más. La segunda: después del afinado el eje es **monótono en las
seis condiciones**. Un jugador de club abre `1. e4` o `1. d4` y poco más; el repertorio se ensancha
con la fuerza. El modelo reproduce esa forma, y la reproduce en orden, sin que nadie se lo pidiera:
el objetivo de entrenamiento no menciona la entropía por ninguna parte.

## Y entonces medí el Elo

Seis condiciones, 160 partidas cada una contra ocho niveles de Stockfish, mismos rivales, misma
semilla, mismas posiciones. Lo único que cambia entre filas son los dos tokens de la cabecera.

| Cabecera  |      Elo |     IC 95 % |
| --------- | -------: | ----------: |
| `<w1200>` |     1425 | 1361 – 1479 |
| `<w1500>` |     1549 | 1481 – 1609 |
| `<w1800>` |     1498 | 1435 – 1552 |
| `<w2000>` |     1538 | 1479 – 1597 |
| `<w2100>` |     1606 | 1543 – 1670 |
| `<w2400>` |     1644 | 1578 – 1707 |

El criterio que me había escrito antes de medir era `Elo(1500) < Elo(2000) < Elo(2400)` con
intervalos. **No se cumple**: el primer par ya va del revés.

Aquí es donde el hito se pone interesante, porque «los intervalos se solapan» tiene dos causas que
se parecen en el informe y no se parecen en nada más: **faltan partidas** o **no hay diferencia**.

## Tres maneras de saber cuál de las dos es

**La primera es aritmética.** Lo que la escalera mide no es Elo sino una tasa de puntos, que es una
proporción, así que su error típico es `sqrt(p(1-p)/n)` y dos intervalos del 95 % dejan de tocarse
cuando la distancia supera `1,96 (se₁ + se₂)`. Despejando: hacen falta unas `3,84 / (Δp)²` partidas
por condición. Metiendo los números medidos:

| Par                     |          tasa |      Δ | partidas por condición |
| ----------------------- | ------------: | -----: | ---------------------: |
| `<w1500>` → `<w2000>`   | 0,466 → 0,453 | −0,013 |             **24 420** |
| `<w2000>` → `<w2400>`   | 0,453 → 0,569 | +0,116 |                    284 |

Las dos filas dicen «no separado» y significan lo contrario. Doscientas ochenta y cuatro partidas
son una hora de máquina. Veinticuatro mil cuatrocientas veinte son sesenta horas de Stockfish para
estrechar un intervalo alrededor de una diferencia que no está.

**La segunda es un grupo de control.** Entre `<w1200>` y `<w2100>` el modelo afinado va de 1425 a
1606, con los intervalos separados. Leído solo, eso es «el condicionamiento da fuerza». Así que
corrí el mismo barrido sobre el modelo **sin afinar**, para el que `<w1200>` es un vector de la
inicialización:

| Cabecera  | Modelo   |  Elo |
| --------- | -------- | ---: |
| `<w1200>` | sin afinar | 1419 |
| `<w1200>` | afinado    | 1425 |
| `<w2100>` | sin afinar | 1580 |
| `<w2100>` | afinado    | 1606 |

El modelo sin afinar recorre 161 puntos por el mismo eje. Y no puede ser condicionamiento, porque su
`<w1200>` nunca recibió un gradiente. Lo que le pasa es otra cosa: un prefijo desconocido le
**estorba**, y estorbarle cuesta unos ciento sesenta puntos. El afinado mueve 181, que está dentro
del ruido de los 161.

**La tercera me la encontré sin buscarla.** El mismo modelo, la misma cabecera y la misma semilla
acabaron midiéndose dos veces por caminos distintos: **1498 y 1558**. La semilla fija mi muestreo,
no el de Stockfish, que juega con un límite de tiempo y con `UCI_LimitStrength`, que además
aleatoriza a propósito. Son 1,18 sigmas — ruido de manual — pero dejan el suelo de
**reproducibilidad** del instrumento en unos 40 puntos de Elo. Los pares contiguos de mi tabla están
a 11, 40, 51 y 38. Están por debajo de lo que el instrumento repite.

## Lo que el afinado sí hizo

La misma tirada, a la misma cabecera, comparando el modelo afinado con el que no lo está:

| A `<w1200>`                  | sin afinar | afinado     |
| ---------------------------- | ---------: | ----------: |
| jugadas ilegales sin máscara |   4 de mil |       **0** |
| top-1 contra jugada humana   |     49,8 % | **51,4 %**  |
| puzles                       |     36,4 % | **37,0 %**  |
| entropía de 1.ª jugada       | 2,856 bits | **1,596**   |

Cero jugadas imposibles contra cuatro de mil. Punto y medio más de acierto. Y 1,26 bits menos de
entropía. El afinado convirtió «ruido en la entrada» en «un jugador de club decidido» — y eso se ve
en todo menos en el resultado contra Stockfish.

Y a `<w2100>`, donde la cabecera siempre estuvo entrenada, los dos modelos coinciden en las cinco
columnas. Que no se tocara lo que ya funcionaba es cómo se comprueba que no hubo olvido
catastrófico.

## El caso más claro: 1,6 MB que cambian una decisión y no cuestan nada

La otra mitad del hito fue LoRA: en vez de mover los 115 millones de pesos, aprender una corrección
de rango bajo y guardarla aparte. Dos adaptadores, uno entrenado sobre 200 000 partidas que empiezan
`1. e4` y otro sobre 200 000 que empiezan `1. d4`. **393 216 números cada uno, el 0,34 % del
modelo, 1,6 MB en disco.**

| Modelo             | entropía 1.ª jugada |      `e2e4` |      `d2d4` |
| ------------------ | ------------------: | ----------: | ----------: |
| base               |         1,7695 bits |     59,64 % |     26,68 % |
| con `lora-e4`      |          **0,0209** | **99,85 %** |      0,06 % |
| con `lora-d4`      |          **0,0160** |      0,06 % | **99,88 %** |

Y lo que costó, medido con la misma suite:

|               | legales | top-1  | puzles |
| ------------- | ------: | -----: | -----: |
| base          |  99,8 % | 54,4 % | 37,5 % |
| con `lora-e4` |  99,8 % | 54,7 % | 37,8 % |

Nada. Ni una décima. Y con la primera jugada clavada en `1. e4`, doscientas auto-partidas de doce
jugadas produjeron **199 líneas distintas**: el adaptador fijó el primer movimiento y no tocó
ninguno de los once siguientes.

Que el rango no es una metáfora se comprueba con una línea de álgebra: los valores singulares de la
corrección son exactamente **ocho**, y el noveno no es pequeño, es cero. En norma pesa el 4,41 % de
la matriz que corrige.

## Un adaptador de 1,6 MB no debería costar 221 MB al desplegarlo

Aquí hubo un problema de ingeniería que me gustó resolver. Fundir el adaptador en los pesos y
exportar el ONNX cuesta 221 MB **por estilo**. Dos estilos, 442 MB de descarga para mover 1,6 MB de
corrección: el método es barato y el formato se lo come.

La salida fue sacar la corrección de los pesos y meterla en el **grafo**: exportar el modelo con las
dos matrices de LoRA como *entradas*, apiladas por capas. El navegador cambia de estilo **subiendo**
1,6 MB, no descargando otro modelo. Tres propiedades lo hacen honesto, y las tres están en los
tests: con los factores a cero el grafo es el modelo base exactamente; con otro adaptador la
respuesta cambia sin tocar el fichero; y la paridad contra PyTorch se mide sobre mil posiciones
reales.

La prueba de navegador acaba con la aserción que resume el hito entero:

```ts
expect(downloads.filter((url) => url.endsWith('.onnx'))).toHaveLength(1);
```

Un solo modelo descargado en toda la prueba, dos estilos jugados.

## ¿Y merecía la pena escribir un modelo propio?

Esa era la pregunta que el curso llevaba tres módulos sin contestar. La contesté afinando
**Qwen3-0.6B** con las mismas partidas escritas como PGN, con un presupuesto elegido a propósito
para ser generoso con él: rango 16 —el doble que mis adaptadores—, 1 500 pasos, unos 24 millones de
tokens de ajedrez, sobre un modelo de 600 millones de parámetros que ya sabe hablar. Contra mis 115
millones entrenados desde cero.

| Métrica                     | decoder propio | Qwen3-0.6B afinado |
| --------------------------- | -------------: | -----------------: |
| Parámetros                  |          115 M |          **600 M** |
| Jugadas legales sin máscara |     **99,8 %** |            62,50 % |
| Top-1                       |     **54,4 %** |             12,5 % |
| Puzles                      |     **37,5 %** |              0,9 % |
| Elo                         |       **1504** |            < 807   |

Ese último no es una estimación: es una cota. Jugó 160 partidas contra los ocho niveles, incluido el
más flojo, y las perdió **todas**.

Y una de cada tres de sus respuestas es una jugada bien escrita que esa posición no permite; tres de
cada cien ni siquiera son una jugada. Mi decoder no puede cometer ninguno de esos dos errores,
porque su vocabulario **es** el conjunto de jugadas.

Esto no demuestra que Qwen sea malo. Demuestra algo más estrecho y más útil: cuando tu salida tiene
una estructura **cerrada, pequeña y verificable**, meterla en la representación vale más que
quinientos millones de parámetros extra entrenados para otra cosa.

## Lo que me llevo

El hito no cumplió su criterio principal. Lo publico incumplido, con los números, porque el valor
de esto está en que lo que enseñe sea cierto y no en aprobar una condición que escribí antes de
medir.

Y lo que dejó es más útil que la casilla marcada:

- **Afinar cambia el comportamiento y no cambia la competencia.** Un fichero de 1,6 MB controla una
  decisión por completo y gratis; ningún afinado del hito mueve el Elo de forma demostrable.
- **El mecanismo explica las dos mitades.** Predecir el siguiente token sobre partidas humanas
  enseña *qué se juega* a cada nivel; nunca premia *calcular mejor*. Mover la competencia necesita
  otra herramienta, y eso es el hito siguiente: recompensas, DPO, GRPO.
- **Tres formas de distinguir «no hay efecto» de «no lo medí bien»**: la aritmética de las partidas
  necesarias, la corrida de control sobre el modelo sin tratar, y medir cuánto se mueve tu montaje
  cuando no cambias nada.

Esa última es la que más se me va a quedar. Antes de explicar una diferencia pequeña, mide tu propio
ruido. A mí me lo enseñó el mismo experimento midiéndose dos veces sin querer.

---

*El código, los modelos y todas las mediciones están abiertos. El curso completo, con esta lección
entera y sus laboratorios, está en el módulo M4.*
