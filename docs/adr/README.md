# ADR · registros de decisiones de arquitectura

Aquí viven las decisiones de arquitectura que cambian la forma del sistema (tokenización por
defecto, formato de checkpoints, contrato ONNX, protocolo del backend de la fase 2…). Las
decisiones y desviaciones del día a día van al ledger `docs/decisiones-de-ejecucion.md`; las
decisiones de diseño previas al arranque están en `docs/spec/08-riesgos-y-decisiones.md`.

## Formato

Un fichero por decisión, `NNNN-<slug>.md`, numerado en orden de creación:

```
# NNNN · Título

- Fecha: AAAA-MM-DD
- Estado: propuesta | aceptada | sustituida por NNNN

## Contexto
## Decisión
## Consecuencias
```

Una ADR no se edita una vez aceptada: si cambia, se escribe otra que la sustituye y se enlaza.

## Índice

Todavía no hay ADR; la primera llega con P1 (formato de los tokens empaquetados).
