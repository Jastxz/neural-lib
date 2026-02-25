# Neural Lib - Librería de IA para Juegos

Esta librería proporciona implementaciones de **Redes Neuronales** para jugar y predecir movimientos en varios juegos de mesa clásicos: **3 en Raya (Tic-Tac-Toe)**, **Gatos y Ratón (Cats & Mouse)** y **Damas**.

El proyecto incluye dos implementaciones:
- **Red Clásica**: Backpropagation tradicional (optimización matemática)
- **Red Experimental**: Plasticidad hebiana biológicamente inspirada

## Funcionalidades Principales

- **Entrenamiento de Modelos**: Generación de datos de entrenamiento utilizando algoritmos Minimax/Negamax y entrenamiento de redes asociadas.
- **Predicción de Movimientos**: Clase unificada para obtener el mejor movimiento dado un estado del tablero.
- **Micro-Adversarial Search**: Implementaciones eficientes de algoritmos de búsqueda para generación de datasets.

## Juegos Soportados

1. **3 en Raya**: Tablero 3x3. Predicción de movimiento óptimo.
2. **Gatos (Gato vs Ratón)**: Juego asimétrico en tablero 8x8.
3. **Damas**: Juego clásico en tablero 8x8. Predicción de movimientos de origen y destino.

## Uso del Servicio de Predicciones

La clase `es.jastxz.services.ServicioPredicciones` facilita la interacción con los modelos entrenados.

### Formato de Respuesta

Las predicciones devuelven un objeto `Movimiento` que contiene:

- `tablero`: Tablero con el estado del tablero resultante.
- `movimiento`: Movimiento realizado (o destino).

### Ejemplo de Uso

```java
ServicioPredicciones servicio = new ServicioPredicciones();
int[][] tablero = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
Movimiento movimiento = servicio.predecir3enRaya(tablero, 1);
System.out.println("Movimiento: " + movimiento.getPos().getFila() + ", " + movimiento.getPos().getColumna());
```

## Estructura del Proyecto

- `es.jastxz.models`: Definición de modelos y lógica de entrenamiento específicos por juego.
- `es.jastxz.nn`: Implementación de la Red Neuronal (Perceptrón Multicapa).
- `es.jastxz.engine`: Reglas de juego y algoritmos Minimax.
- `es.jastxz.services`: Servicio de predicciones.
- `es.jastxz.tipos`: Clases de soporte (Tablero, Posicion, Mundo).

## Requisitos

- Java 21+
- Maven

---

## 📚 Documentación del Proyecto

### Documentación Principal
- **[HISTORIAL_MEJORAS_RED_EXPERIMENTAL.md](HISTORIAL_MEJORAS_RED_EXPERIMENTAL.md)** - Historial completo de todas las mejoras implementadas en la red experimental
- **[HISTORIAL_REFACTORIZACIONES.md](HISTORIAL_REFACTORIZACIONES.md)** - Todas las refactorizaciones importantes del proyecto
- **[MEJORAS_INHIBICION_LATERAL_Y_UMBRALES.md](MEJORAS_INHIBICION_LATERAL_Y_UMBRALES.md)** - Mejoras más recientes (inhibición lateral y umbrales variables)

### Comparativas
- **[RESUMEN_COMPARATIVA.md](RESUMEN_COMPARATIVA.md)** - Resumen ejecutivo de todas las comparativas
- **[COMPARATIVA_3ENRAYA.md](COMPARATIVA_3ENRAYA.md)** - Comparativa detallada: 3 en Raya
- **[COMPARATIVA_GATOS.md](COMPARATIVA_GATOS.md)** - Comparativa detallada: Gatos
- **[COMPARATIVA_DAMAS.md](COMPARATIVA_DAMAS.md)** - Comparativa detallada: Damas
- **[COMPARATIVA_SUPERVISADO.md](COMPARATIVA_SUPERVISADO.md)** - Backpropagation vs Plasticidad Hebiana

### Recursos Adicionales
- **[EJEMPLOS_USO.md](EJEMPLOS_USO.md)** - Ejemplos de uso de la librería
- **[TAREAS_RECOMENDADAS.md](TAREAS_RECOMENDADAS.md)** - Tareas pendientes y mejoras sugeridas
- **[Apuntes_Neurologicos.md](Apuntes_Neurologicos.md)** - Fundamentos neurológicos del proyecto
- **[EXPLICACION_STACKOVERFLOW.md](EXPLICACION_STACKOVERFLOW.md)** - Explicación del problema de StackOverflow y su solución

### Estado Actual
- **Tests pasando:** 126/126 ✅
- **Rendimiento:** Red experimental 48% mejor que clásica
- **Última actualización:** 25 de febrero de 2026
