# Neural Lib - Librería de IA para Juegos

Esta librería proporciona implementaciones de **Redes Neuronales** para jugar y predecir movimientos en varios juegos de mesa clásicos: **3 en Raya (Tic-Tac-Toe)**, **Gatos y Ratón (Cats & Mouse)** y **Damas**.

## Funcionalidades Principales

- **Entrenamiento de Modelos**: Generación de datos de entrenamiento utilizando algoritmos Minimax/Negamax y entrenamiento de redes asociadas.
- **Predicción de Movimientos**: Clase unificada para obtener el mejor movimiento dado un estado del tablero.
- **Micro-Adversarial Search**: Implementaciones eficientes de algoritmos de búsqueda para generación de datasets.

## Juegos Soportados

1. **3 en Raya**: Tablero 3x3. Predicción de movimiento óptimo.
2. **Gatos (Gato vs Ratón)**: Juego asimétrico en tablero 8x8.
3. **Damas**: Juego clásico en tablero 8x8. Predicción de movimientos de origen y destino.

## Uso del Servicio de Predicciones

La clase `org.javig.services.ServicioPredicciones` facilita la interacción con los modelos entrenados.

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

- `org.javig.models`: Definición de modelos y lógica de entrenamiento específicos por juego.
- `org.javig.nn`: Implementación de la Red Neuronal (Perceptrón Multicapa).
- `org.javig.engine`: Reglas de juego y algoritmos Minimax.
- `org.javig.services`: Servicio de predicciones.
- `org.javig.tipos`: Clases de soporte (Tablero, Posicion, Mundo).

## Requisitos

- Java 21+
- Maven
