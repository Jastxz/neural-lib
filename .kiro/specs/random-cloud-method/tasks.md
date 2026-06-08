# Plan de Implementación: Método de la Nube Aleatoria

## Visión General

Implementar el Método de la Nube Aleatoria como un nuevo módulo en el paquete `es.jastxz.nn.nube`, siguiendo los patrones arquitectónicos del módulo genético. El plan comienza con las modificaciones necesarias a `NeuralNetwork`, luego construye los componentes del módulo de forma incremental (configuración → política de eliminación → motor → informe), y finaliza con la integración y tests de extremo a extremo.

## Tareas

- [x] 1. Añadir getters y constructor con pesos a NeuralNetwork
  - [x] 1.1 Añadir métodos `getTopology()`, `getWeights()` y `getBiases()` a `NeuralNetwork`
    - `getTopology()` retorna una copia defensiva del array `int[] topology`
    - `getWeights()` retorna una copia inmutable de la lista de matrices de pesos
    - `getBiases()` retorna una copia inmutable de la lista de matrices de biases
    - Archivo: `src/main/java/es/jastxz/nn/NeuralNetwork.java`
    - _Requisitos: 8.1, 4.6_

  - [x] 1.2 Añadir constructor `NeuralNetwork(int[] topology, List<Matrix> weights, List<Matrix> biases)`
    - Permite crear una red con pesos y biases pre-existentes (para reconstrucción de redes reducidas)
    - Debe validar que las dimensiones de weights y biases sean coherentes con la topología
    - Archivo: `src/main/java/es/jastxz/nn/NeuralNetwork.java`
    - _Requisitos: 4.6, 8.1_

  - [ ]* 1.3 Escribir tests unitarios para los nuevos getters y constructor de NeuralNetwork
    - Verificar que getters retornan copias defensivas (modificar la copia no afecta la red)
    - Verificar que el constructor con pesos produce una red funcional con feedforward correcto
    - Archivo: `src/test/java/es/jastxz/nn/nube/NeuralNetworkNubeTest.java`
    - _Requisitos: 8.1, 4.6_

- [x] 2. Implementar ConfiguracionNube y ConfiguracionNubeBuilder
  - [x] 2.1 Crear el record `ConfiguracionNube` con validaciones en el constructor compacto
    - Campos: `tamañoNube`, `topologiaInicial`, `umbralAcierto`, `neuronasEliminar`, `epocasRefinamiento`, `tasaAprendizaje`, `semilla`
    - Validaciones: tamañoNube ≥ 1, topologiaInicial.length ≥ 3, umbralAcierto en [0.0, 1.0], neuronasEliminar ≥ 1, capas ocultas con ≥ 1 neurona
    - Archivo: `src/main/java/es/jastxz/nn/nube/ConfiguracionNube.java`
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 2.2 Crear `ConfiguracionNubeBuilder` con valores por defecto
    - Valores por defecto: tamañoNube=10, topologiaInicial={2,4,1}, umbralAcierto=0.5, neuronasEliminar=1, epocasRefinamiento=1000, tasaAprendizaje=0.1, semilla=System.nanoTime()
    - Métodos fluidos para cada parámetro y método `build()` que retorna `ConfiguracionNube`
    - Archivo: `src/main/java/es/jastxz/nn/nube/ConfiguracionNubeBuilder.java`
    - _Requisitos: 1.6_

  - [ ]* 2.3 Escribir test de propiedad para validación de configuración
    - **Propiedad 1: Validación de configuración rechaza parámetros inválidos**
    - **Valida: Requisitos 1.2, 1.3, 1.4, 1.5**
    - Archivo: `src/test/java/es/jastxz/nn/nube/ConfiguracionNubePropertyTest.java`

  - [ ]* 2.4 Escribir test de propiedad para ida y vuelta de campos
    - **Propiedad 2: Ida y vuelta de campos de configuración**
    - **Valida: Requisito 1.1**
    - Archivo: `src/test/java/es/jastxz/nn/nube/ConfiguracionNubePropertyTest.java`

- [x] 3. Checkpoint - Verificar configuración
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

- [x] 4. Implementar PoliticaEliminacion y PoliticaEliminacionSecuencial
  - [x] 4.1 Crear la interfaz funcional `PoliticaEliminacion`
    - Método: `int[] siguienteReduccion(int[] topologiaActual, int neuronasEliminar)`
    - Retorna la nueva topología reducida, o null si no hay más reducciones posibles
    - Archivo: `src/main/java/es/jastxz/nn/nube/PoliticaEliminacion.java`
    - _Requisitos: 5.2_

  - [x] 4.2 Implementar `PoliticaEliminacionSecuencial`
    - Elimina las últimas x neuronas comenzando por la última capa oculta
    - Cuando una capa llega a 0, avanza a la capa oculta anterior
    - Retorna null cuando todas las capas ocultas tienen 0 neuronas
    - Archivo: `src/main/java/es/jastxz/nn/nube/PoliticaEliminacionSecuencial.java`
    - _Requisitos: 5.1, 5.3_

  - [ ]* 4.3 Escribir test de propiedad para reducción de topología
    - **Propiedad 8: Reducción de topología**
    - **Valida: Requisito 4.2**
    - Archivo: `src/test/java/es/jastxz/nn/nube/PoliticaEliminacionPropertyTest.java`

  - [ ]* 4.4 Escribir test de propiedad para orden de la política secuencial
    - **Propiedad 10: Orden de la política secuencial**
    - **Valida: Requisito 5.3**
    - Archivo: `src/test/java/es/jastxz/nn/nube/PoliticaEliminacionPropertyTest.java`

- [x] 5. Implementar MotorNube - Generación y evaluación
  - [x] 5.1 Crear la clase `MotorNube` con constructores y generación de la nube
    - Constructor con `ConfiguracionNube`, `double[][] entradas`, `double[][] objetivos`
    - Constructor adicional que acepta una `PoliticaEliminacion` personalizada
    - Validación de datos de entrada (no vacíos, dimensiones coherentes con topología)
    - Método interno `generarNube()`: crea n instancias de NeuralNetwork con semilla para reproducibilidad
    - Archivo: `src/main/java/es/jastxz/nn/nube/MotorNube.java`
    - _Requisitos: 2.1, 2.2, 2.3, 2.4, 5.2, 5.4, 8.2, 8.4_

  - [x] 5.2 Implementar evaluación de redes candidatas en `MotorNube`
    - Método interno `evaluar(NeuralNetwork red)`: ejecuta feedforward sobre todos los datos y calcula precisión
    - Predicción correcta: índice del valor máximo de la salida == índice del valor máximo del objetivo (argmax)
    - Precisión = predicciones correctas / total de muestras
    - Archivo: `src/main/java/es/jastxz/nn/nube/MotorNube.java`
    - _Requisitos: 3.1, 3.3, 3.4_

  - [ ]* 5.3 Escribir test de propiedad para generación de nube correcta
    - **Propiedad 3: Generación de nube correcta**
    - **Valida: Requisitos 2.1, 2.2**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

  - [ ]* 5.4 Escribir test de propiedad para independencia de redes
    - **Propiedad 4: Independencia de redes en la nube**
    - **Valida: Requisito 2.4**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

  - [ ]* 5.5 Escribir test de propiedad para cálculo de precisión
    - **Propiedad 5: Cálculo de precisión**
    - **Valida: Requisitos 3.1, 3.3**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

  - [ ]* 5.6 Escribir test de propiedad para criterio argmax
    - **Propiedad 6: Criterio de corrección por argmax**
    - **Valida: Requisito 3.4**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

- [x] 6. Checkpoint - Verificar generación y evaluación
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

- [x] 7. Implementar MotorNube - Proceso de reducción
  - [x] 7.1 Implementar reconstrucción de red con topología reducida en `MotorNube`
    - Método interno `reconstruirRed(NeuralNetwork original, int[] nuevaTopologia)`: crea nueva NeuralNetwork con pesos preservados
    - Eliminar filas de W[k-1] y B[k], y columnas de W[k] según las neuronas eliminadas
    - Usar el constructor `NeuralNetwork(topology, weights, biases)` creado en la tarea 1.2
    - Archivo: `src/main/java/es/jastxz/nn/nube/MotorNube.java`
    - _Requisitos: 4.6_

  - [x] 7.2 Implementar el ciclo de reducción en `MotorNube`
    - Método interno `ejecutarReduccion(NeuralNetwork red)`: aplica iterativamente la política de eliminación
    - Para cada red candidata: evaluar → si supera umbral, actualizar mejor configuración → aplicar reducción → repetir
    - Detener reducción cuando todas las capas ocultas llegan a 0 neuronas (política retorna null)
    - Archivo: `src/main/java/es/jastxz/nn/nube/MotorNube.java`
    - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 3.2_

  - [ ]* 7.3 Escribir test de propiedad para preservación de pesos
    - **Propiedad 9: Preservación de pesos durante reducción**
    - **Valida: Requisito 4.6**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

  - [ ]* 7.4 Escribir test de propiedad para invariante de mejor configuración
    - **Propiedad 7: Invariante de mejor configuración**
    - **Valida: Requisitos 3.2, 4.4**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

- [x] 8. Implementar MotorNube - Refinamiento y método ejecutar()
  - [x] 8.1 Implementar refinamiento y orquestar el método `ejecutar()` en `MotorNube`
    - Método interno `refinar(NeuralNetwork red)`: entrena con backpropagation usando épocas y tasa de aprendizaje de la configuración
    - Método público `ejecutar()`: orquesta generarNube → ejecutarReduccion por cada red → refinar mejor → retornar InformeNube
    - Medir tiempo de ejecución total en milisegundos
    - Archivo: `src/main/java/es/jastxz/nn/nube/MotorNube.java`
    - _Requisitos: 6.1, 6.2, 6.3, 6.4, 10.1, 10.2_

- [x] 9. Implementar InformeNube
  - [x] 9.1 Crear el record `InformeNube`
    - Campos: `mejorRed`, `precision`, `topologiaFinal`, `totalRedesEvaluadas`, `totalReducciones`, `tiempoEjecucionMs`, `exitoso`
    - Record inmutable, mejorRed puede ser null si ninguna red superó el umbral
    - Archivo: `src/main/java/es/jastxz/nn/nube/InformeNube.java`
    - _Requisitos: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 9.2 Escribir test de propiedad para informe sin red viable
    - **Propiedad 11: Informe sin red viable**
    - **Valida: Requisitos 6.4, 7.3**
    - Archivo: `src/test/java/es/jastxz/nn/nube/InformeNubePropertyTest.java`

  - [ ]* 9.3 Escribir test de propiedad para completitud del informe
    - **Propiedad 12: Completitud del informe**
    - **Valida: Requisitos 7.1, 7.2**
    - Archivo: `src/test/java/es/jastxz/nn/nube/InformeNubePropertyTest.java`

- [x] 10. Checkpoint - Verificar motor e informe
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

- [x] 11. Integración, serialización y reproducibilidad
  - [x] 11.1 Escribir test de integración completo del flujo con dataset XOR
    - Ejecutar el flujo completo: configurar → ejecutar → verificar informe coherente
    - Verificar que la red resultante produce salidas razonables para XOR
    - Verificar que la política personalizada se aplica correctamente (Requisito 5.4)
    - Archivo: `src/test/java/es/jastxz/nn/nube/IntegracionNubeTest.java`
    - _Requisitos: 5.4, 8.1, 8.2, 8.3, 8.4_

  - [ ]* 11.2 Escribir test de propiedad para serialización ida y vuelta
    - **Propiedad 13: Ida y vuelta de serialización**
    - **Valida: Requisito 9.1**
    - Archivo: `src/test/java/es/jastxz/nn/nube/InformeNubePropertyTest.java`

  - [ ]* 11.3 Escribir test de propiedad para reproducibilidad completa
    - **Propiedad 14: Reproducibilidad completa**
    - **Valida: Requisitos 10.1, 2.3**
    - Archivo: `src/test/java/es/jastxz/nn/nube/MotorNubePropertyTest.java`

- [x] 12. Checkpoint final - Verificar integración completa
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia los requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Los tests de propiedades validan propiedades universales de corrección con jqwik (mínimo 100 iteraciones)
- Los tests unitarios complementan con ejemplos concretos y casos borde
- Todos los archivos del módulo se crean en `src/main/java/es/jastxz/nn/nube/`
- Los tests se crean en `src/test/java/es/jastxz/nn/nube/`
