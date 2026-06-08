# Documento de Requisitos: Método de la Nube Aleatoria

## Introducción

Este documento define los requisitos para implementar el Método de la Nube Aleatoria, un nuevo método de búsqueda y diseño de redes neuronales descrito en #[[file:documents/Método de la Nube Aleatoria.md]]. El método consiste en generar una nube (conjunto) de redes neuronales inicializadas aleatoriamente, evaluar cada una contra un umbral de acierto, reducir progresivamente sus neuronas para encontrar la estructura mínima que supere dicho umbral, y finalmente refinar la mejor red encontrada mediante entrenamiento clásico. La implementación se integrará en la biblioteca Java existente, siguiendo los patrones arquitectónicos del módulo genético (`es.jastxz.nn.genetico`) y reutilizando la clase `NeuralNetwork` para la representación y entrenamiento de redes.

## Glosario

- **Nube_Aleatoria**: Conjunto de al menos n redes neuronales (`NeuralNetwork`) con una topología predefinida y pesos inicializados aleatoriamente.
- **Red_Candidata**: Una instancia individual de `NeuralNetwork` perteneciente a la Nube_Aleatoria.
- **Umbral_Acierto**: Porcentaje mínimo de acierto (valor en [0.0, 1.0]) que una Red_Candidata debe superar para ser considerada viable.
- **Proceso_Reduccion**: Procedimiento iterativo que recorre cada Red_Candidata, evalúa su precisión, elimina neuronas de capas ocultas y registra la mejor configuración encontrada.
- **Politica_Eliminacion**: Estrategia que determina qué neuronas eliminar y de qué capa oculta durante el Proceso_Reduccion.
- **Configuracion_Nube**: Registro inmutable que agrupa todos los hiperparámetros del método: tamaño de la nube, topología inicial, umbral de acierto, política de eliminación y parámetros de refinamiento.
- **Motor_Nube**: Orquestador principal que ejecuta el ciclo completo del método: generación de la Nube_Aleatoria, Proceso_Reduccion y refinamiento final.
- **Mejor_Configuracion**: La Red_Candidata con la mayor precisión que haya superado el Umbral_Acierto durante el Proceso_Reduccion.
- **Refinamiento**: Fase final donde la Mejor_Configuracion se entrena mediante backpropagation clásico hasta alcanzar la precisión objetivo.
- **Informe_Nube**: Registro que contiene los resultados del proceso completo: mejor red encontrada, historial de reducciones y métricas de rendimiento.

## Requisitos

### Requisito 1: Configuración del método

**Historia de usuario:** Como desarrollador, quiero configurar los hiperparámetros del Método de la Nube Aleatoria de forma inmutable y validada, para poder controlar el comportamiento del método de manera segura.

#### Criterios de aceptación

1. THE Configuracion_Nube SHALL almacenar el tamaño de la nube (número de redes), la topología inicial (array de enteros), el Umbral_Acierto, el número de neuronas a eliminar por iteración, las épocas de refinamiento, la tasa de aprendizaje para refinamiento y una semilla para reproducibilidad.
2. WHEN el tamaño de la nube es menor que 1, THE Configuracion_Nube SHALL lanzar una excepción de argumento inválido con un mensaje descriptivo.
3. WHEN la topología inicial contiene menos de 3 capas, THE Configuracion_Nube SHALL lanzar una excepción de argumento inválido indicando que se requieren al menos una capa de entrada, una oculta y una de salida.
4. WHEN el Umbral_Acierto no está en el rango [0.0, 1.0], THE Configuracion_Nube SHALL lanzar una excepción de argumento inválido indicando el rango válido.
5. WHEN el número de neuronas a eliminar por iteración es menor que 1, THE Configuracion_Nube SHALL lanzar una excepción de argumento inválido.
6. THE Configuracion_Nube SHALL proporcionar un builder con valores por defecto razonables para todos los parámetros.

### Requisito 2: Generación de la Nube Aleatoria

**Historia de usuario:** Como desarrollador, quiero generar un conjunto de redes neuronales inicializadas aleatoriamente, para tener un punto de partida diverso en la búsqueda de la estructura óptima.

#### Criterios de aceptación

1. WHEN se inicia el método, THE Motor_Nube SHALL crear exactamente n instancias de Red_Candidata donde n es el tamaño de la nube especificado en la Configuracion_Nube.
2. THE Motor_Nube SHALL inicializar cada Red_Candidata con la topología definida en la Configuracion_Nube y pesos aleatorios.
3. WHEN se proporciona una semilla en la Configuracion_Nube, THE Motor_Nube SHALL generar la misma Nube_Aleatoria para la misma semilla y topología.
4. THE Motor_Nube SHALL crear cada Red_Candidata como una instancia independiente de `NeuralNetwork` sin estado compartido entre redes.

### Requisito 3: Evaluación de redes candidatas

**Historia de usuario:** Como desarrollador, quiero evaluar la precisión de cada red candidata contra un conjunto de datos, para determinar si supera el umbral mínimo de acierto.

#### Criterios de aceptación

1. WHEN se evalúa una Red_Candidata, THE Motor_Nube SHALL ejecutar feedforward con los datos de entrada proporcionados y calcular el porcentaje de acierto comparando las salidas con los valores esperados.
2. WHEN una Red_Candidata supera el Umbral_Acierto, THE Motor_Nube SHALL registrar la configuración actual de la Red_Candidata como Mejor_Configuracion si su precisión es superior a la Mejor_Configuracion registrada previamente.
3. THE Motor_Nube SHALL calcular la precisión como el número de predicciones correctas dividido entre el número total de muestras evaluadas.
4. WHEN se evalúa una predicción, THE Motor_Nube SHALL considerar correcta una predicción cuando el índice del valor máximo de la salida coincide con el índice del valor máximo del objetivo esperado.

### Requisito 4: Proceso de reducción

**Historia de usuario:** Como desarrollador, quiero reducir progresivamente las neuronas de cada red candidata para encontrar la estructura mínima que supere el umbral de acierto.

#### Criterios de aceptación

1. WHEN se inicia el Proceso_Reduccion, THE Motor_Nube SHALL recorrer cada Red_Candidata de la Nube_Aleatoria y aplicar iterativamente la eliminación de neuronas.
2. WHEN se aplica una iteración de reducción a una Red_Candidata, THE Motor_Nube SHALL eliminar x neuronas de una capa oculta según la Politica_Eliminacion, donde x es el número de neuronas a eliminar configurado.
3. WHEN la capa oculta o todas las capas ocultas quedan con 0 neuronas tras una eliminación, THE Motor_Nube SHALL detener la reducción de la Red_Candidata actual y pasar a la siguiente.
4. WHEN la Red_Candidata reducida supera el Umbral_Acierto, THE Motor_Nube SHALL actualizar la Mejor_Configuracion si la precisión obtenida es mayor que la registrada previamente.
5. WHEN la Red_Candidata reducida no supera el Umbral_Acierto, THE Motor_Nube SHALL continuar con la siguiente combinación de reducción según la Politica_Eliminacion.
6. THE Motor_Nube SHALL preservar los pesos originales de las neuronas no eliminadas durante cada reducción, reconstruyendo la red con la topología reducida.

### Requisito 5: Políticas de eliminación

**Historia de usuario:** Como desarrollador, quiero poder elegir entre diferentes estrategias de eliminación de neuronas, para adaptar la búsqueda a diferentes escenarios.

#### Criterios de aceptación

1. THE Motor_Nube SHALL soportar al menos una Politica_Eliminacion: eliminación secuencial (eliminar las últimas x neuronas de cada capa oculta, recorriendo las capas de la última a la primera).
2. THE Motor_Nube SHALL aceptar la Politica_Eliminacion como un parámetro configurable mediante una interfaz funcional o estrategia.
3. WHEN se aplica la política de eliminación secuencial, THE Motor_Nube SHALL eliminar neuronas comenzando por la última capa oculta y avanzando hacia la primera.
4. WHERE se implementa una Politica_Eliminacion personalizada, THE Motor_Nube SHALL aplicar la política proporcionada por el usuario en lugar de la política por defecto.

### Requisito 6: Refinamiento de la red

**Historia de usuario:** Como desarrollador, quiero entrenar la mejor red encontrada mediante backpropagation clásico, para obtener una red optimizada tanto en estructura como en pesos.

#### Criterios de aceptación

1. WHEN el Proceso_Reduccion ha encontrado una Mejor_Configuracion, THE Motor_Nube SHALL entrenar la Mejor_Configuracion mediante backpropagation usando los datos de entrenamiento proporcionados.
2. THE Motor_Nube SHALL entrenar la Mejor_Configuracion durante el número de épocas especificado en la Configuracion_Nube.
3. THE Motor_Nube SHALL usar la tasa de aprendizaje especificada en la Configuracion_Nube durante el refinamiento.
4. WHEN el Proceso_Reduccion no ha encontrado ninguna Red_Candidata que supere el Umbral_Acierto, THE Motor_Nube SHALL retornar un Informe_Nube indicando que no se encontró una red viable.

### Requisito 7: Informe de resultados

**Historia de usuario:** Como desarrollador, quiero obtener un informe detallado del proceso completo, para analizar el rendimiento del método y la red resultante.

#### Criterios de aceptación

1. THE Informe_Nube SHALL contener la Mejor_Configuracion encontrada (o null si ninguna superó el umbral), la precisión alcanzada, la topología final de la red, el número total de redes evaluadas, el número total de reducciones realizadas y el tiempo total de ejecución en milisegundos.
2. WHEN el proceso completa exitosamente, THE Motor_Nube SHALL retornar un Informe_Nube con la red refinada y sus métricas.
3. WHEN ninguna red supera el Umbral_Acierto, THE Motor_Nube SHALL retornar un Informe_Nube con la Mejor_Configuracion como null y un indicador de que el proceso no fue exitoso.
4. THE Informe_Nube SHALL ser un registro inmutable.

### Requisito 8: Integración con la arquitectura existente

**Historia de usuario:** Como desarrollador, quiero que el método se integre con las clases existentes de la biblioteca, para reutilizar la infraestructura de redes neuronales ya implementada.

#### Criterios de aceptación

1. THE Motor_Nube SHALL utilizar instancias de `NeuralNetwork` como Red_Candidata para la generación, evaluación y refinamiento.
2. THE Motor_Nube SHALL residir en el paquete `es.jastxz.nn.nube` siguiendo el patrón de organización de los módulos `genetico` y `spiking`.
3. THE Configuracion_Nube SHALL implementarse como un record de Java siguiendo el patrón de `ConfiguracionAG`.
4. THE Motor_Nube SHALL aceptar los datos de entrenamiento como arrays de `double[][]` para entradas y objetivos, consistente con la interfaz de `NeuralNetworkTrainer`.

### Requisito 9: Serialización de la red resultante

**Historia de usuario:** Como desarrollador, quiero poder guardar y cargar la red resultante del método, para reutilizarla sin repetir el proceso de búsqueda.

#### Criterios de aceptación

1. WHEN el proceso completa exitosamente, THE Informe_Nube SHALL proporcionar acceso a la `NeuralNetwork` resultante que puede ser serializada usando el método `save` existente.
2. WHEN se carga una `NeuralNetwork` previamente guardada, THE Motor_Nube SHALL poder usar la red cargada como punto de partida para un nuevo refinamiento.

### Requisito 10: Reproducibilidad del proceso

**Historia de usuario:** Como desarrollador, quiero que el proceso sea reproducible dado los mismos parámetros, para poder depurar y comparar resultados de forma determinista.

#### Criterios de aceptación

1. WHEN se ejecuta el Motor_Nube dos veces con la misma Configuracion_Nube y los mismos datos de entrenamiento, THE Motor_Nube SHALL producir la misma Mejor_Configuracion y el mismo Informe_Nube.
2. THE Motor_Nube SHALL utilizar la semilla de la Configuracion_Nube para inicializar todos los generadores de números aleatorios utilizados en el proceso.
