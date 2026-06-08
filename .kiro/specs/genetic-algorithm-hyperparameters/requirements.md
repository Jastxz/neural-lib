# Documento de Requisitos: Algoritmo Genético para Optimización de Hiperparámetros de SNN

## Introducción

Algoritmo genético (AG) para la búsqueda y acotación automática de hiperparámetros de la Red Neuronal Spiking (`RedNeuralSpiking`). El objetivo es encontrar configuraciones óptimas de topología y parámetros de red evitando redes "elefantiásicas" (excesivamente grandes), mediante un límite superior estricto en el número total de neuronas por red. El AG codifica cada individuo como un `ConfiguracionRed`, evalúa su fitness ejecutando benchmarks, y evoluciona la población mediante selección de padres (con diversidad garantizada), cruce multi-punto con sesgo hacia el mejor padre y mutación hasta converger a configuraciones eficientes.

## Glosario

- **AG**: Algoritmo Genético, motor de optimización evolutiva para hiperparámetros
- **Individuo**: Representación de una `ConfiguracionRed` completa como cromosoma del AG
- **Cromosoma**: Codificación de todos los hiperparámetros optimizables de un Individuo en genes tipados (enteros, reales, booleanos, enumerados)
- **Gen**: Unidad mínima del Cromosoma que representa un hiperparámetro individual con su rango válido
- **Poblacion**: Conjunto de Individuos que evoluciona en cada generación del AG
- **Fitness**: Valor numérico que mide la calidad de un Individuo, calculado a partir de métricas de benchmark (Precisión, Costo_Energético, tamaño de red)
- **Generacion**: Iteración completa del ciclo evolutivo: evaluación → selección → cruce → mutación
- **Selector_Padres**: Componente que elige Individuos de la Poblacion para reproducción
- **Operador_Cruce**: Componente que combina genes de dos Individuos padres para generar descendencia
- **Operador_Mutacion**: Componente que introduce variaciones aleatorias en los genes de un Individuo
- **Limite_Topologico**: Restricción superior sobre el número máximo de neuronas totales de la red (512 por defecto)
- **ConfiguracionRed**: Clase existente que contiene todos los parámetros de la SNN
- **ConfiguracionRedBuilder**: Builder existente para construir instancias de ConfiguracionRed
- **ResultadoBenchmark**: Record existente con métricas de una ejecución de benchmark
- **Elitismo**: Estrategia que preserva los mejores Individuos de una Generacion sin modificación
- **Bloque_Funcional**: Agrupación de genes dependientes dentro del Cromosoma que deben mantenerse juntos durante el cruce para preservar coherencia (ej: topología, LIF, STDP, codificación, regulación)

## Requisitos

### Requisito 1: Codificación del Cromosoma con Hiperparámetros Acotados

**Historia de Usuario:** Como investigador, quiero que cada individuo del AG represente una configuración completa de la SNN con rangos válidos para cada hiperparámetro, para que el AG solo explore configuraciones factibles.

#### Criterios de Aceptación

1. THE Cromosoma SHALL codificar los siguientes genes de topología: número de capas ocultas (entero, rango [1, 10]) y número de neuronas por capa oculta (entero, rango [1, 512])
2. THE Cromosoma SHALL codificar los siguientes genes neuronales LIF: umbralDisparo (real, rango [-60.0, -40.0] mV), potencialReposo (real, rango [-80.0, -60.0] mV), constanteDecaimiento (real, rango [5.0, 50.0] ms) y duracionRefractario (entero, rango [1, 10] timesteps)
3. THE Cromosoma SHALL codificar los siguientes genes STDP: amplitudLTP (real, rango [0.001, 0.1]), amplitudLTD (real, rango [0.001, 0.1]), tauLTP (real, rango [5.0, 50.0] ms) y tauLTD (real, rango [5.0, 50.0] ms)
4. THE Cromosoma SHALL codificar los siguientes genes de codificación: frecuenciaMaxima (real, rango [10.0, 500.0] Hz), modoCodificacion (enumerado: POISSON, REGULAR, BURST) y ventanaDecodificacion (entero, rango [10, 200] timesteps)
5. THE Cromosoma SHALL codificar los siguientes genes booleanos y asociados: homeostasisActiva (booleano), tasaDisparoObjetivo (real, rango [1.0, 50.0] Hz), tasaAjusteHomeostasis (real, rango [0.001, 0.1]), inhibicionLateralActiva (booleano), radioInhibicion (entero, rango [1, 5]) y fuerzaInhibicion (real, rango [0.1, 2.0])
6. WHEN se genera un Individuo aleatorio, THE AG SHALL respetar la restricción de que umbralDisparo sea estrictamente mayor que potencialReposo
7. WHEN se genera un Individuo aleatorio, THE AG SHALL construir la ConfiguracionRed resultante usando ConfiguracionRedBuilder y validar que no lance IllegalArgumentException

### Requisito 2: Límite Topológico Anti-Elefantiasis por Neuronas Totales

**Historia de Usuario:** Como investigador, quiero imponer un límite superior estricto al número total de neuronas de la red, para evitar que el AG genere configuraciones excesivamente grandes que sean inviables computacionalmente, permitiendo libertad en la distribución de capas y neuronas por capa.

#### Criterios de Aceptación

1. THE AG SHALL imponer un límite máximo de 512 neuronas totales (sumando todas las capas ocultas, entrada y salida) por Individuo como único Limite_Topologico estricto
2. THE AG SHALL permitir cualquier distribución de neuronas entre las capas ocultas siempre que la suma total no exceda el Limite_Topologico de 512 neuronas
3. WHEN un Operador_Cruce o un Operador_Mutacion genera un Individuo que excede el Limite_Topologico de 512 neuronas totales, THE AG SHALL reducir proporcionalmente las neuronas por capa oculta hasta cumplir el límite en lugar de descartar el Individuo
4. THE AG SHALL permitir configurar el Limite_Topologico de neuronas totales mediante un objeto de configuración del AG, usando 512 como valor por defecto

### Requisito 3: Función de Fitness Multi-Objetivo

**Historia de Usuario:** Como investigador, quiero que la función de fitness equilibre precisión, eficiencia energética y tamaño de red, para que el AG favorezca configuraciones que sean precisas pero no innecesariamente grandes.

#### Criterios de Aceptación

1. THE AG SHALL calcular el Fitness de cada Individuo ejecutando un benchmark con la ConfiguracionRed correspondiente y un NivelComplejidad configurable
2. THE AG SHALL calcular el Fitness como una combinación ponderada de tres componentes: precisión (peso por defecto 0.5), eficiencia energética inversa (peso por defecto 0.3) y penalización por tamaño de red (peso por defecto 0.2)
3. THE AG SHALL normalizar cada componente del Fitness al rango [0.0, 1.0] antes de aplicar los pesos
4. THE AG SHALL calcular la penalización por tamaño como la ratio entre el número total de neuronas del Individuo y el Limite_Topologico de neuronas totales (512 por defecto)
5. WHEN un Individuo produce un ResultadoBenchmark con clasificación "timeout" o "limite_no_superado", THE AG SHALL asignar un Fitness de 0.0 a dicho Individuo
6. THE AG SHALL permitir configurar los pesos de los componentes del Fitness, validando que sumen 1.0

### Requisito 4: Selección de Padres por Torneo con Diversidad Garantizada

**Historia de Usuario:** Como investigador, quiero un mecanismo de selección de padres que mantenga presión selectiva sin perder diversidad genética, asegurando que una proporción de padres provenga de individuos no-élite, para que el AG converja sin quedar atrapado en óptimos locales.

#### Criterios de Aceptación

1. THE Selector_Padres SHALL implementar selección por torneo con tamaño de torneo configurable (valor por defecto: 3)
2. WHEN se ejecuta un torneo, THE Selector_Padres SHALL seleccionar aleatoriamente k Individuos de la Poblacion (donde k es el tamaño de torneo) y elegir el Individuo con mayor Fitness como ganador
3. THE Selector_Padres SHALL seleccionar dos padres distintos para cada operación de cruce, repitiendo el torneo si ambos padres resultan ser el mismo Individuo
4. THE AG SHALL preservar los mejores N Individuos de cada Generacion sin modificación mediante Elitismo (N configurable, valor por defecto: 2)
5. THE Selector_Padres SHALL garantizar que al menos el 15% de los padres seleccionados en cada Generacion provengan de Individuos no-élite (Individuos que no están entre los N mejores de la Generacion), seleccionándolos aleatoriamente de dicho subconjunto
6. WHEN el tamaño de torneo es igual al tamaño de la Poblacion, THE Selector_Padres SHALL comportarse como selección determinista del mejor Individuo

### Requisito 5: Operador de Cruce Multi-Punto con Sesgo hacia el Mejor Padre

**Historia de Usuario:** Como investigador, quiero un operador de cruce que combine genes de dos padres mediante cortes aleatorios con preferencia hacia el padre de mayor fitness, agrupando genes dependientes en bloques funcionales, para generar descendencia válida que mantenga presión de mejora.

#### Criterios de Aceptación

1. THE Operador_Cruce SHALL agrupar los genes del Cromosoma en bloques funcionales coherentes: bloque de topología (número de capas ocultas y neuronas por capa), bloque neuronal LIF (umbralDisparo, potencialReposo, constanteDecaimiento, duracionRefractario), bloque STDP (amplitudLTP, amplitudLTD, tauLTP, tauLTD), bloque de codificación (frecuenciaMaxima, modoCodificacion, ventanaDecodificacion) y bloque de regulación (homeostasisActiva, tasaDisparoObjetivo, tasaAjusteHomeostasis, inhibicionLateralActiva, radioInhibicion, fuerzaInhibicion)
2. THE Operador_Cruce SHALL marcar X puntos de corte aleatorios a lo largo del Cromosoma de uno de los padres, donde los puntos de corte solo pueden ubicarse entre bloques funcionales (no dentro de un bloque), creando segmentos compuestos por bloques completos
3. THE Operador_Cruce SHALL seleccionar Y segmentos del padre con mayor Fitness (basándose en la estructura del otro padre) para formar el Cromosoma del descendiente, donde Y es mayor que la mitad del número total de segmentos, introduciendo así un sesgo hacia el mejor padre
4. THE Operador_Cruce SHALL copiar los segmentos restantes (no seleccionados del mejor padre) del padre con menor Fitness
5. WHEN se cruza el bloque de topología y los padres tienen diferente número de capas ocultas, THE Operador_Cruce SHALL usar el número de capas y las neuronas por capa del padre del cual se seleccionó el bloque de topología completo
6. WHEN se completa un cruce, THE Operador_Cruce SHALL validar que el Individuo resultante cumple el Limite_Topologico de 512 neuronas totales y las restricciones de ConfiguracionRed
7. THE AG SHALL aplicar el Operador_Cruce con una probabilidad configurable (valor por defecto: 0.8), pasando los padres sin modificar cuando no se aplica cruce
8. THE Operador_Cruce SHALL usar un número de puntos de corte X configurable (valor por defecto: 2), con X en el rango [1, número de bloques funcionales - 1]

### Requisito 6: Operador de Mutación Adaptativa por Tipo de Gen

**Historia de Usuario:** Como investigador, quiero un operador de mutación que perturbe los hiperparámetros de forma apropiada según su tipo, para explorar el espacio de búsqueda sin generar configuraciones inválidas.

#### Criterios de Aceptación

1. THE Operador_Mutacion SHALL aplicar mutación a cada Gen del Cromosoma con una probabilidad independiente configurable (valor por defecto: 0.1 por Gen)
2. WHEN se muta un Gen entero (capas ocultas, neuronas por capa, duracionRefractario, ventanaDecodificacion, radioInhibicion), THE Operador_Mutacion SHALL sumar o restar un valor aleatorio entre 1 y el 20% del rango del Gen, redondeando al entero más cercano
3. WHEN se muta un Gen real (umbralDisparo, constanteDecaimiento, amplitudLTP, frecuenciaMaxima, etc.), THE Operador_Mutacion SHALL aplicar una perturbación gaussiana con media 0 y desviación estándar igual al 10% del rango del Gen
4. WHEN se muta un Gen booleano, THE Operador_Mutacion SHALL invertir el valor (true→false, false→true)
5. WHEN se muta un Gen enumerado (modoCodificacion), THE Operador_Mutacion SHALL seleccionar aleatoriamente uno de los valores posibles del enum con probabilidad uniforme
6. WHEN se muta el Gen de número de capas ocultas y el nuevo valor es mayor que el anterior, THE Operador_Mutacion SHALL inicializar las capas adicionales con un número aleatorio de neuronas dentro del rango permitido
7. WHEN se muta el Gen de número de capas ocultas y el nuevo valor es menor que el anterior, THE Operador_Mutacion SHALL eliminar las capas sobrantes empezando por la última capa oculta
8. WHEN se completa una mutación, THE Operador_Mutacion SHALL aplicar clamping para garantizar que todos los genes permanezcan dentro de sus rangos válidos definidos en el Requisito 1

### Requisito 7: Ciclo Evolutivo y Criterios de Parada

**Historia de Usuario:** Como investigador, quiero controlar el ciclo evolutivo del AG con criterios de parada claros, para obtener resultados en un tiempo razonable sin desperdiciar recursos computacionales.

#### Criterios de Aceptación

1. THE AG SHALL ejecutar el ciclo evolutivo: evaluar Fitness → seleccionar padres → aplicar cruce → aplicar mutación → formar nueva Poblacion, repitiendo hasta cumplir un criterio de parada
2. THE AG SHALL detenerse cuando se alcance un número máximo de generaciones configurable (valor por defecto: 50)
3. THE AG SHALL detenerse cuando el mejor Fitness no mejore en más de un 1% durante un número configurable de generaciones consecutivas (valor por defecto: 10 generaciones de estancamiento)
4. THE AG SHALL usar un tamaño de Poblacion configurable (valor por defecto: 30 Individuos)
5. WHEN se completa el ciclo evolutivo, THE AG SHALL retornar el Individuo con el mejor Fitness encontrado en cualquier Generacion (mejor global, no solo de la última generación)
6. THE AG SHALL registrar en cada Generacion: número de generación, mejor Fitness, Fitness promedio, peor Fitness y la ConfiguracionRed del mejor Individuo
7. THE AG SHALL aceptar una semilla aleatoria para garantizar reproducibilidad de la evolución

### Requisito 8: Configuración del Algoritmo Genético

**Historia de Usuario:** Como investigador, quiero configurar todos los parámetros del AG mediante un objeto de configuración, para poder experimentar con diferentes estrategias evolutivas sin modificar código.

#### Criterios de Aceptación

1. THE AG SHALL aceptar un objeto ConfiguracionAG que agrupe: tamaño de población, número máximo de generaciones, generaciones de estancamiento, probabilidad de cruce, número de puntos de corte del cruce, probabilidad de mutación por gen, tamaño de torneo, número de élites, porcentaje mínimo de padres no-élite, pesos de fitness y Limite_Topologico de neuronas totales
2. THE AG SHALL proporcionar un builder (ConfiguracionAGBuilder) con valores por defecto razonables para todos los parámetros
3. WHEN se construye una ConfiguracionAG con pesos de fitness que no suman 1.0, THE ConfiguracionAGBuilder SHALL lanzar IllegalArgumentException
4. WHEN se construye una ConfiguracionAG con tamaño de torneo mayor que el tamaño de población, THE ConfiguracionAGBuilder SHALL lanzar IllegalArgumentException
5. THE AG SHALL aceptar un NivelComplejidad que determine el problema de benchmark usado para evaluar el Fitness de los Individuos

### Requisito 9: Integración con el Sistema de Benchmark Existente

**Historia de Usuario:** Como investigador, quiero que el AG reutilice la infraestructura de benchmark existente para evaluar individuos, para evitar duplicación de código y mantener consistencia en las métricas.

#### Criterios de Aceptación

1. THE AG SHALL evaluar cada Individuo construyendo una ConfiguracionBenchmark a partir de la ConfiguracionRed del Individuo y el NivelComplejidad configurado
2. THE AG SHALL reutilizar el RecolectorMetricas existente para obtener las métricas de cada evaluación de Individuo
3. THE AG SHALL reutilizar el DetectorLimites existente para clasificar los ResultadoBenchmark de cada Individuo
4. WHEN se evalúa un Individuo cuya ConfiguracionRed produce una excepción durante la construcción o el entrenamiento, THE AG SHALL asignar Fitness 0.0 y registrar la excepción sin abortar la evolución
5. THE AG SHALL permitir configurar el número de épocas y repeticiones del benchmark usado para evaluación de Fitness, independientemente de la configuración del benchmark general

### Requisito 10: Informe de Resultados de la Evolución

**Historia de Usuario:** Como investigador, quiero obtener un informe detallado del proceso evolutivo y la mejor configuración encontrada, para entender cómo convergió el AG y poder reproducir los resultados.

#### Criterios de Aceptación

1. WHEN se completa la evolución, THE AG SHALL generar un informe con: la ConfiguracionRed del mejor Individuo, su Fitness, las métricas de benchmark asociadas y el número de generación en que se encontró
2. WHEN se completa la evolución, THE AG SHALL incluir en el informe la evolución del mejor Fitness y el Fitness promedio por Generacion
3. WHEN se completa la evolución, THE AG SHALL incluir en el informe el motivo de parada (máximo de generaciones alcanzado o estancamiento detectado)
4. THE AG SHALL generar el informe en formato texto legible por consola a través de System.out
5. WHEN se completa la evolución, THE AG SHALL incluir en el informe el Limite_Topologico utilizado y los parámetros de configuración del AG
