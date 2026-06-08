# Documento de Requisitos: Suite de Benchmarks para SNN

## Introducción

Suite integral de pruebas y marco de métricas para evaluar los límites y el comportamiento de la implementación de Redes Neuronales Spiking (SNN) en función del número de capas, neuronas y la complejidad del espacio de estados de los problemas probados. El objetivo es comprender cómo escalan las SNN ante problemas de complejidad creciente (puertas lógicas → 3 en Raya → Gatos → Damas) y determinar los límites prácticos de la topología de red.

## Glosario

- **SNN**: Red Neuronal Spiking (Spiking Neural Network), implementada en `RedNeuralSpiking`
- **Suite_Benchmark**: Conjunto de tests JUnit 5 que ejecutan benchmarks paramétricos sobre la SNN
- **Espacio_de_Estados**: Número de configuraciones posibles de un problema (ej: 3^9 para 3 en Raya, ~5×10^20 para Damas)
- **Topología**: Configuración de capas y neuronas de la red, definida como array de enteros (ej: [9, 18, 9])
- **Configuración_Benchmark**: Combinación de topología de red y problema a resolver usada en una ejecución de benchmark
- **Recolector_Métricas**: Componente que captura y agrega métricas de rendimiento durante la ejecución de benchmarks
- **Informe_Benchmark**: Salida estructurada con los resultados de todas las ejecuciones de benchmark
- **Precisión**: Porcentaje de salidas correctas sobre el total de muestras evaluadas
- **Error_MSE**: Error cuadrático medio entre la salida de la red y el objetivo esperado
- **Tasa_de_Disparo**: Frecuencia promedio de spikes por neurona en una ventana temporal
- **Costo_Energético**: Estimación del consumo energético proporcional al número total de spikes
- **Dispersión_Actividad**: Desviación estándar del número de spikes entre neuronas, indicando uniformidad de activación
- **Problema_Lógico**: Problema de puertas lógicas (AND, OR, XOR) con espacio de estados mínimo (4 combinaciones)
- **GestorMetricas**: Clase existente que recopila métricas de spikes, tasas de disparo y costos energéticos
- **ConfiguracionRedBuilder**: Builder existente para configurar topología y parámetros de la SNN

## Requisitos

### Requisito 1: Definición de Niveles de Complejidad por Espacio de Estados

**Historia de Usuario:** Como investigador, quiero clasificar los problemas de prueba por su espacio de estados, para poder analizar el rendimiento de la SNN en función de la complejidad del problema.

#### Criterios de Aceptación

1. THE Suite_Benchmark SHALL definir al menos cuatro niveles de complejidad: trivial (Problema_Lógico, ~4 estados), bajo (3 en Raya, ~5.478 estados), medio (Gatos, ~10^4-10^5 estados) y alto (Damas, ~5×10^20 estados)
2. WHEN se ejecuta un benchmark para un nivel de complejidad, THE Suite_Benchmark SHALL registrar el nombre del problema, el tamaño del espacio de estados y las dimensiones de entrada/salida de la red
3. THE Suite_Benchmark SHALL permitir añadir nuevos problemas a cualquier nivel de complejidad sin modificar la estructura del framework

### Requisito 2: Barrido Paramétrico de Topologías de Red

**Historia de Usuario:** Como investigador, quiero probar la SNN con múltiples topologías (variando capas y neuronas), para identificar los límites de configuración donde la red deja de aprender eficazmente.

#### Criterios de Aceptación

1. THE Suite_Benchmark SHALL ejecutar benchmarks con topologías que varíen en número de capas ocultas (1, 2, 3 y 4 capas ocultas)
2. THE Suite_Benchmark SHALL ejecutar benchmarks con topologías que varíen en número de neuronas por capa oculta (factores de 1x, 2x, 4x y 8x respecto al tamaño de entrada)
3. WHEN se define una Configuración_Benchmark, THE Suite_Benchmark SHALL validar que la topología sea compatible con las dimensiones de entrada/salida del problema
4. THE Suite_Benchmark SHALL generar todas las combinaciones de topología × problema como matriz de pruebas

### Requisito 3: Recolección Sistemática de Métricas de Rendimiento

**Historia de Usuario:** Como investigador, quiero recopilar métricas detalladas de cada ejecución de benchmark, para poder comparar cuantitativamente el comportamiento de la SNN bajo distintas configuraciones.

#### Criterios de Aceptación

1. WHEN se completa una ejecución de benchmark, THE Recolector_Métricas SHALL capturar: Precisión final, Error_MSE por época, tiempo de entrenamiento en milisegundos, número total de spikes, Tasa_de_Disparo promedio, Costo_Energético y Dispersión_Actividad
2. WHEN se completa una ejecución de benchmark, THE Recolector_Métricas SHALL capturar el número de neuronas activas (que dispararon al menos un spike) respecto al total de neuronas de la red
3. THE Recolector_Métricas SHALL almacenar las métricas de cada ejecución asociadas a la Configuración_Benchmark correspondiente (topología + problema)
4. WHEN se ejecutan múltiples épocas de entrenamiento, THE Recolector_Métricas SHALL registrar la evolución del Error_MSE época a época para detectar convergencia o divergencia

### Requisito 4: Detección de Límites de la Red

**Historia de Usuario:** Como investigador, quiero identificar automáticamente cuándo la SNN alcanza sus límites operativos, para saber qué configuraciones son viables y cuáles no.

#### Criterios de Aceptación

1. WHEN la Precisión final de un benchmark es inferior al 60% tras completar todas las épocas de entrenamiento, THE Suite_Benchmark SHALL clasificar la Configuración_Benchmark como "límite no superado"
2. WHEN el Error_MSE no disminuye en más de un 1% durante 3 épocas consecutivas, THE Suite_Benchmark SHALL clasificar la ejecución como "convergencia estancada"
3. WHEN el porcentaje de neuronas activas es inferior al 20% del total de neuronas, THE Suite_Benchmark SHALL clasificar la ejecución como "red infrautilizada"
4. WHEN el Costo_Energético supera 10 veces el Costo_Energético de la topología más pequeña para el mismo problema, THE Suite_Benchmark SHALL clasificar la ejecución como "ineficiencia energética"
5. IF el tiempo de entrenamiento de un benchmark supera los 120 segundos, THEN THE Suite_Benchmark SHALL abortar la ejecución y registrar el resultado como "timeout"

### Requisito 5: Generación de Informe Comparativo

**Historia de Usuario:** Como investigador, quiero obtener un informe estructurado que compare todas las configuraciones probadas, para poder visualizar tendencias y tomar decisiones sobre la arquitectura de la SNN.

#### Criterios de Aceptación

1. WHEN se completan todos los benchmarks, THE Informe_Benchmark SHALL presentar una tabla resumen con filas por Configuración_Benchmark y columnas por cada métrica recopilada
2. THE Informe_Benchmark SHALL agrupar los resultados por nivel de complejidad del Espacio_de_Estados para facilitar la comparación entre problemas
3. THE Informe_Benchmark SHALL incluir una sección de "hallazgos de límites" que liste todas las configuraciones clasificadas como "límite no superado", "convergencia estancada", "red infrautilizada" o "ineficiencia energética"
4. THE Informe_Benchmark SHALL generar la salida en formato texto legible por consola a través de System.out durante la ejecución de los tests JUnit 5
5. WHEN se genera el informe, THE Informe_Benchmark SHALL calcular y mostrar la topología óptima (mayor Precisión con menor Costo_Energético) para cada nivel de complejidad

### Requisito 6: Escalabilidad y Reproducibilidad de los Benchmarks

**Historia de Usuario:** Como investigador, quiero que los benchmarks sean reproducibles y configurables, para poder repetir experimentos y ajustar parámetros sin modificar el código de los tests.

#### Criterios de Aceptación

1. THE Suite_Benchmark SHALL utilizar semillas aleatorias fijas para la inicialización de pesos, garantizando resultados reproducibles entre ejecuciones
2. THE Suite_Benchmark SHALL parametrizar el número de épocas de entrenamiento, la duración de timesteps por patrón y el número de repeticiones por configuración
3. WHEN se ejecuta un benchmark con múltiples repeticiones, THE Suite_Benchmark SHALL calcular la media y desviación estándar de cada métrica entre repeticiones
4. THE Suite_Benchmark SHALL integrarse con JUnit 5 usando @ParameterizedTest o @TestFactory para generar dinámicamente los casos de prueba a partir de la matriz de configuraciones

### Requisito 7: Análisis de Eficiencia por Espacio de Estados

**Historia de Usuario:** Como investigador, quiero analizar la relación entre el tamaño del espacio de estados y la eficiencia de la SNN, para entender cómo escala la red ante problemas más complejos.

#### Criterios de Aceptación

1. WHEN se completan los benchmarks para todos los niveles de complejidad, THE Informe_Benchmark SHALL calcular la ratio Precisión/Costo_Energético para cada Configuración_Benchmark
2. THE Informe_Benchmark SHALL mostrar cómo varía la Precisión al incrementar el Espacio_de_Estados manteniendo la misma topología
3. THE Informe_Benchmark SHALL mostrar cómo varía el tiempo de entrenamiento al incrementar el Espacio_de_Estados manteniendo la misma topología
4. WHEN se detecta que una topología mantiene Precisión superior al 70% en un nivel de complejidad pero inferior al 50% en el siguiente nivel, THE Informe_Benchmark SHALL señalar la transición como "frontera de capacidad" de la topología

### Requisito 8: Generación de Gráficas de Visualización

**Historia de Usuario:** Como investigador, quiero generar gráficas a partir de los resultados de los benchmarks, para poder visualizar tendencias, comparar configuraciones y comunicar hallazgos de forma clara y rápida.

#### Criterios de Aceptación

1. WHEN se completan todos los benchmarks, THE GeneradorGraficas SHALL generar gráficas de barras comparando la Precisión final por topología para cada nivel de complejidad del Espacio_de_Estados
2. WHEN se completan todos los benchmarks, THE GeneradorGraficas SHALL generar gráficas de líneas mostrando la evolución del Error_MSE por época para cada Configuración_Benchmark
3. WHEN se completan todos los benchmarks, THE GeneradorGraficas SHALL generar gráficas de barras comparando el Costo_Energético entre topologías para cada nivel de complejidad
4. WHEN se completan todos los benchmarks, THE GeneradorGraficas SHALL generar un histograma de distribución de Tasa_de_Disparo agrupado por topología
5. WHEN se detectan fronteras de capacidad (Requisito 7.4), THE GeneradorGraficas SHALL generar una gráfica que muestre la Precisión por nivel de Espacio_de_Estados para cada topología, señalando visualmente las transiciones de frontera
6. THE GeneradorGraficas SHALL exportar todas las gráficas como archivos PNG en un directorio configurable de salida
7. WHEN se genera una gráfica, THE GeneradorGraficas SHALL incluir título descriptivo, etiquetas en los ejes, leyenda cuando haya múltiples series, y el nombre del nivel de complejidad o configuración correspondiente
8. THE GeneradorGraficas SHALL generar un archivo HTML índice que agrupe todas las gráficas generadas con enlaces de navegación por sección (precisión, MSE, energía, disparo, fronteras)
