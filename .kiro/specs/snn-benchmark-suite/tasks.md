# Plan de Implementación: Suite de Benchmarks para SNN

## Visión General

Implementación incremental de la suite de benchmarks paramétricos para evaluar la Red Neuronal Spiking (`RedNeuralSpiking`) variando topologías y niveles de complejidad de problemas. Se construyen primero los modelos de datos y componentes base, luego la lógica de ejecución y detección de límites, después la generación de informes y gráficas, y finalmente el test principal que orquesta todo.

## Tareas

- [x] 1. Configurar dependencia XChart y crear estructura base del paquete benchmark
  - Añadir dependencia `org.knowm.xchart:xchart:3.8.7` al `pom.xml`
  - Crear paquete `es.jastxz.nn.benchmark` en `src/main/java`
  - Crear paquete `es.jastxz.nn.benchmark` en `src/test/java` para tests de propiedades
  - _Requisitos: 8.1, 8.6_

- [x] 2. Implementar modelos de datos y enumeración de niveles de complejidad
  - [x] 2.1 Crear enum `NivelComplejidad` con los cuatro niveles (TRIVIAL, BAJO, MEDIO, ALTO)
    - Incluir campos: nombreProblema, espacioEstados, dimensionEntrada, dimensionSalida
    - Valores: Puertas Lógicas (4, 2, 1), 3 en Raya (5478, 10, 9), Gatos (100000, 25, 25), Damas (5×10^20, 32, 32)
    - _Requisitos: 1.1, 1.2, 1.3_

  - [x] 2.2 Crear record `ConfiguracionBenchmark`
    - Campos: nivel, topologia, epocas, duracionTimesteps, repeticiones, semilla
    - Validación en constructor compacto: topologia[0] == nivel.dimensionEntrada y topologia[last] == nivel.dimensionSalida
    - Método `etiqueta()` para representación legible
    - _Requisitos: 2.3, 6.1, 6.2_

  - [x] 2.3 Crear record `ResultadoBenchmark`
    - Campos: configuracion, precisionFinal, errorMSEPorEpoca, tiempoEntrenamientoMs, totalSpikes, tasaDisparoPromedio, costoEnergetico, dispersionActividad, neuronasActivas, neuronasTotal, clasificacion
    - _Requisitos: 3.1, 3.2_

  - [x] 2.4 Crear record `ResultadoAgregado` para múltiples repeticiones
    - Campos: configuracion, mediaPrecision, desvPrecision, mediaTiempo, desvTiempo, mediaCostoEnergetico, desvCostoEnergetico, ratioEficiencia, clasificacionMayoritaria
    - Método estático `agregar(List<ResultadoBenchmark>)` que calcula media y desviación estándar
    - _Requisitos: 6.3, 7.1_

  - [x] 2.5 Escribir test de propiedad para validación de compatibilidad topología-problema
    - **Propiedad 2: Validación de compatibilidad topología-problema**
    - **Valida: Requisito 2.3**

  - [x] 2.6 Escribir test de propiedad para corrección de la agregación estadística
    - **Propiedad 7: Corrección de la agregación estadística**
    - **Valida: Requisito 6.3**

- [x] 3. Implementar GeneradorTopologias
  - [x] 3.1 Crear clase `GeneradorTopologias`
    - Método estático `generar(int entrada, int salida)` que retorna `List<int[]>`
    - Generar combinaciones: 4 niveles de capas ocultas (1, 2, 3, 4) × 4 factores de neuronas (1x, 2x, 4x, 8x) = 16 topologías
    - Cada topología comienza con `entrada`, termina con `salida`, capas ocultas con `entrada * factor`
    - _Requisitos: 2.1, 2.2, 2.4_

  - [x] 3.2 Escribir test de propiedad para completitud del generador de topologías
    - **Propiedad 1: Completitud del generador de topologías**
    - **Valida: Requisitos 2.1, 2.2, 2.4**

- [x] 4. Checkpoint - Verificar modelos de datos y generador de topologías
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implementar RecolectorMetricas
  - [x] 5.1 Crear clase `RecolectorMetricas`
    - Método `ejecutarYRecolectar(ConfiguracionBenchmark config, double[][] inputs, double[][] targets)` que retorna `ResultadoBenchmark`
    - Crear `RedNeuralSpiking` usando `ConfiguracionRedBuilder` con topología y semilla de la configuración
    - Entrenar por N épocas registrando MSE por época
    - Evaluar precisión final con `procesarLote`
    - Extraer métricas de `GestorMetricas`: totalSpikes, tasaDisparoPromedio, costoEnergetico, dispersionActividad, neuronasActivas
    - Medir tiempo de entrenamiento en milisegundos
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 6.1_

  - [x] 5.2 Escribir test de propiedad para completitud y validez de métricas
    - **Propiedad 3: Completitud y validez de métricas en resultados**
    - **Valida: Requisitos 1.1, 1.2, 3.1, 3.2**

  - [x] 5.3 Escribir test de propiedad para seguimiento de MSE por época
    - **Propiedad 4: Seguimiento de MSE por época**
    - **Valida: Requisito 3.4**

  - [x] 5.4 Escribir test de propiedad para asociación configuración-resultado
    - **Propiedad 5: Asociación configuración-resultado**
    - **Valida: Requisito 3.3**

- [x] 6. Implementar DetectorLimites
  - [x] 6.1 Crear clase `DetectorLimites`
    - Método estático `clasificar(ResultadoBenchmark resultado)` que retorna String o null
    - Implementar reglas: precisión < 0.6 → "limite_no_superado", MSE no baja >1% en 3 épocas → "convergencia_estancada", neuronas activas < 20% → "red_infrautilizada"
    - Método estático `clasificarIneficiencia(ResultadoBenchmark resultado, double costoMinimo)` para detectar costo > 10x mínimo → "ineficiencia_energetica"
    - _Requisitos: 4.1, 4.2, 4.3, 4.4_

  - [x] 6.2 Escribir test de propiedad para corrección de la detección de límites
    - **Propiedad 6: Corrección de la detección de límites**
    - **Valida: Requisitos 4.1, 4.2, 4.3**

  - [x] 6.3 Escribir test de propiedad para detección de ineficiencia energética
    - **Propiedad 11: Detección de ineficiencia energética**
    - **Valida: Requisito 4.4**

- [x] 7. Checkpoint - Verificar ejecución de benchmarks y detección de límites
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implementar GeneradorInforme
  - [x] 8.1 Crear clase `GeneradorInforme`
    - Método estático `generar(List<ResultadoBenchmark> resultados)` con salida por `System.out`
    - Tabla resumen con filas por ConfiguracionBenchmark y columnas por métrica
    - Agrupación de resultados por NivelComplejidad
    - Sección de "hallazgos de límites" con todas las configuraciones clasificadas
    - Cálculo de topología óptima (mayor ratio Precisión/CostoEnergético) por nivel
    - Análisis de eficiencia: ratio Precisión/CostoEnergético por configuración
    - Detección de fronteras de capacidad: precisión > 0.7 en un nivel y < 0.5 en el siguiente
    - Variación de precisión y tiempo al incrementar espacio de estados con misma topología
    - _Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5, 7.1, 7.2, 7.3, 7.4_

  - [x] 8.2 Escribir test de propiedad para completitud del informe
    - **Propiedad 8: Completitud del informe**
    - **Valida: Requisitos 5.1, 5.2, 5.3**

  - [x] 8.3 Escribir test de propiedad para selección de topología óptima
    - **Propiedad 9: Selección de topología óptima**
    - **Valida: Requisitos 5.5, 7.1**

  - [x] 8.4 Escribir test de propiedad para detección de frontera de capacidad
    - **Propiedad 10: Detección de frontera de capacidad**
    - **Valida: Requisito 7.4**

- [x] 9. Implementar GeneradorGraficas
  - [x] 9.1 Crear clase `GeneradorGraficas` con XChart
    - Constructor con `Path directorioSalida`, crear directorio si no existe
    - Método `generarTodas(List<ResultadoBenchmark> resultados, Map<String, List<String>> fronterasCapacidad)`
    - `generarGraficasPrecisionPorNivel`: gráficas de barras de precisión por topología, una por NivelComplejidad
    - `generarGraficasMSEPorConfiguracion`: gráficas de líneas de evolución MSE por época, una por configuración
    - `generarGraficasCostoEnergetico`: gráficas de barras de costo energético por topología, una por NivelComplejidad
    - `generarHistogramaTasaDisparo`: histograma de distribución de tasa de disparo agrupado por topología
    - `generarGraficaFronterasCapacidad`: gráfica de precisión vs nivel de espacio de estados por topología con marcadores en fronteras
    - `generarIndiceHTML`: archivo HTML índice con enlaces a todas las gráficas por sección
    - Todas las gráficas con título descriptivo, etiquetas en ejes, leyenda cuando hay múltiples series
    - Exportar como PNG en directorio configurable
    - Manejo de errores: capturar IOException, continuar con demás gráficas
    - _Requisitos: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 9.2 Escribir test de propiedad para completitud de gráficas de barras por nivel
    - **Propiedad 12: Completitud de gráficas de barras por nivel de complejidad**
    - **Valida: Requisitos 8.1, 8.3**

  - [x] 9.3 Escribir test de propiedad para correspondencia de datos en gráficas MSE
    - **Propiedad 13: Correspondencia de datos en gráficas de evolución MSE**
    - **Valida: Requisito 8.2**

  - [x] 9.4 Escribir test de propiedad para visualización de fronteras de capacidad
    - **Propiedad 14: Visualización correcta de fronteras de capacidad**
    - **Valida: Requisito 8.5**

  - [x] 9.5 Escribir test de propiedad para metadatos y formato de gráficas
    - **Propiedad 15: Corrección de metadatos y formato de salida de gráficas**
    - **Valida: Requisitos 8.6, 8.7**

- [x] 10. Checkpoint - Verificar informe y gráficas
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implementar BenchmarkSNNTest y orquestación final
  - [x] 11.1 Crear clase de test `BenchmarkSNNTest` con JUnit 5
    - Método `@TestFactory Collection<DynamicTest> benchmarkMatriz()` que genera la matriz completa de configuraciones
    - Para cada NivelComplejidad: obtener datos del problema, generar topologías con `GeneradorTopologias`, crear `ConfiguracionBenchmark` por cada combinación
    - Cada DynamicTest ejecuta `RecolectorMetricas.ejecutarYRecolectar`, aplica `DetectorLimites.clasificar`, almacena resultado
    - Implementar timeout de 120 segundos con `assertTimeoutPreemptively` por cada benchmark individual
    - Usar semillas fijas para reproducibilidad
    - Parametrizar épocas, duracionTimesteps y repeticiones
    - _Requisitos: 4.5, 6.1, 6.2, 6.4_

  - [x] 11.2 Implementar `@AfterAll` para generación de informe y gráficas
    - Invocar `GeneradorInforme.generar(resultados)` para salida por consola
    - Invocar `GeneradorGraficas.generarTodas(resultados, fronteras)` para generar PNGs e índice HTML
    - Calcular fronteras de capacidad y pasarlas al generador de gráficas
    - _Requisitos: 5.4, 8.1, 8.8_

  - [x] 11.3 Implementar proveedores de datos de problemas
    - Crear datos de entrenamiento para cada NivelComplejidad: puertas lógicas (AND/OR/XOR), 3 en Raya, Gatos, Damas
    - Reutilizar datos existentes de los tests comparativos en `src/test/java/es/jastxz/comparativas/`
    - _Requisitos: 1.1, 1.2_

- [x] 12. Checkpoint final - Verificar suite completa de benchmarks
  - Ensure all tests pass, ask the user if questions arise.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Los tests de propiedades validan propiedades universales de corrección con jqwik
- Los tests unitarios validan ejemplos específicos y casos borde
- La dependencia XChart se añade al inicio para estar disponible durante toda la implementación
