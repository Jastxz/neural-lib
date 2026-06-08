# Plan de Implementación: Algoritmo Genético para Optimización de Hiperparámetros de SNN

## Visión General

Implementación bottom-up del módulo `es.jastxz.nn.genetico`: se construyen primero los tipos base (Gen, BloqueFuncional, Cromosoma), luego las entidades (Individuo, FabricaIndividuos), los operadores genéticos (EvaluadorFitness, SelectorTorneo, OperadorCruce, OperadorMutacion), la configuración (ConfiguracionAG/Builder), el motor evolutivo (MotorEvolutivo) y finalmente el informe (InformeEvolucion). Cada tarea incluye tests unitarios y de propiedades asociados.

## Tareas

- [x] 1. Implementar tipos base: Gen\<T\> y BloqueFuncional
  - [x] 1.1 Crear la interfaz sellada `Gen<T>` con los records `GenEntero`, `GenReal`, `GenBooleano` y `GenEnum<E>` en `src/main/java/es/jastxz/nn/genetico/Gen.java`
    - Implementar validación de rangos en constructores compactos
    - Implementar `conValor()` con clamping automático al rango válido
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Escribir tests unitarios para Gen\<T\> en `src/test/java/es/jastxz/nn/genetico/GenTest.java`
    - Verificar construcción válida de cada tipo de gen
    - Verificar que construcción fuera de rango lanza `IllegalArgumentException`
    - Verificar que `conValor()` aplica clamping correctamente
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.3 Crear el enum `BloqueFuncional` en `src/main/java/es/jastxz/nn/genetico/BloqueFuncional.java`
    - Definir los 5 bloques: TOPOLOGIA, LIF, STDP, CODIFICACION, REGULACION
    - _Requisitos: 5.1_

- [x] 2. Implementar Cromosoma
  - [x] 2.1 Crear el record `Cromosoma` en `src/main/java/es/jastxz/nn/genetico/Cromosoma.java`
    - Implementar `genesOrdenados()` con orden TOPOLOGIA → LIF → STDP → CODIFICACION → REGULACION
    - Implementar `genesDeBloque(BloqueFuncional)` para acceso por bloque
    - Implementar `conBloque(BloqueFuncional, List<Gen<?>>)` para reemplazo inmutable de bloque
    - Implementar `neuronasTotal(int entrada, int salida)` para cálculo de neuronas totales
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1_

  - [x] 2.2 Escribir tests unitarios para Cromosoma en `src/test/java/es/jastxz/nn/genetico/CromosomaTest.java`
    - Verificar acceso por bloque, `neuronasTotal`, `conBloque`
    - _Requisitos: 1.1, 2.1_

- [x] 3. Implementar Individuo y FabricaIndividuos
  - [x] 3.1 Crear el record `Individuo` en `src/main/java/es/jastxz/nn/genetico/Individuo.java`
    - Implementar `sinEvaluar(Cromosoma, ConfiguracionRed)` con fitness = -1
    - Implementar `conEvaluacion(double, ResultadoBenchmark)` para asignar fitness
    - Implementar `Comparable<Individuo>` con mayor fitness primero
    - _Requisitos: 1.7, 3.1_

  - [x] 3.2 Crear la clase `FabricaIndividuos` en `src/main/java/es/jastxz/nn/genetico/FabricaIndividuos.java`
    - Implementar `generarAleatorio()` respetando rangos, restricción umbralDisparo > potencialReposo y límite topológico
    - Implementar `construirConfiguracion(Cromosoma)` usando `ConfiguracionRedBuilder`
    - Implementar `repararTopologia(Cromosoma)` con reducción proporcional de neuronas por capa
    - _Requisitos: 1.6, 1.7, 2.1, 2.2, 2.3_

  - [x] 3.3 Escribir test de propiedad: genes dentro de rangos válidos
    - **Propiedad 1: Genes dentro de rangos válidos**
    - Crear generadores jqwik personalizados (`Arbitrary<Cromosoma>`, `Arbitrary<Individuo>`) en clase de test
    - Verificar que `generarAleatorio()` produce individuos con todos los genes dentro de rangos
    - **Valida: Requisitos 1.1, 1.2, 1.3, 1.4, 1.5, 6.8**

  - [x] 3.4 Escribir test de propiedad: restricción umbralDisparo > potencialReposo
    - **Propiedad 2: Restricción umbralDisparo > potencialReposo**
    - Verificar que todo individuo generado cumple umbralDisparo > potencialReposo
    - **Valida: Requisito 1.6**

  - [x] 3.5 Escribir test de propiedad: ConfiguracionRed válida
    - **Propiedad 3: ConfiguracionRed válida**
    - Verificar que `construirConfiguracion()` no lanza excepción para individuos generados
    - **Valida: Requisitos 1.7, 5.6**

  - [x] 3.6 Escribir test de propiedad: límite topológico respetado
    - **Propiedad 4: Límite topológico respetado**
    - Verificar que todo individuo generado tiene neuronasTotal ≤ limiteTopologico
    - **Valida: Requisitos 2.1, 2.2, 5.6**

  - [x] 3.7 Escribir test de propiedad: reparación proporcional preserva el límite
    - **Propiedad 5: Reparación proporcional preserva el límite**
    - Generar cromosomas que excedan el límite y verificar que `repararTopologia()` los corrige manteniendo proporciones y al menos 1 neurona por capa
    - **Valida: Requisito 2.3**

  - [x] 3.8 Escribir tests unitarios para FabricaIndividuos en `src/test/java/es/jastxz/nn/genetico/FabricaIndividuosTest.java`
    - Verificar generación aleatoria produce individuo válido
    - Verificar reparación topológica con caso conocido
    - _Requisitos: 1.6, 1.7, 2.3_

- [x] 4. Checkpoint — Verificar tipos base y fábrica
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

- [x] 5. Implementar EvaluadorFitness
  - [x] 5.1 Crear la clase `EvaluadorFitness` en `src/main/java/es/jastxz/nn/genetico/EvaluadorFitness.java`
    - Implementar `evaluar(Individuo)` con cálculo de fitness ponderado multi-objetivo
    - Implementar `evaluarPoblacion(List<Individuo>)` para evaluación masiva
    - Integrar con `ConfiguracionBenchmark`, `RecolectorMetricas` y `DetectorLimites`
    - Asignar fitness 0.0 para benchmarks con clasificación "timeout" o "limite_no_superado"
    - Capturar excepciones durante construcción/entrenamiento y asignar fitness 0.0
    - _Requisitos: 3.1, 3.2, 3.3, 3.4, 3.5, 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 5.2 Escribir test de propiedad: fitness como combinación ponderada normalizada
    - **Propiedad 6: Fitness como combinación ponderada normalizada**
    - Verificar que el fitness calculado es la suma ponderada correcta con componentes en [0.0, 1.0]
    - **Valida: Requisitos 3.2, 3.3, 3.4**

  - [x] 5.3 Escribir test de propiedad: fitness cero para benchmarks fallidos
    - **Propiedad 7: Fitness cero para benchmarks fallidos**
    - Verificar que individuos con clasificación "timeout" o "limite_no_superado" reciben fitness 0.0
    - **Valida: Requisitos 3.5, 9.4**

  - [x] 5.4 Escribir tests unitarios para EvaluadorFitness en `src/test/java/es/jastxz/nn/genetico/EvaluadorFitnessTest.java`
    - Verificar cálculo de fitness con valores conocidos
    - Verificar fitness 0.0 para benchmark fallido
    - _Requisitos: 3.2, 3.5, 9.4_

- [x] 6. Implementar SelectorTorneo
  - [x] 6.1 Crear la clase `SelectorTorneo` con record `Pareja` en `src/main/java/es/jastxz/nn/genetico/SelectorTorneo.java`
    - Implementar `seleccionarParejas(List<Individuo>)` con torneo de tamaño k
    - Implementar `ejecutarTorneo(List<Individuo>)` privado
    - Garantizar padres distintos repitiendo torneo si coinciden
    - Garantizar al menos 15% de padres no-élite
    - Implementar `Pareja` con `mejorPadre()` y `peorPadre()`
    - _Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 6.2 Escribir test de propiedad: ganador del torneo es el mejor de los k seleccionados
    - **Propiedad 8: Ganador del torneo es el mejor de los k seleccionados**
    - Verificar que el ganador tiene fitness ≥ todos los demás del torneo
    - **Valida: Requisitos 4.1, 4.2**

  - [x] 6.3 Escribir test de propiedad: padres siempre distintos
    - **Propiedad 9: Padres siempre distintos**
    - Verificar que padre1 y padre2 de cada pareja son individuos distintos
    - **Valida: Requisito 4.3**

  - [x] 6.4 Escribir test de propiedad: diversidad mínima de padres no-élite
    - **Propiedad 11: Diversidad mínima de padres no-élite**
    - Verificar que al menos 15% de los padres seleccionados son no-élite
    - **Valida: Requisito 4.5**

  - [x] 6.5 Escribir tests unitarios para SelectorTorneo en `src/test/java/es/jastxz/nn/genetico/SelectorTorneoTest.java`
    - Verificar torneo con k=1 retorna aleatorio, torneo con k=N retorna el mejor
    - _Requisitos: 4.2, 4.6_

- [x] 7. Implementar OperadorCruce
  - [x] 7.1 Crear la clase `OperadorCruce` en `src/main/java/es/jastxz/nn/genetico/OperadorCruce.java`
    - Implementar `cruzar(Individuo, Individuo)` con cruce multi-punto por bloques funcionales
    - Implementar sesgo hacia el mejor padre (Y > mitad de segmentos del mejor padre)
    - Manejar bloque TOPOLOGIA completo del padre seleccionado
    - Aplicar reparación topológica si excede límite
    - Respetar probabilidad de cruce (retornar copia del mejor padre si no se aplica)
    - _Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x] 7.2 Escribir test de propiedad: integridad de bloques funcionales en cruce
    - **Propiedad 12: Integridad de bloques funcionales en cruce**
    - Verificar que cada bloque del descendiente proviene íntegramente de uno de los padres
    - **Valida: Requisitos 5.1, 5.2, 5.4, 5.5**

  - [x] 7.3 Escribir test de propiedad: sesgo hacia el mejor padre en cruce
    - **Propiedad 13: Sesgo hacia el mejor padre en cruce**
    - Verificar que el número de bloques del mejor padre es > mitad del total
    - **Valida: Requisito 5.3**

  - [x] 7.4 Escribir tests unitarios para OperadorCruce en `src/test/java/es/jastxz/nn/genetico/OperadorCruceTest.java`
    - Verificar cruce con probabilidad 0 retorna mejor padre
    - Verificar cruce con 1 punto de corte
    - _Requisitos: 5.7, 5.8_

- [x] 8. Implementar OperadorMutacion
  - [x] 8.1 Crear la clase `OperadorMutacion` en `src/main/java/es/jastxz/nn/genetico/OperadorMutacion.java`
    - Implementar `mutar(Individuo)` con mutación adaptativa por tipo de gen
    - Implementar perturbación entera (±[1, 20% rango]), gaussiana para reales (σ = 10% rango), inversión para booleanos, selección uniforme para enums
    - Manejar mutación de capasOcultas (añadir/eliminar capas)
    - Validar restricción umbralDisparo > potencialReposo post-mutación
    - Aplicar clamping y reparación topológica
    - _Requisitos: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8_

  - [x] 8.2 Escribir test de propiedad: mutación con probabilidad cero no modifica
    - **Propiedad 14: Mutación con probabilidad cero no modifica**
    - Verificar que mutar con probabilidad 0.0 retorna cromosoma idéntico
    - **Valida: Requisito 6.1**

  - [x] 8.3 Escribir test de propiedad: mutación respeta estrategia por tipo de gen
    - **Propiedad 15: Mutación respeta estrategia por tipo de gen**
    - Verificar que cada tipo de gen se muta según su estrategia específica
    - **Valida: Requisitos 6.2, 6.3, 6.4, 6.5**

  - [x] 8.4 Escribir test de propiedad: mutación de capas mantiene coherencia topológica
    - **Propiedad 16: Mutación de capas mantiene coherencia topológica**
    - Verificar que añadir capas inicializa neuronas en [1, 512] y eliminar capas conserva las primeras
    - **Valida: Requisitos 6.6, 6.7**

  - [x] 8.5 Escribir tests unitarios para OperadorMutacion en `src/test/java/es/jastxz/nn/genetico/OperadorMutacionTest.java`
    - Verificar mutación con probabilidad 0 no modifica
    - Verificar mutación de capas (añadir/eliminar)
    - _Requisitos: 6.1, 6.6, 6.7_

- [x] 9. Checkpoint — Verificar operadores genéticos
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

- [x] 10. Implementar ConfiguracionAG y ConfiguracionAGBuilder
  - [x] 10.1 Crear el record `ConfiguracionAG` y la clase `ConfiguracionAGBuilder` en `src/main/java/es/jastxz/nn/genetico/ConfiguracionAG.java`
    - Implementar record con todos los parámetros del AG y validaciones en constructor compacto
    - Implementar builder con valores por defecto y métodos fluidos
    - Validar que pesos de fitness sumen 1.0 (tolerancia 1e-6)
    - Validar que tamañoTorneo ≤ tamañoPoblacion
    - Validar puntosCorte en [1, 4], numElites < tamañoPoblacion, limiteTopologico > 0
    - _Requisitos: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 10.2 Escribir test de propiedad: validación de ConfiguracionAG
    - **Propiedad 20: Validación de ConfiguracionAG**
    - Verificar que pesos que no suman 1.0 lanzan `IllegalArgumentException`
    - Verificar que tamañoTorneo > tamañoPoblacion lanza `IllegalArgumentException`
    - **Valida: Requisitos 8.3, 8.4**

  - [x] 10.3 Escribir tests unitarios para ConfiguracionAGBuilder en `src/test/java/es/jastxz/nn/genetico/ConfiguracionAGTest.java`
    - Verificar valores por defecto
    - Verificar validación de pesos y torneo
    - _Requisitos: 8.2, 8.3, 8.4_

- [x] 11. Implementar MotorEvolutivo
  - [x] 11.1 Crear la clase `MotorEvolutivo` en `src/main/java/es/jastxz/nn/genetico/MotorEvolutivo.java`
    - Implementar constructor que instancie todos los componentes internos a partir de `ConfiguracionAG` y `NivelComplejidad`
    - Implementar `evolucionar()` con el ciclo: evaluar → seleccionar → cruzar → mutar → nueva población
    - Implementar preservación de élites (copiar los N mejores sin modificación)
    - Implementar criterio de parada por máximo de generaciones
    - Implementar criterio de parada por estancamiento (mejor fitness no mejora >1% en N generaciones)
    - Registrar estadísticas por generación (mejor, promedio, peor fitness, mejor configuración)
    - Rastrear mejor individuo global a lo largo de toda la evolución
    - Aceptar semilla aleatoria para reproducibilidad
    - _Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 4.4_

  - [x] 11.2 Escribir test de propiedad: preservación de élites
    - **Propiedad 10: Preservación de élites**
    - Verificar que los N mejores individuos de una generación están presentes sin modificación en la siguiente
    - **Valida: Requisito 4.4**

  - [x] 11.3 Escribir test de propiedad: criterios de parada respetados
    - **Propiedad 17: Criterios de parada respetados**
    - Verificar que el número de generaciones no excede maxGeneraciones
    - Verificar que estancamiento se detecta correctamente
    - **Valida: Requisitos 7.2, 7.3**

  - [x] 11.4 Escribir test de propiedad: mejor global es el mejor de toda la evolución
    - **Propiedad 18: Mejor global es el mejor de toda la evolución**
    - Verificar que el fitness del mejor global ≥ mejor fitness de cada generación en el historial
    - **Valida: Requisito 7.5**

  - [x] 11.5 Escribir test de propiedad: reproducibilidad con semilla
    - **Propiedad 19: Reproducibilidad con semilla**
    - Verificar que dos ejecuciones con misma semilla y configuración producen el mismo resultado
    - **Valida: Requisito 7.7**

  - [x] 11.6 Escribir tests unitarios para MotorEvolutivo en `src/test/java/es/jastxz/nn/genetico/MotorEvolutivoTest.java`
    - Verificar evolución con 1 generación
    - Verificar parada por estancamiento
    - _Requisitos: 7.1, 7.2, 7.3_

- [x] 12. Implementar InformeEvolucion
  - [x] 12.1 Crear los records `InformeEvolucion` y `EstadisticaGeneracion` en `src/main/java/es/jastxz/nn/genetico/InformeEvolucion.java`
    - Implementar `imprimir()` con salida legible por consola vía `System.out`
    - Incluir: mejor individuo, generación donde se encontró, motivo de parada, historial de fitness, configuración del AG y límite topológico
    - _Requisitos: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 12.2 Escribir test de propiedad: informe completo
    - **Propiedad 21: Informe completo**
    - Verificar que el informe contiene todos los campos requeridos: mejor individuo con fitness > -1, generación, motivo de parada, historial con una entrada por generación, y configuración del AG
    - **Valida: Requisitos 10.1, 10.2, 10.3, 10.5**

  - [x] 12.3 Escribir tests unitarios para InformeEvolucion en `src/test/java/es/jastxz/nn/genetico/InformeEvolucionTest.java`
    - Verificar que `imprimir()` produce salida no vacía
    - _Requisitos: 10.4_

- [x] 13. Checkpoint — Verificar motor evolutivo e informe
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

- [x] 14. Integración y cableado final
  - [x] 14.1 Verificar integración completa del ciclo evolutivo end-to-end
    - Asegurar que `MotorEvolutivo.evolucionar()` ejecuta el ciclo completo: generación inicial → evaluación → selección → cruce → mutación → nueva población → informe
    - Verificar que todos los componentes están correctamente conectados
    - Verificar que la integración con `ConfiguracionBenchmark`, `RecolectorMetricas` y `DetectorLimites` funciona correctamente
    - _Requisitos: 7.1, 9.1, 9.2, 9.3_

  - [x] 14.2 Escribir tests de integración para el ciclo evolutivo completo en `src/test/java/es/jastxz/nn/genetico/IntegracionAGTest.java`
    - Verificar ejecución completa con configuración mínima (población pequeña, pocas generaciones)
    - Verificar que el informe generado es consistente con la ejecución
    - _Requisitos: 7.1, 9.1, 10.1_

- [x] 15. Checkpoint final — Verificar integración completa
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Los tests de propiedades validan propiedades universales de corrección (jqwik 1.7.4, mínimo 100 iteraciones)
- Los tests unitarios validan ejemplos específicos y casos borde (JUnit 5)
- Todo el código va en el paquete `es.jastxz.nn.genetico` sin modificar paquetes existentes
