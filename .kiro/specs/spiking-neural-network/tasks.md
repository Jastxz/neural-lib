# Plan de Implementación: Spiking Neural Network

## Descripción General

Implementación de una Red Neuronal de Spikes (SNN) independiente en Java con modelo Leaky Integrate-and-Fire (LIF), plasticidad STDP, codificación rate coding, y período refractario. La implementación se organizará en el paquete `es.jastxz.nn.spiking` y seguirá las convenciones del proyecto (Serializable, enums, gestores, builders).

## Tareas

- [x] 1. Configurar infraestructura y dependencias
  - Agregar dependencia jqwik 1.7.4 en pom.xml para property-based testing
  - Crear estructura de paquetes: es.jastxz.nn.spiking
  - Crear subdirectorios para tests: unit/, properties/, integration/
  - _Requisitos: 20.1_

- [x] 2. Implementar enumeraciones y clases de datos básicas
  - [x] 2.1 Crear enumeraciones
    - Crear ModoCodificacion (POISSON, REGULAR, BURST)
    - Crear TipoInicializacion (UNIFORME, NORMAL, CONSTANTE, DESDE_ARRAY)
    - Crear TipoNormalizacion (L1, L2)
    - _Requisitos: 3.3, 11.1, 11.2, 17.3, 17.4_
  
  - [x] 2.2 Crear EventoSpike
    - Implementar Comparable<EventoSpike> y Serializable
    - Campos: neuronaId, capa, indice, timestamp, potencialMembrana
    - Implementar compareTo ordenando por timestamp
    - _Requisitos: 9.1, 9.5_
  
  - [x] 2.3 Crear ColaEventos
    - Usar PriorityQueue<EventoSpike> interna
    - Implementar encolar(EventoSpike)
    - Implementar obtenerEventos(long timestamp) que extrae todos los eventos del timestamp
    - _Requisitos: 8.3, 8.4_


- [x] 3. Implementar modelo neuronal LIF (NeuronaSpiking)
  - [x] 3.1 Crear clase NeuronaSpiking con campos y constructor
    - Implementar Serializable
    - Campos inmutables: id, capa, indice, umbralDisparo, potencialReposo, constanteDecaimiento, duracionRefractario
    - Campos mutables: potencialMembrana, timestampUltimoSpike, timestepsDesdeUltimoSpike
    - Validar umbralDisparo > potencialReposo y constanteDecaimiento > 0
    - _Requisitos: 1.1, 10.1, 10.2, 10.3, 10.4, 10.6, 10.7_
  
  - [x] 3.2 Implementar decaimiento exponencial
    - Método aplicarDecaimiento(double dt)
    - Fórmula: V(t+1) = V(t) * exp(-dt/tau) + V_reposo * (1 - exp(-dt/tau))
    - _Requisitos: 1.1, 7.5_
  
  - [ ]* 3.3 Escribir property test para decaimiento exponencial
    - **Property 1: Decaimiento exponencial del potencial**
    - **Valida: Requisitos 1.1, 1.5**
  
  - [x] 3.4 Implementar recepción de señales y activación
    - Método recibirSeñal(double señal) que incrementa potencialMembrana
    - Método evaluarActivacion(long timestepActual) que verifica umbral y refractario
    - Método generarSpike(long timestamp) que resetea potencial y registra spike
    - _Requisitos: 1.2, 1.3, 1.4, 1.6, 2.2, 2.4_
  
  - [ ]* 3.5 Escribir property tests para modelo neuronal
    - **Property 2: Incremento de potencial por señal sináptica**
    - **Valida: Requisitos 1.2**
    - **Property 3: Generación de spike al alcanzar umbral**
    - **Valida: Requisitos 1.3**
    - **Property 4: Reset de potencial después de spike**
    - **Valida: Requisitos 1.4**
    - **Property 5: Registro de timestamps de spikes**
    - **Valida: Requisitos 1.6**
    - **Property 6: Rechazo de spikes durante período refractario**
    - **Valida: Requisitos 2.2**
    - **Property 7: Recuperación después de período refractario**
    - **Valida: Requisitos 2.4**
  
  - [x] 3.6 Implementar métodos de consulta
    - getPotencialMembrana(), getHistorialSpikes()
    - calcularFrecuencia(long ventanaTemporal)
    - _Requisitos: 9.2, 9.3_
  
  - [ ]* 3.7 Escribir unit tests para casos edge
    - Test: neurona con potencial inicial igual a umbral dispara inmediatamente
    - Test: neurona en refractario ignora señales fuertes
    - Test: frecuencia calculada correctamente con historial vacío

- [x] 4. Implementar sinapsis y propagación (SinapsisSpiking)
  - [x] 4.1 Crear clase SinapsisSpiking
    - Implementar Serializable
    - Campos: presinaptica, postsinaptica, peso (mutable), retardo, pesoMin, pesoMax
    - Campos para STDP: timestampUltimoSpikePresinaptico, timestampUltimoSpikePostsinaptico
    - Validar que neuronas no sean null y retardo >= 0
    - _Requisitos: 5.6, 8.1, 8.5, 15.3_
  
  - [x] 4.2 Implementar propagación de spikes con retardo
    - Método propagarSpike(long timestampOrigen) retorna EventoSpike
    - Calcular timestamp de entrega: timestampOrigen + retardo
    - _Requisitos: 8.2, 8.4_
  
  - [ ]* 4.3 Escribir property test para retardo sináptico
    - **Property 20: Retardo sináptico correcto**
    - **Valida: Requisitos 8.2, 8.4**
  
  - [x] 4.3 Implementar ajuste de peso
    - Método ajustarPeso(double delta)
    - Aplicar clamp a [pesoMin, pesoMax]
    - _Requisitos: 5.6_
  
  - [ ]* 4.4 Escribir property test para límites de peso
    - **Property 16: Límites de peso sináptico**
    - **Valida: Requisitos 5.6**
  
  - [ ]* 4.5 Escribir unit tests para sinapsis
    - Test: retardo cero transmite instantáneamente
    - Test: ajuste de peso respeta límites min/max

- [x] 5. Checkpoint - Verificar modelo neuronal básico
  - Asegurar que todos los tests pasen, preguntar al usuario si surgen dudas.


- [x] 6. Implementar codificación rate coding (GestorCodificacion)
  - [x] 6.1 Crear clase GestorCodificacion
    - Campos: frecuenciaMaxima, modo (ModoCodificacion), Random
    - Constructor con validación de frecuenciaMaxima > 0
    - _Requisitos: 3.4, 16.5_
  
  - [x] 6.2 Implementar codificación Poisson
    - Método codificarPoisson(double valor, int neuronaId, int duracion)
    - Generar spikes probabilísticamente: P = valor * frecuenciaMaxima * dt
    - _Requisitos: 3.3, 16.3_
  
  - [x] 6.3 Implementar codificación regular
    - Método codificarRegular(double valor, int neuronaId, int duracion)
    - Espaciar spikes uniformemente según frecuencia
    - _Requisitos: 16.6_
  
  - [x] 6.4 Implementar codificación burst
    - Método codificarBurst(double valor, int neuronaId, int duracion)
    - Generar ráfagas de spikes consecutivos
    - _Requisitos: 16.5_
  
  - [x] 6.5 Implementar método principal codificar
    - Método codificar(double[] valores, int duracionTimesteps)
    - Delegar a método específico según modo
    - Retornar List<EventoSpike> ordenada por timestamp
    - _Requisitos: 3.1, 3.2, 16.1, 16.2_
  
  - [ ]* 6.6 Escribir property tests para codificación
    - **Property 8: Proporcionalidad en codificación rate coding**
    - **Valida: Requisitos 3.1, 3.2**
    - **Property 9: Distribución Poisson en codificación**
    - **Valida: Requisitos 3.3**
    - **Property 36: Codificación de array a trenes de spikes**
    - **Valida: Requisitos 16.1**
    - **Property 37: Distribución temporal de spikes**
    - **Valida: Requisitos 16.3**
    - **Property 38: Proporcionalidad de frecuencias en codificación**
    - **Valida: Requisitos 16.4**
    - **Property 39: Espaciado uniforme en codificación regular**
    - **Valida: Requisitos 16.6**
  
  - [ ]* 6.7 Escribir unit tests para casos edge
    - Test: valor 0 no genera spikes
    - Test: valor 1 genera frecuencia máxima
    - Test: modo REGULAR produce intervalos uniformes

- [x] 7. Implementar decodificación rate coding (GestorDecodificacion)
  - [x] 7.1 Crear clase GestorDecodificacion
    - Campos: ventanaTemporal (timesteps), frecuenciaMaxima
    - Map<Long, VentanaDeslizante> para mantener ventanas por neurona
    - _Requisitos: 4.6_
  
  - [x] 7.2 Implementar conteo de spikes en ventana
    - Método contarSpikesEnVentana(long neuronaId, long timestampActual)
    - Mantener ventana deslizante de timestamps
    - _Requisitos: 4.1_
  
  - [x] 7.3 Implementar decodificación
    - Método decodificar(List<EventoSpike> spikes)
    - Fórmula: valor = conteo_spikes / (frecuenciaMaxima * duracionVentana)
    - Clamp a [0, 1]
    - _Requisitos: 4.2, 4.3, 4.4, 4.5_
  
  - [x] 7.4 Implementar decodificación de capa
    - Método decodificarCapa(List<List<EventoSpike>> spikesPorNeurona)
    - Retornar double[] con valores decodificados
    - _Requisitos: 4.2_
  
  - [ ]* 7.5 Escribir property tests para decodificación
    - **Property 10: Conteo correcto de spikes en ventana**
    - **Valida: Requisitos 4.1**
    - **Property 11: Rango de decodificación**
    - **Valida: Requisitos 4.2**
    - **Property 12: Fórmula de decodificación**
    - **Valida: Requisitos 4.5**
  
  - [ ]* 7.6 Escribir unit tests para casos edge
    - Test: decodificación sin spikes retorna 0
    - Test: frecuencia máxima retorna 1
    - Test: ventana vacía retorna 0

- [x] 8. Implementar plasticidad STDP (GestorSTDP)
  - [x] 8.1 Crear clase GestorSTDP
    - Campos: amplitudLTP, amplitudLTD, tauLTP, tauLTD
    - Validar amplitudes > 0 y taus > 0
    - _Requisitos: 5.7, 5.8_
  
  - [x] 8.2 Implementar cálculo de cambio de peso
    - Método calcularCambioPeso(long dt, boolean esLTP)
    - Fórmula LTP: dw = A_LTP * exp(-dt/tau_LTP) cuando dt > 0
    - Fórmula LTD: dw = -A_LTD * exp(dt/tau_LTD) cuando dt < 0
    - _Requisitos: 5.3, 5.4, 5.5_
  
  - [x] 8.3 Implementar aplicación de STDP a sinapsis
    - Método aplicarSTDP(SinapsisSpiking sinapsis, long timestampActual)
    - Calcular dt = timestampPost - timestampPre
    - Aplicar LTP o LTD según signo de dt
    - Actualizar peso de sinapsis respetando límites
    - _Requisitos: 5.1, 5.2, 5.6_
  
  - [x] 8.4 Implementar aplicación de STDP a capa
    - Método aplicarSTDPACapa(List<SinapsisSpiking> sinapsis, long timestampActual)
    - Iterar sobre todas las sinapsis aplicando STDP
    - _Requisitos: 5.1, 5.2_
  
  - [ ]* 8.5 Escribir property tests para STDP
    - **Property 13: STDP - Long-Term Potentiation**
    - **Valida: Requisitos 5.1, 5.4**
    - **Property 14: STDP - Long-Term Depression**
    - **Valida: Requisitos 5.2, 5.5**
    - **Property 15: Fórmula STDP con ventana exponencial**
    - **Valida: Requisitos 5.3**
  
  - [ ]* 8.6 Escribir unit tests para STDP
    - Test: dt = 0 no cambia peso
    - Test: dt muy grande produce cambio mínimo
    - Test: amplitudes diferentes para LTP y LTD

- [x] 9. Checkpoint - Verificar codificación, decodificación y STDP
  - Asegurar que todos los tests pasen, preguntar al usuario si surgen dudas.


- [x] 10. Implementar configuración (ConfiguracionRed y Builder)
  - [x] 10.1 Crear clase ConfiguracionRed
    - Implementar Serializable
    - Campos inmutables para todos los parámetros (ver diseño)
    - Constructor que valida todos los parámetros
    - _Requisitos: 10.1, 10.2, 10.3, 10.4, 10.5, 11.4, 17.6, 18.6, 19.4_
  
  - [x] 10.2 Crear ConfiguracionRedBuilder
    - Valores por defecto razonables para todos los parámetros
    - Métodos fluent: topologia(), parametrosLIF(), parametrosSTDP(), etc.
    - Método build() que construye ConfiguracionRed
    - _Requisitos: 10.1, 10.2, 10.3, 10.4_
  
  - [ ]* 10.3 Escribir unit tests para configuración
    - Test: builder con valores por defecto construye configuración válida
    - Test: validación de umbral <= reposo falla
    - Test: validación de tau <= 0 falla
    - Test: validación de topología vacía falla

- [x] 11. Implementar gestores auxiliares
  - [x] 11.1 Crear GestorMetricas
    - Campos: totalSpikes, spikesPorNeurona, timestampsPorNeurona
    - Métodos: registrarSpike(), getTotalSpikes(), getTasaPromedioGlobal()
    - Métodos: calcularCostoEnergetico(), calcularDispersionActividad()
    - Método exportarMetricas() retorna Map<String, Object>
    - _Requisitos: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_
  
  - [ ]* 11.2 Escribir property tests para métricas
    - **Property 29: Conteo total de spikes**
    - **Valida: Requisitos 13.1**
    - **Property 30: Tasa promedio por neurona**
    - **Valida: Requisitos 13.2**
    - **Property 31: Costo energético proporcional a spikes**
    - **Valida: Requisitos 13.4**
    - **Property 32: Dispersión de actividad**
    - **Valida: Requisitos 13.5**
  
  - [x] 11.3 Crear RegistroActividad
    - Implementar Serializable
    - Campos: todosLosSpikes (List), spikesPorNeurona (Map)
    - Métodos: registrar(), obtenerSpikes(), obtenerSpikesEnRango()
    - Método exportarCSV() con formato: timestamp,capa,indice,potencial
    - _Requisitos: 9.1, 9.2, 9.4, 9.5_
  
  - [ ]* 11.4 Escribir property tests para registro
    - **Property 21: Registro completo de spikes**
    - **Valida: Requisitos 9.1**
    - **Property 22: Cálculo de frecuencia de disparo**
    - **Valida: Requisitos 9.3**
    - **Property 23: Completitud de registro de actividad**
    - **Valida: Requisitos 9.5**
  
  - [ ]* 11.5 Escribir unit tests para gestores
    - Test: métricas resetean correctamente
    - Test: exportar CSV con registro vacío
    - Test: registro filtra por rango de timestamps correctamente

- [x] 12. Implementar red neuronal principal (RedNeuralSpiking)
  - [x] 12.1 Crear estructura básica de RedNeuralSpiking
    - Implementar Serializable
    - Campos: capas (List<List<NeuronaSpiking>>), sinapsis (List<SinapsisSpiking>)
    - Campos: timestepActual, duracionTimestep, colaEventos
    - Campos: gestores (codificador, decodificador, gestorSTDP, gestorMetricas, registro)
    - Campo: configuracion (ConfiguracionRed), modoEntrenamiento
    - _Requisitos: 6.1, 6.4, 7.3_
  
  - [x] 12.2 Implementar constructor y validación de arquitectura
    - Constructor que recibe ConfiguracionRed
    - Crear capas según topología
    - Validar: al menos una capa, cada capa con al menos una neurona
    - Validar: índices de capa consecutivos desde 0
    - _Requisitos: 6.1, 6.2, 15.1, 15.2, 15.6_
  
  - [ ]* 12.3 Escribir property tests para validación
    - **Property 24: Validación de umbral mayor que reposo**
    - **Valida: Requisitos 10.6**
    - **Property 25: Validación de constante de decaimiento positiva**
    - **Valida: Requisitos 10.7**
  
  - [ ]* 12.4 Escribir unit tests para validación de arquitectura
    - Test: construcción con cero capas falla
    - Test: construcción con capa vacía falla
    - Test: índices no consecutivos fallan

- [x] 13. Implementar gestión de conexiones
  - [x] 13.1 Implementar creación de conexiones
    - Método crearConexion(NeuronaSpiking pre, NeuronaSpiking post, double peso, int retardo)
    - Validar que neuronas existan en la red
    - Validar que no exista conexión duplicada
    - Agregar sinapsis a lista
    - _Requisitos: 6.3, 15.3, 15.4_
  
  - [x] 13.2 Implementar inicialización de pesos
    - Método inicializarPesos(TipoInicializacion tipo, double min, double max)
    - Implementar inicialización UNIFORME, NORMAL, CONSTANTE
    - Validar que pesos estén en [min, max]
    - _Requisitos: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_
  
  - [ ]* 13.3 Escribir property test para inicialización
    - **Property 26: Validación de pesos dentro de límites**
    - **Valida: Requisitos 11.6**
  
  - [ ]* 13.4 Escribir unit tests para conexiones
    - Test: conexión duplicada falla
    - Test: conexión a neurona inexistente falla
    - Test: inicialización UNIFORME produce pesos en rango


- [x] 14. Implementar simulación temporal
  - [x] 14.1 Implementar avance de timestep
    - Método avanzarTimestep()
    - Aplicar decaimiento a todas las neuronas
    - Obtener eventos pendientes de la cola para timestep actual
    - Procesar cada evento: neurona.recibirSeñal(peso)
    - Evaluar activación de cada neurona en orden de capas
    - Para neuronas que disparan: propagar spike a través de sinapsis
    - Incrementar timestepActual
    - _Requisitos: 7.1, 7.2, 7.5, 7.6, 7.7_
  
  - [ ]* 14.2 Escribir property tests para simulación
    - **Property 17: Propagación de spikes entre capas**
    - **Valida: Requisitos 6.5, 7.6**
    - **Property 18: Orden de operaciones en timestep**
    - **Valida: Requisitos 7.5**
    - **Property 19: Completitud de procesamiento de señales**
    - **Valida: Requisitos 7.7**
  
  - [x] 14.3 Implementar reset de estado temporal
    - Método resetearEstadoTemporal()
    - Resetear potenciales de todas las neuronas a reposo
    - Limpiar cola de eventos
    - Limpiar historiales de spikes
    - Resetear timestepActual a 0
    - _Requisitos: 14.2_
  
  - [ ]* 14.4 Escribir unit tests para simulación
    - Test: timestep avanza correctamente
    - Test: eventos se procesan en orden
    - Test: reset limpia todo el estado temporal

- [x] 15. Implementar procesamiento de patrones
  - [x] 15.1 Implementar procesamiento de patrón único
    - Método procesar(double[] inputs, int duracionTimesteps)
    - Codificar inputs a eventos de spike
    - Encolar eventos en cola
    - Ejecutar simulación por duracionTimesteps
    - Decodificar spikes de capa de salida
    - Retornar double[] con valores decodificados
    - _Requisitos: 3.1, 4.2, 7.1, 16.2_
  
  - [x] 15.2 Implementar procesamiento por lotes
    - Método procesarLote(double[][] batchInputs, int duracionPorPatron, boolean resetEntrePat rones)
    - Iterar sobre patrones
    - Si resetEntrePatrones: llamar resetearEstadoTemporal() entre patrones
    - Acumular métricas a través del lote
    - Retornar double[][] con salidas
    - _Requisitos: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [ ]* 15.3 Escribir property tests para procesamiento
    - **Property 33: Aislamiento entre patrones en lote**
    - **Valida: Requisitos 14.2**
    - **Property 34: Acumulación de métricas en lote**
    - **Valida: Requisitos 14.3**
    - **Property 35: Número correcto de salidas en lote**
    - **Valida: Requisitos 14.4**
  
  - [ ]* 15.4 Escribir unit tests para procesamiento
    - Test: procesar patrón simple produce salida en rango [0,1]
    - Test: lote con reset aísla patrones
    - Test: lote sin reset mantiene estado

- [x] 16. Checkpoint - Verificar red completa funcional
  - Asegurar que todos los tests pasen, preguntar al usuario si surgen dudas.

- [x] 17. Implementar entrenamiento con STDP
  - [x] 17.1 Implementar modo entrenamiento
    - Método setModoEntrenamiento(boolean activar)
    - Durante simulación: si modoEntrenamiento, aplicar STDP después de cada spike
    - _Requisitos: 5.1, 5.2_
  
  - [x] 17.2 Implementar entrenamiento supervisado
    - Método entrenar(double[][] batchInputs, double[][] batchTargets, int duracionPorPatron)
    - Activar modo entrenamiento
    - Procesar cada patrón
    - Aplicar STDP basado en diferencia entre salida y target
    - _Requisitos: 5.1, 5.2, 5.3_
  
  - [ ]* 17.3 Escribir integration test para entrenamiento
    - Test: red aprende patrón simple con STDP
    - Test: pesos cambian después de entrenamiento
    - Test: error disminuye con más épocas

- [x] 18. Implementar normalización de pesos
  - [x] 18.1 Implementar normalización L1 y L2
    - Método normalizarPesos(TipoNormalizacion tipo, int[] capasObjetivo)
    - Para cada neurona en capas objetivo: normalizar pesos entrantes
    - L1: suma de valores absolutos = valor objetivo
    - L2: suma de cuadrados = valor objetivo
    - Preservar signo de pesos
    - _Requisitos: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_
  
  - [ ]* 18.2 Escribir property tests para normalización
    - **Property 40: Normalización L1 de pesos**
    - **Valida: Requisitos 17.2**
    - **Property 41: Preservación de signo en normalización**
    - **Valida: Requisitos 17.5**
  
  - [ ]* 18.3 Escribir unit tests para normalización
    - Test: normalización L1 suma correctamente
    - Test: normalización L2 suma cuadrados correctamente
    - Test: signos se preservan


- [x] 19. Implementar homeostasis sináptica
  - [x] 19.1 Agregar campos de homeostasis a NeuronaSpiking
    - Campos: tasaDisparoObjetivo, tasaDisparoPromedio, tasaAjusteHomeostasis
    - Inicializar desde ConfiguracionRed si homeostasisActiva
    - _Requisitos: 18.6_
  
  - [x] 19.2 Implementar ajuste homeostático
    - Método aplicarHomeostasis() en NeuronaSpiking
    - Calcular tasa actual usando ventana deslizante
    - Si tasa < objetivo: decrementar umbralDisparo
    - Si tasa > objetivo: incrementar umbralDisparo
    - Aplicar con tasa de aprendizaje configurable
    - _Requisitos: 18.1, 18.2, 18.3, 18.4, 18.5, 18.7_
  
  - [x] 19.3 Integrar homeostasis en simulación
    - Llamar aplicarHomeostasis() periódicamente durante simulación
    - Configurar frecuencia de aplicación
    - _Requisitos: 18.1_
  
  - [ ]* 19.4 Escribir property test para homeostasis
    - **Property 42: Ajuste homeostático del umbral**
    - **Valida: Requisitos 18.2, 18.3, 18.4**
  
  - [ ]* 19.5 Escribir unit tests para homeostasis
    - Test: tasa baja decrementa umbral
    - Test: tasa alta incrementa umbral
    - Test: tasa en objetivo no cambia umbral

- [x] 20. Implementar inhibición lateral
  - [x] 20.1 Implementar cálculo de vecindad
    - Método calcularVecinas(NeuronaSpiking neurona, int radio)
    - Organizar neuronas en grid 2D dentro de capa
    - Calcular distancia euclidiana
    - Retornar neuronas dentro del radio
    - _Requisitos: 19.6_
  
  - [x] 20.2 Implementar señales inhibitorias
    - Cuando neurona dispara con inhibición lateral activa:
    - Calcular vecinas en radio de inhibición
    - Enviar señal inhibitoria: -fuerzaInhibicion * (1 - distancia/radio)
    - Solo afectar neuronas de la misma capa
    - _Requisitos: 19.1, 19.2, 19.3, 19.4, 19.5_
  
  - [ ]* 20.3 Escribir property tests para inhibición lateral
    - **Property 43: Inhibición lateral a vecinas**
    - **Valida: Requisitos 19.1**
    - **Property 44: Efecto de señal inhibitoria**
    - **Valida: Requisitos 19.2**
    - **Property 45: Inhibición lateral dentro de capa**
    - **Valida: Requisitos 19.5**
  
  - [ ]* 20.4 Escribir unit tests para inhibición lateral
    - Test: neurona fuera de radio no recibe inhibición
    - Test: señal inhibitoria decrementa potencial
    - Test: inhibición no cruza capas

- [x] 21. Implementar persistencia y serialización
  - [x] 21.1 Implementar serialización binaria
    - Método guardar(String filename) usando ObjectOutputStream
    - Método estático cargar(String filename) usando ObjectInputStream
    - Validar integridad después de cargar
    - Resetear estado temporal después de cargar
    - _Requisitos: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  
  - [ ]* 21.2 Escribir property tests para persistencia
    - **Property 27: Round-trip de serialización**
    - **Valida: Requisitos 12.5**
    - **Property 28: Reset de estado temporal al cargar**
    - **Valida: Requisitos 12.7**
  
  - [x] 21.3 Implementar serialización JSON
    - Método toJSON() que serializa configuración a JSON
    - Método estático desdeJSON(String jsonConfig) que parsea y construye red
    - Validar campos requeridos durante parseo
    - _Requisitos: 20.1, 20.2, 20.3, 20.5, 20.6_
  
  - [ ]* 21.4 Escribir property tests para JSON
    - **Property 46: Round-trip de configuración JSON**
    - **Valida: Requisitos 20.4**
    - **Property 47: Completitud de configuración serializada**
    - **Valida: Requisitos 20.7**
  
  - [ ]* 21.5 Escribir unit tests para persistencia
    - Test: guardar y cargar preserva arquitectura
    - Test: cargar resetea estado temporal
    - Test: JSON inválido falla con mensaje descriptivo
    - Test: parseo valida campos requeridos

- [x] 22. Checkpoint - Verificar features avanzadas
  - Asegurar que todos los tests pasen, preguntar al usuario si surgen dudas.


- [ ] 23. Implementar tests de integración comprehensivos
  - [ ]* 23.1 Test de red simple feed-forward
    - Crear red 2-3-1
    - Procesar patrón de entrada
    - Verificar que produce salida válida
    - Verificar que métricas se registran
  
  - [ ]* 23.2 Test de red con conexiones recurrentes
    - Crear red con conexiones dentro de la misma capa
    - Verificar que spikes recurrentes se procesan correctamente
    - Verificar que no hay ciclos infinitos
  
  - [ ]* 23.3 Test de aprendizaje con STDP
    - Entrenar red con patrones simples
    - Verificar que pesos cambian
    - Verificar que error disminuye
  
  - [ ]* 23.4 Test de procesamiento temporal
    - Procesar secuencia temporal de patrones
    - Verificar que timing de spikes es correcto
    - Verificar que retardos se respetan
  
  - [ ]* 23.5 Test de red con todas las features
    - Crear red con homeostasis e inhibición lateral activas
    - Entrenar con STDP
    - Aplicar normalización de pesos
    - Guardar y cargar
    - Verificar que todo funciona correctamente

- [x] 24. Implementar métodos de consulta y utilidad
  - [x] 24.1 Agregar métodos de consulta a RedNeuralSpiking
    - obtenerNeurona(int capa, int indice)
    - obtenerSpikes(int capa, int neurona)
    - obtenerFrecuenciaDisparo(int capa, int neurona)
    - obtenerMetricas() retorna Map<String, Object>
    - exportarRegistroCSV()
    - getTopologia(), getSinapsis(), getPesoPromedioGlobal()
    - _Requisitos: 9.2, 9.3, 9.4, 13.7_
  
  - [x] 24.2 Implementar métodos de utilidad
    - existeNeurona(long id)
    - existeConexion(NeuronaSpiking pre, NeuronaSpiking post)
    - validarIntegridad() para verificar consistencia interna
    - _Requisitos: 12.6, 15.3_
  
  - [ ]* 24.3 Escribir unit tests para métodos de consulta
    - Test: obtenerNeurona con índice inválido retorna null
    - Test: obtenerSpikes retorna historial correcto
    - Test: exportarRegistroCSV produce formato válido

- [x] 25. Documentación y ejemplos
  - [x] 25.1 Agregar Javadoc a todas las clases públicas
    - Documentar propósito de cada clase
    - Documentar parámetros y retornos de métodos públicos
    - Incluir ejemplos de uso en clases principales
  
  - [x] 25.2 Crear clase de ejemplo SpikingNetworkExample
    - Ejemplo 1: Red simple para clasificación XOR
    - Ejemplo 2: Red con STDP para aprendizaje temporal
    - Ejemplo 3: Guardar y cargar modelo
    - Ejemplo 4: Configuración desde JSON
  
  - [x] 25.3 Crear archivo README.md para el paquete
    - Descripción general de la SNN
    - Características principales
    - Ejemplos de uso básico
    - Referencias a documentación completa

- [x] 26. Optimización y refinamiento
  - [x] 26.1 Revisar y optimizar ColaEventos
    - Verificar eficiencia de PriorityQueue
    - Considerar estructura alternativa si hay muchos eventos simultáneos
  
  - [x] 26.2 Optimizar aplicación de decaimiento
    - Considerar aplicar decaimiento solo a neuronas activas
    - Cachear exp(-dt/tau) si dt y tau son constantes
  
  - [x] 26.3 Revisar manejo de memoria
    - Verificar que historiales de spikes no crezcan indefinidamente
    - Implementar límite de tamaño o ventana deslizante
  
  - [ ]* 26.4 Ejecutar tests de rendimiento
    - Medir tiempo de simulación para redes de diferentes tamaños
    - Medir uso de memoria
    - Identificar cuellos de botella

- [x] 27. Checkpoint final - Verificación completa
  - Ejecutar todos los tests (unit, property, integration)
  - Verificar cobertura de código (objetivo: >85% líneas, >80% branches)
  - Verificar que todas las 47 propiedades tienen tests
  - Revisar que todos los 20 requisitos están cubiertos
  - Asegurar que todos los tests pasen, preguntar al usuario si surgen dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia los requisitos específicos que implementa para trazabilidad
- Los checkpoints aseguran validación incremental
- Los property tests validan propiedades universales (47 propiedades del diseño)
- Los unit tests validan ejemplos específicos y casos edge
- Los integration tests validan el sistema completo funcionando en conjunto
- La implementación sigue las convenciones del proyecto: Serializable, enums, gestores, builders
- Se usa jqwik para property-based testing con mínimo 100 iteraciones por propiedad
- Todos los tests deben incluir tag de referencia: `// Feature: spiking-neural-network, Property X: [texto]`
