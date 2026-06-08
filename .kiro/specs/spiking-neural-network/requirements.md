# Documento de Requisitos: Spiking Neural Network

## Introducción

Este documento define los requisitos para implementar una Red Neuronal de Spikes (Spiking Neural Network - SNN) independiente en Java. La SNN implementará el modelo Leaky Integrate-and-Fire (LIF), codificación por tasa de disparo (rate coding), plasticidad dependiente del timing de spikes (STDP), y período refractario. Esta implementación será completamente independiente de la clase RedNeuralExperimental existente, permitiendo experimentación con procesamiento temporal y principios neuromorfológicos.

## Glosario

- **RedNeuralSpiking**: Sistema principal que implementa la red neuronal de spikes
- **NeuronaSpiking**: Unidad computacional que implementa el modelo Leaky Integrate-and-Fire
- **Spike**: Evento discreto de disparo neuronal que ocurre cuando el potencial de membrana alcanza el umbral
- **Potencial_Membrana**: Valor que representa el estado eléctrico de una neurona
- **Umbral_Disparo**: Valor del potencial de membrana que desencadena un spike
- **Periodo_Refractario**: Intervalo temporal después de un spike durante el cual la neurona no puede disparar
- **STDP**: Spike-Timing-Dependent Plasticity, mecanismo de aprendizaje basado en diferencias temporales entre spikes
- **Rate_Coding**: Esquema de codificación donde la información se representa mediante la frecuencia de spikes
- **Timestep**: Unidad discreta de tiempo en la simulación
- **Tren_Spikes**: Secuencia temporal de spikes de una neurona
- **Ventana_Temporal**: Intervalo de tiempo usado para contar spikes o calcular frecuencias
- **LTP**: Long-Term Potentiation, fortalecimiento sináptico cuando pre-spike precede a post-spike
- **LTD**: Long-Term Depression, debilitamiento sináptico cuando post-spike precede a pre-spike
- **Codificador**: Componente que convierte valores continuos en trenes de spikes
- **Decodificador**: Componente que convierte trenes de spikes en valores continuos
- **Sinapsis_Spiking**: Conexión entre neuronas que transmite spikes con peso y retardo
- **Potencial_Reposo**: Valor base del potencial de membrana al que tiende por decaimiento
- **Constante_Decaimiento**: Parámetro que controla la velocidad de decaimiento del potencial hacia reposo

## Requisitos

### Requisito 1: Modelo Neuronal Leaky Integrate-and-Fire

**User Story:** Como desarrollador de redes neuronales, quiero implementar el modelo LIF, para que las neuronas exhiban comportamiento biológicamente plausible con integración y decaimiento.

#### Criterios de Aceptación

1. THE NeuronaSpiking SHALL mantener un Potencial_Membrana que decae exponencialmente hacia el Potencial_Reposo
2. WHEN la NeuronaSpiking recibe una señal sináptica, THE NeuronaSpiking SHALL incrementar su Potencial_Membrana por el peso de la señal
3. WHEN el Potencial_Membrana alcanza o supera el Umbral_Disparo, THE NeuronaSpiking SHALL generar un Spike
4. WHEN la NeuronaSpiking genera un Spike, THE NeuronaSpiking SHALL resetear su Potencial_Membrana al Potencial_Reposo
5. THE NeuronaSpiking SHALL aplicar decaimiento exponencial usando la fórmula: V(t+1) = V(t) * exp(-dt/tau) + V_reposo * (1 - exp(-dt/tau))
6. THE NeuronaSpiking SHALL registrar el timestamp de cada Spike generado

### Requisito 2: Período Refractario

**User Story:** Como desarrollador de redes neuronales, quiero implementar período refractario, para que las neuronas respeten limitaciones biológicas de frecuencia de disparo.

#### Criterios de Aceptación

1. WHEN la NeuronaSpiking genera un Spike, THE NeuronaSpiking SHALL entrar en Periodo_Refractario
2. WHILE la NeuronaSpiking está en Periodo_Refractario, THE NeuronaSpiking SHALL rechazar cualquier intento de generar un nuevo Spike
3. THE NeuronaSpiking SHALL configurar la duración del Periodo_Refractario en timesteps
4. WHEN el Periodo_Refractario expira, THE NeuronaSpiking SHALL permitir generar nuevos Spikes
5. THE NeuronaSpiking SHALL mantener un contador de timesteps desde el último Spike

### Requisito 3: Codificación por Tasa de Disparo

**User Story:** Como usuario de la red neuronal, quiero codificar valores continuos como frecuencias de spikes, para que la red procese información mediante rate coding.

#### Criterios de Aceptación

1. THE Codificador SHALL convertir valores continuos en el rango [0, 1] a frecuencias de disparo
2. WHEN el Codificador recibe un valor de entrada, THE Codificador SHALL generar spikes con probabilidad proporcional al valor
3. THE Codificador SHALL implementar generación de spikes tipo Poisson
4. THE Codificador SHALL permitir configurar la frecuencia máxima de disparo
5. WHEN el valor de entrada es 0, THE Codificador SHALL generar cero spikes
6. WHEN el valor de entrada es 1, THE Codificador SHALL generar spikes a la frecuencia máxima configurada

### Requisito 4: Decodificación de Trenes de Spikes

**User Story:** Como usuario de la red neuronal, quiero decodificar trenes de spikes a valores continuos, para que pueda interpretar las salidas de la red.

#### Criterios de Aceptación

1. THE Decodificador SHALL contar spikes dentro de una Ventana_Temporal configurable
2. THE Decodificador SHALL convertir el conteo de spikes a un valor continuo en el rango [0, 1]
3. WHEN no hay spikes en la Ventana_Temporal, THE Decodificador SHALL retornar 0
4. WHEN la frecuencia de spikes alcanza el máximo configurado, THE Decodificador SHALL retornar 1
5. THE Decodificador SHALL calcular el valor de salida como: conteo_spikes / (frecuencia_maxima * duracion_ventana)
6. THE Decodificador SHALL permitir configurar el tamaño de la Ventana_Temporal en timesteps

### Requisito 5: Plasticidad STDP

**User Story:** Como desarrollador de redes neuronales, quiero implementar STDP, para que la red aprenda basándose en el timing preciso de spikes pre y post-sinápticos.

#### Criterios de Aceptación

1. WHEN una NeuronaSpiking pre-sináptica dispara antes que una post-sináptica, THE Sinapsis_Spiking SHALL incrementar su peso (LTP)
2. WHEN una NeuronaSpiking post-sináptica dispara antes que una pre-sináptica, THE Sinapsis_Spiking SHALL decrementar su peso (LTD)
3. THE Sinapsis_Spiking SHALL calcular el cambio de peso usando una ventana temporal exponencial: dw = A * exp(-|dt|/tau)
4. THE Sinapsis_Spiking SHALL aplicar LTP cuando dt > 0 (pre antes que post)
5. THE Sinapsis_Spiking SHALL aplicar LTD cuando dt < 0 (post antes que pre)
6. THE Sinapsis_Spiking SHALL mantener pesos dentro de límites configurables [peso_min, peso_max]
7. THE Sinapsis_Spiking SHALL permitir configurar constantes de tiempo separadas para LTP y LTD
8. THE Sinapsis_Spiking SHALL permitir configurar amplitudes separadas para LTP y LTD

### Requisito 6: Arquitectura de Red Multicapa

**User Story:** Como desarrollador de redes neuronales, quiero organizar neuronas en capas, para que pueda construir arquitecturas feed-forward y recurrentes.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL organizar NeuronaSpiking en capas numeradas
2. THE RedNeuralSpiking SHALL permitir crear conexiones entre neuronas de diferentes capas
3. THE RedNeuralSpiking SHALL permitir crear conexiones recurrentes dentro de la misma capa
4. THE RedNeuralSpiking SHALL mantener una lista ordenada de capas
5. THE RedNeuralSpiking SHALL propagar spikes desde capas de entrada hacia capas de salida
6. WHERE se configuran conexiones recurrentes, THE RedNeuralSpiking SHALL procesar spikes recurrentes en el mismo timestep

### Requisito 7: Simulación Temporal Discreta

**User Story:** Como usuario de la red neuronal, quiero ejecutar simulaciones paso a paso, para que pueda procesar secuencias temporales con precisión de timing.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL avanzar la simulación en timesteps discretos
2. WHEN se ejecuta un timestep, THE RedNeuralSpiking SHALL actualizar todas las NeuronaSpiking en orden de capas
3. THE RedNeuralSpiking SHALL mantener un contador global de timesteps
4. THE RedNeuralSpiking SHALL permitir configurar la duración del timestep en milisegundos
5. WHEN se ejecuta un timestep, THE RedNeuralSpiking SHALL aplicar decaimiento de potencial antes de procesar nuevas señales
6. WHEN se ejecuta un timestep, THE RedNeuralSpiking SHALL propagar spikes generados a neuronas conectadas
7. THE RedNeuralSpiking SHALL procesar todas las señales sinápticas pendientes en cada timestep

### Requisito 8: Retardo Sináptico

**User Story:** Como desarrollador de redes neuronales, quiero configurar retardos en sinapsis, para que pueda modelar latencias de transmisión realistas.

#### Criterios de Aceptación

1. THE Sinapsis_Spiking SHALL permitir configurar un retardo en timesteps
2. WHEN una NeuronaSpiking pre-sináptica genera un Spike, THE Sinapsis_Spiking SHALL entregar la señal después del retardo configurado
3. THE Sinapsis_Spiking SHALL mantener una cola de spikes pendientes con sus timestamps de entrega
4. WHEN llega el timestamp de entrega, THE Sinapsis_Spiking SHALL transmitir el spike a la neurona post-sináptica
5. THE Sinapsis_Spiking SHALL permitir retardo cero para transmisión instantánea

### Requisito 9: Registro de Actividad Neuronal

**User Story:** Como investigador, quiero registrar la actividad de spikes, para que pueda analizar y visualizar patrones temporales.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL registrar todos los Spikes generados con timestamp y ID de neurona
2. THE RedNeuralSpiking SHALL permitir consultar el Tren_Spikes de cualquier NeuronaSpiking
3. THE RedNeuralSpiking SHALL calcular la frecuencia de disparo promedio de cada NeuronaSpiking
4. THE RedNeuralSpiking SHALL permitir exportar registros de spikes en formato CSV
5. THE RedNeuralSpiking SHALL incluir en el registro: timestamp, capa, índice de neurona, potencial de membrana
6. WHERE se solicita, THE RedNeuralSpiking SHALL generar visualización de raster plot de spikes

### Requisito 10: Configuración de Parámetros Neuronales

**User Story:** Como desarrollador de redes neuronales, quiero configurar parámetros del modelo LIF, para que pueda experimentar con diferentes comportamientos neuronales.

#### Criterios de Aceptación

1. THE NeuronaSpiking SHALL permitir configurar el Umbral_Disparo
2. THE NeuronaSpiking SHALL permitir configurar el Potencial_Reposo
3. THE NeuronaSpiking SHALL permitir configurar la Constante_Decaimiento (tau)
4. THE NeuronaSpiking SHALL permitir configurar la duración del Periodo_Refractario
5. THE NeuronaSpiking SHALL permitir configurar el Potencial_Membrana inicial
6. THE NeuronaSpiking SHALL validar que Umbral_Disparo > Potencial_Reposo
7. THE NeuronaSpiking SHALL validar que Constante_Decaimiento > 0

### Requisito 11: Inicialización de Pesos Sinápticos

**User Story:** Como desarrollador de redes neuronales, quiero inicializar pesos sinápticos, para que la red comience con una configuración apropiada.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL permitir inicialización aleatoria de pesos con distribución uniforme
2. THE RedNeuralSpiking SHALL permitir inicialización aleatoria de pesos con distribución normal
3. THE RedNeuralSpiking SHALL permitir inicialización de pesos con valores constantes
4. THE RedNeuralSpiking SHALL permitir especificar rangos [min, max] para inicialización aleatoria
5. THE RedNeuralSpiking SHALL permitir inicialización de pesos desde un array de valores
6. WHEN se inicializan pesos, THE RedNeuralSpiking SHALL validar que los valores estén dentro de límites permitidos

### Requisito 12: Persistencia de Modelos

**User Story:** Como usuario de la red neuronal, quiero guardar y cargar modelos entrenados, para que pueda reutilizar redes sin reentrenar.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL implementar Serializable para persistencia
2. THE RedNeuralSpiking SHALL guardar la arquitectura completa (capas, neuronas, conexiones)
3. THE RedNeuralSpiking SHALL guardar todos los pesos sinápticos
4. THE RedNeuralSpiking SHALL guardar todos los parámetros de configuración
5. WHEN se carga un modelo, THE RedNeuralSpiking SHALL restaurar el estado completo de la red
6. WHEN se carga un modelo, THE RedNeuralSpiking SHALL validar la integridad de los datos
7. THE RedNeuralSpiking SHALL resetear el estado temporal (potenciales, spikes pendientes) al cargar

### Requisito 13: Métricas de Rendimiento

**User Story:** Como investigador, quiero calcular métricas de rendimiento, para que pueda evaluar la eficiencia y precisión de la red.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL calcular el número total de spikes generados
2. THE RedNeuralSpiking SHALL calcular la tasa de spikes promedio por neurona
3. THE RedNeuralSpiking SHALL calcular la precisión temporal promedio de spikes (jitter)
4. THE RedNeuralSpiking SHALL calcular el costo energético estimado (proporcional a número de spikes)
5. THE RedNeuralSpiking SHALL calcular la dispersión de actividad entre neuronas
6. THE RedNeuralSpiking SHALL permitir resetear contadores de métricas
7. THE RedNeuralSpiking SHALL exportar métricas en formato estructurado (JSON o Map)

### Requisito 14: Procesamiento por Lotes

**User Story:** Como usuario de la red neuronal, quiero procesar múltiples patrones de entrada, para que pueda entrenar o evaluar eficientemente.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL procesar un lote de patrones de entrada secuencialmente
2. WHEN se procesa un lote, THE RedNeuralSpiking SHALL resetear el estado temporal entre patrones
3. THE RedNeuralSpiking SHALL acumular métricas de rendimiento a través del lote
4. THE RedNeuralSpiking SHALL retornar las salidas decodificadas para cada patrón del lote
5. WHERE se solicita, THE RedNeuralSpiking SHALL mantener el estado temporal entre patrones del lote

### Requisito 15: Validación de Arquitectura

**User Story:** Como desarrollador de redes neuronales, quiero validar la arquitectura de la red, para que pueda detectar configuraciones inválidas antes de entrenar.

#### Criterios de Aceptación

1. WHEN se construye la red, THE RedNeuralSpiking SHALL validar que exista al menos una capa
2. WHEN se construye la red, THE RedNeuralSpiking SHALL validar que cada capa tenga al menos una neurona
3. WHEN se crea una conexión, THE RedNeuralSpiking SHALL validar que las neuronas existan
4. WHEN se crea una conexión, THE RedNeuralSpiking SHALL validar que no se creen conexiones duplicadas
5. IF una validación falla, THEN THE RedNeuralSpiking SHALL lanzar IllegalArgumentException con mensaje descriptivo
6. THE RedNeuralSpiking SHALL validar que los índices de capa sean consecutivos comenzando en 0

### Requisito 16: Codificación Temporal de Entrada

**User Story:** Como usuario de la red neuronal, quiero codificar patrones de entrada como secuencias temporales, para que pueda procesar información con estructura temporal.

#### Criterios de Aceptación

1. THE Codificador SHALL convertir un array de valores en una secuencia de trenes de spikes
2. THE Codificador SHALL permitir especificar la duración de la presentación en timesteps
3. WHEN se codifica un patrón, THE Codificador SHALL generar spikes distribuidos a lo largo de la duración
4. THE Codificador SHALL mantener la proporcionalidad entre valores de entrada y frecuencias de spike
5. THE Codificador SHALL permitir configurar el modo de codificación (Poisson, regular, burst)
6. WHERE se usa codificación regular, THE Codificador SHALL espaciar spikes uniformemente según la frecuencia

### Requisito 17: Normalización de Pesos

**User Story:** Como desarrollador de redes neuronales, quiero normalizar pesos sinápticos, para que pueda mantener estabilidad durante el aprendizaje.

#### Criterios de Aceptación

1. THE RedNeuralSpiking SHALL permitir normalizar pesos de conexiones entrantes a cada neurona
2. WHEN se normaliza, THE RedNeuralSpiking SHALL escalar pesos para que su suma sea un valor objetivo
3. THE RedNeuralSpiking SHALL permitir normalización L1 (suma de valores absolutos)
4. THE RedNeuralSpiking SHALL permitir normalización L2 (suma de cuadrados)
5. THE RedNeuralSpiking SHALL aplicar normalización preservando el signo de los pesos
6. WHERE se especifica, THE RedNeuralSpiking SHALL aplicar normalización solo a capas específicas

### Requisito 18: Homeostasis Sináptica

**User Story:** Como desarrollador de redes neuronales, quiero implementar homeostasis, para que las neuronas mantengan tasas de disparo objetivo.

#### Criterios de Aceptación

1. THE NeuronaSpiking SHALL mantener un registro de su tasa de disparo promedio reciente
2. WHEN la tasa de disparo se desvía del objetivo, THE NeuronaSpiking SHALL ajustar su Umbral_Disparo
3. WHEN la tasa de disparo es muy baja, THE NeuronaSpiking SHALL decrementar el Umbral_Disparo
4. WHEN la tasa de disparo es muy alta, THE NeuronaSpiking SHALL incrementar el Umbral_Disparo
5. THE NeuronaSpiking SHALL aplicar ajustes homeostáticos con una tasa de aprendizaje configurable
6. THE NeuronaSpiking SHALL permitir configurar la tasa de disparo objetivo
7. WHERE se configura, THE NeuronaSpiking SHALL calcular la tasa promedio usando una ventana temporal deslizante

### Requisito 19: Inhibición Lateral

**User Story:** Como desarrollador de redes neuronales, quiero implementar inhibición lateral, para que pueda crear competencia entre neuronas y mejorar la selectividad.

#### Criterios de Aceptación

1. WHERE se configura inhibición lateral, WHEN una NeuronaSpiking dispara, THE NeuronaSpiking SHALL enviar señales inhibitorias a neuronas vecinas
2. WHEN una NeuronaSpiking recibe una señal inhibitoria, THE NeuronaSpiking SHALL decrementar su Potencial_Membrana
3. THE RedNeuralSpiking SHALL permitir configurar el radio de inhibición lateral
4. THE RedNeuralSpiking SHALL permitir configurar la fuerza de inhibición lateral
5. THE RedNeuralSpiking SHALL aplicar inhibición lateral solo dentro de la misma capa
6. WHERE se usa topología espacial, THE RedNeuralSpiking SHALL calcular vecindad basada en distancia euclidiana

### Requisito 20: Parseo y Serialización de Configuración

**User Story:** Como usuario de la red neuronal, quiero definir arquitecturas mediante archivos de configuración, para que pueda crear redes sin código programático.

#### Criterios de Aceptación

1. WHEN se proporciona un archivo de configuración válido, THE RedNeuralSpiking SHALL parsear la arquitectura de red
2. WHEN se proporciona un archivo de configuración inválido, THE RedNeuralSpiking SHALL retornar un error descriptivo
3. THE RedNeuralSpiking SHALL serializar la configuración actual a formato JSON
4. FOR ALL configuraciones válidas, parsear y luego serializar y luego parsear SHALL producir una configuración equivalente (propiedad round-trip)
5. THE RedNeuralSpiking SHALL validar todos los parámetros durante el parseo
6. THE RedNeuralSpiking SHALL soportar formato JSON para archivos de configuración
7. THE RedNeuralSpiking SHALL incluir en la configuración: arquitectura de capas, parámetros neuronales, parámetros de STDP, esquema de conexiones

