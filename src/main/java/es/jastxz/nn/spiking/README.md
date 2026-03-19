# Red Neuronal de Spikes (SNN)

Implementación en Java de una Red Neuronal de Spikes con modelo Leaky Integrate-and-Fire (LIF).

## Características

- **Modelo LIF**: Decaimiento exponencial, umbral de disparo, período refractario
- **STDP**: Plasticidad dependiente del timing de spikes (LTP/LTD)
- **Codificación Rate Coding**: Poisson, Regular, Burst
- **Decodificación**: Ventana deslizante con normalización a [0,1]
- **Homeostasis**: Ajuste adaptativo del umbral de disparo
- **Inhibición Lateral**: Señales inhibitorias a neuronas vecinas
- **Normalización de Pesos**: L1 y L2
- **Persistencia**: Serialización binaria y JSON

## Uso Básico

```java
// Crear configuración
ConfiguracionRed config = new ConfiguracionRedBuilder()
    .topologia(10, 20, 5)
    .parametrosLIF(-55.0, -70.0, 20.0, 2)
    .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
    .build();

// Crear red
RedNeuralSpiking red = new RedNeuralSpiking(config);

// Crear conexiones entre capas
for (int i = 0; i < 10; i++)
    for (int j = 0; j < 20; j++)
        red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);

// Procesar patrón
double[] salida = red.procesar(new double[]{0.8, 0.3, ...}, 100);

// Entrenar con STDP
double error = red.entrenar(inputs, targets, 100);

// Guardar/cargar
red.guardar("modelo.snn");
RedNeuralSpiking cargada = RedNeuralSpiking.cargar("modelo.snn");
```

## Clases Principales

| Clase | Descripción |
|-------|-------------|
| `RedNeuralSpiking` | Red neuronal principal |
| `NeuronaSpiking` | Neurona LIF |
| `SinapsisSpiking` | Conexión sináptica |
| `ConfiguracionRedBuilder` | Builder para configuración |
| `GestorCodificacion` | Codificación rate coding |
| `GestorDecodificacion` | Decodificación de spikes |
| `GestorSTDP` | Plasticidad STDP |
| `GestorMetricas` | Métricas de rendimiento |

## Ejemplos

Ver `SpikingNetworkExample.java` para ejemplos completos de:
1. Clasificación XOR
2. Aprendizaje STDP
3. Persistencia binaria
4. Configuración JSON

## Referencias

- Gerstner, W. & Kistler, W. (2002). Spiking Neuron Models
- Bi, G. & Poo, M. (1998). Synaptic Modifications in Cultured Hippocampal Neurons
