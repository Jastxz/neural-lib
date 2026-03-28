/**
 * Paquete para la implementación de Redes Neuronales de Spikes (Spiking Neural Networks - SNN).
 * 
 * <p>Este paquete contiene una implementación independiente de SNN basada en el modelo
 * Leaky Integrate-and-Fire (LIF) con plasticidad STDP (Spike-Timing-Dependent Plasticity),
 * codificación por tasa de disparo (rate coding), y período refractario.</p>
 * 
 * <p>Componentes principales:</p>
 * <ul>
 *   <li>{@code RedNeuralSpiking} - Clase principal que orquesta la simulación temporal</li>
 *   <li>{@code NeuronaSpiking} - Unidad computacional que implementa el modelo LIF</li>
 *   <li>{@code SinapsisSpiking} - Conexión entre neuronas con peso y retardo</li>
 *   <li>{@code GestorCodificacion} - Convierte valores continuos a trenes de spikes</li>
 *   <li>{@code GestorDecodificacion} - Convierte trenes de spikes a valores continuos</li>
 *   <li>{@code GestorSTDP} - Aplica plasticidad dependiente del timing</li>
 * </ul>
 * 
 * @since 1.0
 */
package es.jastxz.nn.spiking;
