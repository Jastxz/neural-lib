package es.jastxz.nn.spiking;

/**
 * Modos de codificación para convertir valores continuos en trenes de spikes.
 * 
 * <p>Define los diferentes esquemas de codificación temporal disponibles para
 * transformar valores de entrada en patrones de spikes.</p>
 * 
 * @since 1.0
 */
public enum ModoCodificacion {
    /**
     * Generación probabilística de spikes siguiendo un proceso de Poisson.
     * Los spikes se generan con probabilidad proporcional al valor de entrada,
     * resultando en un patrón biológicamente realista con variabilidad temporal.
     */
    POISSON,
    
    /**
     * Generación determinística con espaciado uniforme entre spikes.
     * Los spikes se distribuyen uniformemente en el tiempo según la frecuencia
     * calculada, produciendo intervalos regulares y predecibles.
     */
    REGULAR,
    
    /**
     * Generación de ráfagas (bursts) de spikes consecutivos.
     * Los spikes se agrupan en ráfagas concentradas temporalmente,
     * simulando patrones de disparo en ráfaga observados en neuronas biológicas.
     */
    BURST
}
