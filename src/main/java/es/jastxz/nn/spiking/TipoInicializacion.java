package es.jastxz.nn.spiking;

/**
 * Tipos de inicialización para pesos sinápticos.
 * 
 * <p>Define los diferentes métodos disponibles para inicializar los pesos
 * de las conexiones sinápticas en la red neuronal de spikes.</p>
 * 
 * @since 1.0
 */
public enum TipoInicializacion {
    /**
     * Inicialización con distribución uniforme en un rango [min, max].
     * Todos los valores tienen la misma probabilidad de ser seleccionados
     * dentro del rango especificado.
     */
    UNIFORME,
    
    /**
     * Inicialización con distribución normal (gaussiana).
     * Los valores se generan siguiendo una distribución normal con media
     * y desviación estándar configurables.
     */
    NORMAL,
    
    /**
     * Inicialización con un valor constante.
     * Todos los pesos se establecen al mismo valor especificado.
     */
    CONSTANTE,
    
    /**
     * Inicialización desde un array de valores proporcionado.
     * Los pesos se cargan directamente desde un array predefinido,
     * permitiendo configuraciones personalizadas o restauración de estados previos.
     */
    DESDE_ARRAY
}
