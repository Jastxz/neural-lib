package es.jastxz.nn.spiking;

/**
 * Tipos de normalización para pesos sinápticos.
 * 
 * <p>Define los métodos de normalización disponibles para escalar los pesos
 * de las conexiones sinápticas, manteniendo la estabilidad durante el aprendizaje.</p>
 * 
 * @since 1.0
 */
public enum TipoNormalizacion {
    /**
     * Normalización L1 (norma Manhattan).
     * Escala los pesos para que la suma de sus valores absolutos sea igual
     * a un valor objetivo. Fórmula: Σ|w_i| = valor_objetivo
     */
    L1,
    
    /**
     * Normalización L2 (norma Euclidiana).
     * Escala los pesos para que la suma de sus cuadrados sea igual
     * a un valor objetivo. Fórmula: Σ(w_i²) = valor_objetivo
     */
    L2
}
