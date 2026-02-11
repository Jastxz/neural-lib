package es.jastxz.nn.enums;

/**
 * Estados posibles de la red neuronal experimental
 */
public enum EstadoRed {
    ACTIVO,        // Procesando información
    CONSOLIDANDO,  // Fase de "sueño" - consolidación de engramas
    DEGRADANDO     // Fase de mantenimiento - poda y limpieza
}
