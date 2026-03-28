package es.jastxz.nn.nube;

import es.jastxz.nn.NeuralNetwork;
import java.util.Arrays;

/**
 * Informe inmutable con los resultados del Método de la Nube Aleatoria.
 */
public record InformeNube(
    NeuralNetwork mejorRed,        // null si ninguna superó el umbral
    double precision,               // precisión de la mejor red (0.0 si no hay)
    int[] topologiaFinal,          // topología de la mejor red (null si no hay)
    int totalRedesEvaluadas,       // número total de redes procesadas
    int totalReducciones,          // número total de reducciones realizadas
    long tiempoEjecucionMs,        // tiempo total en milisegundos
    boolean exitoso                // true si se encontró red viable
) {
    // Compact constructor: defensive copy of topologiaFinal
    public InformeNube {
        topologiaFinal = topologiaFinal != null ? Arrays.copyOf(topologiaFinal, topologiaFinal.length) : null;
    }

    // Override accessor for defensive copy
    @Override
    public int[] topologiaFinal() {
        return topologiaFinal != null ? Arrays.copyOf(topologiaFinal, topologiaFinal.length) : null;
    }
}
