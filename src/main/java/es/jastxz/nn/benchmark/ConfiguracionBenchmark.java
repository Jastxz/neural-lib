package es.jastxz.nn.benchmark;

import java.util.Arrays;

/**
 * Configuración inmutable para una ejecución de benchmark de la SNN.
 *
 * <p>Combina un nivel de complejidad de problema con una topología de red
 * y parámetros de entrenamiento. El constructor compacto valida que la
 * topología sea compatible con las dimensiones del problema.</p>
 *
 * @param nivel             nivel de complejidad del problema
 * @param topologia         array con tamaños de cada capa [entrada, ocultas..., salida]
 * @param epocas            número de épocas de entrenamiento
 * @param duracionTimesteps timesteps por patrón de entrenamiento
 * @param repeticiones      número de repeticiones para estadísticas
 * @param semilla           semilla para reproducibilidad
 * @since 1.1
 */
public record ConfiguracionBenchmark(
    NivelComplejidad nivel,
    int[] topologia,
    int epocas,
    int duracionTimesteps,
    int repeticiones,
    long semilla
) {

    /**
     * Constructor compacto que valida la compatibilidad entre la topología
     * y las dimensiones de entrada/salida del problema.
     *
     * @throws IllegalArgumentException si la primera capa no coincide con
     *         la dimensión de entrada del problema, o si la última capa no
     *         coincide con la dimensión de salida
     */
    public ConfiguracionBenchmark {
        if (topologia[0] != nivel.getDimensionEntrada()) {
            throw new IllegalArgumentException(
                "Dimensión de entrada incompatible: topologia[0]=" + topologia[0]
                + " pero el problema '" + nivel.getNombreProblema()
                + "' requiere dimensionEntrada=" + nivel.getDimensionEntrada());
        }
        if (topologia[topologia.length - 1] != nivel.getDimensionSalida()) {
            throw new IllegalArgumentException(
                "Dimensión de salida incompatible: topologia[último]="
                + topologia[topologia.length - 1]
                + " pero el problema '" + nivel.getNombreProblema()
                + "' requiere dimensionSalida=" + nivel.getDimensionSalida());
        }
        topologia = topologia.clone();
    }

    /**
     * Genera una etiqueta legible para esta configuración.
     *
     * @return representación legible con el nombre del problema y la topología
     */
    public String etiqueta() {
        return nivel.getNombreProblema() + " | " + Arrays.toString(topologia);
    }
}
