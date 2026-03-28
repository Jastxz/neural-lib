package es.jastxz.nn.nube;

import java.util.Arrays;

/**
 * Configuración inmutable del Método de la Nube Aleatoria.
 *
 * <p>Agrupa todos los hiperparámetros del método: tamaño de la nube, topología inicial,
 * umbral de acierto, número de neuronas a eliminar por iteración, épocas de refinamiento,
 * tasa de aprendizaje y semilla para reproducibilidad.</p>
 *
 * <p>Validaciones del constructor compacto:</p>
 * <ul>
 *   <li>El tamaño de la nube debe ser al menos 1</li>
 *   <li>La topología inicial debe tener al menos 3 capas (entrada, oculta, salida)</li>
 *   <li>El umbral de acierto debe estar en [0.0, 1.0]</li>
 *   <li>El número de neuronas a eliminar debe ser al menos 1</li>
 *   <li>Las capas ocultas deben tener al menos 1 neurona</li>
 * </ul>
 *
 * @param tamañoNube         número de redes en la nube (≥ 1)
 * @param topologiaInicial   topología de cada red (≥ 3 capas)
 * @param umbralAcierto      umbral mínimo de acierto [0.0, 1.0]
 * @param neuronasEliminar   neuronas a eliminar por iteración (≥ 1)
 * @param epocasRefinamiento épocas de backpropagation para refinamiento
 * @param tasaAprendizaje    learning rate para refinamiento
 * @param semilla            semilla para reproducibilidad
 */
public record ConfiguracionNube(
        int tamañoNube,
        int[] topologiaInicial,
        double umbralAcierto,
        int neuronasEliminar,
        int epocasRefinamiento,
        double tasaAprendizaje,
        long semilla
) {

    /**
     * Constructor compacto con validaciones y copia defensiva.
     */
    public ConfiguracionNube {
        if (tamañoNube < 1) {
            throw new IllegalArgumentException(
                    "El tamaño de la nube debe ser al menos 1 (valor: " + tamañoNube + ")");
        }
        if (topologiaInicial.length < 3) {
            throw new IllegalArgumentException(
                    "La topología debe tener al menos 3 capas (entrada, oculta, salida)");
        }
        if (umbralAcierto < 0.0 || umbralAcierto > 1.0) {
            throw new IllegalArgumentException(
                    "El umbral de acierto debe estar en [0.0, 1.0] (valor: " + umbralAcierto + ")");
        }
        if (neuronasEliminar < 1) {
            throw new IllegalArgumentException(
                    "El número de neuronas a eliminar debe ser al menos 1 (valor: " + neuronasEliminar + ")");
        }
        for (int i = 1; i < topologiaInicial.length - 1; i++) {
            if (topologiaInicial[i] < 1) {
                throw new IllegalArgumentException(
                        "Las capas ocultas deben tener al menos 1 neurona");
            }
        }
        topologiaInicial = Arrays.copyOf(topologiaInicial, topologiaInicial.length);
    }

    /**
     * Retorna una copia defensiva de la topología inicial.
     *
     * @return copia del array de topología
     */
    @Override
    public int[] topologiaInicial() {
        return Arrays.copyOf(topologiaInicial, topologiaInicial.length);
    }
}
