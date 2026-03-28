package es.jastxz.nn.nube;

/**
 * Interfaz funcional que define la estrategia de eliminación de neuronas
 * durante el proceso de reducción del Método de la Nube Aleatoria.
 *
 * <p>Una política de eliminación determina cómo se reducen las capas ocultas
 * de una red neuronal en cada iteración del proceso de reducción. La capa de
 * entrada (primera posición) y la capa de salida (última posición) deben
 * permanecer inalteradas en la topología retornada.</p>
 *
 * @see PoliticaEliminacionSecuencial
 */
@FunctionalInterface
public interface PoliticaEliminacion {

    /**
     * Determina la siguiente topología reducida eliminando neuronas de las capas ocultas.
     *
     * <p>Recibe la topología actual de la red (incluyendo las capas de entrada y salida)
     * y el número de neuronas a eliminar en esta iteración. Retorna una nueva topología
     * con las neuronas eliminadas según la estrategia implementada, o {@code null} si no
     * es posible realizar más reducciones (por ejemplo, cuando todas las capas ocultas
     * tienen 0 neuronas).</p>
     *
     * <p>El tamaño de la capa de entrada (primera posición del array) y el tamaño de la
     * capa de salida (última posición del array) deben permanecer sin cambios en la
     * topología retornada.</p>
     *
     * @param topologiaActual topología actual de la red, donde cada posición representa
     *                        el número de neuronas de cada capa (incluye entrada y salida)
     * @param neuronasEliminar número de neuronas a eliminar en esta iteración (≥ 1)
     * @return nueva topología con las neuronas eliminadas, o {@code null} si no hay más
     *         reducciones posibles
     */
    int[] siguienteReduccion(int[] topologiaActual, int neuronasEliminar);
}
