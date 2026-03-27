package es.jastxz.nn.benchmark;

import java.util.ArrayList;
import java.util.List;

/**
 * Generador paramétrico de topologías de red para benchmarks.
 *
 * <p>Genera todas las combinaciones de capas ocultas (1, 2, 3, 4) y
 * factores de neuronas (1x, 2x, 4x, 8x) respecto al tamaño de entrada,
 * produciendo exactamente 16 topologías por cada par entrada/salida.</p>
 *
 * @since 1.1
 */
public class GeneradorTopologias {

    private static final int[] CAPAS_OCULTAS = {1, 2, 3, 4};
    private static final int[] FACTORES_NEURONAS = {1, 2};

    private GeneradorTopologias() {
        // Clase utilitaria, no instanciable
    }

    /**
     * Genera todas las combinaciones de topologías para las dimensiones dadas.
     *
     * <p>Para cada combinación de número de capas ocultas y factor de neuronas,
     * crea un array donde el primer elemento es {@code entrada}, el último es
     * {@code salida}, y las capas ocultas tienen {@code entrada * factor} neuronas.</p>
     *
     * @param entrada número de neuronas en la capa de entrada
     * @param salida  número de neuronas en la capa de salida
     * @return lista con exactamente 8 topologías (4 capas × 2 factores)
     */
    public static List<int[]> generar(int entrada, int salida) {
        List<int[]> topologias = new ArrayList<>();

        for (int numCapas : CAPAS_OCULTAS) {
            for (int factor : FACTORES_NEURONAS) {
                int[] topologia = new int[numCapas + 2];
                topologia[0] = entrada;
                topologia[topologia.length - 1] = salida;
                int neuronasOcultas = entrada * factor;
                for (int i = 1; i <= numCapas; i++) {
                    topologia[i] = neuronasOcultas;
                }
                topologias.add(topologia);
            }
        }

        return topologias;
    }
}
