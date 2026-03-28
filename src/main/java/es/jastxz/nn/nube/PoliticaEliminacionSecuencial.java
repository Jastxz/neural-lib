package es.jastxz.nn.nube;

import java.util.Arrays;

/**
 * Política de eliminación secuencial que reduce neuronas comenzando por la última
 * capa oculta y avanzando hacia la primera cuando una capa se agota.
 *
 * <p>En cada invocación, localiza la última capa oculta (de derecha a izquierda)
 * que aún tenga neuronas y le resta {@code neuronasEliminar}, con un mínimo de 0.
 * Retorna {@code null} cuando todas las capas ocultas han quedado en 0 neuronas.</p>
 *
 * @see PoliticaEliminacion
 */
public class PoliticaEliminacionSecuencial implements PoliticaEliminacion {

    @Override
    public int[] siguienteReduccion(int[] topologiaActual, int neuronasEliminar) {
        // Si todas las capas ocultas ya están en 0, no hay más reducciones posibles
        boolean todasVacias = true;
        for (int i = 1; i < topologiaActual.length - 1; i++) {
            if (topologiaActual[i] > 0) {
                todasVacias = false;
                break;
            }
        }
        if (todasVacias) {
            return null;
        }

        int[] nueva = Arrays.copyOf(topologiaActual, topologiaActual.length);

        // Buscar la última capa oculta con neuronas > 0 (de derecha a izquierda)
        for (int i = nueva.length - 2; i >= 1; i--) {
            if (nueva[i] > 0) {
                nueva[i] = Math.max(0, nueva[i] - neuronasEliminar);
                break;
            }
        }

        // Si tras la reducción todas las capas ocultas quedaron en 0, retornar null
        for (int i = 1; i < nueva.length - 1; i++) {
            if (nueva[i] > 0) {
                return nueva;
            }
        }
        return null;
    }
}
