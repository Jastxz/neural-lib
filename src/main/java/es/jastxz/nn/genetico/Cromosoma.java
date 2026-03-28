package es.jastxz.nn.genetico;

import java.util.*;

/**
 * Colección ordenada de genes agrupados por bloque funcional.
 *
 * <p>El cromosoma es inmutable: el constructor realiza copias defensivas
 * del mapa y de cada lista de genes, y todos los accesores devuelven
 * vistas o copias no modificables.</p>
 *
 * @param bloques mapa de bloque funcional a lista de genes
 */
public record Cromosoma(Map<BloqueFuncional, List<Gen<?>>> bloques) {

    /** Orden canónico de los bloques funcionales. */
    private static final List<BloqueFuncional> ORDEN_BLOQUES = List.of(
            BloqueFuncional.TOPOLOGIA,
            BloqueFuncional.LIF,
            BloqueFuncional.STDP,
            BloqueFuncional.CODIFICACION,
            BloqueFuncional.REGULACION,
            BloqueFuncional.COMPETICION
    );

    /**
     * Constructor compacto que realiza copias defensivas profundas.
     */
    public Cromosoma {
        var copia = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        for (var entry : bloques.entrySet()) {
            copia.put(entry.getKey(), List.copyOf(entry.getValue()));
        }
        bloques = Collections.unmodifiableMap(copia);
    }

    /**
     * Retorna todos los genes en orden de bloque:
     * TOPOLOGIA → LIF → STDP → CODIFICACION → REGULACION.
     *
     * @return lista inmutable con todos los genes ordenados
     */
    public List<Gen<?>> genesOrdenados() {
        return ORDEN_BLOQUES.stream()
                .flatMap(b -> bloques.getOrDefault(b, List.of()).stream())
                .toList();
    }

    /**
     * Retorna los genes de un bloque específico.
     *
     * @param bloque bloque funcional a consultar
     * @return lista inmutable de genes del bloque (vacía si el bloque no existe)
     */
    public List<Gen<?>> genesDeBloque(BloqueFuncional bloque) {
        return bloques.getOrDefault(bloque, List.of());
    }

    /**
     * Crea un nuevo Cromosoma reemplazando un bloque completo.
     *
     * @param bloque bloque funcional a reemplazar
     * @param genes  nueva lista de genes para el bloque
     * @return nuevo cromosoma con el bloque reemplazado
     */
    public Cromosoma conBloque(BloqueFuncional bloque, List<Gen<?>> genes) {
        var nuevo = new EnumMap<BloqueFuncional, List<Gen<?>>>(bloques);
        nuevo.put(bloque, genes);
        return new Cromosoma(nuevo);
    }

    /**
     * Calcula el número total de neuronas (entrada + ocultas + salida).
     *
     * <p>Las neuronas ocultas se obtienen del bloque TOPOLOGIA: el primer gen
     * es "capasOcultas" y los siguientes son "neuronasCapa_0", "neuronasCapa_1", etc.
     * Se suman los valores de los genes de neuronas por capa.</p>
     *
     * @param entrada número de neuronas de entrada
     * @param salida  número de neuronas de salida
     * @return total de neuronas de la red
     */
    public int neuronasTotal(int entrada, int salida) {
        List<Gen<?>> topologia = genesDeBloque(BloqueFuncional.TOPOLOGIA);
        int ocultas = 0;
        // El primer gen es capasOcultas; los restantes son neuronasPorCapa
        for (int i = 1; i < topologia.size(); i++) {
            ocultas += ((Number) topologia.get(i).valor()).intValue();
        }
        return entrada + ocultas + salida;
    }
}
