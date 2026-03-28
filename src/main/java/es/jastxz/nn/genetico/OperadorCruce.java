package es.jastxz.nn.genetico;

import java.util.*;

/**
 * Operador de cruce multi-punto por bloques funcionales.
 *
 * <p>Combina genes de dos padres mediante puntos de corte aleatorios entre
 * bloques funcionales, con sesgo hacia el padre de mayor fitness. Los bloques
 * funcionales se mantienen íntegros durante el cruce para preservar coherencia.</p>
 *
 * <p>Algoritmo:</p>
 * <ol>
 *   <li>Si random &gt; probabilidadCruce → retornar copia del mejor padre</li>
 *   <li>Generar X puntos de corte entre posiciones [1, 4] (entre bloques)</li>
 *   <li>Dividir los 5 bloques en (X+1) segmentos</li>
 *   <li>Seleccionar Y segmentos del mejor padre donde Y &gt; (X+1)/2</li>
 *   <li>Ensamblar cromosoma descendiente con bloques seleccionados</li>
 *   <li>Reparar topología si excede límite</li>
 *   <li>Construir ConfiguracionRed y retornar nuevo Individuo sin evaluar</li>
 * </ol>
 */
public class OperadorCruce {

    /** Orden canónico de los bloques funcionales para el cruce. */
    private static final BloqueFuncional[] BLOQUES = BloqueFuncional.values();

    private final double probabilidadCruce;
    private final int puntosCorte;
    private final FabricaIndividuos fabrica;
    private final Random random;
    private final Set<BloqueFuncional> bloquesCongelados;

    /**
     * @param probabilidadCruce probabilidad de aplicar cruce (default 0.8)
     * @param puntosCorte       número de puntos de corte (default 2, rango [1, 4])
     * @param limiteTopologico  máximo de neuronas totales permitidas (delegado a fabrica)
     * @param fabrica           fábrica para construir configuraciones y reparar topología
     * @param random            generador de números aleatorios
     */
    public OperadorCruce(double probabilidadCruce, int puntosCorte,
                         int limiteTopologico, FabricaIndividuos fabrica,
                         Random random) {
        this(probabilidadCruce, puntosCorte, limiteTopologico, fabrica, random, Set.of());
    }

    /**
     * Constructor con bloques congelados que siempre se heredan del mejor padre.
     *
     * @param probabilidadCruce  probabilidad de aplicar cruce
     * @param puntosCorte        número de puntos de corte
     * @param limiteTopologico   máximo de neuronas totales permitidas
     * @param fabrica            fábrica para construir configuraciones y reparar topología
     * @param random             generador de números aleatorios
     * @param bloquesCongelados  bloques que siempre se heredan del mejor padre
     */
    public OperadorCruce(double probabilidadCruce, int puntosCorte,
                         int limiteTopologico, FabricaIndividuos fabrica,
                         Random random, Set<BloqueFuncional> bloquesCongelados) {
        this.probabilidadCruce = probabilidadCruce;
        this.puntosCorte = puntosCorte;
        this.fabrica = fabrica;
        this.random = random;
        this.bloquesCongelados = Set.copyOf(bloquesCongelados);
    }

    /**
     * Cruza dos padres produciendo un descendiente.
     *
     * <p>Si el cruce no se aplica (por probabilidad), retorna una copia sin evaluar
     * del mejor padre. Si se aplica, genera puntos de corte entre bloques funcionales,
     * selecciona más segmentos del mejor padre, y ensambla el cromosoma descendiente.</p>
     *
     * @param padre1 primer padre
     * @param padre2 segundo padre
     * @return nuevo individuo descendiente sin evaluar
     */
    /**
     * Cruza dos padres produciendo un descendiente.
     *
     * <p>Si el cruce no se aplica (por probabilidad), retorna una copia sin evaluar
     * del mejor padre. Si se aplica, genera puntos de corte entre bloques funcionales,
     * selecciona más segmentos del mejor padre (ensuring more blocks from best parent),
     * y ensambla el cromosoma descendiente.</p>
     *
     * @param padre1 primer padre
     * @param padre2 segundo padre
     * @return nuevo individuo descendiente sin evaluar
     */
    public Individuo cruzar(Individuo padre1, Individuo padre2) {
        // Determinar mejor y peor padre por fitness
        Individuo mejorPadre = padre1.fitness() >= padre2.fitness() ? padre1 : padre2;
        Individuo peorPadre = padre1.fitness() >= padre2.fitness() ? padre2 : padre1;

        // 1. Si random > probabilidadCruce → retornar copia del mejor padre
        if (random.nextDouble() > probabilidadCruce) {
            return Individuo.sinEvaluar(
                    mejorPadre.cromosoma(),
                    fabrica.construirConfiguracion(mejorPadre.cromosoma()));
        }

        // 2. Generar X puntos de corte aleatorios entre posiciones [1, 4]
        int numBloques = BLOQUES.length; // 5
        int maxPosiciones = numBloques - 1; // 4
        int x = Math.min(puntosCorte, maxPosiciones);

        List<Integer> posicionesPosibles = new ArrayList<>();
        for (int i = 1; i <= maxPosiciones; i++) {
            posicionesPosibles.add(i);
        }
        Collections.shuffle(posicionesPosibles, random);
        List<Integer> puntosDeCorte = new ArrayList<>(posicionesPosibles.subList(0, x));
        Collections.sort(puntosDeCorte);

        // 3. Los puntos de corte dividen los 5 bloques en (X+1) segmentos
        List<int[]> segmentos = new ArrayList<>();
        int inicio = 0;
        for (int corte : puntosDeCorte) {
            segmentos.add(new int[]{inicio, corte});
            inicio = corte;
        }
        segmentos.add(new int[]{inicio, numBloques});

        int totalSegmentos = segmentos.size();

        // 4. Seleccionar Y segmentos del mejor padre where Y > totalSegmentos / 2
        //    Y = ceil((totalSegmentos + 1) / 2)
        int y = (int) Math.ceil((totalSegmentos + 1) / 2.0);

        // Randomly select Y segments for best parent
        List<Integer> indicesSegmentos = new ArrayList<>();
        for (int i = 0; i < totalSegmentos; i++) {
            indicesSegmentos.add(i);
        }
        Collections.shuffle(indicesSegmentos, random);
        Set<Integer> segmentosMejorPadre = new HashSet<>(indicesSegmentos.subList(0, y));

        // Count blocks from best parent; if not > numBloques/2, swap segments
        int bloquesMejor = 0;
        for (int s : segmentosMejorPadre) {
            bloquesMejor += segmentos.get(s)[1] - segmentos.get(s)[0];
        }

        // If bias not satisfied, swap the largest segment from worst parent to best parent
        while (bloquesMejor <= numBloques / 2) {
            int mejorCandidato = -1;
            int maxTamaño = 0;
            for (int s = 0; s < totalSegmentos; s++) {
                if (!segmentosMejorPadre.contains(s)) {
                    int tamaño = segmentos.get(s)[1] - segmentos.get(s)[0];
                    if (tamaño > maxTamaño) {
                        maxTamaño = tamaño;
                        mejorCandidato = s;
                    }
                }
            }
            if (mejorCandidato < 0) break;
            segmentosMejorPadre.add(mejorCandidato);
            bloquesMejor += maxTamaño;
        }

        // 5. Ensamblar cromosoma descendiente
        var bloquesMapa = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);

        for (int s = 0; s < totalSegmentos; s++) {
            int[] segmento = segmentos.get(s);
            Individuo padreOrigen = segmentosMejorPadre.contains(s) ? mejorPadre : peorPadre;

            for (int b = segmento[0]; b < segmento[1]; b++) {
                BloqueFuncional bloque = BLOQUES[b];
                bloquesMapa.put(bloque, padreOrigen.cromosoma().genesDeBloque(bloque));
            }
        }

        Cromosoma cromosomaHijo = new Cromosoma(bloquesMapa);

        // 6. Forzar bloques congelados del mejor padre
        for (BloqueFuncional bloque : bloquesCongelados) {
            cromosomaHijo = cromosomaHijo.conBloque(bloque,
                    mejorPadre.cromosoma().genesDeBloque(bloque));
        }

        // 7. Reparar topología si excede límite
        cromosomaHijo = fabrica.repararTopologia(cromosomaHijo);

        // 8. Construir ConfiguracionRed y retornar nuevo Individuo sin evaluar
        var config = fabrica.construirConfiguracion(cromosomaHijo);
        return Individuo.sinEvaluar(cromosomaHijo, config);
    }
}
