package es.jastxz.nn.genetico;

import java.util.*;

/**
 * Operador de mutación adaptativa por tipo de gen.
 *
 * <p>Aplica perturbaciones a los genes de un individuo según su tipo:</p>
 * <ul>
 *   <li>GenEntero: perturbación ±[1, ceil(0.20 * rango)]</li>
 *   <li>GenReal: perturbación gaussiana con σ = 10% del rango</li>
 *   <li>GenBooleano: inversión del valor</li>
 *   <li>GenEnum: selección uniforme entre valores posibles</li>
 * </ul>
 *
 * <p>Tratamiento especial para el gen "capasOcultas": añade o elimina capas
 * ocultas según el nuevo valor, inicializando neuronas aleatorias para capas
 * nuevas y eliminando las últimas capas sobrantes.</p>
 *
 * <p>Post-mutación se valida umbralDisparo &gt; potencialReposo y se repara
 * la topología si excede el límite.</p>
 */
public class OperadorMutacion {

    private final double probabilidadMutacion;
    private final int limiteTopologico;
    private final FabricaIndividuos fabrica;
    private final Random random;
    private final Set<BloqueFuncional> bloquesCongelados;

    /**
     * @param probabilidadMutacion probabilidad de mutar cada gen (default 0.1)
     * @param limiteTopologico     máximo de neuronas totales permitidas
     * @param fabrica              fábrica para construir configuraciones y reparar topología
     * @param random               generador de números aleatorios
     */
    public OperadorMutacion(double probabilidadMutacion, int limiteTopologico,
                            FabricaIndividuos fabrica, Random random) {
        this(probabilidadMutacion, limiteTopologico, fabrica, random, Set.of());
    }

    /**
     * Constructor con bloques congelados que no se mutarán.
     *
     * @param probabilidadMutacion probabilidad de mutar cada gen
     * @param limiteTopologico     máximo de neuronas totales permitidas
     * @param fabrica              fábrica para construir configuraciones y reparar topología
     * @param random               generador de números aleatorios
     * @param bloquesCongelados    bloques funcionales que no se mutarán
     */
    public OperadorMutacion(double probabilidadMutacion, int limiteTopologico,
                            FabricaIndividuos fabrica, Random random,
                            Set<BloqueFuncional> bloquesCongelados) {
        this.probabilidadMutacion = probabilidadMutacion;
        this.limiteTopologico = limiteTopologico;
        this.fabrica = fabrica;
        this.random = random;
        this.bloquesCongelados = Set.copyOf(bloquesCongelados);
    }

    /**
     * Muta un individuo aplicando perturbaciones por tipo de gen.
     *
     * <p>Algoritmo:</p>
     * <ol>
     *   <li>Para cada gen: si random &gt; probabilidadMutacion → mantener gen</li>
     *   <li>Según tipo: perturbación entera, gaussiana, inversión o selección uniforme</li>
     *   <li>Tratamiento especial para capasOcultas (añadir/eliminar capas)</li>
     *   <li>Validar umbralDisparo &gt; potencialReposo</li>
     *   <li>Reparar topología si excede límite</li>
     *   <li>Construir ConfiguracionRed y retornar nuevo Individuo sin evaluar</li>
     * </ol>
     *
     * @param individuo individuo a mutar
     * @return nuevo individuo mutado sin evaluar
     */
    public Individuo mutar(Individuo individuo) {
        Cromosoma cromosoma = individuo.cromosoma();

        // 1. Mutar cada bloque (saltar bloques congelados)
        var nuevosBloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesOriginales = cromosoma.genesDeBloque(bloque);
            if (bloquesCongelados.contains(bloque)) {
                nuevosBloques.put(bloque, genesOriginales);
                continue;
            }
            List<Gen<?>> genesMutados = new ArrayList<>();
            for (Gen<?> gen : genesOriginales) {
                if (random.nextDouble() < probabilidadMutacion) {
                    genesMutados.add(mutarGen(gen));
                } else {
                    genesMutados.add(gen);
                }
            }
            nuevosBloques.put(bloque, genesMutados);
        }

        // 2. Tratamiento especial para capasOcultas
        nuevosBloques.put(BloqueFuncional.TOPOLOGIA,
                ajustarCapasOcultas(
                        cromosoma.genesDeBloque(BloqueFuncional.TOPOLOGIA),
                        nuevosBloques.get(BloqueFuncional.TOPOLOGIA)));

        // 3. Validar restricción umbralDisparo > potencialReposo
        nuevosBloques.put(BloqueFuncional.LIF,
                validarUmbralReposo(nuevosBloques.get(BloqueFuncional.LIF)));

        Cromosoma cromosomaResultado = new Cromosoma(nuevosBloques);

        // 4. Reparar topología si excede límite
        cromosomaResultado = fabrica.repararTopologia(cromosomaResultado);

        // 5. Construir ConfiguracionRed y retornar nuevo Individuo sin evaluar
        var config = fabrica.construirConfiguracion(cromosomaResultado);
        return Individuo.sinEvaluar(cromosomaResultado, config);
    }

    // ==================== Mutación por tipo de gen ====================

    private Gen<?> mutarGen(Gen<?> gen) {
        return switch (gen) {
            case Gen.GenEntero g -> mutarEntero(g);
            case Gen.GenReal g -> mutarReal(g);
            case Gen.GenBooleano g -> mutarBooleano(g);
            case Gen.GenEnum<?> g -> mutarEnum(g);
        };
    }

    /**
     * Muta un gen entero: delta = random(1, ceil(0.20 * rango)), nuevoValor = valor ± delta.
     */
    private Gen.GenEntero mutarEntero(Gen.GenEntero gen) {
        int rango = gen.maximo() - gen.minimo();
        int maxDelta = Math.max(1, (int) Math.ceil(0.20 * rango));
        int delta = 1 + random.nextInt(maxDelta); // [1, maxDelta]
        int signo = random.nextBoolean() ? 1 : -1;
        int nuevoValor = gen.valor() + signo * delta;
        return (Gen.GenEntero) gen.conValor(nuevoValor);
    }

    /**
     * Muta un gen real: perturbación gaussiana con σ = 10% del rango.
     */
    private Gen.GenReal mutarReal(Gen.GenReal gen) {
        double rango = gen.maximo() - gen.minimo();
        double sigma = 0.10 * rango;
        double perturbacion = random.nextGaussian() * sigma;
        double nuevoValor = gen.valor() + perturbacion;
        return (Gen.GenReal) gen.conValor(nuevoValor);
    }

    /**
     * Muta un gen booleano: invierte el valor.
     */
    private Gen.GenBooleano mutarBooleano(Gen.GenBooleano gen) {
        return (Gen.GenBooleano) gen.conValor(!gen.valor());
    }

    /**
     * Muta un gen enumerado: selección uniforme entre valores posibles.
     */
    @SuppressWarnings("unchecked")
    private <E extends Enum<E>> Gen.GenEnum<E> mutarEnum(Gen.GenEnum<?> gen) {
        Gen.GenEnum<E> typed = (Gen.GenEnum<E>) gen;
        E[] valores = typed.tipoEnum().getEnumConstants();
        E nuevoValor = valores[random.nextInt(valores.length)];
        return (Gen.GenEnum<E>) typed.conValor(nuevoValor);
    }

    // ==================== Tratamiento especial de capas ====================

    /**
     * Ajusta las capas ocultas tras mutación del gen capasOcultas.
     * Si nuevasCapas &gt; capasAnteriores: añade capas con neuronas aleatorias en [1, 512].
     * Si nuevasCapas &lt; capasAnteriores: elimina las últimas capas sobrantes.
     */
    private List<Gen<?>> ajustarCapasOcultas(List<Gen<?>> topologiaOriginal,
                                              List<Gen<?>> topologiaMutada) {
        int capasOriginales = ((Number) topologiaOriginal.get(0).valor()).intValue();
        int capasNuevas = ((Number) topologiaMutada.get(0).valor()).intValue();

        if (capasNuevas == capasOriginales) {
            return topologiaMutada;
        }

        List<Gen<?>> resultado = new ArrayList<>();
        resultado.add(topologiaMutada.get(0)); // gen capasOcultas (ya mutado)

        if (capasNuevas > capasOriginales) {
            // Conservar capas existentes (mutadas)
            for (int i = 1; i <= capasOriginales && i < topologiaMutada.size(); i++) {
                resultado.add(topologiaMutada.get(i));
            }
            // Añadir capas nuevas con neuronas aleatorias en [1, 512]
            for (int i = capasOriginales; i < capasNuevas; i++) {
                int neuronas = 1 + random.nextInt(512);
                resultado.add(new Gen.GenEntero("neuronasCapa_" + i, neuronas, 1, 512));
            }
        } else {
            // capasNuevas < capasOriginales: conservar solo las primeras capasNuevas capas
            for (int i = 1; i <= capasNuevas && i < topologiaMutada.size(); i++) {
                resultado.add(topologiaMutada.get(i));
            }
        }

        return resultado;
    }

    // ==================== Validación de restricciones ====================

    /**
     * Valida que umbralDisparo &gt; potencialReposo.
     * Si se viola, ajusta umbralDisparo = potencialReposo + 1.0.
     */
    private List<Gen<?>> validarUmbralReposo(List<Gen<?>> genesLIF) {
        double umbralDisparo = ((Number) genesLIF.get(0).valor()).doubleValue();
        double potencialReposo = ((Number) genesLIF.get(1).valor()).doubleValue();

        if (umbralDisparo <= potencialReposo) {
            double nuevoUmbral = potencialReposo + 1.0;
            Gen.GenReal genUmbral = (Gen.GenReal) genesLIF.get(0);
            List<Gen<?>> resultado = new ArrayList<>(genesLIF);
            resultado.set(0, genUmbral.conValor(nuevoUmbral));
            return resultado;
        }
        return genesLIF;
    }
}
