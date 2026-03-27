package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ModoCodificacion;
import net.jqwik.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link OperadorCruce}.
 *
 * <p>Verifica propiedades universales del cruce multi-punto por bloques funcionales:
 * integridad de bloques y sesgo hacia el mejor padre.</p>
 */
class OperadorCrucePropertyTest {

    private static final int DIM_ENTRADA = 10;
    private static final int DIM_SALIDA = 5;
    private static final int LIMITE_TOPOLOGICO = 512;
    private static final BloqueFuncional[] BLOQUES = BloqueFuncional.values();

    // ==================== Property 12: Integridad de bloques funcionales en cruce ====================

    // Feature: genetic-algorithm-hyperparameters, Property 12: Integridad de bloques funcionales en cruce
    /**
     * Para cualquier descendiente producido por cruce, cada bloque funcional
     * (TOPOLOGIA, LIF, STDP, CODIFICACION, REGULACION) debe provenir íntegramente
     * de uno de los dos padres, sin mezclar genes de diferentes padres dentro del
     * mismo bloque.
     *
     * <p><b>Validates: Requirements 5.1, 5.2, 5.4, 5.5</b></p>
     */
    @Property(tries = 100)
    void integridadBloquesFuncionalesEnCruce(
            @ForAll("semilla") long semilla,
            @ForAll("puntosCorte") int puntosCorte) {

        Random rng = new Random(semilla);
        FabricaIndividuos fabrica = new FabricaIndividuos(DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));

        // Create two parents with distinct gene values per block
        Cromosoma cromPadre1 = crearCromosomaConValores(1);
        Cromosoma cromPadre2 = crearCromosomaConValores(2);

        ConfiguracionRed config1 = fabrica.construirConfiguracion(cromPadre1);
        ConfiguracionRed config2 = fabrica.construirConfiguracion(cromPadre2);

        Individuo padre1 = Individuo.sinEvaluar(cromPadre1, config1).conEvaluacion(0.8, null);
        Individuo padre2 = Individuo.sinEvaluar(cromPadre2, config2).conEvaluacion(0.5, null);

        // Force crossover to always apply (probability = 1.0)
        OperadorCruce cruce = new OperadorCruce(1.0, puntosCorte, LIMITE_TOPOLOGICO, fabrica, rng);
        Individuo hijo = cruce.cruzar(padre1, padre2);

        // Verify each block comes entirely from one parent
        for (BloqueFuncional bloque : BLOQUES) {
            List<Gen<?>> genesHijo = hijo.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesPadre1 = padre1.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesPadre2 = padre2.cromosoma().genesDeBloque(bloque);

            boolean coincidePadre1 = bloquesIguales(genesHijo, genesPadre1);
            boolean coincidePadre2 = bloquesIguales(genesHijo, genesPadre2);

            assertTrue(coincidePadre1 || coincidePadre2,
                    "Bloque " + bloque + " del hijo no coincide íntegramente con ningún padre. "
                            + "Hijo=" + genesAString(genesHijo)
                            + ", Padre1=" + genesAString(genesPadre1)
                            + ", Padre2=" + genesAString(genesPadre2));
        }
    }

    // ==================== Property 13: Sesgo hacia el mejor padre en cruce ====================

    // Feature: genetic-algorithm-hyperparameters, Property 13: Sesgo hacia el mejor padre en cruce
    /**
     * Para cualquier cruce entre dos padres con fitness distintos, el número de bloques
     * funcionales tomados del padre con mayor fitness debe ser estrictamente mayor que
     * la mitad del número total de bloques.
     *
     * <p><b>Validates: Requirement 5.3</b></p>
     */
    @Property(tries = 100)
    void sesgoHaciaMejorPadreEnCruce(
            @ForAll("semilla") long semilla,
            @ForAll("puntosCorte") int puntosCorte) {

        Random rng = new Random(semilla);
        FabricaIndividuos fabrica = new FabricaIndividuos(DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));

        // Create two parents with distinct gene values and different fitness
        Cromosoma cromPadre1 = crearCromosomaConValores(1);
        Cromosoma cromPadre2 = crearCromosomaConValores(2);

        ConfiguracionRed config1 = fabrica.construirConfiguracion(cromPadre1);
        ConfiguracionRed config2 = fabrica.construirConfiguracion(cromPadre2);

        Individuo padre1 = Individuo.sinEvaluar(cromPadre1, config1).conEvaluacion(0.9, null);
        Individuo padre2 = Individuo.sinEvaluar(cromPadre2, config2).conEvaluacion(0.3, null);

        // Force crossover to always apply (probability = 1.0)
        OperadorCruce cruce = new OperadorCruce(1.0, puntosCorte, LIMITE_TOPOLOGICO, fabrica, rng);
        Individuo hijo = cruce.cruzar(padre1, padre2);

        // Count blocks from best parent (padre1 has higher fitness)
        Individuo mejorPadre = padre1;
        int bloquesMejor = 0;
        for (BloqueFuncional bloque : BLOQUES) {
            List<Gen<?>> genesHijo = hijo.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesMejor = mejorPadre.cromosoma().genesDeBloque(bloque);
            if (bloquesIguales(genesHijo, genesMejor)) {
                bloquesMejor++;
            }
        }

        int totalBloques = BLOQUES.length; // 5
        assertTrue(bloquesMejor > totalBloques / 2,
                "Bloques del mejor padre (" + bloquesMejor + ") debe ser > "
                        + (totalBloques / 2) + " (mitad de " + totalBloques + " bloques). "
                        + "Semilla=" + semilla + ", puntosCorte=" + puntosCorte);
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Long> semilla() {
        return Arbitraries.longs();
    }

    @Provide
    Arbitrary<Integer> puntosCorte() {
        return Arbitraries.integers().between(1, 4);
    }

    // ==================== Helpers ====================

    /**
     * Creates a chromosome with deterministic values based on a variant number,
     * so that parent1 and parent2 have clearly distinguishable gene values.
     */
    private static Cromosoma crearCromosomaConValores(int variante) {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);

        // TOPOLOGIA: 2 hidden layers with different neuron counts per variant
        int neuronas1 = 20 + variante * 10; // variant 1: 30, variant 2: 40
        int neuronas2 = 10 + variante * 5;  // variant 1: 15, variant 2: 20
        bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                new Gen.GenEntero("capasOcultas", 2, 1, 10),
                new Gen.GenEntero("neuronasCapa_0", neuronas1, 1, 512),
                new Gen.GenEntero("neuronasCapa_1", neuronas2, 1, 512)));

        // LIF: different values per variant
        double umbral = -55.0 + variante;   // variant 1: -54.0, variant 2: -53.0
        double reposo = -70.0 - variante;   // variant 1: -71.0, variant 2: -72.0
        bloques.put(BloqueFuncional.LIF, List.of(
                new Gen.GenReal("umbralDisparo", umbral, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", reposo, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", 20.0 + variante, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", variante + 1, 1, 10)));

        // STDP: different values per variant
        bloques.put(BloqueFuncional.STDP, List.of(
                new Gen.GenReal("amplitudLTP", 0.01 + variante * 0.005, 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", 0.012 + variante * 0.005, 0.001, 0.1),
                new Gen.GenReal("tauLTP", 20.0 + variante * 2, 5.0, 50.0),
                new Gen.GenReal("tauLTD", 20.0 + variante * 3, 5.0, 50.0)));

        // CODIFICACION: different values per variant
        ModoCodificacion modo = variante == 1 ? ModoCodificacion.POISSON : ModoCodificacion.REGULAR;
        bloques.put(BloqueFuncional.CODIFICACION, List.of(
                new Gen.GenReal("frecuenciaMaxima", 100.0 + variante * 50, 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion", modo, ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", 50 + variante * 10, 10, 200)));

        // REGULACION: different values per variant
        boolean homeostasis = variante == 1;
        boolean inhibicion = variante == 2;
        bloques.put(BloqueFuncional.REGULACION, List.of(
                new Gen.GenBooleano("homeostasisActiva", homeostasis),
                new Gen.GenReal("tasaDisparoObjetivo", 10.0 + variante * 5, 1.0, 50.0),
                new Gen.GenReal("tasaAjusteHomeostasis", 0.01 + variante * 0.005, 0.001, 0.1),
                new Gen.GenBooleano("inhibicionLateralActiva", inhibicion),
                new Gen.GenEntero("radioInhibicion", Math.min(variante + 1, 5), 1, 5),
                new Gen.GenReal("fuerzaInhibicion", 0.5 + variante * 0.2, 0.1, 2.0)));

        boolean wta = variante == 1;
        bloques.put(BloqueFuncional.COMPETICION, List.of(
                new Gen.GenBooleano("wtaActivo", wta),
                new Gen.GenBooleano("wtaCapaSalida", wta),
                new Gen.GenBooleano("wtaCapasOcultas", false),
                new Gen.GenEntero("radioWTA", variante, 0, 10),
                new Gen.GenReal("fuerzaWTA", 2.0 + variante * 0.5, 0.5, 5.0),
                new Gen.GenReal("umbralActivacionWTA", 0.1 + variante * 0.05, 0.0, 0.5)));

        return new Cromosoma(bloques);
    }

    /**
     * Compares two gene lists for equality by name and value.
     */
    private static boolean bloquesIguales(List<Gen<?>> genes1, List<Gen<?>> genes2) {
        if (genes1.size() != genes2.size()) return false;
        for (int i = 0; i < genes1.size(); i++) {
            if (!Objects.equals(genes1.get(i).valor(), genes2.get(i).valor())) {
                return false;
            }
            if (!Objects.equals(genes1.get(i).nombre(), genes2.get(i).nombre())) {
                return false;
            }
        }
        return true;
    }

    private static String genesAString(List<Gen<?>> genes) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < genes.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(genes.get(i).nombre()).append("=").append(genes.get(i).valor());
        }
        return sb.append("]").toString();
    }
}
