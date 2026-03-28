package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link OperadorCruce}.
 */
class OperadorCruceTest {

    private static final int DIM_ENTRADA = 10;
    private static final int DIM_SALIDA = 5;
    private static final int LIMITE_TOPOLOGICO = 512;

    @Test
    void cruceConProbabilidad0_retornaMejorPadre() {
        FabricaIndividuos fabrica = new FabricaIndividuos(DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(42));

        Cromosoma crom1 = crearCromosomaConValores(1);
        Cromosoma crom2 = crearCromosomaConValores(2);
        ConfiguracionRed config1 = fabrica.construirConfiguracion(crom1);
        ConfiguracionRed config2 = fabrica.construirConfiguracion(crom2);

        Individuo padre1 = Individuo.sinEvaluar(crom1, config1).conEvaluacion(0.9, null);
        Individuo padre2 = Individuo.sinEvaluar(crom2, config2).conEvaluacion(0.3, null);

        // Probability 0 → crossover never applied → return copy of best parent
        OperadorCruce cruce = new OperadorCruce(0.0, 2, LIMITE_TOPOLOGICO, fabrica, new Random(42));
        Individuo hijo = cruce.cruzar(padre1, padre2);

        // Offspring should have the same chromosome as the best parent (padre1)
        assertEquals(-1.0, hijo.fitness(), "Hijo debe estar sin evaluar");
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesHijo = hijo.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesMejor = padre1.cromosoma().genesDeBloque(bloque);
            assertEquals(genesMejor.size(), genesHijo.size(),
                    "Bloque " + bloque + " debe tener mismo tamaño");
            for (int i = 0; i < genesHijo.size(); i++) {
                assertEquals(genesMejor.get(i).valor(), genesHijo.get(i).valor(),
                        "Gen " + genesHijo.get(i).nombre() + " del bloque " + bloque
                                + " debe coincidir con el mejor padre");
            }
        }
    }

    @Test
    void cruceConProbabilidad0_retornaMejorPadre_cuandoPadre2EsMejor() {
        FabricaIndividuos fabrica = new FabricaIndividuos(DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(42));

        Cromosoma crom1 = crearCromosomaConValores(1);
        Cromosoma crom2 = crearCromosomaConValores(2);
        ConfiguracionRed config1 = fabrica.construirConfiguracion(crom1);
        ConfiguracionRed config2 = fabrica.construirConfiguracion(crom2);

        // padre2 has higher fitness
        Individuo padre1 = Individuo.sinEvaluar(crom1, config1).conEvaluacion(0.3, null);
        Individuo padre2 = Individuo.sinEvaluar(crom2, config2).conEvaluacion(0.9, null);

        OperadorCruce cruce = new OperadorCruce(0.0, 2, LIMITE_TOPOLOGICO, fabrica, new Random(42));
        Individuo hijo = cruce.cruzar(padre1, padre2);

        // Offspring should have the same chromosome as the best parent (padre2)
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesHijo = hijo.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesMejor = padre2.cromosoma().genesDeBloque(bloque);
            for (int i = 0; i < genesHijo.size(); i++) {
                assertEquals(genesMejor.get(i).valor(), genesHijo.get(i).valor(),
                        "Gen " + genesHijo.get(i).nombre() + " del bloque " + bloque
                                + " debe coincidir con el mejor padre (padre2)");
            }
        }
    }

    @Test
    void cruceCon1PuntoDeCorte_produceBloquesMezclados() {
        FabricaIndividuos fabrica = new FabricaIndividuos(DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(42));

        Cromosoma crom1 = crearCromosomaConValores(1);
        Cromosoma crom2 = crearCromosomaConValores(2);
        ConfiguracionRed config1 = fabrica.construirConfiguracion(crom1);
        ConfiguracionRed config2 = fabrica.construirConfiguracion(crom2);

        Individuo padre1 = Individuo.sinEvaluar(crom1, config1).conEvaluacion(0.8, null);
        Individuo padre2 = Individuo.sinEvaluar(crom2, config2).conEvaluacion(0.4, null);

        // 1 cut point, probability 1.0 → always apply crossover
        OperadorCruce cruce = new OperadorCruce(1.0, 1, LIMITE_TOPOLOGICO, fabrica, new Random(42));
        Individuo hijo = cruce.cruzar(padre1, padre2);

        // With 1 cut point → 2 segments → best parent gets ceil((2+1)/2) = 2 segments
        // This means ALL blocks come from best parent (since 2 segments cover all 5 blocks)
        // But with different seeds, the cut point position varies.
        // We verify: each block comes from one parent, and offspring is unevaluated
        assertEquals(-1.0, hijo.fitness(), "Hijo debe estar sin evaluar");

        int bloquesPadre1 = 0;
        int bloquesPadre2 = 0;
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesHijo = hijo.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesPadre1 = padre1.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesPadre2 = padre2.cromosoma().genesDeBloque(bloque);

            boolean esPadre1 = bloquesIguales(genesHijo, genesPadre1);
            boolean esPadre2 = bloquesIguales(genesHijo, genesPadre2);

            assertTrue(esPadre1 || esPadre2,
                    "Bloque " + bloque + " debe provenir de uno de los padres");

            if (esPadre1) bloquesPadre1++;
            if (esPadre2) bloquesPadre2++;
        }

        // With 1 cut point, best parent gets more blocks
        assertTrue(bloquesPadre1 > bloquesPadre2,
                "Mejor padre (padre1) debe tener más bloques: padre1=" + bloquesPadre1
                        + ", padre2=" + bloquesPadre2);
    }

    // ==================== Helpers ====================

    private static Cromosoma crearCromosomaConValores(int variante) {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);

        int neuronas1 = 20 + variante * 10;
        int neuronas2 = 10 + variante * 5;
        bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                new Gen.GenEntero("capasOcultas", 2, 1, 10),
                new Gen.GenEntero("neuronasCapa_0", neuronas1, 1, 512),
                new Gen.GenEntero("neuronasCapa_1", neuronas2, 1, 512)));

        double umbral = -55.0 + variante;
        double reposo = -70.0 - variante;
        bloques.put(BloqueFuncional.LIF, List.of(
                new Gen.GenReal("umbralDisparo", umbral, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", reposo, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", 20.0 + variante, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", variante + 1, 1, 10)));

        bloques.put(BloqueFuncional.STDP, List.of(
                new Gen.GenReal("amplitudLTP", 0.01 + variante * 0.005, 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", 0.012 + variante * 0.005, 0.001, 0.1),
                new Gen.GenReal("tauLTP", 20.0 + variante * 2, 5.0, 50.0),
                new Gen.GenReal("tauLTD", 20.0 + variante * 3, 5.0, 50.0)));

        ModoCodificacion modo = variante == 1 ? ModoCodificacion.POISSON : ModoCodificacion.REGULAR;
        bloques.put(BloqueFuncional.CODIFICACION, List.of(
                new Gen.GenReal("frecuenciaMaxima", 100.0 + variante * 50, 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion", modo, ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", 50 + variante * 10, 10, 200)));

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

    private static boolean bloquesIguales(List<Gen<?>> genes1, List<Gen<?>> genes2) {
        if (genes1.size() != genes2.size()) return false;
        for (int i = 0; i < genes1.size(); i++) {
            if (!Objects.equals(genes1.get(i).valor(), genes2.get(i).valor())) return false;
            if (!Objects.equals(genes1.get(i).nombre(), genes2.get(i).nombre())) return false;
        }
        return true;
    }
}
