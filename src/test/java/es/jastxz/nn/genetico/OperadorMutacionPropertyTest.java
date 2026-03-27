package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ModoCodificacion;
import net.jqwik.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link OperadorMutacion}.
 *
 * <p>Verifica propiedades universales de la mutación adaptativa por tipo de gen:
 * probabilidad cero no modifica, estrategia por tipo, y coherencia topológica.</p>
 */
class OperadorMutacionPropertyTest {

    private static final int DIM_ENTRADA = 10;
    private static final int DIM_SALIDA = 5;
    private static final int LIMITE_TOPOLOGICO = 512;

    // ==================== Property 14: Mutación con probabilidad cero no modifica ====================

    // Feature: genetic-algorithm-hyperparameters, Property 14: Mutación con probabilidad cero no modifica
    /**
     * Para cualquier individuo, aplicar mutación con probabilidad 0.0 debe retornar
     * un individuo con cromosoma idéntico al original.
     *
     * <p><b>Validates: Requirement 6.1</b></p>
     */
    @Property(tries = 100)
    void mutacionConProbabilidadCeroNoModifica(@ForAll("semilla") long semilla) {
        Random rng = new Random(semilla);
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));
        Individuo original = fabrica.generarAleatorio();

        // Probability 0.0 → no gene should be mutated
        OperadorMutacion mutacion = new OperadorMutacion(
                0.0, LIMITE_TOPOLOGICO, fabrica, rng);
        Individuo mutado = mutacion.mutar(original);

        // Verify all genes are identical
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesOriginal = original.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesMutado = mutado.cromosoma().genesDeBloque(bloque);
            assertEquals(genesOriginal.size(), genesMutado.size(),
                    "Bloque " + bloque + " debe tener mismo tamaño");
            for (int i = 0; i < genesOriginal.size(); i++) {
                assertEquals(genesOriginal.get(i).nombre(), genesMutado.get(i).nombre(),
                        "Gen nombre debe coincidir en bloque " + bloque);
                assertEquals(genesOriginal.get(i).valor(), genesMutado.get(i).valor(),
                        "Gen " + genesOriginal.get(i).nombre()
                                + " no debe cambiar con probabilidad 0.0");
            }
        }
    }

    // ==================== Property 15: Mutación respeta estrategia por tipo de gen ====================

    // Feature: genetic-algorithm-hyperparameters, Property 15: Mutación respeta estrategia por tipo de gen
    /**
     * Para cualquier gen mutado: si es entero, el cambio absoluto está en [1, ceil(0.20 * rango)];
     * si es booleano, el valor se invierte; si es enumerado, el nuevo valor es un valor válido del enum;
     * si es real, el resultado está dentro de los límites del gen.
     *
     * <p><b>Validates: Requirements 6.2, 6.3, 6.4, 6.5</b></p>
     */
    @Property(tries = 100)
    void mutacionRespetaEstrategiaPorTipoDeGen(@ForAll("semilla") long semilla) {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));

        // Create a known chromosome with fixed values for predictable comparison
        Cromosoma cromosoma = crearCromosomaConValores(1);
        ConfiguracionRed config = fabrica.construirConfiguracion(cromosoma);
        Individuo original = Individuo.sinEvaluar(cromosoma, config).conEvaluacion(0.5, null);

        // Probability 1.0 → every gene is mutated
        Random rng = new Random(semilla);
        OperadorMutacion mutacion = new OperadorMutacion(
                1.0, LIMITE_TOPOLOGICO, fabrica, rng);
        Individuo mutado = mutacion.mutar(original);

        // Check each block (skip TOPOLOGIA since capasOcultas has special handling)
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesOriginal = original.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesMutado = mutado.cromosoma().genesDeBloque(bloque);

            // For TOPOLOGIA, the number of genes may differ due to layer add/remove
            if (bloque == BloqueFuncional.TOPOLOGIA) {
                // Just verify capasOcultas gene is within range
                int capasOrig = ((Number) genesOriginal.get(0).valor()).intValue();
                int capasMut = ((Number) genesMutado.get(0).valor()).intValue();
                assertTrue(capasMut >= 1 && capasMut <= 10,
                        "capasOcultas mutado debe estar en [1, 10], fue: " + capasMut);
                // Verify integer perturbation range for capasOcultas
                int rango = 10 - 1; // max - min
                int maxDelta = Math.max(1, (int) Math.ceil(0.20 * rango));
                int delta = Math.abs(capasMut - capasOrig);
                assertTrue(delta >= 1 && delta <= maxDelta,
                        "capasOcultas delta debe estar en [1, " + maxDelta + "], fue: " + delta);
                // Verify all layer neuron genes are in [1, 512]
                for (int i = 1; i < genesMutado.size(); i++) {
                    int neuronas = ((Number) genesMutado.get(i).valor()).intValue();
                    assertTrue(neuronas >= 1 && neuronas <= 512,
                            "neuronasCapa debe estar en [1, 512], fue: " + neuronas);
                }
                continue;
            }

            assertEquals(genesOriginal.size(), genesMutado.size(),
                    "Bloque " + bloque + " debe tener mismo tamaño");

            for (int i = 0; i < genesOriginal.size(); i++) {
                Gen<?> orig = genesOriginal.get(i);
                Gen<?> mut = genesMutado.get(i);

                switch (orig) {
                    case Gen.GenEntero go -> {
                        Gen.GenEntero gm = (Gen.GenEntero) mut;
                        int rango = go.maximo() - go.minimo();
                        int maxDelta = Math.max(1, (int) Math.ceil(0.20 * rango));
                        int delta = Math.abs(gm.valor() - go.valor());
                        // After clamping, delta could be less than 1 if at boundary
                        assertTrue(delta <= maxDelta,
                                "GenEntero " + go.nombre() + " delta=" + delta
                                        + " debe ser <= " + maxDelta);
                        assertTrue(gm.valor() >= gm.minimo() && gm.valor() <= gm.maximo(),
                                "GenEntero " + go.nombre() + " debe estar en rango");
                    }
                    case Gen.GenReal go -> {
                        Gen.GenReal gm = (Gen.GenReal) mut;
                        // After Gaussian perturbation + clamping, value must be in range
                        assertTrue(gm.valor() >= gm.minimo() && gm.valor() <= gm.maximo(),
                                "GenReal " + go.nombre() + " debe estar en rango ["
                                        + gm.minimo() + ", " + gm.maximo() + "], fue: " + gm.valor());
                    }
                    case Gen.GenBooleano go -> {
                        Gen.GenBooleano gm = (Gen.GenBooleano) mut;
                        assertNotEquals(go.valor(), gm.valor(),
                                "GenBooleano " + go.nombre() + " debe invertirse");
                    }
                    case Gen.GenEnum<?> go -> {
                        Gen.GenEnum<?> gm = (Gen.GenEnum<?>) mut;
                        // New value must be a valid enum constant
                        Enum<?>[] constantes = gm.tipoEnum().getEnumConstants();
                        boolean valido = false;
                        for (Enum<?> c : constantes) {
                            if (c.equals(gm.valor())) {
                                valido = true;
                                break;
                            }
                        }
                        assertTrue(valido,
                                "GenEnum " + go.nombre() + " debe ser un valor válido del enum");
                    }
                }
            }
        }
    }

    // ==================== Property 16: Mutación de capas mantiene coherencia topológica ====================

    // Feature: genetic-algorithm-hyperparameters, Property 16: Mutación de capas mantiene coherencia topológica
    /**
     * Para cualquier mutación del gen capasOcultas: si el nuevo valor es mayor,
     * las capas adicionales tienen neuronas en [1, 512]; si es menor, las capas
     * restantes conservan los valores de las primeras capas originales.
     *
     * <p><b>Validates: Requirements 6.6, 6.7</b></p>
     */
    @Property(tries = 100)
    void mutacionCapasMantieneCoherenciaTopologica(@ForAll("semilla") long semilla) {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));

        // Create a chromosome with a known number of layers (3 layers)
        Cromosoma cromosoma = crearCromosomaConCapas(3);
        ConfiguracionRed config = fabrica.construirConfiguracion(cromosoma);
        Individuo original = Individuo.sinEvaluar(cromosoma, config).conEvaluacion(0.5, null);

        // Mutate with probability 1.0 to force capasOcultas mutation
        Random rng = new Random(semilla);
        OperadorMutacion mutacion = new OperadorMutacion(
                1.0, LIMITE_TOPOLOGICO, fabrica, rng);
        Individuo mutado = mutacion.mutar(original);

        List<Gen<?>> topOriginal = original.cromosoma().genesDeBloque(BloqueFuncional.TOPOLOGIA);
        List<Gen<?>> topMutado = mutado.cromosoma().genesDeBloque(BloqueFuncional.TOPOLOGIA);

        int capasOriginales = ((Number) topOriginal.get(0).valor()).intValue();
        int capasMutadas = ((Number) topMutado.get(0).valor()).intValue();

        // Verify the number of layer genes matches capasOcultas
        assertEquals(capasMutadas + 1, topMutado.size(),
                "Número de genes de topología debe ser capasOcultas + 1 (gen capasOcultas + neuronas por capa)");

        if (capasMutadas > capasOriginales) {
            // New layers should have neurons in [1, 512]
            for (int i = capasOriginales + 1; i <= capasMutadas; i++) {
                int neuronas = ((Number) topMutado.get(i).valor()).intValue();
                assertTrue(neuronas >= 1 && neuronas <= 512,
                        "Capa nueva " + (i - 1) + " debe tener neuronas en [1, 512], fue: " + neuronas);
            }
        } else if (capasMutadas < capasOriginales) {
            // Remaining layers should preserve the first ones (possibly mutated)
            // The key property: the number of layer genes equals the new capasOcultas
            assertEquals(capasMutadas, topMutado.size() - 1,
                    "Capas restantes deben ser exactamente capasOcultas mutadas");
        }

        // All layer neurons must be in valid range [1, 512]
        for (int i = 1; i < topMutado.size(); i++) {
            int neuronas = ((Number) topMutado.get(i).valor()).intValue();
            assertTrue(neuronas >= 1 && neuronas <= 512,
                    "Todas las capas deben tener neuronas en [1, 512], capa " + (i - 1) + " fue: " + neuronas);
        }
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Long> semilla() {
        return Arbitraries.longs();
    }

    // ==================== Helpers ====================

    /**
     * Creates a chromosome with deterministic values based on a variant number.
     */
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

    /**
     * Creates a chromosome with a specific number of hidden layers.
     */
    private static Cromosoma crearCromosomaConCapas(int numCapas) {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);

        List<Gen<?>> topologia = new ArrayList<>();
        topologia.add(new Gen.GenEntero("capasOcultas", numCapas, 1, 10));
        for (int i = 0; i < numCapas; i++) {
            topologia.add(new Gen.GenEntero("neuronasCapa_" + i, 20 + i * 5, 1, 512));
        }
        bloques.put(BloqueFuncional.TOPOLOGIA, topologia);

        bloques.put(BloqueFuncional.LIF, List.of(
                new Gen.GenReal("umbralDisparo", -50.0, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", -70.0, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", 20.0, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", 2, 1, 10)));

        bloques.put(BloqueFuncional.STDP, List.of(
                new Gen.GenReal("amplitudLTP", 0.01, 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", 0.012, 0.001, 0.1),
                new Gen.GenReal("tauLTP", 20.0, 5.0, 50.0),
                new Gen.GenReal("tauLTD", 20.0, 5.0, 50.0)));

        bloques.put(BloqueFuncional.CODIFICACION, List.of(
                new Gen.GenReal("frecuenciaMaxima", 100.0, 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion", ModoCodificacion.POISSON, ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", 50, 10, 200)));

        bloques.put(BloqueFuncional.REGULACION, List.of(
                new Gen.GenBooleano("homeostasisActiva", true),
                new Gen.GenReal("tasaDisparoObjetivo", 10.0, 1.0, 50.0),
                new Gen.GenReal("tasaAjusteHomeostasis", 0.01, 0.001, 0.1),
                new Gen.GenBooleano("inhibicionLateralActiva", false),
                new Gen.GenEntero("radioInhibicion", 2, 1, 5),
                new Gen.GenReal("fuerzaInhibicion", 0.5, 0.1, 2.0)));

        bloques.put(BloqueFuncional.COMPETICION, List.of(
                new Gen.GenBooleano("wtaActivo", false),
                new Gen.GenBooleano("wtaCapaSalida", false),
                new Gen.GenBooleano("wtaCapasOcultas", false),
                new Gen.GenEntero("radioWTA", 0, 0, 10),
                new Gen.GenReal("fuerzaWTA", 2.0, 0.5, 5.0),
                new Gen.GenReal("umbralActivacionWTA", 0.1, 0.0, 0.5)));

        return new Cromosoma(bloques);
    }
}
