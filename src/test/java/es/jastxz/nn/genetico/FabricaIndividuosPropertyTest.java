package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ModoCodificacion;
import net.jqwik.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link FabricaIndividuos}.
 *
 * <p>Cada test genera individuos usando {@code FabricaIndividuos} con semillas
 * aleatorias proporcionadas por jqwik y verifica propiedades universales.</p>
 */
class FabricaIndividuosPropertyTest {

    private static final int DIMENSION_ENTRADA = 10;
    private static final int DIMENSION_SALIDA = 5;
    private static final int LIMITE_TOPOLOGICO = 512;

    // ==================== Property 1: Genes dentro de rangos válidos ====================

    // Feature: genetic-algorithm-hyperparameters, Property 1: Genes dentro de rangos válidos
    /**
     * Para cualquier individuo producido por generarAleatorio(), todos los genes
     * del cromosoma deben estar dentro de sus rangos definidos.
     *
     * <p><b>Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 6.8</b></p>
     */
    @Property(tries = 100)
    void genesEnRangosValidos(@ForAll("semilla") long semilla) {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));
        Individuo individuo = fabrica.generarAleatorio();
        Cromosoma cromosoma = individuo.cromosoma();

        // TOPOLOGIA: capasOcultas en [1, 10], neuronasPorCapa en [1, 512]
        List<Gen<?>> topologia = cromosoma.genesDeBloque(BloqueFuncional.TOPOLOGIA);
        int capasOcultas = ((Number) topologia.get(0).valor()).intValue();
        assertTrue(capasOcultas >= 1 && capasOcultas <= 10,
                "capasOcultas=" + capasOcultas + " fuera de [1, 10]");
        assertEquals(capasOcultas + 1, topologia.size(),
                "Debe haber capasOcultas+1 genes en TOPOLOGIA (capasOcultas + neuronasPorCapa)");
        for (int i = 1; i <= capasOcultas; i++) {
            int neuronas = ((Number) topologia.get(i).valor()).intValue();
            assertTrue(neuronas >= 1 && neuronas <= 512,
                    "neuronasCapa_" + (i - 1) + "=" + neuronas + " fuera de [1, 512]");
        }

        // LIF
        List<Gen<?>> lif = cromosoma.genesDeBloque(BloqueFuncional.LIF);
        double umbralDisparo = ((Number) lif.get(0).valor()).doubleValue();
        double potencialReposo = ((Number) lif.get(1).valor()).doubleValue();
        double constanteDecaimiento = ((Number) lif.get(2).valor()).doubleValue();
        int duracionRefractario = ((Number) lif.get(3).valor()).intValue();
        assertTrue(umbralDisparo >= -60.0 && umbralDisparo <= -40.0,
                "umbralDisparo=" + umbralDisparo + " fuera de [-60.0, -40.0]");
        assertTrue(potencialReposo >= -80.0 && potencialReposo <= -60.0,
                "potencialReposo=" + potencialReposo + " fuera de [-80.0, -60.0]");
        assertTrue(constanteDecaimiento >= 5.0 && constanteDecaimiento <= 50.0,
                "constanteDecaimiento=" + constanteDecaimiento + " fuera de [5.0, 50.0]");
        assertTrue(duracionRefractario >= 1 && duracionRefractario <= 10,
                "duracionRefractario=" + duracionRefractario + " fuera de [1, 10]");

        // STDP
        List<Gen<?>> stdp = cromosoma.genesDeBloque(BloqueFuncional.STDP);
        double amplitudLTP = ((Number) stdp.get(0).valor()).doubleValue();
        double amplitudLTD = ((Number) stdp.get(1).valor()).doubleValue();
        double tauLTP = ((Number) stdp.get(2).valor()).doubleValue();
        double tauLTD = ((Number) stdp.get(3).valor()).doubleValue();
        assertTrue(amplitudLTP >= 0.001 && amplitudLTP <= 0.1,
                "amplitudLTP=" + amplitudLTP + " fuera de [0.001, 0.1]");
        assertTrue(amplitudLTD >= 0.001 && amplitudLTD <= 0.1,
                "amplitudLTD=" + amplitudLTD + " fuera de [0.001, 0.1]");
        assertTrue(tauLTP >= 5.0 && tauLTP <= 50.0,
                "tauLTP=" + tauLTP + " fuera de [5.0, 50.0]");
        assertTrue(tauLTD >= 5.0 && tauLTD <= 50.0,
                "tauLTD=" + tauLTD + " fuera de [5.0, 50.0]");

        // CODIFICACION
        List<Gen<?>> codificacion = cromosoma.genesDeBloque(BloqueFuncional.CODIFICACION);
        double frecuenciaMaxima = ((Number) codificacion.get(0).valor()).doubleValue();
        @SuppressWarnings("unchecked")
        ModoCodificacion modo = ((Gen.GenEnum<ModoCodificacion>) codificacion.get(1)).valor();
        int ventanaDecodificacion = ((Number) codificacion.get(2).valor()).intValue();
        assertTrue(frecuenciaMaxima >= 10.0 && frecuenciaMaxima <= 500.0,
                "frecuenciaMaxima=" + frecuenciaMaxima + " fuera de [10.0, 500.0]");
        assertNotNull(modo, "modoCodificacion no debe ser null");
        assertTrue(ventanaDecodificacion >= 10 && ventanaDecodificacion <= 200,
                "ventanaDecodificacion=" + ventanaDecodificacion + " fuera de [10, 200]");

        // REGULACION
        List<Gen<?>> regulacion = cromosoma.genesDeBloque(BloqueFuncional.REGULACION);
        assertNotNull(regulacion.get(0).valor(), "homeostasisActiva no debe ser null");
        double tasaDisparoObjetivo = ((Number) regulacion.get(1).valor()).doubleValue();
        double tasaAjusteHomeostasis = ((Number) regulacion.get(2).valor()).doubleValue();
        assertNotNull(regulacion.get(3).valor(), "inhibicionLateralActiva no debe ser null");
        int radioInhibicion = ((Number) regulacion.get(4).valor()).intValue();
        double fuerzaInhibicion = ((Number) regulacion.get(5).valor()).doubleValue();
        assertTrue(tasaDisparoObjetivo >= 1.0 && tasaDisparoObjetivo <= 50.0,
                "tasaDisparoObjetivo=" + tasaDisparoObjetivo + " fuera de [1.0, 50.0]");
        assertTrue(tasaAjusteHomeostasis >= 0.001 && tasaAjusteHomeostasis <= 0.1,
                "tasaAjusteHomeostasis=" + tasaAjusteHomeostasis + " fuera de [0.001, 0.1]");
        assertTrue(radioInhibicion >= 1 && radioInhibicion <= 5,
                "radioInhibicion=" + radioInhibicion + " fuera de [1, 5]");
        assertTrue(fuerzaInhibicion >= 0.1 && fuerzaInhibicion <= 2.0,
                "fuerzaInhibicion=" + fuerzaInhibicion + " fuera de [0.1, 2.0]");
    }

    // ==================== Property 2: Restricción umbralDisparo > potencialReposo ====================

    // Feature: genetic-algorithm-hyperparameters, Property 2: Restricción umbralDisparo > potencialReposo
    /**
     * Para cualquier individuo producido por generarAleatorio(), el valor del gen
     * umbralDisparo debe ser estrictamente mayor que el valor del gen potencialReposo.
     *
     * <p><b>Validates: Requirement 1.6</b></p>
     */
    @Property(tries = 100)
    void umbralMayorQueReposo(@ForAll("semilla") long semilla) {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));
        Individuo individuo = fabrica.generarAleatorio();

        List<Gen<?>> lif = individuo.cromosoma().genesDeBloque(BloqueFuncional.LIF);
        double umbralDisparo = ((Number) lif.get(0).valor()).doubleValue();
        double potencialReposo = ((Number) lif.get(1).valor()).doubleValue();

        assertTrue(umbralDisparo > potencialReposo,
                "umbralDisparo=" + umbralDisparo + " debe ser > potencialReposo=" + potencialReposo);
    }

    // ==================== Property 3: ConfiguracionRed válida ====================

    // Feature: genetic-algorithm-hyperparameters, Property 3: ConfiguracionRed válida
    /**
     * Para cualquier individuo producido por generarAleatorio(), la construcción de
     * ConfiguracionRed mediante construirConfiguracion() no debe lanzar excepción.
     *
     * <p><b>Validates: Requirements 1.7, 5.6</b></p>
     */
    @Property(tries = 100)
    void configuracionRedValida(@ForAll("semilla") long semilla) {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));
        Individuo individuo = fabrica.generarAleatorio();

        assertDoesNotThrow(
                () -> fabrica.construirConfiguracion(individuo.cromosoma()),
                "construirConfiguracion no debe lanzar excepción para individuos generados");
    }

    // ==================== Property 4: Límite topológico respetado ====================

    // Feature: genetic-algorithm-hyperparameters, Property 4: Límite topológico respetado
    /**
     * Para cualquier individuo producido por generarAleatorio(), el número total de
     * neuronas (entrada + ocultas + salida) no debe exceder el límite topológico.
     *
     * <p><b>Validates: Requirements 2.1, 2.2, 5.6</b></p>
     */
    @Property(tries = 100)
    void limiteTopologicoRespetado(@ForAll("semilla") long semilla) {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, new Random(semilla));
        Individuo individuo = fabrica.generarAleatorio();

        int total = individuo.cromosoma().neuronasTotal(DIMENSION_ENTRADA, DIMENSION_SALIDA);
        assertTrue(total <= LIMITE_TOPOLOGICO,
                "neuronasTotal=" + total + " excede limiteTopologico=" + LIMITE_TOPOLOGICO);
    }

    // ==================== Property 5: Reparación proporcional preserva el límite ====================

    // Feature: genetic-algorithm-hyperparameters, Property 5: Reparación proporcional preserva el límite
    /**
     * Para cualquier cromosoma cuyo número total de neuronas excede el límite topológico,
     * después de aplicar repararTopologia(), el total debe ser ≤ límite, cada capa oculta
     * debe tener al menos 1 neurona, y las proporciones relativas se mantienen aproximadamente.
     *
     * <p><b>Validates: Requirement 2.3</b></p>
     */
    @Property(tries = 100)
    void reparacionProporcionalPreservaLimite(
            @ForAll("capasOcultasExcedentes") int numCapas,
            @ForAll("semilla") long semilla) {

        Random rng = new Random(semilla);
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, rng);

        // Crear cromosoma que excede el límite con neuronas grandes
        int maxOcultas = LIMITE_TOPOLOGICO - DIMENSION_ENTRADA - DIMENSION_SALIDA;
        List<Gen<?>> topologia = new ArrayList<>();
        topologia.add(new Gen.GenEntero("capasOcultas", numCapas, 1, 10));

        int[] neuronasOriginales = new int[numCapas];
        for (int i = 0; i < numCapas; i++) {
            // Generar neuronas que garanticen exceder el límite
            int neuronas = (maxOcultas / numCapas) + 50 + rng.nextInt(100);
            neuronas = Math.min(neuronas, 512);
            neuronasOriginales[i] = neuronas;
            topologia.add(new Gen.GenEntero("neuronasCapa_" + i, neuronas, 1, 512));
        }

        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        bloques.put(BloqueFuncional.TOPOLOGIA, topologia);
        bloques.put(BloqueFuncional.LIF, crearGenesLIF());
        bloques.put(BloqueFuncional.STDP, crearGenesSTDP());
        bloques.put(BloqueFuncional.CODIFICACION, crearGenesCodificacion());
        bloques.put(BloqueFuncional.REGULACION, crearGenesRegulacion());
        Cromosoma cromosoma = new Cromosoma(bloques);

        // Verify it actually exceeds the limit before repair
        int totalAntes = cromosoma.neuronasTotal(DIMENSION_ENTRADA, DIMENSION_SALIDA);
        Assume.that(totalAntes > LIMITE_TOPOLOGICO);

        // Apply repair
        Cromosoma reparado = fabrica.repararTopologia(cromosoma);

        // 1. Total must be within limit
        int totalDespues = reparado.neuronasTotal(DIMENSION_ENTRADA, DIMENSION_SALIDA);
        assertTrue(totalDespues <= LIMITE_TOPOLOGICO,
                "neuronasTotal=" + totalDespues + " excede limiteTopologico=" + LIMITE_TOPOLOGICO);

        // 2. Each hidden layer must have at least 1 neuron
        List<Gen<?>> topReparada = reparado.genesDeBloque(BloqueFuncional.TOPOLOGIA);
        for (int i = 1; i <= numCapas; i++) {
            int neuronas = ((Number) topReparada.get(i).valor()).intValue();
            assertTrue(neuronas >= 1,
                    "Capa " + (i - 1) + " tiene " + neuronas + " neuronas, debe tener al menos 1");
        }

        // 3. Proportions approximately maintained (ratio original ± 1 neuron per rounding)
        if (numCapas >= 2) {
            int[] neuronasReparadas = new int[numCapas];
            for (int i = 0; i < numCapas; i++) {
                neuronasReparadas[i] = ((Number) topReparada.get(i + 1).valor()).intValue();
            }
            for (int i = 0; i < numCapas - 1; i++) {
                for (int j = i + 1; j < numCapas; j++) {
                    if (neuronasOriginales[j] > 0 && neuronasReparadas[j] > 0) {
                        double ratioOriginal = (double) neuronasOriginales[i] / neuronasOriginales[j];
                        double ratioReparado = (double) neuronasReparadas[i] / neuronasReparadas[j];
                        // Allow tolerance for rounding: proportions should be close
                        // With max(1, round(n*factor)), the ratio can differ by at most
                        // a factor related to rounding, so we use a generous tolerance
                        double tolerancia = 1.0 + (2.0 / Math.min(neuronasReparadas[i], neuronasReparadas[j]));
                        assertTrue(ratioReparado <= ratioOriginal * tolerancia + 0.01
                                        && ratioReparado >= ratioOriginal / tolerancia - 0.01,
                                "Proporción entre capas " + i + " y " + j + " no se mantiene: "
                                        + "original=" + ratioOriginal + " reparado=" + ratioReparado);
                    }
                }
            }
        }
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Long> semilla() {
        return Arbitraries.longs();
    }

    @Provide
    Arbitrary<Integer> capasOcultasExcedentes() {
        return Arbitraries.integers().between(1, 10);
    }

    // ==================== Helpers ====================

    private List<Gen<?>> crearGenesLIF() {
        return List.of(
                new Gen.GenReal("umbralDisparo", -50.0, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", -70.0, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", 20.0, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", 2, 1, 10)
        );
    }

    private List<Gen<?>> crearGenesSTDP() {
        return List.of(
                new Gen.GenReal("amplitudLTP", 0.01, 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", 0.012, 0.001, 0.1),
                new Gen.GenReal("tauLTP", 20.0, 5.0, 50.0),
                new Gen.GenReal("tauLTD", 20.0, 5.0, 50.0)
        );
    }

    private List<Gen<?>> crearGenesCodificacion() {
        return List.of(
                new Gen.GenReal("frecuenciaMaxima", 100.0, 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion",
                        ModoCodificacion.POISSON, ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", 50, 10, 200)
        );
    }

    private List<Gen<?>> crearGenesRegulacion() {
        return List.of(
                new Gen.GenBooleano("homeostasisActiva", true),
                new Gen.GenReal("tasaDisparoObjetivo", 10.0, 1.0, 50.0),
                new Gen.GenReal("tasaAjusteHomeostasis", 0.01, 0.001, 0.1),
                new Gen.GenBooleano("inhibicionLateralActiva", false),
                new Gen.GenEntero("radioInhibicion", 2, 1, 5),
                new Gen.GenReal("fuerzaInhibicion", 0.5, 0.1, 2.0)
        );
    }
}
