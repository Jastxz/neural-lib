package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.*;
import net.jqwik.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link EvaluadorFitness}.
 *
 * <p>Verifica las propiedades 6 y 7 del diseño: cálculo de fitness como
 * combinación ponderada normalizada y fitness cero para benchmarks fallidos.</p>
 *
 * <p>Los tests verifican la lógica de cálculo directamente usando el método
 * package-private {@code calcularFitness()} para evitar ejecutar benchmarks reales.</p>
 */
class EvaluadorFitnessPropertyTest {

    private static final int LIMITE_TOPOLOGICO = 512;

    // ==================== Property 6: Fitness como combinación ponderada normalizada ====================

    // Feature: genetic-algorithm-hyperparameters, Property 6: Fitness como combinación ponderada normalizada
    /**
     * Para cualquier individuo con un ResultadoBenchmark válido (clasificación distinta
     * de "timeout" y "limite_no_superado"), el fitness debe ser igual a
     * pesoPrecision × precisionNorm + pesoEnergia × energiaNorm + pesoTamanio × tamanioNorm,
     * donde cada componente normalizado está en [0.0, 1.0].
     *
     * <p><b>Validates: Requirements 3.2, 3.3, 3.4</b></p>
     */
    @Property(tries = 100)
    void fitnessCombinacionPonderada(
            @ForAll("precisionFinal") double precisionFinal,
            @ForAll("costoEnergetico") double costoEnergetico,
            @ForAll("neuronasTotal") int neuronasTotal,
            @ForAll("pesosValidos") double[] pesos) {

        double pesoPrecision = pesos[0];
        double pesoEnergia = pesos[1];
        double pesoTamanio = pesos[2];

        EvaluadorFitness evaluador = new EvaluadorFitness(
                NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                pesoPrecision, pesoEnergia, pesoTamanio,
                LIMITE_TOPOLOGICO, 10, 1, 42L);

        // Create a valid ResultadoBenchmark (no failed classification)
        // Use TRIVIAL level with correct dimensions (entrada=2, salida=1)
        ConfiguracionBenchmark configBenchmark = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 10, 1}, 10, 1, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                configBenchmark, precisionFinal, new double[]{0.1},
                100L, 50L, 5.0, costoEnergetico, 0.5,
                neuronasTotal, neuronasTotal, null);

        double fitness = evaluador.calcularFitness(resultado);

        // Calculate expected components
        double precisionNorm = Math.max(0.0, Math.min(1.0, precisionFinal));
        double energiaNorm = 1.0 - Math.min(1.0, costoEnergetico / EvaluadorFitness.COSTO_MAX_REFERENCIA);
        energiaNorm = Math.max(0.0, Math.min(1.0, energiaNorm));
        double tamanioNorm = 1.0 - ((double) neuronasTotal / LIMITE_TOPOLOGICO);
        tamanioNorm = Math.max(0.0, Math.min(1.0, tamanioNorm));

        // Verify each component is in [0.0, 1.0]
        assertTrue(precisionNorm >= 0.0 && precisionNorm <= 1.0,
                "precisionNorm=" + precisionNorm + " fuera de [0.0, 1.0]");
        assertTrue(energiaNorm >= 0.0 && energiaNorm <= 1.0,
                "energiaNorm=" + energiaNorm + " fuera de [0.0, 1.0]");
        assertTrue(tamanioNorm >= 0.0 && tamanioNorm <= 1.0,
                "tamanioNorm=" + tamanioNorm + " fuera de [0.0, 1.0]");

        // Verify fitness equals weighted sum
        double expected = pesoPrecision * precisionNorm
                        + pesoEnergia * energiaNorm
                        + pesoTamanio * tamanioNorm;

        assertEquals(expected, fitness, 1e-10,
                "fitness=" + fitness + " no coincide con la combinación ponderada esperada=" + expected);

        // Fitness should be non-negative
        assertTrue(fitness >= 0.0, "fitness=" + fitness + " debe ser >= 0.0");
    }

    // ==================== Providers ====================

    // ==================== Property 7: Fitness cero para benchmarks fallidos ====================

    // Feature: genetic-algorithm-hyperparameters, Property 7: Fitness cero para benchmarks fallidos
    /**
     * Para cualquier individuo cuyo ResultadoBenchmark tiene clasificación "timeout"
     * o "limite_no_superado", el fitness asignado debe ser exactamente 0.0.
     *
     * <p><b>Validates: Requirements 3.5, 9.4</b></p>
     */
    @Property(tries = 100)
    void fitnessCeroParaFallidos(
            @ForAll("clasificacionFallida") String clasificacion,
            @ForAll("precisionFinal") double precisionFinal,
            @ForAll("costoEnergetico") double costoEnergetico,
            @ForAll("neuronasTotal") int neuronasTotal) {

        EvaluadorFitness evaluador = new EvaluadorFitness(
                NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                0.5, 0.3, 0.2,
                LIMITE_TOPOLOGICO, 10, 1, 42L);

        // Create ResultadoBenchmark with failed classification
        ConfiguracionBenchmark configBenchmark = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 10, 1}, 10, 1, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                configBenchmark, precisionFinal, new double[]{0.1},
                100L, 50L, 5.0, costoEnergetico, 0.5,
                neuronasTotal, neuronasTotal, clasificacion);

        double fitness = evaluador.calcularFitness(resultado);

        assertEquals(0.0, fitness,
                "fitness debe ser 0.0 para clasificación '" + clasificacion
                + "' pero fue " + fitness);
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<String> clasificacionFallida() {
        return Arbitraries.of("timeout", "limite_no_superado");
    }

    @Provide
    Arbitrary<double[]> pesosValidos() {
        // Generate three weights that sum to 1.0 with each >= 0.05
        return Arbitraries.integers().between(5, 90).flatMap(p1Int -> {
            int maxP2 = 100 - p1Int - 5;
            if (maxP2 < 5) maxP2 = 5;
            return Arbitraries.integers().between(5, maxP2).map(p2Int -> {
                double p1 = p1Int / 100.0;
                double p2 = p2Int / 100.0;
                double p3 = 1.0 - p1 - p2;
                return new double[]{p1, p2, p3};
            });
        });
    }

    @Provide
    Arbitrary<Double> precisionFinal() {
        return Arbitraries.doubles().between(0.0, 1.0);
    }

    @Provide
    Arbitrary<Double> costoEnergetico() {
        return Arbitraries.doubles().between(0.0, 2000.0);
    }

    @Provide
    Arbitrary<Integer> neuronasTotal() {
        return Arbitraries.integers().between(1, LIMITE_TOPOLOGICO);
    }
}
