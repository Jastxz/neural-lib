package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link EvaluadorFitness}.
 */
class EvaluadorFitnessTest {

    private static final int LIMITE_TOPOLOGICO = 512;
    private EvaluadorFitness evaluador;

    @BeforeEach
    void setUp() {
        evaluador = new EvaluadorFitness(
                NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                0.5, 0.3, 0.2,
                LIMITE_TOPOLOGICO, 10, 1, 42L);
    }

    @Test
    void calcularFitness_valoresConocidos() {
        // precision = 0.8, costoEnergetico = 500.0, neuronasTotal = 100
        // precisionNorm = 0.8
        // energiaNorm = 1.0 - min(1.0, 500.0/1000.0) = 1.0 - 0.5 = 0.5
        // tamanioNorm = 1.0 - (100.0/512) = 1.0 - 0.1953125 = 0.8046875
        // fitness = 0.5*0.8 + 0.3*0.5 + 0.2*0.8046875
        //         = 0.4 + 0.15 + 0.1609375 = 0.7109375
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 98, 1}, 10, 1, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                config, 0.8, new double[]{0.5, 0.4, 0.3},
                200L, 100L, 10.0, 500.0, 1.0,
                80, 100, null);

        double fitness = evaluador.calcularFitness(resultado);

        assertEquals(0.7109375, fitness, 1e-10,
                "fitness con valores conocidos");
    }

    @Test
    void calcularFitness_precisionPerfecta() {
        // precision = 1.0, costoEnergetico = 0.0, neuronasTotal = 15 (minimal)
        // precisionNorm = 1.0
        // energiaNorm = 1.0 - 0.0 = 1.0
        // tamanioNorm = 1.0 - (15.0/512) ≈ 0.9707
        // fitness = 0.5*1.0 + 0.3*1.0 + 0.2*0.9707 ≈ 0.9941
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 12, 1}, 10, 1, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                config, 1.0, new double[]{0.1},
                50L, 10L, 2.0, 0.0, 0.1,
                15, 15, null);

        double fitness = evaluador.calcularFitness(resultado);

        double expectedTamanioNorm = 1.0 - (15.0 / 512.0);
        double expected = 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * expectedTamanioNorm;
        assertEquals(expected, fitness, 1e-10);
        assertTrue(fitness > 0.9, "fitness con precisión perfecta debe ser alto");
    }

    @Test
    void calcularFitness_ceroParaTimeout() {
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 10, 1}, 10, 1, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                config, 0.9, new double[]{0.1},
                100L, 50L, 5.0, 100.0, 0.5,
                10, 12, "timeout");

        double fitness = evaluador.calcularFitness(resultado);

        assertEquals(0.0, fitness, "fitness debe ser 0.0 para timeout");
    }

    @Test
    void calcularFitness_ceroParaLimiteNoSuperado() {
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 10, 1}, 10, 1, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                config, 0.3, new double[]{0.9, 0.8, 0.7},
                100L, 50L, 5.0, 100.0, 0.5,
                10, 12, "limite_no_superado");

        double fitness = evaluador.calcularFitness(resultado);

        assertEquals(0.0, fitness, "fitness debe ser 0.0 para limite_no_superado");
    }

    @Test
    void calcularFitness_costoEnergeticoAltoReduceFitness() {
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 10, 1}, 10, 1, 1, 42L);

        // Low energy cost
        ResultadoBenchmark resultadoBajo = new ResultadoBenchmark(
                config, 0.8, new double[]{0.1},
                100L, 50L, 5.0, 100.0, 0.5,
                10, 12, null);

        // High energy cost
        ResultadoBenchmark resultadoAlto = new ResultadoBenchmark(
                config, 0.8, new double[]{0.1},
                100L, 50L, 5.0, 900.0, 0.5,
                10, 12, null);

        double fitnessBajo = evaluador.calcularFitness(resultadoBajo);
        double fitnessAlto = evaluador.calcularFitness(resultadoAlto);

        assertTrue(fitnessBajo > fitnessAlto,
                "fitness con bajo costo energético (" + fitnessBajo
                + ") debe ser mayor que con alto costo (" + fitnessAlto + ")");
    }

    @Test
    void evaluarPoblacion_evaluaTodosLosIndividuos() {
        // This test verifies evaluarPoblacion processes all individuals
        // We use a minimal setup - the actual benchmark may fail but
        // the evaluator should handle exceptions gracefully
        java.util.Random rng = new java.util.Random(42);
        FabricaIndividuos fabrica = new FabricaIndividuos(2, 1, LIMITE_TOPOLOGICO, rng);

        java.util.List<Individuo> poblacion = new java.util.ArrayList<>();
        for (int i = 0; i < 3; i++) {
            poblacion.add(fabrica.generarAleatorio());
        }

        java.util.List<Individuo> evaluados = evaluador.evaluarPoblacion(poblacion);

        assertEquals(3, evaluados.size(), "debe evaluar todos los individuos");
        for (Individuo ind : evaluados) {
            assertTrue(ind.fitness() >= 0.0,
                    "fitness debe ser >= 0.0 después de evaluación");
        }
    }
}
