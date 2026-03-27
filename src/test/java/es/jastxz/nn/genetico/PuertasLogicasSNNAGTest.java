package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.OptimizadorSNN;
import es.jastxz.nn.spiking.ResultadoOptimizacion;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de integración que verifica que {@link OptimizadorSNN} resuelve
 * los problemas lógicos OR, AND y XOR usando AG + SNN en dos fases.
 *
 * <p>Fase 1: optimiza hiperparámetros (LIF, STDP, codificación, regulación)
 * con topología fija al máximo. Fase 2: optimiza topología con los mejores
 * hiperparámetros encontrados.</p>
 */
class PuertasLogicasSNNAGTest {

    /** Número de intentos con semillas aleatorias distintas. */
    private static final int NUM_INTENTOS = 10;

    /** Precisión mínima exigida para OR y AND (linealmente separables). */
    private static final double PRECISION_MINIMA_LINEAL = 1.0;

    /** Precisión mínima exigida para XOR (no linealmente separable). */
    private static final double PRECISION_MINIMA_XOR = 1.0;

    // ==================== Datos de entrenamiento ====================

    private static final double[][] INPUTS = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };

    private static final double[][] TARGETS_OR  = {{0}, {1}, {1}, {1}};
    private static final double[][] TARGETS_AND = {{0}, {0}, {0}, {1}};
    private static final double[][] TARGETS_XOR = {{0}, {1}, {1}, {0}};

    // ==================== Helpers ====================

    private ResultadoOptimizacion buscarMejorSolucion(double[][] targets, String nombre) {
        ResultadoOptimizacion mejorResultado = null;

        for (int i = 0; i < NUM_INTENTOS; i++) {
            ResultadoOptimizacion resultado = new OptimizadorSNN.Builder()
                    .tamañoPoblacion(30)
                    .maxGeneraciones(50)
                    .generacionesEstancamiento(10)
                    .limiteTopologico(32)
                    .pesoPrecision(0.9)
                    .pesoEnergia(0.0)
                    .pesoTamanio(0.1)
                    .duracionTimesteps(100)
                    .epocasEntrenamiento(10)
                    .numIntentos(1)
                    .optimizar(INPUTS, targets);

            System.out.printf("[%s] Intento %d → precisión=%.0f%% fitness=%.4f topología=%s%n",
                    nombre, i + 1, resultado.precision() * 100, resultado.fitness(),
                    Arrays.toString(resultado.configuracionRed().getTopologia()));

            if (mejorResultado == null || resultado.precision() > mejorResultado.precision()) {
                mejorResultado = resultado;
            }

            if (resultado.precision() >= 1.0) break;
        }

        return mejorResultado;
    }

    // ==================== Tests ====================

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_resuelvePuertaOR() {
        ResultadoOptimizacion resultado = buscarMejorSolucion(TARGETS_OR, "OR");

        System.out.printf("==> OR: precisión=%.0f%% topología=%s%n",
                resultado.precision() * 100,
                Arrays.toString(resultado.configuracionRed().getTopologia()));

        assertTrue(resultado.precision() >= PRECISION_MINIMA_LINEAL,
                "OR debe alcanzar al menos " + (PRECISION_MINIMA_LINEAL * 100)
                        + "% de precisión, obtuvo: " + (resultado.precision() * 100) + "%");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_resuelvePuertaAND() {
        ResultadoOptimizacion resultado = buscarMejorSolucion(TARGETS_AND, "AND");

        System.out.printf("==> AND: precisión=%.0f%% topología=%s%n",
                resultado.precision() * 100,
                Arrays.toString(resultado.configuracionRed().getTopologia()));

        assertTrue(resultado.precision() >= PRECISION_MINIMA_LINEAL,
                "AND debe alcanzar al menos " + (PRECISION_MINIMA_LINEAL * 100)
                        + "% de precisión, obtuvo: " + (resultado.precision() * 100) + "%");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_resuelvePuertaXOR() {
        ResultadoOptimizacion resultado = buscarMejorSolucion(TARGETS_XOR, "XOR");

        System.out.printf("==> XOR: precisión=%.0f%% topología=%s%n",
                resultado.precision() * 100,
                Arrays.toString(resultado.configuracionRed().getTopologia()));

        assertTrue(resultado.precision() >= PRECISION_MINIMA_XOR,
                "XOR debe alcanzar al menos " + (PRECISION_MINIMA_XOR * 100)
                        + "% de precisión, obtuvo: " + (resultado.precision() * 100) + "%");
    }
}
