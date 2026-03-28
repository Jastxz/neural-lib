package es.jastxz.nn.spiking.integration;

import es.jastxz.nn.spiking.OptimizadorSNN;
import es.jastxz.nn.spiking.RedNeuralSpiking;
import es.jastxz.nn.spiking.ResultadoOptimizacion;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de integración para {@link OptimizadorSNN}.
 *
 * <p>Verifica que la fachada de optimización automática funciona
 * correctamente con la API mínima y con configuración personalizada.</p>
 */
class OptimizadorSNNTest {

    private static final double[][] INPUTS = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    private static final double[][] TARGETS_OR = {{0}, {1}, {1}, {1}};

    /**
     * Verifica que la API mínima (una sola línea) produce un resultado válido.
     * No exige 100% de precisión — solo que el flujo completo funcione.
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void apiMinima_produceResultadoValido() {
        ResultadoOptimizacion resultado = new OptimizadorSNN.Builder()
                .tamañoPoblacion(10)
                .maxGeneraciones(5)
                .numIntentos(1)
                .epocasEntrenamiento(3)
                .semilla(42L)
                .optimizar(INPUTS, TARGETS_OR);

        assertNotNull(resultado, "El resultado no debe ser null");
        assertNotNull(resultado.red(), "La red no debe ser null");
        assertNotNull(resultado.configuracionRed(), "La configuración no debe ser null");
        assertNotNull(resultado.informeEvolucion(), "El informe no debe ser null");
        assertTrue(resultado.precision() >= 0.0 && resultado.precision() <= 1.0,
                "La precisión debe estar en [0, 1]: " + resultado.precision());
        assertTrue(resultado.fitness() >= 0.0,
                "El fitness debe ser >= 0: " + resultado.fitness());

        // La red devuelta debe poder procesar entradas
        RedNeuralSpiking red = resultado.red();
        red.setModoEntrenamiento(false);
        red.resetearEstadoTemporal();
        double[] salida = red.procesar(new double[]{1.0, 0.0}, 100);
        assertNotNull(salida, "La salida de procesar no debe ser null");
        assertEquals(1, salida.length, "La salida debe tener 1 elemento");
    }

    /**
     * Verifica que datos inválidos lanzan excepciones claras.
     */
    @Test
    void datosInvalidos_lanzanExcepcion() {
        assertThrows(IllegalArgumentException.class,
                () -> OptimizadorSNN.optimizar(null, TARGETS_OR));
        assertThrows(IllegalArgumentException.class,
                () -> OptimizadorSNN.optimizar(INPUTS, null));
        assertThrows(IllegalArgumentException.class,
                () -> OptimizadorSNN.optimizar(new double[0][0], TARGETS_OR));
        assertThrows(IllegalArgumentException.class,
                () -> OptimizadorSNN.optimizar(INPUTS, new double[][]{{0}, {1}}));
    }

    /**
     * Verifica que el resultado incluye un informe de evolución imprimible.
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void resultado_tieneInformeImprimible() {
        ResultadoOptimizacion resultado = new OptimizadorSNN.Builder()
                .tamañoPoblacion(6)
                .maxGeneraciones(3)
                .numIntentos(1)
                .epocasEntrenamiento(2)
                .semilla(123L)
                .optimizar(INPUTS, TARGETS_OR);

        // No debe lanzar excepción
        assertDoesNotThrow(resultado::imprimir);
        assertDoesNotThrow(() -> resultado.informeEvolucion().imprimir());
    }
}
