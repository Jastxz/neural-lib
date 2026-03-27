package es.jastxz.nn.benchmark;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link DetectorLimites}.
 */
class DetectorLimitesTest {

    private static final ConfiguracionBenchmark CONFIG_TRIVIAL =
        new ConfiguracionBenchmark(NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 50, 1, 42L);

    // --- Helper para crear ResultadoBenchmark con valores específicos ---

    private static ResultadoBenchmark resultado(double precision, double[] mse,
                                                 int neuronasActivas, int neuronasTotal,
                                                 double costoEnergetico) {
        return new ResultadoBenchmark(
            CONFIG_TRIVIAL, precision, mse, 100L, 50L, 0.5, costoEnergetico, 0.1,
            neuronasActivas, neuronasTotal, null
        );
    }

    @Nested
    @DisplayName("clasificar()")
    class Clasificar {

        @Test
        @DisplayName("Precisión < 0.6 → limite_no_superado")
        void precisionBaja() {
            var r = resultado(0.59, new double[]{1.0, 0.5}, 10, 20, 1.0);
            assertEquals("limite_no_superado", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Precisión = 0.0 → limite_no_superado")
        void precisionCero() {
            var r = resultado(0.0, new double[]{1.0}, 10, 20, 1.0);
            assertEquals("limite_no_superado", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Precisión >= 0.6 no dispara limite_no_superado")
        void precisionSuficiente() {
            var r = resultado(0.6, new double[]{1.0, 0.5}, 10, 20, 1.0);
            assertNull(DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("MSE estancado en 3 pares consecutivos → convergencia_estancada")
        void mseEstancado() {
            // 4 épocas, MSE constante → 3 pares con mejora 0% (<=1%)
            var r = resultado(0.8, new double[]{1.0, 1.0, 1.0, 1.0}, 10, 20, 1.0);
            assertEquals("convergencia_estancada", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("MSE con mejora mínima (<=1%) en 3 pares → convergencia_estancada")
        void mseMejoraInsuficiente() {
            // Mejora de exactamente 1% o menos en cada par
            double[] mse = {1.0, 0.991, 0.982, 0.974};
            // Par 0: (1.0-0.991)/1.0 = 0.009 <= 0.01 ✓
            // Par 1: (0.991-0.982)/0.991 ≈ 0.00908 <= 0.01 ✓
            // Par 2: (0.982-0.974)/0.982 ≈ 0.00815 <= 0.01 ✓
            var r = resultado(0.8, mse, 10, 20, 1.0);
            assertEquals("convergencia_estancada", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("MSE con buena mejora no dispara convergencia_estancada")
        void mseMejorando() {
            // Mejora >1% en cada par
            double[] mse = {1.0, 0.8, 0.6, 0.4};
            var r = resultado(0.8, mse, 10, 20, 1.0);
            assertNull(DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Menos de 4 épocas no dispara convergencia_estancada")
        void pocasEpocas() {
            var r = resultado(0.8, new double[]{1.0, 1.0, 1.0}, 10, 20, 1.0);
            assertNull(DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Neuronas activas < 20% → red_infrautilizada")
        void redInfrautilizada() {
            // 1 de 10 = 10% < 20%
            var r = resultado(0.8, new double[]{1.0, 0.5}, 1, 10, 1.0);
            assertEquals("red_infrautilizada", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Neuronas activas = 0 → red_infrautilizada")
        void sinNeuronasActivas() {
            var r = resultado(0.8, new double[]{1.0, 0.5}, 0, 10, 1.0);
            assertEquals("red_infrautilizada", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Neuronas activas >= 20% no dispara red_infrautilizada")
        void neuronasActivas() {
            // 2 de 10 = 20%, no es < 20%
            var r = resultado(0.8, new double[]{1.0, 0.5}, 2, 10, 1.0);
            assertNull(DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Prioridad: precisión se evalúa antes que MSE")
        void prioridadPrecisionSobreMse() {
            // Precisión baja Y MSE estancado → debe retornar limite_no_superado
            var r = resultado(0.3, new double[]{1.0, 1.0, 1.0, 1.0}, 10, 20, 1.0);
            assertEquals("limite_no_superado", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Prioridad: MSE se evalúa antes que neuronas")
        void prioridadMseSobreNeuronas() {
            // MSE estancado Y neuronas infrautilizadas → debe retornar convergencia_estancada
            var r = resultado(0.8, new double[]{1.0, 1.0, 1.0, 1.0}, 1, 10, 1.0);
            assertEquals("convergencia_estancada", DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("Sin límites detectados → null")
        void sinLimites() {
            var r = resultado(0.9, new double[]{1.0, 0.5}, 10, 20, 1.0);
            assertNull(DetectorLimites.clasificar(r));
        }

        @Test
        @DisplayName("neuronasTotal = 0 no dispara red_infrautilizada")
        void neuronaTotalCero() {
            var r = resultado(0.8, new double[]{1.0, 0.5}, 0, 0, 1.0);
            assertNull(DetectorLimites.clasificar(r));
        }
    }

    @Nested
    @DisplayName("clasificarIneficiencia()")
    class ClasificarIneficiencia {

        @Test
        @DisplayName("Costo > 10x mínimo → ineficiencia_energetica")
        void costoExcesivo() {
            var r = resultado(0.8, new double[]{1.0}, 10, 20, 101.0);
            assertEquals("ineficiencia_energetica",
                DetectorLimites.clasificarIneficiencia(r, 10.0));
        }

        @Test
        @DisplayName("Costo = 10x mínimo no dispara ineficiencia")
        void costoExacto() {
            var r = resultado(0.8, new double[]{1.0}, 10, 20, 100.0);
            assertNull(DetectorLimites.clasificarIneficiencia(r, 10.0));
        }

        @Test
        @DisplayName("Costo < 10x mínimo → null")
        void costoAceptable() {
            var r = resultado(0.8, new double[]{1.0}, 10, 20, 50.0);
            assertNull(DetectorLimites.clasificarIneficiencia(r, 10.0));
        }

        @Test
        @DisplayName("Costo mínimo = 0 y costo > 0 → ineficiencia")
        void costoMinimoCero() {
            var r = resultado(0.8, new double[]{1.0}, 10, 20, 1.0);
            assertEquals("ineficiencia_energetica",
                DetectorLimites.clasificarIneficiencia(r, 0.0));
        }
    }
}
