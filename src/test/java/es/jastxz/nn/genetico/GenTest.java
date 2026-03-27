package es.jastxz.nn.genetico;

import es.jastxz.nn.genetico.Gen.GenBooleano;
import es.jastxz.nn.genetico.Gen.GenEntero;
import es.jastxz.nn.genetico.Gen.GenEnum;
import es.jastxz.nn.genetico.Gen.GenReal;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link Gen} y sus implementaciones.
 * Valida: Requisitos 1.1, 1.2, 1.3, 1.4, 1.5
 */
class GenTest {

    // ── GenEntero ────────────────────────────────────────────────────────

    @Nested
    class GenEnteroTest {

        @Test
        void construccionValidaDentroDeRango() {
            var gen = new GenEntero("capasOcultas", 5, 1, 10);
            assertEquals("capasOcultas", gen.nombre());
            assertEquals(5, gen.valor());
            assertEquals(1, gen.minimo());
            assertEquals(10, gen.maximo());
        }

        @Test
        void construccionEnLimitesDelRango() {
            assertDoesNotThrow(() -> new GenEntero("min", 1, 1, 10));
            assertDoesNotThrow(() -> new GenEntero("max", 10, 1, 10));
        }

        @Test
        void construccionFueraDeRangoLanzaExcepcion() {
            assertThrows(IllegalArgumentException.class,
                    () -> new GenEntero("capas", 0, 1, 10));
            assertThrows(IllegalArgumentException.class,
                    () -> new GenEntero("capas", 11, 1, 10));
        }

        @Test
        void construccionMinimoMayorQueMaximoLanzaExcepcion() {
            assertThrows(IllegalArgumentException.class,
                    () -> new GenEntero("inv", 5, 10, 1));
        }

        @Test
        void conValorAplicaClampingInferior() {
            var gen = new GenEntero("neuronas", 32, 1, 512);
            Gen<Integer> resultado = gen.conValor(-10);
            assertEquals(1, resultado.valor());
        }

        @Test
        void conValorAplicaClampingSuperior() {
            var gen = new GenEntero("neuronas", 32, 1, 512);
            Gen<Integer> resultado = gen.conValor(1000);
            assertEquals(512, resultado.valor());
        }

        @Test
        void conValorDentroDeRangoNoModifica() {
            var gen = new GenEntero("neuronas", 32, 1, 512);
            Gen<Integer> resultado = gen.conValor(100);
            assertEquals(100, resultado.valor());
        }

        @Test
        void conValorPreservaNombreYRango() {
            var gen = new GenEntero("dur", 3, 1, 10);
            Gen<Integer> resultado = gen.conValor(7);
            assertEquals("dur", resultado.nombre());
            assertEquals(1, resultado.minimo());
            assertEquals(10, resultado.maximo());
        }
    }

    // ── GenReal ─────────────────────────────────────────────────────────

    @Nested
    class GenRealTest {

        @Test
        void construccionValidaDentroDeRango() {
            var gen = new GenReal("umbralDisparo", -55.0, -60.0, -40.0);
            assertEquals("umbralDisparo", gen.nombre());
            assertEquals(-55.0, gen.valor());
            assertEquals(-60.0, gen.minimo());
            assertEquals(-40.0, gen.maximo());
        }

        @Test
        void construccionEnLimitesDelRango() {
            assertDoesNotThrow(() -> new GenReal("min", -60.0, -60.0, -40.0));
            assertDoesNotThrow(() -> new GenReal("max", -40.0, -60.0, -40.0));
        }

        @Test
        void construccionFueraDeRangoLanzaExcepcion() {
            assertThrows(IllegalArgumentException.class,
                    () -> new GenReal("umbral", -61.0, -60.0, -40.0));
            assertThrows(IllegalArgumentException.class,
                    () -> new GenReal("umbral", -39.0, -60.0, -40.0));
        }

        @Test
        void construccionMinimoMayorQueMaximoLanzaExcepcion() {
            assertThrows(IllegalArgumentException.class,
                    () -> new GenReal("inv", 0.0, 10.0, 1.0));
        }

        @Test
        void conValorAplicaClampingInferior() {
            var gen = new GenReal("tau", 20.0, 5.0, 50.0);
            Gen<Double> resultado = gen.conValor(1.0);
            assertEquals(5.0, resultado.valor());
        }

        @Test
        void conValorAplicaClampingSuperior() {
            var gen = new GenReal("tau", 20.0, 5.0, 50.0);
            Gen<Double> resultado = gen.conValor(100.0);
            assertEquals(50.0, resultado.valor());
        }

        @Test
        void conValorDentroDeRangoNoModifica() {
            var gen = new GenReal("freq", 100.0, 10.0, 500.0);
            Gen<Double> resultado = gen.conValor(250.0);
            assertEquals(250.0, resultado.valor(), 1e-9);
        }

        @Test
        void conValorPreservaNombreYRango() {
            var gen = new GenReal("amp", 0.01, 0.001, 0.1);
            Gen<Double> resultado = gen.conValor(0.05);
            assertEquals("amp", resultado.nombre());
            assertEquals(0.001, resultado.minimo());
            assertEquals(0.1, resultado.maximo());
        }
    }

    // ── GenBooleano ─────────────────────────────────────────────────────

    @Nested
    class GenBooleanoTest {

        @Test
        void construccionValida() {
            var genTrue = new GenBooleano("homeostasis", true);
            assertEquals("homeostasis", genTrue.nombre());
            assertTrue(genTrue.valor());

            var genFalse = new GenBooleano("inhibicion", false);
            assertFalse(genFalse.valor());
        }

        @Test
        void minimoEsFalseMaximoEsTrue() {
            var gen = new GenBooleano("flag", true);
            assertFalse(gen.minimo());
            assertTrue(gen.maximo());
        }

        @Test
        void conValorCambiaValor() {
            var gen = new GenBooleano("homeostasis", true);
            Gen<Boolean> resultado = gen.conValor(false);
            assertFalse(resultado.valor());
        }

        @Test
        void conValorPreservaNombre() {
            var gen = new GenBooleano("flag", false);
            Gen<Boolean> resultado = gen.conValor(true);
            assertEquals("flag", resultado.nombre());
        }
    }

    // ── GenEnum ─────────────────────────────────────────────────────────

    @Nested
    class GenEnumTest {

        @Test
        void construccionValida() {
            var gen = new GenEnum<>("modo", ModoCodificacion.POISSON, ModoCodificacion.class);
            assertEquals("modo", gen.nombre());
            assertEquals(ModoCodificacion.POISSON, gen.valor());
        }

        @Test
        void minimoEsPrimerValorMaximoEsUltimo() {
            var gen = new GenEnum<>("modo", ModoCodificacion.REGULAR, ModoCodificacion.class);
            assertEquals(ModoCodificacion.POISSON, gen.minimo());
            assertEquals(ModoCodificacion.BURST, gen.maximo());
        }

        @Test
        void conValorCambiaValor() {
            var gen = new GenEnum<>("modo", ModoCodificacion.POISSON, ModoCodificacion.class);
            Gen<ModoCodificacion> resultado = gen.conValor(ModoCodificacion.BURST);
            assertEquals(ModoCodificacion.BURST, resultado.valor());
        }

        @Test
        void conValorPreservaNombreYTipo() {
            var gen = new GenEnum<>("modo", ModoCodificacion.POISSON, ModoCodificacion.class);
            var resultado = (GenEnum<ModoCodificacion>) gen.conValor(ModoCodificacion.REGULAR);
            assertEquals("modo", resultado.nombre());
            assertEquals(ModoCodificacion.class, resultado.tipoEnum());
        }
    }
}
