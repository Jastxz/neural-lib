package es.jastxz.nn.benchmark;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link ConfiguracionBenchmark}.
 */
class ConfiguracionBenchmarkTest {

    @Test
    void etiquetaDevuelveFormatoLegible() {
        var config = new ConfiguracionBenchmark(
            NivelComplejidad.BAJO, new int[]{10, 20, 9}, 100, 50, 3, 42L);
        assertEquals("3 en Raya | [10, 20, 9]", config.etiqueta());
    }

    @Test
    void constructorAceptaTopologiaCompatible() {
        assertDoesNotThrow(() -> new ConfiguracionBenchmark(
            NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 50, 25, 1, 1L));
    }

    @Test
    void constructorRechazaEntradaIncompatible() {
        var ex = assertThrows(IllegalArgumentException.class, () ->
            new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{5, 4, 1}, 50, 25, 1, 1L));
        assertTrue(ex.getMessage().contains("Dimensión de entrada incompatible"));
    }

    @Test
    void constructorRechazaSalidaIncompatible() {
        var ex = assertThrows(IllegalArgumentException.class, () ->
            new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 4, 3}, 50, 25, 1, 1L));
        assertTrue(ex.getMessage().contains("Dimensión de salida incompatible"));
    }

    @Test
    void topologiaEsCopiaDeFensiva() {
        int[] original = {10, 20, 9};
        var config = new ConfiguracionBenchmark(
            NivelComplejidad.BAJO, original, 100, 50, 3, 42L);
        original[1] = 999;
        assertEquals(20, config.topologia()[1],
            "La topología interna no debe verse afectada por cambios en el array original");
    }

    @Test
    void etiquetaParaCadaNivel() {
        var trivial = new ConfiguracionBenchmark(
            NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 5, 1, 0L);
        assertTrue(trivial.etiqueta().startsWith("Puertas Lógicas"));

        var alto = new ConfiguracionBenchmark(
            NivelComplejidad.ALTO, new int[]{32, 64, 32}, 10, 5, 1, 0L);
        assertTrue(alto.etiqueta().startsWith("Damas"));
    }
}
