package es.jastxz.nn.benchmark;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link ResultadoAgregado}.
 */
class ResultadoAgregadoTest {

    private static final ConfiguracionBenchmark CONFIG = new ConfiguracionBenchmark(
        NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 25, 3, 42L);

    private static ResultadoBenchmark resultado(double precision, long tiempoMs,
                                                 double costo, String clasificacion) {
        return new ResultadoBenchmark(
            CONFIG, precision, new double[]{0.5, 0.3}, tiempoMs,
            100L, 0.5, costo, 0.1, 3, 5, clasificacion);
    }

    @Test
    void agregarConUnSoloResultado() {
        var r = resultado(0.8, 100, 2.0, null);
        var agregado = ResultadoAgregado.agregar(List.of(r));

        assertEquals(0.8, agregado.mediaPrecision(), 1e-9);
        assertEquals(0.0, agregado.desvPrecision(), 1e-9);
        assertEquals(100.0, agregado.mediaTiempo(), 1e-9);
        assertEquals(0.0, agregado.desvTiempo(), 1e-9);
        assertEquals(2.0, agregado.mediaCostoEnergetico(), 1e-9);
        assertEquals(0.0, agregado.desvCostoEnergetico(), 1e-9);
        assertEquals(0.4, agregado.ratioEficiencia(), 1e-9);
        assertNull(agregado.clasificacionMayoritaria());
    }

    @Test
    void agregarCalculaMediaCorrectamente() {
        var resultados = List.of(
            resultado(0.6, 100, 1.0, null),
            resultado(0.8, 200, 3.0, null),
            resultado(1.0, 300, 2.0, null));
        var agregado = ResultadoAgregado.agregar(resultados);

        assertEquals(0.8, agregado.mediaPrecision(), 1e-9);
        assertEquals(200.0, agregado.mediaTiempo(), 1e-9);
        assertEquals(2.0, agregado.mediaCostoEnergetico(), 1e-9);
    }

    @Test
    void agregarCalculaDesviacionEstandarMuestral() {
        var resultados = List.of(
            resultado(0.6, 100, 1.0, null),
            resultado(0.8, 200, 3.0, null),
            resultado(1.0, 300, 2.0, null));
        var agregado = ResultadoAgregado.agregar(resultados);

        // Desviación estándar muestral (N-1) para precisión: [0.6, 0.8, 1.0], media=0.8
        // varianza = ((0.04 + 0 + 0.04) / 2) = 0.04, desv = 0.2
        assertEquals(0.2, agregado.desvPrecision(), 1e-9);
    }

    @Test
    void ratioEficienciaConCostoCero() {
        var r = resultado(0.9, 100, 0.0, null);
        var agregado = ResultadoAgregado.agregar(List.of(r));

        assertEquals(Double.POSITIVE_INFINITY, agregado.ratioEficiencia());
    }

    @Test
    void clasificacionMayoritariaConVotosMixtos() {
        var resultados = List.of(
            resultado(0.3, 100, 1.0, "limite_no_superado"),
            resultado(0.4, 200, 2.0, "limite_no_superado"),
            resultado(0.5, 300, 3.0, "convergencia_estancada"));
        var agregado = ResultadoAgregado.agregar(resultados);

        assertEquals("limite_no_superado", agregado.clasificacionMayoritaria());
    }

    @Test
    void clasificacionMayoritariaTodosNull() {
        var resultados = List.of(
            resultado(0.9, 100, 1.0, null),
            resultado(0.8, 200, 2.0, null));
        var agregado = ResultadoAgregado.agregar(resultados);

        assertNull(agregado.clasificacionMayoritaria());
    }

    @Test
    void agregarConListaNulaLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class,
            () -> ResultadoAgregado.agregar(null));
    }

    @Test
    void agregarConListaVaciaLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class,
            () -> ResultadoAgregado.agregar(List.of()));
    }

    @Test
    void agregarPreservaConfiguracion() {
        var r = resultado(0.7, 150, 1.5, null);
        var agregado = ResultadoAgregado.agregar(List.of(r));

        assertSame(CONFIG, agregado.configuracion());
    }
}
