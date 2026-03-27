package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de propiedad jqwik para la completitud y validez de métricas
 * generadas por {@link RecolectorMetricas}.
 *
 * <p><b>Validates: Requisitos 1.1, 1.2, 3.1, 3.2</b></p>
 *
 * <p>Usa exclusivamente {@link NivelComplejidad#TRIVIAL} con datos de puerta
 * lógica AND para mantener la ejecución rápida, ya que cada intento implica
 * entrenamiento real de la SNN.</p>
 *
 * @since 1.1
 */
class RecolectorMetricasPropertyTest {

    // Feature: snn-benchmark-suite, Property 3: Completitud y validez de métricas en resultados

    /** Datos de entrada de la puerta AND: [0,0], [0,1], [1,0], [1,1]. */
    private static final double[][] AND_INPUTS = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    /** Salidas esperadas de la puerta AND: 0, 0, 0, 1. */
    private static final double[][] AND_TARGETS = {
        {0}, {0}, {0}, {1}
    };

    /**
     * Propiedad: para cualquier configuración TRIVIAL válida con topología pequeña,
     * el {@link ResultadoBenchmark} generado por {@link RecolectorMetricas} debe
     * contener métricas completas y dentro de rangos válidos.
     *
     * <p>Verifica:</p>
     * <ul>
     *   <li>precisionFinal ∈ [0.0, 1.0]</li>
     *   <li>errorMSEPorEpoca no vacío y longitud == config.epocas()</li>
     *   <li>tiempoEntrenamientoMs &ge; 0</li>
     *   <li>totalSpikes &ge; 0</li>
     *   <li>tasaDisparoPromedio &ge; 0</li>
     *   <li>costoEnergetico &ge; 0</li>
     *   <li>dispersionActividad &ge; 0</li>
     *   <li>neuronasActivas ∈ [0, neuronasTotal]</li>
     *   <li>configuracion del resultado coincide con la configuración de entrada</li>
     * </ul>
     */
    @Property(tries = 10)
    void metricasCompletasYValidas(
            @ForAll("topologiaTrivial") int[] topologia,
            @ForAll("epocasPequenas") int epocas,
            @ForAll("duracionCorta") int duracion,
            @ForAll("semillaAleatoria") long semilla) {

        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, topologia, epocas, duracion, 1, semilla);

        RecolectorMetricas recolector = new RecolectorMetricas();
        ResultadoBenchmark resultado = recolector.ejecutarYRecolectar(config, AND_INPUTS, AND_TARGETS);

        // 1. precisionFinal in [0.0, 1.0]
        assertTrue(resultado.precisionFinal() >= 0.0 && resultado.precisionFinal() <= 1.0,
                "precisionFinal debe estar en [0.0, 1.0], fue: " + resultado.precisionFinal());

        // 2. errorMSEPorEpoca not empty and length == config.epocas()
        assertNotNull(resultado.errorMSEPorEpoca(), "errorMSEPorEpoca no debe ser null");
        assertEquals(epocas, resultado.errorMSEPorEpoca().length,
                "errorMSEPorEpoca debe tener longitud == epocas");

        // 3. tiempoEntrenamientoMs >= 0
        assertTrue(resultado.tiempoEntrenamientoMs() >= 0,
                "tiempoEntrenamientoMs debe ser >= 0, fue: " + resultado.tiempoEntrenamientoMs());

        // 4. totalSpikes >= 0
        assertTrue(resultado.totalSpikes() >= 0,
                "totalSpikes debe ser >= 0, fue: " + resultado.totalSpikes());

        // 5. tasaDisparoPromedio >= 0
        assertTrue(resultado.tasaDisparoPromedio() >= 0,
                "tasaDisparoPromedio debe ser >= 0, fue: " + resultado.tasaDisparoPromedio());

        // 6. costoEnergetico >= 0
        assertTrue(resultado.costoEnergetico() >= 0,
                "costoEnergetico debe ser >= 0, fue: " + resultado.costoEnergetico());

        // 7. dispersionActividad >= 0
        assertTrue(resultado.dispersionActividad() >= 0,
                "dispersionActividad debe ser >= 0, fue: " + resultado.dispersionActividad());

        // 8. neuronasActivas in [0, neuronasTotal]
        assertTrue(resultado.neuronasActivas() >= 0,
                "neuronasActivas debe ser >= 0, fue: " + resultado.neuronasActivas());
        assertTrue(resultado.neuronasActivas() <= resultado.neuronasTotal(),
                "neuronasActivas (" + resultado.neuronasActivas()
                + ") debe ser <= neuronasTotal (" + resultado.neuronasTotal() + ")");

        // 9. configuracion in the result matches the input config
        assertSame(config, resultado.configuracion(),
                "La configuración del resultado debe ser la misma instancia que la de entrada");
        assertArrayEquals(topologia, resultado.configuracion().topologia(),
                "La topología en el resultado debe coincidir con la configuración de entrada");
        assertEquals(epocas, resultado.configuracion().epocas(),
                "Las épocas en el resultado deben coincidir con la configuración de entrada");
    }

    // Feature: snn-benchmark-suite, Property 4: Seguimiento de MSE por época

    /**
     * Propiedad: para cualquier ejecución de benchmark configurada con N épocas,
     * el array {@code errorMSEPorEpoca} del resultado debe tener exactamente N
     * elementos, y cada elemento debe ser un valor no negativo.
     *
     * <p><b>Validates: Requisito 3.4</b></p>
     */
    @Property(tries = 10)
    void seguimientoMSEPorEpoca(
            @ForAll("topologiaTrivial") int[] topologia,
            @ForAll("epocasParaMSE") int epocas,
            @ForAll("duracionCorta") int duracion,
            @ForAll("semillaAleatoria") long semilla) {

        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, topologia, epocas, duracion, 1, semilla);

        RecolectorMetricas recolector = new RecolectorMetricas();
        ResultadoBenchmark resultado = recolector.ejecutarYRecolectar(config, AND_INPUTS, AND_TARGETS);

        // 1. errorMSEPorEpoca must have exactly N elements
        assertNotNull(resultado.errorMSEPorEpoca(),
                "errorMSEPorEpoca no debe ser null");
        assertEquals(epocas, resultado.errorMSEPorEpoca().length,
                "errorMSEPorEpoca debe tener exactamente " + epocas + " elementos");

        // 2. Each MSE value must be non-negative
        for (int i = 0; i < resultado.errorMSEPorEpoca().length; i++) {
            assertTrue(resultado.errorMSEPorEpoca()[i] >= 0.0,
                    "MSE en época " + i + " debe ser >= 0, fue: " + resultado.errorMSEPorEpoca()[i]);
        }
    }

    // Feature: snn-benchmark-suite, Property 5: Asociación configuración-resultado

    /**
     * Propiedad: para cualquier {@link ResultadoBenchmark}, la configuración
     * almacenada en el resultado debe ser idéntica a la configuración usada
     * para ejecutar el benchmark (misma topología, mismo nivel, mismos parámetros).
     *
     * <p><b>Validates: Requisito 3.3</b></p>
     */
    @Property(tries = 10)
    void asociacionConfiguracionResultado(
            @ForAll("topologiaTrivial") int[] topologia,
            @ForAll("epocasPequenas") int epocas,
            @ForAll("duracionCorta") int duracion,
            @ForAll("semillaAleatoria") long semilla) {

        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, topologia, epocas, duracion, 1, semilla);

        RecolectorMetricas recolector = new RecolectorMetricas();
        ResultadoBenchmark resultado = recolector.ejecutarYRecolectar(config, AND_INPUTS, AND_TARGETS);

        // 1. The result's configuracion must be the same object as the input config
        assertSame(config, resultado.configuracion(),
                "La configuración del resultado debe ser la misma instancia que la de entrada");

        // 2. Verify all config fields match
        assertEquals(NivelComplejidad.TRIVIAL, resultado.configuracion().nivel(),
                "El nivel debe coincidir con la configuración de entrada");
        assertArrayEquals(topologia, resultado.configuracion().topologia(),
                "La topología debe coincidir con la configuración de entrada");
        assertEquals(epocas, resultado.configuracion().epocas(),
                "Las épocas deben coincidir con la configuración de entrada");
        assertEquals(duracion, resultado.configuracion().duracionTimesteps(),
                "La duración de timesteps debe coincidir con la configuración de entrada");
        assertEquals(1, resultado.configuracion().repeticiones(),
                "Las repeticiones deben coincidir con la configuración de entrada");
        assertEquals(semilla, resultado.configuracion().semilla(),
                "La semilla debe coincidir con la configuración de entrada");
    }



    // --- Proveedores (Arbitraries) ---

    /**
     * Genera topologías pequeñas compatibles con TRIVIAL (entrada=2, salida=1).
     * Topologías: [2, h1, 1] o [2, h1, h2, 1] con capas ocultas de 2-8 neuronas.
     */
    @Provide
    Arbitrary<int[]> topologiaTrivial() {
        Arbitrary<int[]> unaCapaOculta = Arbitraries.integers().between(2, 8)
                .map(h -> new int[]{2, h, 1});

        Arbitrary<int[]> dosCapasOcultas = Arbitraries.integers().between(2, 8)
                .flatMap(h1 -> Arbitraries.integers().between(2, 8)
                        .map(h2 -> new int[]{2, h1, h2, 1}));

        return Arbitraries.oneOf(unaCapaOculta, dosCapasOcultas);
    }

    /** Genera número de épocas entre 1 y 5. */
    @Provide
    Arbitrary<Integer> epocasPequenas() {
        return Arbitraries.integers().between(1, 5);
    }

    /** Genera número de épocas entre 1 y 10 para test de seguimiento MSE. */
    @Provide
    Arbitrary<Integer> epocasParaMSE() {
        return Arbitraries.integers().between(1, 10);
    }


    /** Genera duración de timesteps entre 5 y 15. */
    @Provide
    Arbitrary<Integer> duracionCorta() {
        return Arbitraries.integers().between(5, 15);
    }

    /** Genera semillas aleatorias. */
    @Provide
    Arbitrary<Long> semillaAleatoria() {
        return Arbitraries.longs().between(1L, 10_000L);
    }
}
