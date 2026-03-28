package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para la validación de compatibilidad
 * topología-problema en {@link ConfiguracionBenchmark}.
 *
 * <p><b>Validates: Requisito 2.3</b></p>
 *
 * @since 1.1
 */
class ConfiguracionBenchmarkPropertyTest {

    // Feature: snn-benchmark-suite, Property 2: Validación de compatibilidad topología-problema

    /**
     * Propiedad: configuraciones válidas (donde topologia[0] == nivel.dimensionEntrada
     * y topologia[last] == nivel.dimensionSalida) son aceptadas sin excepción.
     */
    @Property(tries = 100)
    void configuracionesValidasSonAceptadas(
            @ForAll("nivelAleatorio") NivelComplejidad nivel,
            @ForAll("capasOcultasAleatorias") int[] capasOcultas) {

        int entrada = nivel.getDimensionEntrada();
        int salida = nivel.getDimensionSalida();

        int[] topologia = construirTopologia(entrada, capasOcultas, salida);

        ConfiguracionBenchmark config = assertDoesNotThrow(() ->
            new ConfiguracionBenchmark(nivel, topologia, 10, 25, 1, 42L));

        assertEquals(entrada, config.topologia()[0]);
        assertEquals(salida, config.topologia()[config.topologia().length - 1]);
    }

    /**
     * Propiedad: configuraciones con dimensión de entrada incorrecta
     * lanzan IllegalArgumentException.
     */
    @Property(tries = 100)
    void entradaIncompatibleEsRechazada(
            @ForAll("nivelAleatorio") NivelComplejidad nivel,
            @ForAll("capasOcultasAleatorias") int[] capasOcultas,
            @ForAll("offsetNoZero") int offset) {

        int entradaIncorrecta = nivel.getDimensionEntrada() + offset;
        int salida = nivel.getDimensionSalida();

        int[] topologia = construirTopologia(entradaIncorrecta, capasOcultas, salida);

        assertThrows(IllegalArgumentException.class, () ->
            new ConfiguracionBenchmark(nivel, topologia, 10, 25, 1, 42L));
    }

    /**
     * Propiedad: configuraciones con dimensión de salida incorrecta
     * lanzan IllegalArgumentException.
     */
    @Property(tries = 100)
    void salidaIncompatibleEsRechazada(
            @ForAll("nivelAleatorio") NivelComplejidad nivel,
            @ForAll("capasOcultasAleatorias") int[] capasOcultas,
            @ForAll("offsetNoZero") int offset) {

        int entrada = nivel.getDimensionEntrada();
        int salidaIncorrecta = nivel.getDimensionSalida() + offset;

        int[] topologia = construirTopologia(entrada, capasOcultas, salidaIncorrecta);

        assertThrows(IllegalArgumentException.class, () ->
            new ConfiguracionBenchmark(nivel, topologia, 10, 25, 1, 42L));
    }

    // --- Proveedores (Arbitraries) ---

    @Provide
    Arbitrary<NivelComplejidad> nivelAleatorio() {
        return Arbitraries.of(NivelComplejidad.values());
    }

    @Provide
    Arbitrary<int[]> capasOcultasAleatorias() {
        return Arbitraries.integers().between(1, 64)
            .array(int[].class).ofMinSize(0).ofMaxSize(4);
    }

    @Provide
    Arbitrary<Integer> offsetNoZero() {
        return Arbitraries.integers().between(1, 50)
            .map(v -> Arbitraries.of(-1, 1).sample() * v);
    }

    // --- Utilidades ---

    private static int[] construirTopologia(int entrada, int[] capasOcultas, int salida) {
        int[] topologia = new int[capasOcultas.length + 2];
        topologia[0] = entrada;
        System.arraycopy(capasOcultas, 0, topologia, 1, capasOcultas.length);
        topologia[topologia.length - 1] = salida;
        return topologia;
    }
}
