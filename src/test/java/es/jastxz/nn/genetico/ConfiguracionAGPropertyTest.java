package es.jastxz.nn.genetico;

import net.jqwik.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link ConfiguracionAG}.
 *
 * <p>Verifica que las validaciones del constructor compacto rechazan
 * configuraciones inválidas con {@link IllegalArgumentException}.</p>
 */
class ConfiguracionAGPropertyTest {

    // ==================== Property 20: Validación de ConfiguracionAG ====================

    // Feature: genetic-algorithm-hyperparameters, Property 20: Validación de ConfiguracionAG
    /**
     * Para cualquier tripleta de pesos (p1, p2, p3) donde |p1 + p2 + p3 - 1.0| > 1e-6,
     * construir ConfiguracionAG debe lanzar IllegalArgumentException.
     *
     * <p><b>Validates: Requirements 8.3, 8.4</b></p>
     */
    @Property(tries = 100)
    void pesosQueNoSumanUnoLanzanExcepcion(
            @ForAll("pesoInvalido1") double peso1,
            @ForAll("pesoInvalido2") double peso2,
            @ForAll("pesoInvalido3") double peso3) {

        double suma = peso1 + peso2 + peso3;
        Assume.that(Math.abs(suma - 1.0) > 1e-6);

        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .pesosFitness(peso1, peso2, peso3)
                        .build(),
                "Pesos que no suman 1.0 deben lanzar IllegalArgumentException (suma=" + suma + ")");
    }

    // Feature: genetic-algorithm-hyperparameters, Property 20: Validación de ConfiguracionAG
    /**
     * Para cualquier tamañoTorneo > tamañoPoblacion, construir ConfiguracionAG
     * debe lanzar IllegalArgumentException.
     *
     * <p><b>Validates: Requirements 8.3, 8.4</b></p>
     */
    @Property(tries = 100)
    void torneoMayorQuePoblacionLanzaExcepcion(
            @ForAll("tamañoPoblacion") int tamPoblacion,
            @ForAll("incrementoTorneo") int incremento) {

        int tamTorneo = tamPoblacion + incremento;

        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .tamañoPoblacion(tamPoblacion)
                        .tamañoTorneo(tamTorneo)
                        .build(),
                "tamañoTorneo > tamañoPoblacion debe lanzar IllegalArgumentException"
                        + " (torneo=" + tamTorneo + ", poblacion=" + tamPoblacion + ")");
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Double> pesoInvalido1() {
        return Arbitraries.doubles().between(0.0, 1.0);
    }

    @Provide
    Arbitrary<Double> pesoInvalido2() {
        return Arbitraries.doubles().between(0.0, 1.0);
    }

    @Provide
    Arbitrary<Double> pesoInvalido3() {
        return Arbitraries.doubles().between(0.0, 1.0);
    }

    @Provide
    Arbitrary<Integer> tamañoPoblacion() {
        return Arbitraries.integers().between(5, 100);
    }

    @Provide
    Arbitrary<Integer> incrementoTorneo() {
        return Arbitraries.integers().between(1, 50);
    }
}
