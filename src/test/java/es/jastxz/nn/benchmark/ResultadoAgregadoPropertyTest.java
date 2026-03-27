package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para la corrección de la agregación estadística
 * en {@link ResultadoAgregado}.
 *
 * <p><b>Validates: Requisito 6.3</b></p>
 *
 * @since 1.1
 */
class ResultadoAgregadoPropertyTest {

    // Feature: snn-benchmark-suite, Property 7: Corrección de la agregación estadística

    private static final ConfiguracionBenchmark CONFIG = new ConfiguracionBenchmark(
        NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 25, 1, 42L);

    private static final double EPSILON = 1e-9;

    /**
     * Propiedad: para cualquier lista de N resultados (N >= 1), la media de precisión
     * calculada por agregar() debe ser igual a la suma de precisiones dividida por N.
     * Lo mismo aplica para tiempo y costo energético.
     */
    @Property(tries = 100)
    void mediaEsSumaDivididaPorN(
            @ForAll("listaResultados") List<ResultadoBenchmark> resultados) {

        int n = resultados.size();
        ResultadoAgregado agregado = ResultadoAgregado.agregar(resultados);

        double sumaPrecision = resultados.stream()
            .mapToDouble(ResultadoBenchmark::precisionFinal).sum();
        double sumaTiempo = resultados.stream()
            .mapToDouble(r -> r.tiempoEntrenamientoMs()).sum();
        double sumaCosto = resultados.stream()
            .mapToDouble(ResultadoBenchmark::costoEnergetico).sum();

        assertEquals(sumaPrecision / n, agregado.mediaPrecision(), EPSILON,
            "Media de precisión debe ser suma/N");
        assertEquals(sumaTiempo / n, agregado.mediaTiempo(), EPSILON,
            "Media de tiempo debe ser suma/N");
        assertEquals(sumaCosto / n, agregado.mediaCostoEnergetico(), EPSILON,
            "Media de costo energético debe ser suma/N");
    }

    /**
     * Propiedad: para cualquier lista de N resultados (N >= 2), la desviación estándar
     * debe ser la raíz cuadrada de la varianza muestral (dividida por N-1).
     */
    @Property(tries = 100)
    void desviacionEsRaizDeVarianzaMuestral(
            @ForAll("listaResultadosMultiples") List<ResultadoBenchmark> resultados) {

        int n = resultados.size();
        ResultadoAgregado agregado = ResultadoAgregado.agregar(resultados);

        // Verificar desviación estándar de precisión
        double mediaPrecision = resultados.stream()
            .mapToDouble(ResultadoBenchmark::precisionFinal).sum() / n;
        double varianzaPrecision = resultados.stream()
            .mapToDouble(r -> Math.pow(r.precisionFinal() - mediaPrecision, 2))
            .sum() / (n - 1);
        assertEquals(Math.sqrt(varianzaPrecision), agregado.desvPrecision(), EPSILON,
            "Desviación de precisión debe ser sqrt(varianza muestral)");

        // Verificar desviación estándar de tiempo
        double mediaTiempo = resultados.stream()
            .mapToDouble(r -> r.tiempoEntrenamientoMs()).sum() / n;
        double varianzaTiempo = resultados.stream()
            .mapToDouble(r -> Math.pow(r.tiempoEntrenamientoMs() - mediaTiempo, 2))
            .sum() / (n - 1);
        assertEquals(Math.sqrt(varianzaTiempo), agregado.desvTiempo(), EPSILON,
            "Desviación de tiempo debe ser sqrt(varianza muestral)");

        // Verificar desviación estándar de costo energético
        double mediaCosto = resultados.stream()
            .mapToDouble(ResultadoBenchmark::costoEnergetico).sum() / n;
        double varianzaCosto = resultados.stream()
            .mapToDouble(r -> Math.pow(r.costoEnergetico() - mediaCosto, 2))
            .sum() / (n - 1);
        assertEquals(Math.sqrt(varianzaCosto), agregado.desvCostoEnergetico(), EPSILON,
            "Desviación de costo energético debe ser sqrt(varianza muestral)");
    }

    /**
     * Propiedad: para N=1, todas las desviaciones estándar deben ser 0.
     */
    @Property(tries = 100)
    void desviacionEsCeroParaUnSoloResultado(
            @ForAll("resultadoAleatorio") ResultadoBenchmark resultado) {

        ResultadoAgregado agregado = ResultadoAgregado.agregar(List.of(resultado));

        assertEquals(0.0, agregado.desvPrecision(), EPSILON,
            "Desviación de precisión debe ser 0 para N=1");
        assertEquals(0.0, agregado.desvTiempo(), EPSILON,
            "Desviación de tiempo debe ser 0 para N=1");
        assertEquals(0.0, agregado.desvCostoEnergetico(), EPSILON,
            "Desviación de costo energético debe ser 0 para N=1");
    }

    // --- Proveedores (Arbitraries) ---

    @Provide
    Arbitrary<ResultadoBenchmark> resultadoAleatorio() {
        return Combinators.combine(
            Arbitraries.doubles().between(0.0, 1.0),       // precisionFinal
            Arbitraries.longs().between(0L, 10_000L),      // tiempoEntrenamientoMs
            Arbitraries.doubles().between(0.0, 100.0)      // costoEnergetico
        ).as((precision, tiempo, costo) ->
            new ResultadoBenchmark(
                CONFIG, precision, new double[]{0.5}, tiempo,
                50L, 0.3, costo, 0.1, 3, 5, null));
    }

    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultados() {
        return resultadoAleatorio().list().ofMinSize(1).ofMaxSize(20);
    }

    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultadosMultiples() {
        return resultadoAleatorio().list().ofMinSize(2).ofMaxSize(20);
    }
}
