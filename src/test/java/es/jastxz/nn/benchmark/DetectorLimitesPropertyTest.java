package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para la corrección de la detección de límites
 * en {@link DetectorLimites}.
 *
 * <p><b>Validates: Requisitos 4.1, 4.2, 4.3</b></p>
 *
 * @since 1.1
 */
class DetectorLimitesPropertyTest {

    // Feature: snn-benchmark-suite, Property 6: Corrección de la detección de límites

    private static final ConfiguracionBenchmark CONFIG_TRIVIAL =
        new ConfiguracionBenchmark(NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 50, 1, 42L);

    // --- Propiedad 6a: Precisión < 0.6 → "limite_no_superado" ---

    /**
     * Para cualquier resultado con precisión &lt; 0.6, clasificar debe retornar
     * {@code "limite_no_superado"} independientemente de las demás métricas.
     */
    @Property(tries = 100)
    void precisionBajaClasificaComoLimiteNoSuperado(
            @ForAll("precisionBaja") double precision,
            @ForAll("mseMejorando") double[] mse,
            @ForAll("ratioNeuronasAlta") int[] neuronas) {

        var resultado = crearResultado(precision, mse, neuronas[0], neuronas[1]);
        assertEquals("limite_no_superado", DetectorLimites.clasificar(resultado));
    }

    // --- Propiedad 6b: MSE estancado → "convergencia_estancada" ---

    /**
     * Para cualquier resultado con precisión &ge; 0.6, MSE estancado (valores
     * constantes, al menos 4 épocas) y neuronas activas &ge; 20%, clasificar
     * debe retornar {@code "convergencia_estancada"}.
     */
    @Property(tries = 100)
    void mseEstancadoClasificaComoConvergenciaEstancada(
            @ForAll("precisionAlta") double precision,
            @ForAll("mseEstancado") double[] mse,
            @ForAll("ratioNeuronasAlta") int[] neuronas) {

        var resultado = crearResultado(precision, mse, neuronas[0], neuronas[1]);
        assertEquals("convergencia_estancada", DetectorLimites.clasificar(resultado));
    }

    // --- Propiedad 6c: Neuronas activas < 20% → "red_infrautilizada" ---

    /**
     * Para cualquier resultado con precisión &ge; 0.6, MSE mejorando y
     * neuronas activas &lt; 20% del total, clasificar debe retornar
     * {@code "red_infrautilizada"}.
     */
    @Property(tries = 100)
    void neuronasInfrautilizadasClasificaComoRedInfrautilizada(
            @ForAll("precisionAlta") double precision,
            @ForAll("mseMejorando") double[] mse,
            @ForAll("ratioNeuronasBaja") int[] neuronas) {

        var resultado = crearResultado(precision, mse, neuronas[0], neuronas[1]);
        assertEquals("red_infrautilizada", DetectorLimites.clasificar(resultado));
    }

    // --- Propiedad 6d: Sin condiciones → null ---

    /**
     * Para cualquier resultado con precisión &ge; 0.6, MSE mejorando y
     * neuronas activas &ge; 20%, clasificar debe retornar {@code null}.
     */
    @Property(tries = 100)
    void sinLimitesClasificaComoNull(
            @ForAll("precisionAlta") double precision,
            @ForAll("mseMejorando") double[] mse,
            @ForAll("ratioNeuronasAlta") int[] neuronas) {

        var resultado = crearResultado(precision, mse, neuronas[0], neuronas[1]);
        assertNull(DetectorLimites.clasificar(resultado));
    }

    // --- Proveedores (Arbitraries) ---

    /** Precisión en [0.0, 0.6) — dispara "limite_no_superado". */
    @Provide
    Arbitrary<Double> precisionBaja() {
        return Arbitraries.integers().between(0, 59)
            .map(i -> i / 100.0);
    }

    /** Precisión en [0.6, 1.0] — no dispara "limite_no_superado". */
    @Provide
    Arbitrary<Double> precisionAlta() {
        return Arbitraries.doubles().between(0.6, 1.0);
    }

    /**
     * MSE con mejora significativa (&gt;1%) entre épocas consecutivas.
     * Genera un array de 4+ valores decrecientes con al menos 5% de mejora
     * entre cada par, evitando que se detecte estancamiento.
     */
    @Provide
    Arbitrary<double[]> mseMejorando() {
        return Arbitraries.integers().between(4, 10).flatMap(size ->
            Arbitraries.doubles().between(0.5, 2.0).map(start -> {
                double[] mse = new double[size];
                mse[0] = start;
                for (int i = 1; i < size; i++) {
                    // Reducción del 5-15% por época — siempre supera el umbral del 1%
                    mse[i] = mse[i - 1] * (1.0 - 0.05 - (i * 0.01));
                }
                return mse;
            })
        );
    }

    /**
     * MSE estancado: valores constantes (misma base) durante al menos 4 épocas,
     * garantizando 3 pares consecutivos con mejora &le; 1%.
     */
    @Provide
    Arbitrary<double[]> mseEstancado() {
        return Arbitraries.integers().between(4, 10).flatMap(size ->
            Arbitraries.doubles().between(0.1, 2.0).map(base -> {
                double[] mse = new double[size];
                for (int i = 0; i < size; i++) {
                    mse[i] = base; // Constante → mejora 0% en cada par
                }
                return mse;
            })
        );
    }

    /**
     * Neuronas activas &ge; 20% del total.
     * Retorna [neuronasActivas, neuronasTotal].
     */
    @Provide
    Arbitrary<int[]> ratioNeuronasAlta() {
        return Arbitraries.integers().between(5, 100).flatMap(total ->
            Arbitraries.integers().between((int) Math.ceil(total * 0.20), total)
                .map(activas -> new int[]{activas, total})
        );
    }

    /**
     * Neuronas activas &lt; 20% del total (con total &gt; 0).
     * Retorna [neuronasActivas, neuronasTotal].
     */
    @Provide
    Arbitrary<int[]> ratioNeuronasBaja() {
        return Arbitraries.integers().between(5, 100).flatMap(total -> {
            int maxActivas = (int) Math.ceil(total * 0.20) - 1;
            if (maxActivas < 0) maxActivas = 0;
            return Arbitraries.integers().between(0, maxActivas)
                .map(activas -> new int[]{activas, total});
        });
    }

    // --- Propiedad 11: Detección de ineficiencia energética ---

    // Feature: snn-benchmark-suite, Property 11: Detección de ineficiencia energética

    /**
     * Para cualquier resultado cuyo costo energético supera 10 veces el costo
     * mínimo, clasificarIneficiencia debe retornar {@code "ineficiencia_energetica"}.
     *
     * <p><b>Validates: Requisito 4.4</b></p>
     */
    @Property(tries = 100)
    void costoSuperiorA10xMinimoClasificaComoIneficiencia(
            @ForAll("costoMinimo") double costoMinimo,
            @ForAll("factorIneficiente") double factor) {

        double costoEnergetico = costoMinimo * 10 * factor; // factor > 1.0 → supera 10x
        var resultado = crearResultadoConCosto(costoEnergetico);
        assertEquals("ineficiencia_energetica",
            DetectorLimites.clasificarIneficiencia(resultado, costoMinimo));
    }

    /**
     * Para cualquier resultado cuyo costo energético no supera 10 veces el costo
     * mínimo, clasificarIneficiencia debe retornar {@code null}.
     *
     * <p><b>Validates: Requisito 4.4</b></p>
     */
    @Property(tries = 100)
    void costoIgualOMenorA10xMinimoNoClasificaComoIneficiencia(
            @ForAll("costoMinimo") double costoMinimo,
            @ForAll("factorEficiente") double factor) {

        double costoEnergetico = costoMinimo * 10 * factor; // factor in (0, 1] → no supera 10x
        var resultado = crearResultadoConCosto(costoEnergetico);
        assertNull(DetectorLimites.clasificarIneficiencia(resultado, costoMinimo));
    }

    // --- Proveedores para Propiedad 11 ---

    /** Costo mínimo positivo en rango [0.01, 100.0]. Usa enteros mapeados para evitar problemas de escala. */
    @Provide
    Arbitrary<Double> costoMinimo() {
        return Arbitraries.integers().between(1, 10000)
            .map(i -> i / 100.0);
    }

    /** Factor &gt; 1.0 para generar costos que superan 10× el mínimo. Usa enteros mapeados para evitar problemas de escala. */
    @Provide
    Arbitrary<Double> factorIneficiente() {
        return Arbitraries.integers().between(101, 1000)
            .map(i -> i / 100.0);
    }

    /** Factor en (0.0, 1.0] para generar costos que no superan 10× el mínimo. Usa enteros mapeados para evitar problemas de escala. */
    @Provide
    Arbitrary<Double> factorEficiente() {
        return Arbitraries.integers().between(1, 100)
            .map(i -> i / 100.0);
    }

    // --- Utilidades ---

    private static ResultadoBenchmark crearResultado(double precision, double[] mse,
                                                      int neuronasActivas, int neuronasTotal) {
        return new ResultadoBenchmark(
            CONFIG_TRIVIAL, precision, mse, 100L, 50L, 0.5, 1.0, 0.1,
            neuronasActivas, neuronasTotal, null
        );
    }

    private static ResultadoBenchmark crearResultadoConCosto(double costoEnergetico) {
        return new ResultadoBenchmark(
            CONFIG_TRIVIAL, 0.8, new double[]{1.0, 0.5, 0.3, 0.1}, 100L, 50L,
            0.5, costoEnergetico, 0.1, 5, 10, null
        );
    }
}
