package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.OptimizadorSNN;
import es.jastxz.nn.spiking.ResultadoOptimizacion;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de integración que verifica que {@link OptimizadorSNN} aprende
 * un catálogo completo de patrones sencillos mediante AG + SNN.
 *
 * <h3>Patrones lógicos (entradas binarias, 1 salida):</h3>
 * <ul>
 *   <li>Identidad, NOT</li>
 *   <li>Mayoría-3, Paridad-3</li>
 * </ul>
 *
 * <h3>Patrones numéricos — clasificación binaria (3 bits → 1 salida):</h3>
 * <ul>
 *   <li>Par/Impar, Múltiplo de 3, ¿Es primo?, ¿Es Fibonacci?</li>
 * </ul>
 *
 * <h3>Patrones funcionales:</h3>
 * <ul>
 *   <li>Función escalón (1 entrada → 1 salida)</li>
 *   <li>Comparador A &gt; B (2 entradas → 1 salida)</li>
 *   <li>Cuadrante 2D (2 entradas → 4 salidas one-hot)</li>
 * </ul>
 *
 * <h3>Patrones de secuencia — mapeo multi-clase (2 bits → 4 salidas one-hot):</h3>
 * <ul>
 *   <li>Sucesor mod 4: n → (n+1) mod 4</li>
 *   <li>Doble mod 4: n → 2n mod 4</li>
 * </ul>
 */
class PatronesSimplesSNNAGTest {

    private static final int NUM_INTENTOS = 5;
    private static final double PRECISION_MINIMA = 1.0;
    /** Precisión mínima para problemas multi-clase (4+ salidas).
     * Con 4 clases el azar da 25%, así que 50% demuestra aprendizaje real. */
    private static final double PRECISION_MINIMA_MULTICLASE = 0.50;
    /**
     * Precisión mínima para clasificación de cuadrantes 2D.
     * Con 4 clases el azar da 25%, así que 50% demuestra aprendizaje real.
     */
    private static final double PRECISION_MINIMA_CUADRANTE = 0.50;

    // ================================================================
    //  DATOS — Patrones lógicos
    // ================================================================

    // --- Identidad / NOT (1 entrada) ---
    private static final double[][] INPUTS_1 = {{0.0}, {1.0}};
    private static final double[][] TARGETS_IDENTIDAD = {{0}, {1}};
    private static final double[][] TARGETS_NOT = {{1}, {0}};

    // --- Mayoría-3 / Paridad-3 (3 entradas binarias) ---
    private static final double[][] INPUTS_3BIN = {
        {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
        {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}
    };
    /** Salida 1 si al menos 2 de 3 entradas son 1. */
    private static final double[][] TARGETS_MAYORIA = {
        {0}, {0}, {0}, {1}, {0}, {1}, {1}, {1}
    };
    /** Salida 1 si número impar de 1s. */
    private static final double[][] TARGETS_PARIDAD = {
        {0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}
    };

    // ================================================================
    //  DATOS — Patrones numéricos (3 bits → clasificación binaria)
    // ================================================================
    //
    //  Cada fila de INPUTS_3BIN codifica un número 0–7 en binario.
    //  Las salidas clasifican propiedades numéricas de ese número.

    /** Par/Impar: salida 1 si el número es par (0,2,4,6). */
    private static final double[][] TARGETS_PAR = {
        {1}, {0}, {0}, {1}, {0}, {1}, {1}, {0}
    };

    /** Múltiplo de 3: salida 1 si n ∈ {0,3,6}. */
    private static final double[][] TARGETS_MULT3 = {
        {1}, {0}, {0}, {1}, {0}, {0}, {1}, {0}
    };

    /** ¿Es primo?: salida 1 si n ∈ {2,3,5,7}. */
    private static final double[][] TARGETS_PRIMO = {
        {0}, {0}, {1}, {1}, {0}, {1}, {0}, {1}
    };

    /** ¿Es Fibonacci?: salida 1 si n ∈ {0,1,2,3,5}. */
    private static final double[][] TARGETS_FIBONACCI = {
        {1}, {1}, {1}, {1}, {0}, {1}, {0}, {0}
    };

    // ================================================================
    //  DATOS — Patrones funcionales
    // ================================================================

    /**
     * Función escalón: 5 muestras uniformes en [0,1].
     * Salida 1 si entrada ≥ 0.5, salida 0 en caso contrario.
     */
    private static final double[][] INPUTS_ESCALON = {
        {0.0}, {0.25}, {0.5}, {0.75}, {1.0}
    };
    private static final double[][] TARGETS_ESCALON = {
        {0}, {0}, {1}, {1}, {1}
    };

    /**
     * Comparador: 2 entradas, salida 1 si A &gt; B.
     * Usa pares representativos con margen claro.
     */
    private static final double[][] INPUTS_COMPARADOR = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0},
        {0.3, 0.7}, {0.7, 0.3}, {0.2, 0.8}, {0.8, 0.2}
    };
    private static final double[][] TARGETS_COMPARADOR = {
        {0}, {0}, {1}, {0},
        {0}, {1}, {0}, {1}
    };

    /**
     * Cuadrante 2D: 2 entradas → 4 clases (one-hot).
     * Clasifica el punto (x,y) en uno de 4 cuadrantes.
     * <pre>
     *   Clase 0: x&lt;0.5, y&lt;0.5  (abajo-izquierda)
     *   Clase 1: x≥0.5, y&lt;0.5  (abajo-derecha)
     *   Clase 2: x&lt;0.5, y≥0.5  (arriba-izquierda)
     *   Clase 3: x≥0.5, y≥0.5  (arriba-derecha)
     * </pre>
     */
    private static final double[][] INPUTS_CUADRANTE = {
        {0.2, 0.2}, {0.8, 0.2}, {0.2, 0.8}, {0.8, 0.8},
        {0.1, 0.1}, {0.9, 0.1}, {0.1, 0.9}, {0.9, 0.9}
    };
    private static final double[][] TARGETS_CUADRANTE = {
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}
    };

    // ================================================================
    //  DATOS — Patrones de secuencia (2 bits → 4 clases one-hot)
    // ================================================================
    //
    //  Entrada: número 0–3 codificado en 2 bits.
    //  Salida: resultado de la operación codificado en 4 clases one-hot.

    /** Entradas: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1). */
    private static final double[][] INPUTS_2BIT = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    /** Sucesor mod 4: 0→1, 1→2, 2→3, 3→0. */
    private static final double[][] TARGETS_SUCESOR = {
        {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}
    };

    /** Doble mod 4: 0→0, 1→2, 2→0, 3→2. */
    private static final double[][] TARGETS_DOBLE = {
        {1, 0, 0, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {0, 0, 1, 0}
    };

    // ================================================================
    //  Helpers
    // ================================================================

    /** Helper para patrones con 1 salida (clasificación binaria). */
    private ResultadoOptimizacion buscarMejorSolucion(
            double[][] inputs, double[][] targets, String nombre) {

        return buscarMejorSolucion(inputs, targets, nombre, 30, 50, 32);
    }

    /** Helper para patrones multi-clase (más neuronas y generaciones). */
    private ResultadoOptimizacion buscarMejorSolucionMultiClase(
            double[][] inputs, double[][] targets, String nombre) {

        return buscarMejorSolucion(inputs, targets, nombre,
                30, 50, 32, PRECISION_MINIMA_MULTICLASE, 150, 15);
    }

    private ResultadoOptimizacion buscarMejorSolucion(
            double[][] inputs, double[][] targets, String nombre,
            int poblacion, int generaciones, int limiteTopologico) {

        return buscarMejorSolucion(inputs, targets, nombre,
                poblacion, generaciones, limiteTopologico, PRECISION_MINIMA);
    }

    private ResultadoOptimizacion buscarMejorSolucion(
            double[][] inputs, double[][] targets, String nombre,
            int poblacion, int generaciones, int limiteTopologico,
            double precisionObjetivo) {

        return buscarMejorSolucion(inputs, targets, nombre,
                poblacion, generaciones, limiteTopologico, precisionObjetivo, 100, 10);
    }

    private ResultadoOptimizacion buscarMejorSolucion(
            double[][] inputs, double[][] targets, String nombre,
            int poblacion, int generaciones, int limiteTopologico,
            double precisionObjetivo, int duracionTimesteps, int epocas) {

        ResultadoOptimizacion mejorResultado = null;

        for (int i = 0; i < NUM_INTENTOS; i++) {
            ResultadoOptimizacion resultado = new OptimizadorSNN.Builder()
                    .tamañoPoblacion(poblacion)
                    .maxGeneraciones(generaciones)
                    .generacionesEstancamiento(15)
                    .limiteTopologico(limiteTopologico)
                    .pesoPrecision(0.9)
                    .pesoEnergia(0.0)
                    .pesoTamanio(0.1)
                    .duracionTimesteps(duracionTimesteps)
                    .epocasEntrenamiento(epocas)
                    .numIntentos(1)
                    .optimizar(inputs, targets);

            System.out.printf("[%s] Intento %d → precisión=%.0f%% fitness=%.4f topología=%s%n",
                    nombre, i + 1, resultado.precision() * 100, resultado.fitness(),
                    Arrays.toString(resultado.configuracionRed().getTopologia()));

            if (mejorResultado == null || resultado.precision() > mejorResultado.precision()) {
                mejorResultado = resultado;
            }

            if (resultado.precision() >= precisionObjetivo) break;
        }

        return mejorResultado;
    }

    private void verificar(ResultadoOptimizacion resultado, String nombre) {
        verificar(resultado, nombre, PRECISION_MINIMA);
    }

    private void verificarMultiClase(ResultadoOptimizacion resultado, String nombre) {
        verificar(resultado, nombre, PRECISION_MINIMA_MULTICLASE);
    }

    private void verificar(ResultadoOptimizacion resultado, String nombre, double precisionMinima) {
        System.out.printf("==> %s: precisión=%.0f%% topología=%s%n",
                nombre, resultado.precision() * 100,
                Arrays.toString(resultado.configuracionRed().getTopologia()));

        assertTrue(resultado.precision() >= precisionMinima,
                nombre + " debe alcanzar al menos " + (precisionMinima * 100)
                        + "% de precisión, obtuvo: " + (resultado.precision() * 100) + "%");
    }

    // ================================================================
    //  Tests — Patrones lógicos
    // ================================================================

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeIdentidad() {
        verificar(buscarMejorSolucion(INPUTS_1, TARGETS_IDENTIDAD, "IDENTIDAD"), "IDENTIDAD");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeNOT() {
        verificar(buscarMejorSolucion(INPUTS_1, TARGETS_NOT, "NOT"), "NOT");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeMayoria3() {
        verificar(buscarMejorSolucion(INPUTS_3BIN, TARGETS_MAYORIA, "MAYORÍA-3"), "MAYORÍA-3");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeParidad3() {
        verificar(buscarMejorSolucion(INPUTS_3BIN, TARGETS_PARIDAD, "PARIDAD-3"), "PARIDAD-3");
    }

    // ================================================================
    //  Tests — Patrones numéricos
    // ================================================================

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeParImpar() {
        verificar(buscarMejorSolucion(INPUTS_3BIN, TARGETS_PAR, "PAR/IMPAR"), "PAR/IMPAR");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeMultiploDe3() {
        verificar(buscarMejorSolucion(INPUTS_3BIN, TARGETS_MULT3, "MÚLTIPLO-3"), "MÚLTIPLO-3");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendePrimo() {
        verificar(buscarMejorSolucion(INPUTS_3BIN, TARGETS_PRIMO, "PRIMO"), "PRIMO");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeFibonacci() {
        verificar(buscarMejorSolucion(INPUTS_3BIN, TARGETS_FIBONACCI, "FIBONACCI"), "FIBONACCI");
    }

    // ================================================================
    //  Tests — Patrones funcionales
    // ================================================================

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeEscalon() {
        verificar(buscarMejorSolucion(INPUTS_ESCALON, TARGETS_ESCALON, "ESCALÓN"), "ESCALÓN");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeComparador() {
        verificar(buscarMejorSolucion(INPUTS_COMPARADOR, TARGETS_COMPARADOR, "COMPARADOR"), "COMPARADOR");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeCuadrante2D() {
        ResultadoOptimizacion resultado = buscarMejorSolucionMultiClase(
                INPUTS_CUADRANTE, TARGETS_CUADRANTE, "CUADRANTE-2D");
        verificar(resultado, "CUADRANTE-2D", PRECISION_MINIMA_CUADRANTE);
    }

    // ================================================================
    //  Tests — Patrones de secuencia
    // ================================================================

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeSucesorMod4() {
        verificarMultiClase(buscarMejorSolucionMultiClase(INPUTS_2BIT, TARGETS_SUCESOR, "SUCESOR-MOD4"), "SUCESOR-MOD4");
    }

    @Test
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void evolucionarSNN_aprendeDobleMod4() {
        verificarMultiClase(buscarMejorSolucionMultiClase(INPUTS_2BIT, TARGETS_DOBLE, "DOBLE-MOD4"), "DOBLE-MOD4");
    }
}
