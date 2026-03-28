package es.jastxz.nn.spiking.integration;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de integración que verifica que una SNN es capaz de resolver
 * los problemas lógicos OR, AND y XOR usando una sola capa oculta
 * con el número mínimo teórico de neuronas:
 * <ul>
 *   <li>OR  → topología [2, 1, 1] — 1 neurona oculta (linealmente separable)</li>
 *   <li>AND → topología [2, 1, 1] — 1 neurona oculta (linealmente separable)</li>
 *   <li>XOR → topología [2, 2, 1] — 2 neuronas ocultas (no linealmente separable)</li>
 * </ul>
 *
 * <p>Se usa {@link ConfiguracionRedBuilder#autoconfigurar()} para calcular
 * automáticamente el rango de pesos que garantiza la propagación de señal.
 * Los pesos de cada puerta se asignan usando {@code calcularPesoRelay()} y
 * {@code calcularPesoMinimoPropagacion(n)}.</p>
 */
class PuertasLogicasSNNTest {

    // ==================== Datos ====================

    private static final double[][] INPUTS = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    private static final double[][] TARGETS_OR  = {{0}, {1}, {1}, {1}};
    private static final double[][] TARGETS_AND = {{0}, {0}, {0}, {1}};
    private static final double[][] TARGETS_XOR = {{0}, {1}, {1}, {0}};

    private static final int DURACION = 100;

    // ==================== Helpers ====================

    /**
     * Crea un builder preconfigurado. Sin llamar a rangoPesos ni
     * inicializacionPesos, build() aplicará autoconfigurar() automáticamente.
     */
    private ConfiguracionRedBuilder builderBase(int... topologia) {
        return new ConfiguracionRedBuilder()
            .topologia(topologia)
            .modoCodificacion(ModoCodificacion.REGULAR)
            .ventanaDecodificacion(DURACION)
            .retardos(1, 1)
            .sinHomeostasis()
            .sinInhibicionLateral();
    }

    /**
     * Evalúa la precisión de la red sobre los 4 patrones de entrada.
     */
    private double evaluar(RedNeuralSpiking red, double[][] targets) {
        red.setModoEntrenamiento(false);
        int correctas = 0;
        for (int i = 0; i < INPUTS.length; i++) {
            red.resetearEstadoTemporal();
            double[] salida = red.procesar(INPUTS[i], DURACION);
            if (Math.round(salida[0]) == Math.round(targets[i][0])) {
                correctas++;
            }
        }
        return (double) correctas / INPUTS.length;
    }

    // ==================== Tests de puertas lógicas ====================

    /**
     * OR con topología [2, 1, 1] — 1 neurona oculta.
     *
     * <p>Peso relay: cada spike de entrada genera un spike en la oculta.
     * Una sola entrada activa basta para que la salida dispare.</p>
     */
    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void snn_resuelvePuertaOR_conUnaNeuronaOculta() {
        ConfiguracionRedBuilder builder = builderBase(2, 1, 1);
        double wRelay = builder.calcularPesoRelay();

        RedNeuralSpiking red = new RedNeuralSpiking(builder.build());

        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), wRelay, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 0), wRelay, 1);
        red.crearConexion(red.getNeurona(1, 0), red.getNeurona(2, 0), wRelay, 1);

        double precision = evaluar(red, TARGETS_OR);

        System.out.printf("==> OR [2,1,1]: precisión=%.0f%% (wRelay=%.2f)%n",
                precision * 100, wRelay);
        assertEquals(1.0, precision,
                "OR con topología [2,1,1] debe alcanzar 100%%, obtuvo: "
                        + (precision * 100) + "%");
    }

    /**
     * AND con topología [2, 1, 1] — 1 neurona oculta.
     *
     * <p>Peso para fan-in=2: cada entrada aporta la mitad del umbral.
     * Se necesitan ambas entradas activas para que la oculta dispare.</p>
     */
    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void snn_resuelvePuertaAND_conUnaNeuronaOculta() {
        ConfiguracionRedBuilder builder = builderBase(2, 1, 1);
        double wRelay = builder.calcularPesoRelay();
        // Peso para que 2 conexiones juntas activen, pero 1 sola no
        double wAnd = builder.calcularPesoMinimoPropagacion(2) * 1.1;

        RedNeuralSpiking red = new RedNeuralSpiking(builder.build());

        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), wAnd, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 0), wAnd, 1);
        red.crearConexion(red.getNeurona(1, 0), red.getNeurona(2, 0), wRelay, 1);

        double precision = evaluar(red, TARGETS_AND);

        System.out.printf("==> AND [2,1,1]: precisión=%.0f%% (wRelay=%.2f, wAnd=%.2f)%n",
                precision * 100, wRelay, wAnd);
        assertEquals(1.0, precision,
                "AND con topología [2,1,1] debe alcanzar 100%%, obtuvo: "
                        + (precision * 100) + "%");
    }

    /**
     * XOR con topología [2, 2, 1] — 2 neuronas ocultas.
     *
     * <p>Implementa XOR = OR AND NOT(AND):</p>
     * <ul>
     *   <li>Oculta 0 = OR: peso relay desde ambas entradas</li>
     *   <li>Oculta 1 = AND: peso fan-in=2 desde ambas entradas</li>
     *   <li>Salida: excitación (relay) desde OR, inhibición desde AND</li>
     * </ul>
     */
    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void snn_resuelvePuertaXOR_conDosNeuronasOcultas() {
        ConfiguracionRedBuilder builder = builderBase(2, 2, 1);
        double wRelay = builder.calcularPesoRelay();
        double wAnd = builder.calcularPesoMinimoPropagacion(2) * 1.1;
        double wInhib = -wRelay * 1.5;

        // XOR necesita pesos negativos → configurar rango manualmente
        RedNeuralSpiking red = new RedNeuralSpiking(
            builder.rangoPesos(wInhib * 1.2, wRelay * 2.0).build());

        // Oculta 0 = OR
        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), wRelay, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 0), wRelay, 1);

        // Oculta 1 = AND
        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 1), wAnd, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 1), wAnd, 1);

        // Salida: OR excita, AND inhibe
        red.crearConexion(red.getNeurona(1, 0), red.getNeurona(2, 0), wRelay, 1);
        red.crearConexion(red.getNeurona(1, 1), red.getNeurona(2, 0), wInhib, 1);

        double precision = evaluar(red, TARGETS_XOR);

        System.out.printf("==> XOR [2,2,1]: precisión=%.0f%% (wRelay=%.2f, wAnd=%.2f, wInhib=%.2f)%n",
                precision * 100, wRelay, wAnd, wInhib);
        assertEquals(1.0, precision,
                "XOR con topología [2,2,1] debe alcanzar 100%%, obtuvo: "
                        + (precision * 100) + "%");
    }

    // ==================== Tests de autoconfiguración ====================

    /**
     * Verifica que build() autoconfigura pesoMax cuando no se especifican pesos.
     */
    @Test
    void autoconfigurar_seAplicaAutomaticamenteEnBuild() {
        ConfiguracionRedBuilder builder = new ConfiguracionRedBuilder()
            .topologia(2, 4, 1);

        double wRelay = builder.calcularPesoRelay();
        ConfiguracionRed config = builder.build();

        assertTrue(config.pesoMax >= wRelay,
                "pesoMax autoconfigurado (" + config.pesoMax
                        + ") debe ser >= wRelay (" + wRelay + ")");
        assertEquals(0.0, config.pesoMin,
                "pesoMin autoconfigurado debe ser 0.0");
    }

    /**
     * Verifica que la autoconfiguración NO se aplica si el usuario
     * configuró los pesos manualmente.
     */
    @Test
    void autoconfigurar_noSobreescribePesosManuales() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 4, 1)
            .rangoPesos(0.5, 3.0)
            .build();

        assertEquals(0.5, config.pesoMin, "pesoMin manual debe mantenerse");
        assertEquals(3.0, config.pesoMax, "pesoMax manual debe mantenerse");
    }

    /**
     * Verifica que calcularPesoMinimoPropagacion escala inversamente
     * con el número de conexiones presinapticas.
     */
    @Test
    void calcularPesoMinimo_escalaInversamenteConFanIn() {
        ConfiguracionRedBuilder builder = new ConfiguracionRedBuilder()
            .topologia(4, 4, 1);

        double w1 = builder.calcularPesoMinimoPropagacion(1);
        double w2 = builder.calcularPesoMinimoPropagacion(2);
        double w4 = builder.calcularPesoMinimoPropagacion(4);

        assertTrue(w1 > w2, "w(n=1) debe ser mayor que w(n=2)");
        assertTrue(w2 > w4, "w(n=2) debe ser mayor que w(n=4)");
        assertEquals(w1, w2 * 2, 0.001, "w(n=1) ≈ 2 * w(n=2)");
        assertEquals(w1, w4 * 4, 0.001, "w(n=1) ≈ 4 * w(n=4)");
    }

    /**
     * Verifica que calcularPesoRelay es mayor que calcularPesoMinimoPropagacion(1).
     */
    @Test
    void calcularPesoRelay_esMayorQuePesoMinimo() {
        ConfiguracionRedBuilder builder = new ConfiguracionRedBuilder()
            .topologia(2, 4, 1);

        double wRelay = builder.calcularPesoRelay();
        double wMin = builder.calcularPesoMinimoPropagacion(1);

        assertTrue(wRelay > wMin,
                "wRelay (" + wRelay + ") debe ser mayor que wMin(n=1) (" + wMin + ")");
    }
}
