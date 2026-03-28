package es.jastxz.comparativas;

import es.jastxz.engine.FuncionesDamas;
import es.jastxz.models.ModeloDamas;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.spiking.*;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import es.jastxz.util.ModelManager;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa entre Red Neuronal Clásica (Backpropagation) y Red Neuronal Spiking (STDP)
 * para el juego de Damas.
 *
 * El modelo clásico NO se entrena aquí — se carga pre-entrenado.
 * El modelo spiking se entrena con los mismos datos supervisados (.dat).
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComparativaSpikingDamasTest {

    private static NeuralNetwork cerebroClasico;
    private static RedNeuralSpiking redSpiking;
    private static final int DURACION_TIMESTEPS = 50;

    @BeforeAll
    static void cargarModeloClasico() {
        cerebroClasico = ModelManager.loadModel("modeloDamas.nn");
        assertNotNull(cerebroClasico, "El modelo clásico de Damas debe estar pre-entrenado");
    }

    /**
     * Test 1: Verificar que el modelo clásico está entrenado y funciona.
     */
    @Test
    @Order(1)
    void test1_ModeloClasicoPreEntrenado() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: VERIFICAR MODELO CLÁSICO PRE-ENTRENADO (Damas)");
        System.out.println("=".repeat(60));

        Tablero tableroInicial = FuncionesDamas.inicialDamas();
        double[] input = ModeloDamas.tabularToInput(tableroInicial, 1);
        double[] output = cerebroClasico.feedForward(input);

        assertNotNull(output, "La salida no debe ser null");
        assertEquals(128, output.length, "La salida debe tener 128 valores (64 origen + 64 destino)");

        double max = Arrays.stream(output).max().orElse(0);
        assertTrue(max > 0, "Al menos una salida debe ser positiva");

        System.out.println("✓ Modelo clásico de Damas cargado y funcional");
    }

    /**
     * Test 2: Entrenar modelo spiking con datos supervisados de Damas.
     */
    @Test
    @Order(2)
    void test2_EntrenarModeloSpiking() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: ENTRENAMIENTO RED SPIKING (Damas)");
        System.out.println("=".repeat(60));

        // Crear red spiking: 65 entrada, 128 oculta, 128 salida
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(65, 128, 128)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .modoCodificacion(ModoCodificacion.REGULAR)
            .frecuenciaMaxima(100.0)
            .ventanaDecodificacion(DURACION_TIMESTEPS)
            .inicializacionPesos(TipoInicializacion.UNIFORME, 0.1, 0.5)
            .conHomeostasis()
            .tasaDisparoObjetivo(0.1)
            .build();

        redSpiking = new RedNeuralSpiking(config);
        redSpiking.inicializarPesos(TipoInicializacion.UNIFORME, 0.1, 0.5);

        // Generar datos de entrenamiento desde archivos .dat
        ModeloDamas modelo = new ModeloDamas();
        var data = modelo.generateTrainingData();
        double[][] inputs = data.getInputs();
        double[][] outputs = data.getOutputs();

        System.out.println("Datos de entrenamiento: " + inputs.length + " muestras");

        // Usar un subconjunto para no tardar demasiado
        int maxMuestras = Math.min(inputs.length, 150);
        double[][] inputsSub = Arrays.copyOf(inputs, maxMuestras);
        double[][] outputsSub = Arrays.copyOf(outputs, maxMuestras);

        // Normalizar inputs a [0,1]
        double[][] inputsNorm = normalizarInputs(inputsSub);

        long inicio = System.currentTimeMillis();
        int epocas = 3;
        for (int e = 0; e < epocas; e++) {
            double error = redSpiking.entrenar(inputsNorm, outputsSub, DURACION_TIMESTEPS);
            System.out.printf("Época %d/%d - Error MSE: %.6f%n", e + 1, epocas, error);
        }
        long tiempo = System.currentTimeMillis() - inicio;

        System.out.println("\n✓ Red spiking entrenada");
        System.out.println("Tiempo: " + tiempo + " ms");
    }

    /**
     * Test 3: Comparar rendimiento contra jugador aleatorio.
     */
    @Test
    @Order(3)
    void test3_CompararContraAleatorio() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: RENDIMIENTO CONTRA JUGADOR ALEATORIO (Damas)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) test2_EntrenarModeloSpiking();

        int partidasPrueba = 20;

        System.out.println("\n--- Modelo Clásico (Backprop) vs Aleatorio ---");
        ResultadoPartidas resultadoClasico = jugarContraAleatorio(partidasPrueba, true);
        mostrarResultados(resultadoClasico);

        System.out.println("\n--- Modelo Spiking (STDP) vs Aleatorio ---");
        ResultadoPartidas resultadoSpiking = jugarContraAleatorio(partidasPrueba, false);
        mostrarResultados(resultadoSpiking);

        System.out.println("\n--- Comparación ---");
        System.out.printf("Victorias Clásico: %.1f%% vs Spiking: %.1f%%%n",
            resultadoClasico.porcentajeVictorias(),
            resultadoSpiking.porcentajeVictorias());
    }

    /**
     * Test 4: Enfrentamiento directo Clásico vs Spiking.
     */
    @Test
    @Order(4)
    void test4_EnfrentamientoDirecto() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: ENFRENTAMIENTO DIRECTO (Damas)");
        System.out.println("Clásico (Backprop) vs Spiking (STDP)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) test2_EntrenarModeloSpiking();

        int partidasPorRonda = 10;

        // Ronda 1: Clásico como Blancas
        System.out.println("\n--- Ronda 1: Clásico (Blancas) vs Spiking (Negras) ---");
        ResultadoEnfrentamiento ronda1 = enfrentarModelos(partidasPorRonda, true);
        mostrarEnfrentamiento(ronda1, "Clásico", "Spiking");

        // Ronda 2: Spiking como Blancas
        System.out.println("\n--- Ronda 2: Spiking (Blancas) vs Clásico (Negras) ---");
        ResultadoEnfrentamiento ronda2 = enfrentarModelos(partidasPorRonda, false);
        mostrarEnfrentamiento(ronda2, "Spiking", "Clásico");

        int total = partidasPorRonda * 2;
        int victoriasClasico = ronda1.victoriasP1 + ronda2.victoriasP2;
        int victoriasSpiking = ronda1.victoriasP2 + ronda2.victoriasP1;
        int empates = ronda1.empates + ronda2.empates;

        System.out.println("\n--- RESUMEN TOTAL (" + total + " partidas) ---");
        System.out.printf("Clásico: %d (%.1f%%)%n", victoriasClasico, victoriasClasico * 100.0 / total);
        System.out.printf("Spiking: %d (%.1f%%)%n", victoriasSpiking, victoriasSpiking * 100.0 / total);
        System.out.printf("Empates: %d (%.1f%%)%n", empates, empates * 100.0 / total);
    }

    /**
     * Test 5: Análisis de estrategias aprendidas.
     */
    @Test
    @Order(5)
    void test5_AnalisisEstrategias() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 5: ANÁLISIS DE ESTRATEGIAS (Damas)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) test2_EntrenarModeloSpiking();

        // Apertura
        System.out.println("\n--- Situación 1: Apertura (primer movimiento) ---");
        Tablero tableroInicial = FuncionesDamas.inicialDamas();
        analizarDecision(tableroInicial, 1);

        // Medio juego
        System.out.println("\n--- Situación 2: Medio juego ---");
        int[][] estadoMedio = new int[8][8];
        estadoMedio[5][0] = 1;
        estadoMedio[5][2] = 3;
        estadoMedio[6][1] = 5;
        estadoMedio[2][1] = 2;
        estadoMedio[2][3] = 4;
        estadoMedio[3][2] = 6;
        analizarDecision(new Tablero(new SmallMatrix(estadoMedio)), 1);

        // Métricas
        System.out.println("\n--- Métricas Red Spiking ---");
        Map<String, Object> metricas = redSpiking.obtenerMetricas();
        metricas.forEach((k, v) -> System.out.println("  " + k + ": " + v));
    }

    // ==================== MÉTODOS AUXILIARES ====================

    private double[][] normalizarInputs(double[][] inputs) {
        double[][] norm = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            norm[i] = new double[inputs[i].length];
            for (int j = 0; j < inputs[i].length; j++) {
                norm[i][j] = (inputs[i][j] + 1.0) / 2.0;
            }
        }
        return norm;
    }

    private Movimiento obtenerMovimientoClasico(Tablero tablero, int turno, List<Movimiento> movsPosibles) {
        double[] input = ModeloDamas.tabularToInput(tablero, turno);
        double[] output = cerebroClasico.feedForward(input);
        return elegirMejorMovimiento(output, tablero, turno, movsPosibles);
    }

    private Movimiento obtenerMovimientoSpiking(Tablero tablero, int turno, List<Movimiento> movsPosibles) {
        double[] input = ModeloDamas.tabularToInput(tablero, turno);
        double[] inputNorm = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            inputNorm[i] = (input[i] + 1.0) / 2.0;
        }
        redSpiking.resetearEstadoTemporal();
        double[] output = redSpiking.procesar(inputNorm, DURACION_TIMESTEPS);
        return elegirMejorMovimiento(output, tablero, turno, movsPosibles);
    }

    private Movimiento elegirMejorMovimiento(double[] output, Tablero tablero, int turno,
                                              List<Movimiento> movsPosibles) {
        if (movsPosibles.isEmpty()) return null;

        Movimiento mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;

        for (Movimiento mov : movsPosibles) {
            Posicion origen = encontrarOrigen(tablero, mov.getTablero(), turno);
            Posicion destino = mov.getPos();
            if (origen == null) continue;

            int indexOrigen = origen.getFila() * 8 + origen.getColumna();
            int indexDestino = destino.getFila() * 8 + destino.getColumna();
            double valor = output[indexOrigen] + output[64 + indexDestino];

            if (valor > mejorValor) {
                mejorValor = valor;
                mejor = mov;
            }
        }
        return mejor != null ? mejor : movsPosibles.get(0);
    }

    private Posicion encontrarOrigen(Tablero antes, Tablero despues, int turno) {
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int valAntes = antes.getValor(f, c);
                int valDespues = despues.getValor(f, c);
                if (valAntes != 0 && valDespues == 0) {
                    boolean esBlanca = ModeloDamas.contiene(FuncionesDamas.nombresBlancas, valAntes) ||
                                      ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, valAntes);
                    boolean esNegra = ModeloDamas.contiene(FuncionesDamas.nombresNegras, valAntes) ||
                                     ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, valAntes);
                    if ((turno == 1 && esBlanca) || (turno == 2 && esNegra)) {
                        return new Posicion(f, c);
                    }
                }
            }
        }
        return null;
    }

    private int evaluarResultado(Tablero tablero, boolean empate) {
        if (empate) return 0;
        int blancas = 0, negras = 0;
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int val = tablero.getValor(f, c);
                if (ModeloDamas.contiene(FuncionesDamas.nombresBlancas, val) ||
                    ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, val)) blancas++;
                else if (ModeloDamas.contiene(FuncionesDamas.nombresNegras, val) ||
                         ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, val)) negras++;
            }
        }
        if (blancas > negras) return 1;
        if (negras > blancas) return -1;
        return 0;
    }

    private ResultadoPartidas jugarContraAleatorio(int numPartidas, boolean usarClasico) {
        int victorias = 0, derrotas = 0, empates = 0;
        Random rand = new Random(42);

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesDamas.inicialDamas();
            int turno = 1;
            int movimientos = 0;

            while (!FuncionesDamas.finDamas(tablero) && movimientos < 200) {
                List<Movimiento> movsPosibles = FuncionesDamas.movimientosDamas(
                    tablero, FuncionesDamas.bandoMarca(turno));
                if (movsPosibles.isEmpty()) break;

                Movimiento movimiento;
                if (turno == 1) {
                    movimiento = usarClasico ?
                        obtenerMovimientoClasico(tablero, turno, movsPosibles) :
                        obtenerMovimientoSpiking(tablero, turno, movsPosibles);
                } else {
                    movimiento = movsPosibles.get(rand.nextInt(movsPosibles.size()));
                }
                if (movimiento == null) break;

                tablero = movimiento.getTablero();
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }

            int resultado = evaluarResultado(tablero, movimientos >= 200);
            if (resultado == 1) victorias++;
            else if (resultado == -1) derrotas++;
            else empates++;
        }
        return new ResultadoPartidas(victorias, derrotas, empates);
    }

    private ResultadoEnfrentamiento enfrentarModelos(int numPartidas, boolean clasicoEsBlancas) {
        int victoriasP1 = 0, victoriasP2 = 0, empates = 0;

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesDamas.inicialDamas();
            int turno = 1;
            int movimientos = 0;

            while (!FuncionesDamas.finDamas(tablero) && movimientos < 200) {
                List<Movimiento> movsPosibles = FuncionesDamas.movimientosDamas(
                    tablero, FuncionesDamas.bandoMarca(turno));
                if (movsPosibles.isEmpty()) break;

                boolean esTurnoClasico = (clasicoEsBlancas && turno == 1) || (!clasicoEsBlancas && turno == 2);
                Movimiento movimiento = esTurnoClasico ?
                    obtenerMovimientoClasico(tablero, turno, movsPosibles) :
                    obtenerMovimientoSpiking(tablero, turno, movsPosibles);

                if (movimiento == null) break;
                tablero = movimiento.getTablero();
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }

            int resultado = evaluarResultado(tablero, movimientos >= 200);
            if (resultado == 1) victoriasP1++;
            else if (resultado == -1) victoriasP2++;
            else empates++;
        }
        return new ResultadoEnfrentamiento(victoriasP1, victoriasP2, empates);
    }

    private void analizarDecision(Tablero tablero, int turno) {
        imprimirTablero(tablero);
        List<Movimiento> movsPosibles = FuncionesDamas.movimientosDamas(
            tablero, FuncionesDamas.bandoMarca(turno));
        if (movsPosibles.isEmpty()) {
            System.out.println("No hay movimientos posibles");
            return;
        }

        Movimiento movClasico = obtenerMovimientoClasico(tablero, turno, movsPosibles);
        Movimiento movSpiking = obtenerMovimientoSpiking(tablero, turno, movsPosibles);

        Posicion origenC = encontrarOrigen(tablero, movClasico.getTablero(), turno);
        Posicion origenS = encontrarOrigen(tablero, movSpiking.getTablero(), turno);

        System.out.println("Decisión Clásico:  " + posStr(origenC) + " -> " + posStr(movClasico.getPos()));
        System.out.println("Decisión Spiking:  " + posStr(origenS) + " -> " + posStr(movSpiking.getPos()));
        System.out.println("Movimientos posibles: " + movsPosibles.size());
    }

    private void imprimirTablero(Tablero tablero) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if ((i + j) % 2 != 0) { System.out.print("  "); continue; }
                int val = tablero.getValor(i, j);
                if (val == 0) System.out.print(". ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresBlancas, val)) System.out.print("b ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, val)) System.out.print("B ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresNegras, val)) System.out.print("n ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, val)) System.out.print("N ");
                else System.out.print("? ");
            }
            System.out.println();
        }
    }

    private String posStr(Posicion p) {
        return p == null ? "?" : "(" + p.getFila() + "," + p.getColumna() + ")";
    }

    private void mostrarResultados(ResultadoPartidas r) {
        System.out.printf("Victorias: %d (%.1f%%)  Derrotas: %d (%.1f%%)  Empates: %d (%.1f%%)%n",
            r.victorias, r.porcentajeVictorias(),
            r.derrotas, r.porcentajeDerrotas(),
            r.empates, r.porcentajeEmpates());
    }

    private void mostrarEnfrentamiento(ResultadoEnfrentamiento r, String p1, String p2) {
        System.out.printf("%s: %d (%.1f%%)  %s: %d (%.1f%%)  Empates: %d (%.1f%%)%n",
            p1, r.victoriasP1, r.porcentajeP1(),
            p2, r.victoriasP2, r.porcentajeP2(),
            r.empates, r.porcentajeEmpates());
    }

    private static class ResultadoPartidas {
        int victorias, derrotas, empates;
        ResultadoPartidas(int v, int d, int e) { victorias = v; derrotas = d; empates = e; }
        int total() { return victorias + derrotas + empates; }
        double porcentajeVictorias() { return total() > 0 ? victorias * 100.0 / total() : 0; }
        double porcentajeDerrotas() { return total() > 0 ? derrotas * 100.0 / total() : 0; }
        double porcentajeEmpates() { return total() > 0 ? empates * 100.0 / total() : 0; }
    }

    private static class ResultadoEnfrentamiento {
        int victoriasP1, victoriasP2, empates;
        ResultadoEnfrentamiento(int p1, int p2, int e) { victoriasP1 = p1; victoriasP2 = p2; empates = e; }
        int total() { return victoriasP1 + victoriasP2 + empates; }
        double porcentajeP1() { return total() > 0 ? victoriasP1 * 100.0 / total() : 0; }
        double porcentajeP2() { return total() > 0 ? victoriasP2 * 100.0 / total() : 0; }
        double porcentajeEmpates() { return total() > 0 ? empates * 100.0 / total() : 0; }
    }
}
