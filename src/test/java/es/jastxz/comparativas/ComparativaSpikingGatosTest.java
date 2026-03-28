package es.jastxz.comparativas;

import es.jastxz.engine.FuncionesGato;
import es.jastxz.models.ModeloGatos;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.spiking.*;
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
 * para el juego de Gatos (Ratón vs Gatos).
 *
 * El modelo clásico NO se entrena aquí — se carga pre-entrenado.
 * El modelo spiking se entrena con datos supervisados generados por Minimax.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComparativaSpikingGatosTest {

    private static NeuralNetwork cerebroClasico;
    private static RedNeuralSpiking redSpiking;
    private static final int DURACION_TIMESTEPS = 50;
    private static final Posicion POS_MOUSE_INICIAL = new Posicion(0, 2);

    @BeforeAll
    static void cargarModeloClasico() {
        cerebroClasico = ModelManager.loadModel("modeloGatos.nn");
        assertNotNull(cerebroClasico, "El modelo clásico de Gatos debe estar pre-entrenado");
    }

    /**
     * Test 1: Verificar que el modelo clásico está entrenado y funciona.
     */
    @Test
    @Order(1)
    void test1_ModeloClasicoPreEntrenado() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: VERIFICAR MODELO CLÁSICO PRE-ENTRENADO (Gatos)");
        System.out.println("=".repeat(60));

        Tablero tableroInicial = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
        double[] input = ModeloGatos.tabularToInput(tableroInicial, 1);
        double[] output = cerebroClasico.feedForward(input);

        assertNotNull(output, "La salida no debe ser null");
        assertEquals(64, output.length, "La salida debe tener 64 valores (8x8)");

        double max = Arrays.stream(output).max().orElse(0);
        assertTrue(max > 0, "Al menos una salida debe ser positiva");

        System.out.println("✓ Modelo clásico de Gatos cargado y funcional");
    }

    /**
     * Test 2: Entrenar modelo spiking con datos supervisados de Gatos.
     */
    @Test
    @Order(2)
    void test2_EntrenarModeloSpiking() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: ENTRENAMIENTO RED SPIKING (Gatos)");
        System.out.println("=".repeat(60));

        // Crear red spiking: 65 entrada, 64 oculta, 64 salida
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(65, 64, 64)
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

        // Generar datos de entrenamiento
        ModeloGatos modelo = new ModeloGatos();
        var data = modelo.generateTrainingData();
        double[][] inputs = data.getInputs();
        double[][] outputs = data.getOutputs();

        System.out.println("Datos de entrenamiento: " + inputs.length + " muestras");

        // Usar un subconjunto para no tardar demasiado
        int maxMuestras = Math.min(inputs.length, 200);
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
        System.out.println("TEST 3: RENDIMIENTO CONTRA JUGADOR ALEATORIO (Gatos)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) test2_EntrenarModeloSpiking();

        int partidasPrueba = 50;

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
        System.out.println("TEST 4: ENFRENTAMIENTO DIRECTO (Gatos)");
        System.out.println("Clásico (Backprop) vs Spiking (STDP)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) test2_EntrenarModeloSpiking();

        int partidasPorRonda = 20;

        // Ronda 1: Clásico como Ratón
        System.out.println("\n--- Ronda 1: Clásico (Ratón) vs Spiking (Gatos) ---");
        ResultadoEnfrentamiento ronda1 = enfrentarModelos(partidasPorRonda, true);
        mostrarEnfrentamiento(ronda1, "Clásico", "Spiking");

        // Ronda 2: Spiking como Ratón
        System.out.println("\n--- Ronda 2: Spiking (Ratón) vs Clásico (Gatos) ---");
        ResultadoEnfrentamiento ronda2 = enfrentarModelos(partidasPorRonda, false);
        mostrarEnfrentamiento(ronda2, "Spiking", "Clásico");

        int total = partidasPorRonda * 2;
        int victoriasClasico = ronda1.victoriasP1 + ronda2.victoriasP2;
        int victoriasSpiking = ronda1.victoriasP2 + ronda2.victoriasP1;

        System.out.println("\n--- RESUMEN TOTAL (" + total + " partidas) ---");
        System.out.printf("Clásico: %d (%.1f%%)%n", victoriasClasico, victoriasClasico * 100.0 / total);
        System.out.printf("Spiking: %d (%.1f%%)%n", victoriasSpiking, victoriasSpiking * 100.0 / total);
    }

    /**
     * Test 5: Análisis de estrategias aprendidas.
     */
    @Test
    @Order(5)
    void test5_AnalisisEstrategias() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 5: ANÁLISIS DE ESTRATEGIAS (Gatos)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) test2_EntrenarModeloSpiking();

        // Apertura
        System.out.println("\n--- Situación 1: Apertura (primer movimiento Ratón) ---");
        Tablero tableroInicial = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
        compararDecisiones(tableroInicial, 1);

        // Ratón cerca de la meta
        System.out.println("\n--- Situación 2: Ratón cerca de la victoria (fila 6) ---");
        int[][] estadoCerca = new int[8][8];
        estadoCerca[6][2] = FuncionesGato.nombreRaton;
        estadoCerca[7][1] = 1;
        estadoCerca[7][3] = 3;
        estadoCerca[7][5] = 5;
        compararDecisiones(new Tablero(new SmallMatrix(estadoCerca)), 1);

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

    private Posicion obtenerMovimientoClasico(Tablero tablero, int turno) {
        double[] input = ModeloGatos.tabularToInput(tablero, turno);
        double[] output = cerebroClasico.feedForward(input);
        return elegirMejorMovimiento(output, tablero, turno);
    }

    private Posicion obtenerMovimientoSpiking(Tablero tablero, int turno) {
        double[] input = ModeloGatos.tabularToInput(tablero, turno);
        double[] inputNorm = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            inputNorm[i] = (input[i] + 1.0) / 2.0;
        }
        redSpiking.resetearEstadoTemporal();
        double[] output = redSpiking.procesar(inputNorm, DURACION_TIMESTEPS);
        return elegirMejorMovimiento(output, tablero, turno);
    }

    private Posicion elegirMejorMovimiento(double[] output, Tablero tablero, int turno) {
        List<Posicion> movsPosibles = generarMovimientosLegales(tablero, turno);
        if (movsPosibles.isEmpty()) return null;

        Posicion mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        for (Posicion pos : movsPosibles) {
            int index = pos.getFila() * 8 + pos.getColumna();
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejor = pos;
            }
        }
        return mejor != null ? mejor : movsPosibles.get(0);
    }

    private List<Posicion> generarMovimientosLegales(Tablero tablero, int turno) {
        List<Posicion> movimientos = new ArrayList<>();
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int val = tablero.getValor(f, c);
                boolean esMio = (turno == 1 && val == FuncionesGato.nombreRaton) ||
                                (turno == 2 && val != 0 && val != FuncionesGato.nombreRaton);
                if (!esMio) continue;

                int[][] dirs = (turno == 1) ?
                    new int[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}} :
                    new int[][]{{-1, -1}, {-1, 1}};

                for (int[] d : dirs) {
                    int nf = f + d[0], nc = c + d[1];
                    if (nf >= 0 && nf < 8 && nc >= 0 && nc < 8 &&
                        (nf + nc) % 2 == 0 && tablero.getValor(nf, nc) == 0) {
                        movimientos.add(new Posicion(nf, nc));
                    }
                }
            }
        }
        return movimientos;
    }

    private Tablero aplicarMovimiento(Tablero tablero, Posicion destino, int turno) {
        Posicion origen = encontrarOrigen(tablero, destino, turno);
        if (origen == null) return tablero;

        int[][] nuevo = new int[8][8];
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                nuevo[i][j] = tablero.getValor(i, j);

        nuevo[destino.getFila()][destino.getColumna()] = tablero.getValor(origen.getFila(), origen.getColumna());
        nuevo[origen.getFila()][origen.getColumna()] = 0;
        return new Tablero(new SmallMatrix(nuevo));
    }

    private Posicion encontrarOrigen(Tablero tablero, Posicion destino, int turno) {
        int[][] dirs = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        for (int[] d : dirs) {
            int f = destino.getFila() + d[0], c = destino.getColumna() + d[1];
            if (f >= 0 && f < 8 && c >= 0 && c < 8) {
                int val = tablero.getValor(f, c);
                if (turno == 1 && val == FuncionesGato.nombreRaton) return new Posicion(f, c);
                if (turno == 2 && val != 0 && val != FuncionesGato.nombreRaton) return new Posicion(f, c);
            }
        }
        return null;
    }

    private Posicion encontrarRaton(Tablero tablero) {
        for (int f = 0; f < 8; f++)
            for (int c = 0; c < 8; c++)
                if (tablero.getValor(f, c) == FuncionesGato.nombreRaton)
                    return new Posicion(f, c);
        return null;
    }

    private ResultadoPartidas jugarContraAleatorio(int numPartidas, boolean usarClasico) {
        int victorias = 0, derrotas = 0;
        Random rand = new Random(42);

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
            int turno = 1;
            int movimientos = 0;

            while (!FuncionesGato.finGato(tablero) && movimientos < 100) {
                Posicion movimiento;
                if (turno == 1) {
                    movimiento = usarClasico ?
                        obtenerMovimientoClasico(tablero, turno) :
                        obtenerMovimientoSpiking(tablero, turno);
                } else {
                    List<Posicion> movs = generarMovimientosLegales(tablero, turno);
                    movimiento = movs.isEmpty() ? null : movs.get(rand.nextInt(movs.size()));
                }
                if (movimiento == null) break;

                tablero = aplicarMovimiento(tablero, movimiento, turno);
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }

            Posicion raton = encontrarRaton(tablero);
            boolean ratonGano = raton != null && !FuncionesGato.ratonEncerrado(tablero, raton);
            if (ratonGano) victorias++;
            else derrotas++;
        }
        return new ResultadoPartidas(victorias, derrotas, 0);
    }

    private ResultadoEnfrentamiento enfrentarModelos(int numPartidas, boolean clasicoEsRaton) {
        int victoriasP1 = 0, victoriasP2 = 0;

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
            int turno = 1;
            int movimientos = 0;

            while (!FuncionesGato.finGato(tablero) && movimientos < 100) {
                boolean esTurnoClasico = (clasicoEsRaton && turno == 1) || (!clasicoEsRaton && turno == 2);
                Posicion movimiento = esTurnoClasico ?
                    obtenerMovimientoClasico(tablero, turno) :
                    obtenerMovimientoSpiking(tablero, turno);

                if (movimiento == null) break;
                tablero = aplicarMovimiento(tablero, movimiento, turno);
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }

            Posicion raton = encontrarRaton(tablero);
            boolean ratonGano = raton != null && !FuncionesGato.ratonEncerrado(tablero, raton);
            if (clasicoEsRaton) {
                if (ratonGano) victoriasP1++; else victoriasP2++;
            } else {
                if (ratonGano) victoriasP2++; else victoriasP1++;
            }
        }
        return new ResultadoEnfrentamiento(victoriasP1, victoriasP2, 0);
    }

    private void compararDecisiones(Tablero tablero, int turno) {
        imprimirTablero(tablero);
        Posicion movClasico = obtenerMovimientoClasico(tablero, turno);
        Posicion movSpiking = obtenerMovimientoSpiking(tablero, turno);

        System.out.println("Decisión Clásico:  " + posStr(movClasico));
        System.out.println("Decisión Spiking:  " + posStr(movSpiking));
        System.out.println(movClasico != null && movClasico.equals(movSpiking) ? "✓ Coinciden" : "✗ Difieren");
    }

    private void imprimirTablero(Tablero tablero) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if ((i + j) % 2 != 0) { System.out.print("  "); continue; }
                int val = tablero.getValor(i, j);
                if (val == 0) System.out.print(". ");
                else if (val == FuncionesGato.nombreRaton) System.out.print("R ");
                else System.out.print("G ");
            }
            System.out.println();
        }
    }

    private String posStr(Posicion p) {
        return p == null ? "null" : "(" + p.getFila() + "," + p.getColumna() + ")";
    }

    private void mostrarResultados(ResultadoPartidas r) {
        System.out.printf("Victorias: %d (%.1f%%)  Derrotas: %d (%.1f%%)%n",
            r.victorias, r.porcentajeVictorias(), r.derrotas, r.porcentajeDerrotas());
    }

    private void mostrarEnfrentamiento(ResultadoEnfrentamiento r, String p1, String p2) {
        System.out.printf("%s: %d (%.1f%%)  %s: %d (%.1f%%)%n",
            p1, r.victoriasP1, r.porcentajeP1(), p2, r.victoriasP2, r.porcentajeP2());
    }

    private static class ResultadoPartidas {
        int victorias, derrotas, empates;
        ResultadoPartidas(int v, int d, int e) { victorias = v; derrotas = d; empates = e; }
        int total() { return victorias + derrotas + empates; }
        double porcentajeVictorias() { return total() > 0 ? victorias * 100.0 / total() : 0; }
        double porcentajeDerrotas() { return total() > 0 ? derrotas * 100.0 / total() : 0; }
    }

    private static class ResultadoEnfrentamiento {
        int victoriasP1, victoriasP2, empates;
        ResultadoEnfrentamiento(int p1, int p2, int e) { victoriasP1 = p1; victoriasP2 = p2; empates = e; }
        int total() { return victoriasP1 + victoriasP2 + empates; }
        double porcentajeP1() { return total() > 0 ? victoriasP1 * 100.0 / total() : 0; }
        double porcentajeP2() { return total() > 0 ? victoriasP2 * 100.0 / total() : 0; }
    }
}
