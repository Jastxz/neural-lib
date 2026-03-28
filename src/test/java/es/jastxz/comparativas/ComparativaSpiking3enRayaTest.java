package es.jastxz.comparativas;

import es.jastxz.engine.Funciones3enRaya;
import es.jastxz.models.Modelo3enRaya;
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
 * para el juego de 3 en Raya.
 * 
 * El modelo clásico NO se entrena aquí — se carga pre-entrenado.
 * El modelo spiking se entrena con los mismos datos supervisados.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComparativaSpiking3enRayaTest {

    private static NeuralNetwork cerebroClasico;
    private static RedNeuralSpiking redSpiking;
    private static final int DURACION_TIMESTEPS = 50;

    @BeforeAll
    static void cargarModeloClasico() {
        cerebroClasico = ModelManager.loadModel("modelo3enRaya.nn");
        assertNotNull(cerebroClasico, "El modelo clásico de 3 en Raya debe estar pre-entrenado");
    }

    /**
     * Test 1: Verificar que el modelo clásico está entrenado y funciona.
     */
    @Test
    @Order(1)
    void test1_ModeloClasicoPreEntrenado() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: VERIFICAR MODELO CLÁSICO PRE-ENTRENADO");
        System.out.println("=".repeat(60));

        // Verificar que produce salidas válidas en tablero vacío
        Tablero tableroVacio = Funciones3enRaya.inicial3enRaya();
        double[] input = Modelo3enRaya.tabularToInput(tableroVacio, 1);
        double[] output = cerebroClasico.feedForward(input);

        assertNotNull(output, "La salida no debe ser null");
        assertEquals(9, output.length, "La salida debe tener 9 valores (una por casilla)");

        // Verificar que al menos una salida es positiva
        double max = Arrays.stream(output).max().orElse(0);
        assertTrue(max > 0, "Al menos una salida debe ser positiva");

        System.out.println("✓ Modelo clásico cargado y funcional");
        System.out.println("Salida para tablero vacío: " + Arrays.toString(output));
    }

    /**
     * Test 2: Entrenar modelo spiking con datos supervisados de 3 en Raya.
     */
    @Test
    @Order(2)
    void test2_EntrenarModeloSpiking() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: ENTRENAMIENTO RED SPIKING (STDP supervisado)");
        System.out.println("=".repeat(60));

        // Crear red spiking: 10 entrada, 20 oculta, 9 salida
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 20, 9)
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
        Modelo3enRaya modelo = new Modelo3enRaya();
        var data = modelo.generateTrainingData(modelo.getMundo());
        double[][] inputs = data.getInputs();
        double[][] outputs = data.getOutputs();

        System.out.println("Datos de entrenamiento: " + inputs.length + " muestras");

        // Normalizar inputs a rango [0,1] para codificación spiking
        double[][] inputsNorm = normalizarInputs(inputs);

        // Entrenar por varias épocas
        long inicio = System.currentTimeMillis();
        int epocas = 3;
        for (int e = 0; e < epocas; e++) {
            double error = redSpiking.entrenar(inputsNorm, outputs, DURACION_TIMESTEPS);
            System.out.printf("Época %d/%d - Error MSE: %.6f%n", e + 1, epocas, error);
        }
        long tiempo = System.currentTimeMillis() - inicio;

        System.out.println("\n✓ Red spiking entrenada");
        System.out.println("Tiempo: " + tiempo + " ms");
        System.out.println("Método: STDP + corrección supervisada");
    }

    /**
     * Test 3: Comparar rendimiento contra jugador aleatorio.
     */
    @Test
    @Order(3)
    void test3_CompararContraAleatorio() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: RENDIMIENTO CONTRA JUGADOR ALEATORIO");
        System.out.println("=".repeat(60));

        // Asegurar que la red spiking está entrenada
        if (redSpiking == null) {
            test2_EntrenarModeloSpiking();
        }

        int partidasPrueba = 100;

        System.out.println("\n--- Modelo Clásico (Backprop) vs Aleatorio ---");
        ResultadoPartidas resultadoClasico = jugarContraAleatorio(
            cerebroClasico, null, partidasPrueba, true);
        mostrarResultados(resultadoClasico);

        System.out.println("\n--- Modelo Spiking (STDP) vs Aleatorio ---");
        ResultadoPartidas resultadoSpiking = jugarContraAleatorio(
            null, redSpiking, partidasPrueba, false);
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
        System.out.println("TEST 4: ENFRENTAMIENTO DIRECTO");
        System.out.println("Clásico (Backprop) vs Spiking (STDP)");
        System.out.println("=".repeat(60));

        if (redSpiking == null) {
            test2_EntrenarModeloSpiking();
        }

        int partidasPorRonda = 50;

        // Ronda 1: Clásico juega primero (P1)
        System.out.println("\n--- Ronda 1: Clásico (P1) vs Spiking (P2) ---");
        ResultadoEnfrentamiento ronda1 = enfrentarModelos(partidasPorRonda, true);
        mostrarEnfrentamiento(ronda1, "Clásico", "Spiking");

        // Ronda 2: Spiking juega primero (P1)
        System.out.println("\n--- Ronda 2: Spiking (P1) vs Clásico (P2) ---");
        ResultadoEnfrentamiento ronda2 = enfrentarModelos(partidasPorRonda, false);
        mostrarEnfrentamiento(ronda2, "Spiking", "Clásico");

        // Resumen
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
        System.out.println("TEST 5: ANÁLISIS DE ESTRATEGIAS");
        System.out.println("=".repeat(60));

        if (redSpiking == null) {
            test2_EntrenarModeloSpiking();
        }

        // Tablero vacío
        System.out.println("\n--- Situación 1: Tablero vacío (primer movimiento) ---");
        Tablero tableroVacio = Funciones3enRaya.inicial3enRaya();
        compararDecisiones(tableroVacio, 1);

        // Bloquear victoria inminente
        System.out.println("\n--- Situación 2: Bloquear victoria del oponente ---");
        int[][] estadoBloqueo = {{1, 1, 0}, {2, 0, 0}, {0, 0, 0}};
        compararDecisiones(new Tablero(new SmallMatrix(estadoBloqueo)), 2);

        // Oportunidad de ganar
        System.out.println("\n--- Situación 3: Oportunidad de ganar ---");
        int[][] estadoGanar = {{1, 1, 0}, {2, 2, 0}, {0, 0, 0}};
        compararDecisiones(new Tablero(new SmallMatrix(estadoGanar)), 1);

        // Métricas de la red spiking
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
                // Mapear [-1, 1] a [0, 1]
                norm[i][j] = (inputs[i][j] + 1.0) / 2.0;
            }
        }
        return norm;
    }

    private Posicion obtenerMovimientoClasico(Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] output = cerebroClasico.feedForward(input);
        return elegirMejorMovimiento(output, tablero, turno);
    }

    private Posicion obtenerMovimientoSpiking(Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] inputNorm = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            inputNorm[i] = (input[i] + 1.0) / 2.0;
        }
        redSpiking.resetearEstadoTemporal();
        double[] output = redSpiking.procesar(inputNorm, DURACION_TIMESTEPS);
        return elegirMejorMovimiento(output, tablero, turno);
    }

    private Posicion elegirMejorMovimiento(double[] output, Tablero tablero, int turno) {
        List<Movimiento> movsPosibles = Funciones3enRaya.movs3enRaya(tablero, turno);
        Posicion mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;

        for (Movimiento mov : movsPosibles) {
            int index = mov.getPos().getFila() * 3 + mov.getPos().getColumna();
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejor = mov.getPos();
            }
        }
        return mejor != null ? mejor : movsPosibles.get(0).getPos();
    }

    private ResultadoPartidas jugarContraAleatorio(
            NeuralNetwork clasico, RedNeuralSpiking spiking, int numPartidas, boolean usarClasico) {
        int victorias = 0, derrotas = 0, empates = 0;
        Random rand = new Random(42);

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                if (turno == 1) {
                    movimiento = usarClasico ?
                        obtenerMovimientoClasico(tablero, turno) :
                        obtenerMovimientoSpiking(tablero, turno);
                } else {
                    List<Movimiento> movs = Funciones3enRaya.movs3enRaya(tablero, turno);
                    movimiento = movs.get(rand.nextInt(movs.size())).getPos();
                }

                int[][] nuevo = copiarTablero(tablero);
                nuevo[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevo));
                turno = (turno == 1) ? 2 : 1;
            }

            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victorias++;
            else if (resultado == -1) derrotas++;
            else empates++;
        }
        return new ResultadoPartidas(victorias, derrotas, empates);
    }

    private ResultadoEnfrentamiento enfrentarModelos(int numPartidas, boolean clasicoEsP1) {
        int victoriasP1 = 0, victoriasP2 = 0, empates = 0;

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                boolean esTurnoClasico = (clasicoEsP1 && turno == 1) || (!clasicoEsP1 && turno == 2);

                movimiento = esTurnoClasico ?
                    obtenerMovimientoClasico(tablero, turno) :
                    obtenerMovimientoSpiking(tablero, turno);

                int[][] nuevo = copiarTablero(tablero);
                nuevo[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevo));
                turno = (turno == 1) ? 2 : 1;
            }

            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victoriasP1++;
            else if (resultado == -1) victoriasP2++;
            else empates++;
        }
        return new ResultadoEnfrentamiento(victoriasP1, victoriasP2, empates);
    }

    private void compararDecisiones(Tablero tablero, int turno) {
        imprimirTablero(tablero);
        Posicion movClasico = obtenerMovimientoClasico(tablero, turno);
        Posicion movSpiking = obtenerMovimientoSpiking(tablero, turno);

        System.out.println("Decisión Clásico:  (" + movClasico.getFila() + "," + movClasico.getColumna() + ")");
        System.out.println("Decisión Spiking:  (" + movSpiking.getFila() + "," + movSpiking.getColumna() + ")");
        System.out.println(movClasico.equals(movSpiking) ? "✓ Coinciden" : "✗ Difieren");
    }

    private int[][] copiarTablero(Tablero tablero) {
        int[][] copia = new int[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                copia[i][j] = tablero.getValor(i, j);
        return copia;
    }

    private int evaluarResultado(Tablero tablero) {
        if (!Funciones3enRaya.hay3EnRaya(tablero)) return 0;
        int c1 = 0, c2 = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                if (tablero.getValor(i, j) == 1) c1++;
                else if (tablero.getValor(i, j) == 2) c2++;
            }
        return (c1 == c2) ? -1 : 1;
    }

    private void imprimirTablero(Tablero tablero) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int val = tablero.getValor(i, j);
                System.out.print(val == 0 ? "." : (val == 1 ? "X" : "O"));
                System.out.print(" ");
            }
            System.out.println();
        }
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

    // Clases auxiliares
    private static class ResultadoPartidas {
        int victorias, derrotas, empates;
        ResultadoPartidas(int v, int d, int e) { victorias = v; derrotas = d; empates = e; }
        int total() { return victorias + derrotas + empates; }
        double porcentajeVictorias() { return victorias * 100.0 / total(); }
        double porcentajeDerrotas() { return derrotas * 100.0 / total(); }
        double porcentajeEmpates() { return empates * 100.0 / total(); }
    }

    private static class ResultadoEnfrentamiento {
        int victoriasP1, victoriasP2, empates;
        ResultadoEnfrentamiento(int p1, int p2, int e) { victoriasP1 = p1; victoriasP2 = p2; empates = e; }
        int total() { return victoriasP1 + victoriasP2 + empates; }
        double porcentajeP1() { return victoriasP1 * 100.0 / total(); }
        double porcentajeP2() { return victoriasP2 * 100.0 / total(); }
        double porcentajeEmpates() { return empates * 100.0 / total(); }
    }
}
