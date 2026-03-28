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
import org.junit.jupiter.api.Timeout;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa entre Red Neuronal Clásica (Backpropagation), Red Spiking (STDP)
 * y Red Spiking optimizada con Algoritmo Genético (AG+SNN) para 3 en Raya.
 *
 * <p>El modelo clásico se carga pre-entrenado. El modelo STDP se entrena
 * directamente. El modelo AG+SNN usa {@link OptimizadorSNN} para encontrar
 * la mejor arquitectura y parámetros automáticamente.</p>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComparativaAGSpiking3enRayaTest {

    private static NeuralNetwork cerebroClasico;
    private static RedNeuralSpiking redSTDP;
    private static RedNeuralSpiking redAGSNN;
    private static ConfiguracionRed configAGSNN;

    private static double[][] todosInputsNorm;
    private static double[][] todosOutputs;

    private static final int DURACION_TIMESTEPS = 100;
    /** Subconjunto de muestras para el AG (el dataset completo es ~4500). */
    private static final int MUESTRAS_AG = 1000;

    @BeforeAll
    static void prepararDatos() {
        // Cargar modelo clásico
        cerebroClasico = ModelManager.loadModel("modelo3enRaya.nn");
        assertNotNull(cerebroClasico, "El modelo clásico debe estar pre-entrenado");

        // Generar datos de entrenamiento
        Modelo3enRaya modelo = new Modelo3enRaya();
        var data = modelo.generateTrainingData(modelo.getMundo());
        double[][] inputs = data.getInputs();
        todosOutputs = data.getOutputs();
        todosInputsNorm = normalizarInputs(inputs);

        System.out.println("Datos de entrenamiento generados: " + inputs.length + " muestras");
        System.out.println("Dimensiones: " + inputs[0].length + " entradas → "
                + todosOutputs[0].length + " salidas");
    }

    // ================================================================
    //  Test 1: Verificar modelo clásico
    // ================================================================

    @Test
    @Order(1)
    void test1_ModeloClasicoFuncional() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: VERIFICAR MODELO CLÁSICO PRE-ENTRENADO");
        System.out.println("=".repeat(60));

        Tablero tableroVacio = Funciones3enRaya.inicial3enRaya();
        double[] input = Modelo3enRaya.tabularToInput(tableroVacio, 1);
        double[] output = cerebroClasico.feedForward(input);

        assertNotNull(output);
        assertEquals(9, output.length);
        assertTrue(Arrays.stream(output).max().orElse(0) > 0);

        System.out.println("✓ Modelo clásico cargado y funcional");
    }

    // ================================================================
    //  Test 2: Entrenar modelo STDP (supervisado directo)
    // ================================================================

    @Test
    @Order(2)
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void test2_EntrenarModeloSTDP() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: ENTRENAMIENTO RED SPIKING (STDP supervisado)");
        System.out.println("=".repeat(60));

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

        redSTDP = new RedNeuralSpiking(config);
        redSTDP.inicializarPesos(TipoInicializacion.UNIFORME, 0.1, 0.5);

        long inicio = System.currentTimeMillis();
        int epocas = 3;
        for (int e = 0; e < epocas; e++) {
            double error = redSTDP.entrenar(todosInputsNorm, todosOutputs, DURACION_TIMESTEPS);
            System.out.printf("  Época %d/%d — Error MSE: %.6f%n", e + 1, epocas, error);
        }
        long tiempo = System.currentTimeMillis() - inicio;

        System.out.println("✓ Red STDP entrenada en " + tiempo + " ms");
    }

    // ================================================================
    //  Test 3: Optimizar modelo AG+SNN
    // ================================================================

    @Test
    @Order(3)
    void test3_OptimizarModeloAGSNN() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: OPTIMIZACIÓN AG+SNN");
        System.out.println("=".repeat(60));

        // Seleccionar subconjunto representativo para el AG
        double[][] inputsSub = new double[MUESTRAS_AG][];
        double[][] outputsSub = new double[MUESTRAS_AG][];
        seleccionarSubconjunto(todosInputsNorm, todosOutputs, inputsSub, outputsSub);

        System.out.println("Subconjunto para AG: " + MUESTRAS_AG + " muestras");
        System.out.println("Dimensiones: " + inputsSub[0].length + " → " + outputsSub[0].length);

        long inicio = System.currentTimeMillis();

        ResultadoOptimizacion resultado = new OptimizadorSNN.Builder()
                .tamañoPoblacion(50)
                .maxGeneraciones(100)
                .generacionesEstancamiento(10)
                .limiteTopologico(32)
                .pesoPrecision(0.9)
                .pesoEnergia(0.0)
                .pesoTamanio(0.1)
                .duracionTimesteps(100)
                .epocasEntrenamiento(5)
                .optimizar(inputsSub, outputsSub);

        long tiempo = System.currentTimeMillis() - inicio;

        redAGSNN = resultado.red();
        configAGSNN = resultado.configuracionRed();

        System.out.printf("✓ AG+SNN optimizado en %d ms%n", tiempo);
        System.out.printf("  Precisión: %.1f%%%n", resultado.precision() * 100);
        System.out.printf("  Fitness:   %.4f%n", resultado.fitness());
        System.out.printf("  Topología: %s%n", Arrays.toString(configAGSNN.getTopologia()));

        assertNotNull(redAGSNN);
        assertTrue(resultado.precision() > 0, "AG+SNN debe aprender algo");
    }

    // ================================================================
    //  Test 4: Comparar contra jugador aleatorio
    // ================================================================

    @Test
    @Order(4)
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void test4_CompararContraAleatorio() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: RENDIMIENTO CONTRA JUGADOR ALEATORIO");
        System.out.println("=".repeat(60));

        asegurarModelos();
        int partidas = 100;

        System.out.println("\n--- Clásico (Backprop) vs Aleatorio ---");
        int[] rClasico = jugarContraAleatorio(partidas, "clasico");
        mostrarResultados(rClasico, partidas);

        System.out.println("\n--- STDP vs Aleatorio ---");
        int[] rSTDP = jugarContraAleatorio(partidas, "stdp");
        mostrarResultados(rSTDP, partidas);

        System.out.println("\n--- AG+SNN vs Aleatorio ---");
        int[] rAG = jugarContraAleatorio(partidas, "ag");
        mostrarResultados(rAG, partidas);

        System.out.println("\n--- Comparación ---");
        System.out.printf("  Clásico: %.1f%% victorias%n", rClasico[0] * 100.0 / partidas);
        System.out.printf("  STDP:    %.1f%% victorias%n", rSTDP[0] * 100.0 / partidas);
        System.out.printf("  AG+SNN:  %.1f%% victorias%n", rAG[0] * 100.0 / partidas);
    }

    // ================================================================
    //  Test 5: Enfrentamiento directo entre los 3 modelos
    // ================================================================

    @Test
    @Order(5)
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void test5_EnfrentamientoDirecto() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 5: ENFRENTAMIENTO DIRECTO (3 modelos)");
        System.out.println("=".repeat(60));

        asegurarModelos();
        int partidas = 50;

        // Clásico vs STDP
        System.out.println("\n--- Clásico vs STDP ---");
        int[] r1 = enfrentar(partidas, "clasico", "stdp");
        System.out.printf("  Clásico: %d  STDP: %d  Empates: %d%n", r1[0], r1[1], r1[2]);

        // Clásico vs AG+SNN
        System.out.println("\n--- Clásico vs AG+SNN ---");
        int[] r2 = enfrentar(partidas, "clasico", "ag");
        System.out.printf("  Clásico: %d  AG+SNN: %d  Empates: %d%n", r2[0], r2[1], r2[2]);

        // STDP vs AG+SNN
        System.out.println("\n--- STDP vs AG+SNN ---");
        int[] r3 = enfrentar(partidas, "stdp", "ag");
        System.out.printf("  STDP: %d  AG+SNN: %d  Empates: %d%n", r3[0], r3[1], r3[2]);

        // Resumen
        int vClasico = r1[0] + r2[0];
        int vSTDP = r1[1] + r3[0];
        int vAG = r2[1] + r3[1];

        System.out.println("\n--- RESUMEN GLOBAL ---");
        System.out.printf("  Clásico: %d victorias%n", vClasico);
        System.out.printf("  STDP:    %d victorias%n", vSTDP);
        System.out.printf("  AG+SNN:  %d victorias%n", vAG);
    }

    // ================================================================
    //  Test 6: Análisis de estrategias
    // ================================================================

    @Test
    @Order(6)
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void test6_AnalisisEstrategias() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 6: ANÁLISIS DE ESTRATEGIAS");
        System.out.println("=".repeat(60));

        asegurarModelos();

        // Tablero vacío
        System.out.println("\n--- Tablero vacío (primer movimiento) ---");
        Tablero vacio = Funciones3enRaya.inicial3enRaya();
        compararDecisiones(vacio, 1);

        // Bloquear victoria
        System.out.println("\n--- Bloquear victoria del oponente ---");
        int[][] bloqueo = {{1, 1, 0}, {2, 0, 0}, {0, 0, 0}};
        compararDecisiones(new Tablero(new SmallMatrix(bloqueo)), 2);

        // Oportunidad de ganar
        System.out.println("\n--- Oportunidad de ganar ---");
        int[][] ganar = {{1, 1, 0}, {2, 2, 0}, {0, 0, 0}};
        compararDecisiones(new Tablero(new SmallMatrix(ganar)), 1);

        // Centro ocupado
        System.out.println("\n--- Centro ocupado por oponente ---");
        int[][] centro = {{0, 0, 0}, {0, 2, 0}, {0, 0, 0}};
        compararDecisiones(new Tablero(new SmallMatrix(centro)), 1);

        // Métricas AG+SNN
        System.out.println("\n--- Configuración AG+SNN ---");
        System.out.println("  Topología: " + Arrays.toString(configAGSNN.getTopologia()));
        Map<String, Object> metricas = redAGSNN.obtenerMetricas();
        metricas.forEach((k, v) -> System.out.println("  " + k + ": " + v));
    }

    // ==================== MÉTODOS AUXILIARES ====================

    private static double[][] normalizarInputs(double[][] inputs) {
        double[][] norm = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            norm[i] = new double[inputs[i].length];
            for (int j = 0; j < inputs[i].length; j++) {
                norm[i][j] = (inputs[i][j] + 1.0) / 2.0;
            }
        }
        return norm;
    }

    private void seleccionarSubconjunto(double[][] inputs, double[][] outputs,
                                         double[][] inputsSub, double[][] outputsSub) {
        // Selección equiespaciada para representatividad
        int total = inputs.length;
        int paso = Math.max(1, total / MUESTRAS_AG);
        for (int i = 0; i < MUESTRAS_AG; i++) {
            int idx = (i * paso) % total;
            inputsSub[i] = inputs[idx];
            outputsSub[i] = outputs[idx];
        }
    }

    private void asegurarModelos() {
        if (redSTDP == null) {
            test2_EntrenarModeloSTDP();
        }
        if (redAGSNN == null) {
            test3_OptimizarModeloAGSNN();
        }
    }

    private Posicion obtenerMovimiento(String tipo, Tablero tablero, int turno) {
        return switch (tipo) {
            case "clasico" -> obtenerMovimientoClasico(tablero, turno);
            case "stdp" -> obtenerMovimientoSTDP(tablero, turno);
            case "ag" -> obtenerMovimientoAG(tablero, turno);
            default -> throw new IllegalArgumentException("Tipo desconocido: " + tipo);
        };
    }

    private Posicion obtenerMovimientoClasico(Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] output = cerebroClasico.feedForward(input);
        return elegirMejorMovimiento(output, tablero, turno);
    }

    private Posicion obtenerMovimientoSTDP(Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] inputNorm = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            inputNorm[i] = (input[i] + 1.0) / 2.0;
        }
        redSTDP.resetearEstadoTemporal();
        double[] output = redSTDP.procesar(inputNorm, DURACION_TIMESTEPS);
        return elegirMejorMovimiento(output, tablero, turno);
    }

    private Posicion obtenerMovimientoAG(Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] inputNorm = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            inputNorm[i] = (input[i] + 1.0) / 2.0;
        }
        redAGSNN.resetearEstadoTemporal();
        double[] output = redAGSNN.procesar(inputNorm, DURACION_TIMESTEPS);
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

    private int[] jugarContraAleatorio(int numPartidas, String tipo) {
        int victorias = 0, derrotas = 0, empates = 0;
        Random rand = new Random(42);

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                if (turno == 1) {
                    movimiento = obtenerMovimiento(tipo, tablero, turno);
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
        return new int[]{victorias, derrotas, empates};
    }

    private int[] enfrentar(int numPartidas, String tipoP1, String tipoP2) {
        int victoriasP1 = 0, victoriasP2 = 0, empates = 0;

        // Ida: P1 empieza
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento = (turno == 1)
                        ? obtenerMovimiento(tipoP1, tablero, turno)
                        : obtenerMovimiento(tipoP2, tablero, turno);

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

        // Vuelta: P2 empieza
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento = (turno == 1)
                        ? obtenerMovimiento(tipoP2, tablero, turno)
                        : obtenerMovimiento(tipoP1, tablero, turno);

                int[][] nuevo = copiarTablero(tablero);
                nuevo[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevo));
                turno = (turno == 1) ? 2 : 1;
            }

            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victoriasP2++;
            else if (resultado == -1) victoriasP1++;
            else empates++;
        }

        return new int[]{victoriasP1, victoriasP2, empates};
    }

    private void compararDecisiones(Tablero tablero, int turno) {
        imprimirTablero(tablero);
        Posicion movClasico = obtenerMovimientoClasico(tablero, turno);
        Posicion movSTDP = obtenerMovimientoSTDP(tablero, turno);
        Posicion movAG = obtenerMovimientoAG(tablero, turno);

        System.out.println("  Clásico: (" + movClasico.getFila() + "," + movClasico.getColumna() + ")");
        System.out.println("  STDP:    (" + movSTDP.getFila() + "," + movSTDP.getColumna() + ")");
        System.out.println("  AG+SNN:  (" + movAG.getFila() + "," + movAG.getColumna() + ")");

        boolean coinciden = movClasico.equals(movSTDP) && movSTDP.equals(movAG);
        System.out.println(coinciden ? "  ✓ Los 3 coinciden" : "  ✗ Difieren");
    }

    private void mostrarResultados(int[] r, int total) {
        System.out.printf("  Victorias: %d (%.1f%%)  Derrotas: %d (%.1f%%)  Empates: %d (%.1f%%)%n",
                r[0], r[0] * 100.0 / total,
                r[1], r[1] * 100.0 / total,
                r[2], r[2] * 100.0 / total);
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
            System.out.print("  ");
            for (int j = 0; j < 3; j++) {
                int val = tablero.getValor(i, j);
                System.out.print(val == 0 ? "." : (val == 1 ? "X" : "O"));
                System.out.print(" ");
            }
            System.out.println();
        }
    }
}
