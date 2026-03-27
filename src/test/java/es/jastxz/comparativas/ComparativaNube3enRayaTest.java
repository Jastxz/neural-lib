package es.jastxz.comparativas;

import es.jastxz.engine.Funciones3enRaya;
import es.jastxz.models.Modelo3enRaya;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.nube.ConfiguracionNube;
import es.jastxz.nn.nube.ConfiguracionNubeBuilder;
import es.jastxz.nn.nube.InformeNube;
import es.jastxz.nn.nube.MotorNube;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import es.jastxz.util.DataContainer;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa de entrenamiento de 3 en Raya: Método de la Nube Aleatoria
 * vs entrenamiento clásico por backpropagation.
 *
 * <p>Compara velocidad de entrenamiento, precisión sobre datos de entrenamiento,
 * rendimiento contra jugador aleatorio, tamaño de la red resultante y
 * enfrentamiento directo entre ambos modelos.</p>
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComparativaNube3enRayaTest {

    // Datos de entrenamiento compartidos (generados una vez)
    private static double[][] entradas;
    private static double[][] objetivos;

    // Redes entrenadas (compartidas entre tests)
    private static NeuralNetwork redClasica;
    private static NeuralNetwork redNube;
    private static InformeNube informeNube;

    // Métricas de entrenamiento
    private static long tiempoClasico;
    private static long tiempoNube;

    // Topología del modelo clásico de 3 en raya
    private static final int[] TOPOLOGIA_CLASICA = {10, 30, 9};
    // Topología inicial para la nube (más grande, para que la reducción tenga margen)
    private static final int[] TOPOLOGIA_NUBE = {10, 30, 15, 9};
    private static final int EPOCAS = 500;
    private static final int NUM_PARTIDAS = 200;
    private static final long SEMILLA = 42L;

    @BeforeAll
    static void generarDatos() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("PREPARACIÓN: Generando datos de 3 en Raya");
        System.out.println("=".repeat(60));

        Modelo3enRaya modelo = new Modelo3enRaya();
        DataContainer data = modelo.generateTrainingData(modelo.getMundo());
        entradas = data.getInputs();
        objetivos = data.getOutputs();

        System.out.println("Muestras de entrenamiento: " + entradas.length);
        System.out.println("Dimensión entrada: " + entradas[0].length);
        System.out.println("Dimensión salida:  " + objetivos[0].length);
    }

    /**
     * Test 1: Entrenar red clásica con backpropagation.
     */
    @Test
    @Order(1)
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void test1_EntrenamientoClasico() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: ENTRENAMIENTO CLÁSICO (Backpropagation)");
        System.out.println("=".repeat(60));

        redClasica = new NeuralNetwork(TOPOLOGIA_CLASICA);
        redClasica.setLearningRate(0.1);

        long inicio = System.nanoTime();
        for (int epoca = 0; epoca < EPOCAS; epoca++) {
            for (int i = 0; i < entradas.length; i++) {
                redClasica.train(entradas[i], objetivos[i]);
            }
        }
        tiempoClasico = (System.nanoTime() - inicio) / 1_000_000;

        double precision = calcularPrecisionEntrenamiento(redClasica);

        System.out.println("\n--- Resultados ---");
        System.out.println("Topología:   " + formatTopologia(TOPOLOGIA_CLASICA));
        System.out.println("Épocas:      " + EPOCAS);
        System.out.println("Precisión:   " + String.format("%.2f%%", precision * 100));
        System.out.println("Tiempo:      " + tiempoClasico + " ms");
        System.out.println("Parámetros:  " + contarParametros(TOPOLOGIA_CLASICA));

        assertNotNull(redClasica);
    }

    /**
     * Test 2: Entrenar con el Método de la Nube Aleatoria.
     */
    @Test
    @Order(2)
    @Timeout(value = 300, unit = TimeUnit.SECONDS)
    void test2_MetodoNubeAleatoria() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: MÉTODO DE LA NUBE ALEATORIA");
        System.out.println("=".repeat(60));

        ConfiguracionNube config = new ConfiguracionNubeBuilder()
                .tamañoNube(10)
                .topologiaInicial(TOPOLOGIA_NUBE)
                .umbralAcierto(0.15)
                .neuronasEliminar(1)
                .epocasRefinamiento(EPOCAS)
                .tasaAprendizaje(0.1)
                .semilla(SEMILLA)
                .build();

        MotorNube motor = new MotorNube(config, entradas, objetivos);

        long inicio = System.nanoTime();
        informeNube = motor.ejecutar();
        tiempoNube = (System.nanoTime() - inicio) / 1_000_000;

        System.out.println("\n--- Resultados ---");
        System.out.println("Redes en nube:       " + config.tamañoNube());
        System.out.println("Umbral acierto:      " + String.format("%.2f%%", config.umbralAcierto() * 100));
        System.out.println("Exitoso:             " + (informeNube.exitoso() ? "Sí" : "No"));
        System.out.println("Redes evaluadas:     " + informeNube.totalRedesEvaluadas());
        System.out.println("Reducciones totales: " + informeNube.totalReducciones());

        if (informeNube.exitoso()) {
            redNube = informeNube.mejorRed();
            double precision = calcularPrecisionEntrenamiento(redNube);

            System.out.println("Topología inicial:   " + formatTopologia(TOPOLOGIA_NUBE));
            System.out.println("Topología final:     " + formatTopologia(informeNube.topologiaFinal()));
            System.out.println("Precisión final:     " + String.format("%.2f%%", precision * 100));
            System.out.println("Parámetros inicial:  " + contarParametros(TOPOLOGIA_NUBE));
            System.out.println("Parámetros final:    " + contarParametros(informeNube.topologiaFinal()));
            System.out.printf("Reducción params:    %.1f%%%n",
                    (1.0 - (double) contarParametros(informeNube.topologiaFinal()) / contarParametros(TOPOLOGIA_NUBE)) * 100);
        }

        System.out.println("Tiempo total:        " + tiempoNube + " ms");

        assertNotNull(informeNube);
    }

    /**
     * Test 3: Rendimiento contra jugador aleatorio.
     */
    @Test
    @Order(3)
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void test3_ContraJugadorAleatorio() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: RENDIMIENTO CONTRA JUGADOR ALEATORIO");
        System.out.println("=".repeat(60));

        assertNotNull(redClasica, "La red clásica debe haberse entrenado en test1");

        System.out.println("\n--- Red Clásica vs Aleatorio (" + NUM_PARTIDAS + " partidas) ---");
        ResultadoPartidas resClasico = jugarContraAleatorio(redClasica, NUM_PARTIDAS);
        mostrarResultados(resClasico);

        if (redNube != null) {
            System.out.println("\n--- Red Nube Aleatoria vs Aleatorio (" + NUM_PARTIDAS + " partidas) ---");
            ResultadoPartidas resNube = jugarContraAleatorio(redNube, NUM_PARTIDAS);
            mostrarResultados(resNube);

            System.out.println("\n┌─────────────────────────┬──────────────────┬──────────────────┐");
            System.out.println("│ Métrica                 │ Clásica          │ Nube Aleatoria   │");
            System.out.println("├─────────────────────────┼──────────────────┼──────────────────┤");
            System.out.printf("│ Victorias               │ %15.1f%% │ %15.1f%% │%n",
                    resClasico.porcentajeVictorias(), resNube.porcentajeVictorias());
            System.out.printf("│ Derrotas                │ %15.1f%% │ %15.1f%% │%n",
                    resClasico.porcentajeDerrotas(), resNube.porcentajeDerrotas());
            System.out.printf("│ Empates                 │ %15.1f%% │ %15.1f%% │%n",
                    resClasico.porcentajeEmpates(), resNube.porcentajeEmpates());
            System.out.println("└─────────────────────────┴──────────────────┴──────────────────┘");
        } else {
            System.out.println("\n⚠ La Nube Aleatoria no encontró red viable, no se puede comparar.");
        }
    }

    /**
     * Test 4: Enfrentamiento directo entre ambos modelos.
     */
    @Test
    @Order(4)
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void test4_EnfrentamientoDirecto() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: ENFRENTAMIENTO DIRECTO");
        System.out.println("=".repeat(60));

        assertNotNull(redClasica, "La red clásica debe haberse entrenado en test1");

        if (redNube == null) {
            System.out.println("⚠ La Nube Aleatoria no encontró red viable, no se puede enfrentar.");
            return;
        }

        System.out.println("\n--- Clásica (P1) vs Nube (P2) — " + NUM_PARTIDAS + " partidas ---");
        ResultadoEnfrentamiento res1 = enfrentarModelos(redClasica, redNube, NUM_PARTIDAS, true);
        mostrarEnfrentamiento(res1, "Clásica", "Nube");

        System.out.println("\n--- Nube (P1) vs Clásica (P2) — " + NUM_PARTIDAS + " partidas ---");
        ResultadoEnfrentamiento res2 = enfrentarModelos(redNube, redClasica, NUM_PARTIDAS, true);
        mostrarEnfrentamiento(res2, "Nube", "Clásica");

        int totalVicClasica = res1.victoriasP1 + res2.victoriasP2;
        int totalVicNube = res1.victoriasP2 + res2.victoriasP1;
        int totalEmpates = res1.empates + res2.empates;
        int totalPartidas = NUM_PARTIDAS * 2;

        System.out.println("\n--- Resumen global (" + totalPartidas + " partidas) ---");
        System.out.printf("Clásica: %d victorias (%.1f%%)%n",
                totalVicClasica, totalVicClasica * 100.0 / totalPartidas);
        System.out.printf("Nube:    %d victorias (%.1f%%)%n",
                totalVicNube, totalVicNube * 100.0 / totalPartidas);
        System.out.printf("Empates: %d (%.1f%%)%n",
                totalEmpates, totalEmpates * 100.0 / totalPartidas);
    }

    /**
     * Test 5: Tabla resumen comparativa completa.
     */
    @Test
    @Order(5)
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void test5_ResumenComparativo() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 5: RESUMEN COMPARATIVO FINAL");
        System.out.println("=".repeat(60));

        assertNotNull(redClasica, "La red clásica debe haberse entrenado");

        double precClasica = calcularPrecisionEntrenamiento(redClasica);
        double precNube = redNube != null ? calcularPrecisionEntrenamiento(redNube) : 0.0;

        System.out.println("\n┌─────────────────────────┬──────────────────┬──────────────────┐");
        System.out.println("│ Métrica                 │ Clásica          │ Nube Aleatoria   │");
        System.out.println("├─────────────────────────┼──────────────────┼──────────────────┤");
        System.out.printf("│ Tiempo entrenamiento    │ %13d ms │ %13d ms │%n", tiempoClasico, tiempoNube);
        System.out.printf("│ Precisión datos train   │ %15.2f%% │ %15.2f%% │%n",
                precClasica * 100, precNube * 100);
        System.out.printf("│ Topología               │ %16s │ %16s │%n",
                formatTopologia(TOPOLOGIA_CLASICA),
                informeNube != null && informeNube.exitoso() ? formatTopologia(informeNube.topologiaFinal()) : "N/A");
        System.out.printf("│ Parámetros              │ %16d │ %16s │%n",
                contarParametros(TOPOLOGIA_CLASICA),
                informeNube != null && informeNube.exitoso()
                        ? String.valueOf(contarParametros(informeNube.topologiaFinal())) : "N/A");
        System.out.printf("│ Épocas refinamiento     │ %16d │ %16d │%n", EPOCAS, EPOCAS);
        System.out.println("└─────────────────────────┴──────────────────┴──────────────────┘");

        if (tiempoNube > 0 && tiempoClasico > 0) {
            double ratio = (double) tiempoNube / tiempoClasico;
            System.out.printf("\nRatio tiempo (Nube/Clásico): %.2fx%n", ratio);
        }
        if (informeNube != null && informeNube.exitoso()) {
            int paramsClasica = contarParametros(TOPOLOGIA_CLASICA);
            int paramsNube = contarParametros(informeNube.topologiaFinal());
            System.out.printf("Ratio parámetros (Nube/Clásica): %.2fx (%s)%n",
                    (double) paramsNube / paramsClasica,
                    paramsNube < paramsClasica ? "más compacta" : "más grande");
        }
    }

    // ==================== Utilidades ====================

    private double calcularPrecisionEntrenamiento(NeuralNetwork red) {
        int correctas = 0;
        for (int i = 0; i < entradas.length; i++) {
            double[] salida = red.feedForward(entradas[i]);
            // Para 3 en raya, comparamos el movimiento de mayor valor
            // entre los movimientos legales (argmax sobre posiciones válidas)
            if (argmax(salida) == argmax(objetivos[i])) {
                correctas++;
            }
        }
        return (double) correctas / entradas.length;
    }

    private ResultadoPartidas jugarContraAleatorio(NeuralNetwork cerebro, int numPartidas) {
        int victorias = 0, derrotas = 0, empates = 0;
        Random rand = new Random(SEMILLA);

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                if (turno == 1) {
                    movimiento = obtenerMovimiento(cerebro, tablero, turno);
                } else {
                    List<Movimiento> movs = Funciones3enRaya.movs3enRaya(tablero, turno);
                    movimiento = movs.get(rand.nextInt(movs.size())).getPos();
                }
                int[][] nuevoEstado = copiarTablero(tablero);
                nuevoEstado[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevoEstado));
                turno = (turno == 1) ? 2 : 1;
            }

            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victorias++;
            else if (resultado == -1) derrotas++;
            else empates++;
        }
        return new ResultadoPartidas(victorias, derrotas, empates);
    }

    private ResultadoEnfrentamiento enfrentarModelos(
            NeuralNetwork cerebroP1, NeuralNetwork cerebroP2, int numPartidas, boolean p1EsP1) {
        int victoriasP1 = 0, victoriasP2 = 0, empates = 0;

        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;

            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                if (turno == 1) {
                    movimiento = obtenerMovimiento(cerebroP1, tablero, turno);
                } else {
                    movimiento = obtenerMovimiento(cerebroP2, tablero, turno);
                }
                int[][] nuevoEstado = copiarTablero(tablero);
                nuevoEstado[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevoEstado));
                turno = (turno == 1) ? 2 : 1;
            }

            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victoriasP1++;
            else if (resultado == -1) victoriasP2++;
            else empates++;
        }
        return new ResultadoEnfrentamiento(victoriasP1, victoriasP2, empates);
    }

    private Posicion obtenerMovimiento(NeuralNetwork cerebro, Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] output = cerebro.feedForward(input);

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

    private int[][] copiarTablero(Tablero tablero) {
        int[][] copia = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                copia[i][j] = tablero.getValor(i, j);
            }
        }
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

    private static int argmax(double[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }

    private static String formatTopologia(int[] topologia) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < topologia.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(topologia[i]);
        }
        return sb.append("]").toString();
    }

    private static int contarParametros(int[] topologia) {
        int total = 0;
        for (int i = 0; i < topologia.length - 1; i++) {
            total += topologia[i] * topologia[i + 1] + topologia[i + 1];
        }
        return total;
    }

    private void mostrarResultados(ResultadoPartidas r) {
        System.out.printf("Victorias: %d (%.1f%%)  Derrotas: %d (%.1f%%)  Empates: %d (%.1f%%)%n",
                r.victorias, r.porcentajeVictorias(),
                r.derrotas, r.porcentajeDerrotas(),
                r.empates, r.porcentajeEmpates());
    }

    private void mostrarEnfrentamiento(ResultadoEnfrentamiento r, String p1, String p2) {
        System.out.printf("%s: %d victorias (%.1f%%)  %s: %d victorias (%.1f%%)  Empates: %d (%.1f%%)%n",
                p1, r.victoriasP1, r.porcentajeP1(),
                p2, r.victoriasP2, r.porcentajeP2(),
                r.empates, r.porcentajeEmpates());
    }

    // ==================== Records auxiliares ====================

    private record ResultadoPartidas(int victorias, int derrotas, int empates) {
        int total() { return victorias + derrotas + empates; }
        double porcentajeVictorias() { return victorias * 100.0 / total(); }
        double porcentajeDerrotas() { return derrotas * 100.0 / total(); }
        double porcentajeEmpates() { return empates * 100.0 / total(); }
    }

    private record ResultadoEnfrentamiento(int victoriasP1, int victoriasP2, int empates) {
        int total() { return victoriasP1 + victoriasP2 + empates; }
        double porcentajeP1() { return victoriasP1 * 100.0 / total(); }
        double porcentajeP2() { return victoriasP2 * 100.0 / total(); }
        double porcentajeEmpates() { return empates * 100.0 / total(); }
    }
}
