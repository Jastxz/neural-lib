package es.jastxz.comparativas;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.nube.ConfiguracionNube;
import es.jastxz.nn.nube.ConfiguracionNubeBuilder;
import es.jastxz.nn.nube.InformeNube;
import es.jastxz.nn.nube.MotorNube;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa de velocidad y eficiencia entre el Método de la Nube Aleatoria
 * y el entrenamiento clásico por backpropagation para el problema XOR.
 *
 * <p>Mide tiempo de ejecución, precisión final y topología resultante
 * para ambos enfoques bajo condiciones equivalentes.</p>
 */
public class ComparativaNubeAleatoria3enRayaTest {

    // Dataset XOR con codificación one-hot: [NOT XOR, XOR]
    private static final double[][] XOR_ENTRADAS = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    private static final double[][] XOR_OBJETIVOS = {
        {1, 0}, {0, 1}, {0, 1}, {1, 0}
    };

    // Topología común para ambos métodos
    private static final int[] TOPOLOGIA = {2, 8, 4, 2};

    // Hiperparámetros compartidos
    private static final int EPOCAS = 2000;
    private static final double LEARNING_RATE = 0.5;
    private static final long SEMILLA = 42L;

    /**
     * Test 1: Entrenamiento clásico por backpropagation puro.
     * Crea una red con la topología fija y la entrena directamente.
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void test1_EntrenamientoClasico() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: ENTRENAMIENTO CLÁSICO (Backpropagation)");
        System.out.println("=".repeat(60));

        NeuralNetwork red = new NeuralNetwork(TOPOLOGIA);
        red.setLearningRate(LEARNING_RATE);

        long inicio = System.nanoTime();

        for (int epoca = 0; epoca < EPOCAS; epoca++) {
            for (int i = 0; i < XOR_ENTRADAS.length; i++) {
                red.train(XOR_ENTRADAS[i], XOR_OBJETIVOS[i]);
            }
        }

        long tiempoMs = (System.nanoTime() - inicio) / 1_000_000;

        // Evaluar precisión
        int correctas = 0;
        System.out.println("\nPredicciones:");
        for (int i = 0; i < XOR_ENTRADAS.length; i++) {
            double[] salida = red.feedForward(XOR_ENTRADAS[i]);
            int prediccion = argmax(salida);
            int esperado = argmax(XOR_OBJETIVOS[i]);
            boolean correcto = prediccion == esperado;
            if (correcto) correctas++;
            System.out.printf("  Entrada: [%.0f, %.0f] → Salida: [%.4f, %.4f] → Predicción: %d (esperado: %d) %s%n",
                    XOR_ENTRADAS[i][0], XOR_ENTRADAS[i][1],
                    salida[0], salida[1], prediccion, esperado,
                    correcto ? "✓" : "✗");
        }
        double precision = (double) correctas / XOR_ENTRADAS.length;

        System.out.println("\n--- Resultados Clásico ---");
        System.out.println("Topología:  " + formatTopologia(TOPOLOGIA));
        System.out.println("Épocas:     " + EPOCAS);
        System.out.println("Precisión:  " + String.format("%.2f%%", precision * 100));
        System.out.println("Tiempo:     " + tiempoMs + " ms");
        System.out.println("Parámetros: " + contarParametros(TOPOLOGIA));

        assertTrue(precision >= 0.0, "La precisión debe ser no negativa");
    }

    /**
     * Test 2: Método de la Nube Aleatoria.
     * Genera una nube de redes, reduce y refina la mejor.
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void test2_MetodoNubeAleatoria() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: MÉTODO DE LA NUBE ALEATORIA");
        System.out.println("=".repeat(60));

        ConfiguracionNube config = new ConfiguracionNubeBuilder()
                .tamañoNube(10)
                .topologiaInicial(TOPOLOGIA)
                .umbralAcierto(0.25)
                .neuronasEliminar(1)
                .epocasRefinamiento(EPOCAS)
                .tasaAprendizaje(LEARNING_RATE)
                .semilla(SEMILLA)
                .build();

        MotorNube motor = new MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS);

        long inicio = System.nanoTime();
        InformeNube informe = motor.ejecutar();
        long tiempoMs = (System.nanoTime() - inicio) / 1_000_000;

        System.out.println("\n--- Resultados Nube Aleatoria ---");
        System.out.println("Redes en nube:       " + config.tamañoNube());
        System.out.println("Umbral acierto:      " + String.format("%.2f%%", config.umbralAcierto() * 100));
        System.out.println("Exitoso:             " + (informe.exitoso() ? "Sí" : "No"));
        System.out.println("Redes evaluadas:     " + informe.totalRedesEvaluadas());
        System.out.println("Reducciones totales: " + informe.totalReducciones());

        if (informe.exitoso()) {
            System.out.println("Topología inicial:   " + formatTopologia(TOPOLOGIA));
            System.out.println("Topología final:     " + formatTopologia(informe.topologiaFinal()));
            System.out.println("Precisión final:     " + String.format("%.2f%%", informe.precision() * 100));
            System.out.println("Parámetros inicial:  " + contarParametros(TOPOLOGIA));
            System.out.println("Parámetros final:    " + contarParametros(informe.topologiaFinal()));
            System.out.printf("Reducción params:    %.1f%%%n",
                    (1.0 - (double) contarParametros(informe.topologiaFinal()) / contarParametros(TOPOLOGIA)) * 100);

            // Mostrar predicciones
            System.out.println("\nPredicciones:");
            for (int i = 0; i < XOR_ENTRADAS.length; i++) {
                double[] salida = informe.mejorRed().feedForward(XOR_ENTRADAS[i]);
                int prediccion = argmax(salida);
                int esperado = argmax(XOR_OBJETIVOS[i]);
                boolean correcto = prediccion == esperado;
                System.out.printf("  Entrada: [%.0f, %.0f] → Salida: [%.4f, %.4f] → Predicción: %d (esperado: %d) %s%n",
                        XOR_ENTRADAS[i][0], XOR_ENTRADAS[i][1],
                        salida[0], salida[1], prediccion, esperado,
                        correcto ? "✓" : "✗");
            }
        }

        System.out.println("Tiempo total:        " + tiempoMs + " ms");

        assertNotNull(informe);
    }

    /**
     * Test 3: Comparativa directa lado a lado.
     * Ejecuta ambos métodos y compara métricas clave.
     */
    @Test
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    void test3_ComparativaDirecta() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: COMPARATIVA DIRECTA");
        System.out.println("=".repeat(60));

        // --- Clásico ---
        NeuralNetwork redClasica = new NeuralNetwork(TOPOLOGIA);
        redClasica.setLearningRate(LEARNING_RATE);

        long inicioClasico = System.nanoTime();
        for (int epoca = 0; epoca < EPOCAS; epoca++) {
            for (int i = 0; i < XOR_ENTRADAS.length; i++) {
                redClasica.train(XOR_ENTRADAS[i], XOR_OBJETIVOS[i]);
            }
        }
        long tiempoClasico = (System.nanoTime() - inicioClasico) / 1_000_000;
        double precisionClasica = calcularPrecision(redClasica);

        // --- Nube Aleatoria ---
        ConfiguracionNube config = new ConfiguracionNubeBuilder()
                .tamañoNube(10)
                .topologiaInicial(TOPOLOGIA)
                .umbralAcierto(0.25)
                .neuronasEliminar(1)
                .epocasRefinamiento(EPOCAS)
                .tasaAprendizaje(LEARNING_RATE)
                .semilla(SEMILLA)
                .build();

        long inicioNube = System.nanoTime();
        InformeNube informe = new MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS).ejecutar();
        long tiempoNube = (System.nanoTime() - inicioNube) / 1_000_000;

        // --- Tabla comparativa ---
        System.out.println("\n┌─────────────────────────┬──────────────────┬──────────────────┐");
        System.out.println("│ Métrica                 │ Clásico          │ Nube Aleatoria   │");
        System.out.println("├─────────────────────────┼──────────────────┼──────────────────┤");
        System.out.printf("│ Tiempo (ms)             │ %16d │ %16d │%n", tiempoClasico, tiempoNube);
        System.out.printf("│ Precisión               │ %15.2f%% │ %15.2f%% │%n",
                precisionClasica * 100,
                informe.exitoso() ? informe.precision() * 100 : 0.0);
        System.out.printf("│ Topología final         │ %16s │ %16s │%n",
                formatTopologia(TOPOLOGIA),
                informe.exitoso() ? formatTopologia(informe.topologiaFinal()) : "N/A");
        System.out.printf("│ Parámetros              │ %16d │ %16s │%n",
                contarParametros(TOPOLOGIA),
                informe.exitoso() ? String.valueOf(contarParametros(informe.topologiaFinal())) : "N/A");
        System.out.printf("│ Épocas refinamiento     │ %16d │ %16d │%n", EPOCAS, EPOCAS);
        System.out.printf("│ Redes evaluadas         │ %16d │ %16d │%n", 1, informe.totalRedesEvaluadas());
        System.out.printf("│ Reducciones             │ %16d │ %16d │%n", 0, informe.totalReducciones());
        System.out.println("└─────────────────────────┴──────────────────┴──────────────────┘");

        // Análisis
        System.out.println("\n--- Análisis ---");
        if (tiempoNube > 0 && tiempoClasico > 0) {
            double ratio = (double) tiempoNube / tiempoClasico;
            System.out.printf("Ratio tiempo (Nube/Clásico): %.2fx%n", ratio);
            if (ratio > 1.0) {
                System.out.printf("La Nube Aleatoria tardó %.1f%% más que el clásico%n", (ratio - 1) * 100);
            } else {
                System.out.printf("La Nube Aleatoria fue %.1f%% más rápida que el clásico%n", (1 - ratio) * 100);
            }
        }

        if (informe.exitoso()) {
            int paramsInicial = contarParametros(TOPOLOGIA);
            int paramsFinal = contarParametros(informe.topologiaFinal());
            if (paramsFinal < paramsInicial) {
                System.out.printf("La Nube Aleatoria encontró una red %.1f%% más compacta%n",
                        (1.0 - (double) paramsFinal / paramsInicial) * 100);
            }
        }
    }

    /**
     * Test 4: Escalabilidad — comparar con topologías más grandes.
     */
    @Test
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    void test4_Escalabilidad() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: ESCALABILIDAD CON TOPOLOGÍA MAYOR");
        System.out.println("=".repeat(60));

        int[] topologiaGrande = {2, 16, 8, 2};
        int epocas = 1000;

        // --- Clásico ---
        NeuralNetwork redClasica = new NeuralNetwork(topologiaGrande);
        redClasica.setLearningRate(LEARNING_RATE);

        long inicioClasico = System.nanoTime();
        for (int epoca = 0; epoca < epocas; epoca++) {
            for (int i = 0; i < XOR_ENTRADAS.length; i++) {
                redClasica.train(XOR_ENTRADAS[i], XOR_OBJETIVOS[i]);
            }
        }
        long tiempoClasico = (System.nanoTime() - inicioClasico) / 1_000_000;
        double precisionClasica = calcularPrecision(redClasica);

        // --- Nube Aleatoria ---
        ConfiguracionNube config = new ConfiguracionNubeBuilder()
                .tamañoNube(15)
                .topologiaInicial(topologiaGrande)
                .umbralAcierto(0.25)
                .neuronasEliminar(1)
                .epocasRefinamiento(epocas)
                .tasaAprendizaje(LEARNING_RATE)
                .semilla(SEMILLA)
                .build();

        long inicioNube = System.nanoTime();
        InformeNube informe = new MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS).ejecutar();
        long tiempoNube = (System.nanoTime() - inicioNube) / 1_000_000;

        System.out.println("\nTopología inicial: " + formatTopologia(topologiaGrande));
        System.out.println("Parámetros iniciales: " + contarParametros(topologiaGrande));
        System.out.println();

        System.out.println("┌─────────────────────────┬──────────────────┬──────────────────┐");
        System.out.println("│ Métrica                 │ Clásico          │ Nube Aleatoria   │");
        System.out.println("├─────────────────────────┼──────────────────┼──────────────────┤");
        System.out.printf("│ Tiempo (ms)             │ %16d │ %16d │%n", tiempoClasico, tiempoNube);
        System.out.printf("│ Precisión               │ %15.2f%% │ %15.2f%% │%n",
                precisionClasica * 100,
                informe.exitoso() ? informe.precision() * 100 : 0.0);
        System.out.printf("│ Topología final         │ %16s │ %16s │%n",
                formatTopologia(topologiaGrande),
                informe.exitoso() ? formatTopologia(informe.topologiaFinal()) : "N/A");
        System.out.printf("│ Parámetros              │ %16d │ %16s │%n",
                contarParametros(topologiaGrande),
                informe.exitoso() ? String.valueOf(contarParametros(informe.topologiaFinal())) : "N/A");
        System.out.println("└─────────────────────────┴──────────────────┴──────────────────┘");

        if (informe.exitoso()) {
            int paramsInicial = contarParametros(topologiaGrande);
            int paramsFinal = contarParametros(informe.topologiaFinal());
            System.out.printf("\nReducción de parámetros: %d → %d (%.1f%% menos)%n",
                    paramsInicial, paramsFinal,
                    (1.0 - (double) paramsFinal / paramsInicial) * 100);
        }
    }

    // --- Utilidades ---

    private double calcularPrecision(NeuralNetwork red) {
        int correctas = 0;
        for (int i = 0; i < XOR_ENTRADAS.length; i++) {
            if (argmax(red.feedForward(XOR_ENTRADAS[i])) == argmax(XOR_OBJETIVOS[i])) {
                correctas++;
            }
        }
        return (double) correctas / XOR_ENTRADAS.length;
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

    /**
     * Cuenta el total de parámetros entrenables (pesos + biases) de una topología.
     */
    private static int contarParametros(int[] topologia) {
        int total = 0;
        for (int i = 0; i < topologia.length - 1; i++) {
            total += topologia[i] * topologia[i + 1]; // pesos
            total += topologia[i + 1];                 // biases
        }
        return total;
    }
}
