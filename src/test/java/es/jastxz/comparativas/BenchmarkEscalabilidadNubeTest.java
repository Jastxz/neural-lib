package es.jastxz.comparativas;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.nube.ConfiguracionNube;
import es.jastxz.nn.nube.ConfiguracionNubeBuilder;
import es.jastxz.nn.nube.InformeNube;
import es.jastxz.nn.nube.MotorNube;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Benchmark de escalabilidad: mide cómo crece el tiempo y los parámetros
 * de ambos métodos (clásico vs Nube Aleatoria) al aumentar la complejidad
 * de la topología.
 *
 * <p>Genera un CSV en {@code target/benchmark_escalabilidad.csv} con los datos
 * para montar gráficas de ratios de tiempo y parámetros.</p>
 *
 * <p>Dimensiones de escalado:</p>
 * <ul>
 *   <li>Ancho de capas ocultas (neuronas por capa)</li>
 *   <li>Profundidad (número de capas ocultas)</li>
 *   <li>Tamaño de la nube</li>
 * </ul>
 */
public class BenchmarkEscalabilidadNubeTest {

    // Dataset XOR — problema fijo para aislar el efecto de la topología
    private static final double[][] ENTRADAS = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    private static final double[][] OBJETIVOS = {
        {1, 0}, {0, 1}, {0, 1}, {1, 0}
    };

    private static final int EPOCAS = 1000;
    private static final double LR = 0.5;
    private static final long SEMILLA = 42L;
    private static final int WARMUP_RUNS = 2;
    private static final int MEASURE_RUNS = 3;

    /**
     * Escala el ancho de las capas ocultas: [2, N, 2] con N creciente.
     * Mide cómo afecta el número de neuronas por capa al tiempo de cada método.
     */
    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void benchmarkEscaladoAncho() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("BENCHMARK 1: ESCALADO POR ANCHO DE CAPA OCULTA");
        System.out.println("=".repeat(70));

        int[] anchos = {4, 8, 16, 32, 64, 128};
        List<ResultadoBenchmark> resultados = new ArrayList<>();

        for (int ancho : anchos) {
            int[] topologia = {2, ancho, 2};
            ResultadoBenchmark r = ejecutarComparativa("ancho", topologia, 10);
            resultados.add(r);
        }

        imprimirTabla(resultados);
        imprimirDatosGrafica(resultados, "ANCHO");
    }

    /**
     * Escala la profundidad: [2, 8, ..., 8, 2] con capas ocultas crecientes.
     * Mide cómo afecta el número de capas al tiempo de cada método.
     */
    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void benchmarkEscaladoProfundidad() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("BENCHMARK 2: ESCALADO POR PROFUNDIDAD (CAPAS OCULTAS)");
        System.out.println("=".repeat(70));

        int[] profundidades = {1, 2, 3, 4, 5, 6};
        List<ResultadoBenchmark> resultados = new ArrayList<>();

        for (int prof : profundidades) {
            int[] topologia = new int[prof + 2]; // entrada + ocultas + salida
            topologia[0] = 2;
            for (int i = 1; i <= prof; i++) topologia[i] = 8;
            topologia[prof + 1] = 2;
            ResultadoBenchmark r = ejecutarComparativa("prof", topologia, 10);
            resultados.add(r);
        }

        imprimirTabla(resultados);
        imprimirDatosGrafica(resultados, "PROFUNDIDAD");
    }

    /**
     * Escala el tamaño de la nube con topología fija.
     * Mide el overhead de evaluar más redes candidatas.
     */
    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void benchmarkEscaladoNube() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("BENCHMARK 3: ESCALADO POR TAMAÑO DE NUBE");
        System.out.println("=".repeat(70));

        int[] tamanosNube = {5, 10, 20, 40, 80};
        int[] topologia = {2, 16, 8, 2};
        List<ResultadoBenchmark> resultados = new ArrayList<>();

        for (int tamano : tamanosNube) {
            ResultadoBenchmark r = ejecutarComparativa("nube", topologia, tamano);
            resultados.add(r);
        }

        imprimirTabla(resultados);
        imprimirDatosGrafica(resultados, "TAMAÑO_NUBE");
    }

    /**
     * Escalado combinado: ancho + profundidad crecientes simultáneamente.
     */
    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void benchmarkEscaladoCombinado() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("BENCHMARK 4: ESCALADO COMBINADO (ANCHO + PROFUNDIDAD)");
        System.out.println("=".repeat(70));

        int[][] topologias = {
            {2, 4, 2},
            {2, 8, 4, 2},
            {2, 16, 8, 2},
            {2, 32, 16, 8, 2},
            {2, 64, 32, 16, 2},
            {2, 64, 32, 16, 8, 2}
        };
        List<ResultadoBenchmark> resultados = new ArrayList<>();

        for (int[] topologia : topologias) {
            ResultadoBenchmark r = ejecutarComparativa("combinado", topologia, 10);
            resultados.add(r);
        }

        imprimirTabla(resultados);
        imprimirDatosGrafica(resultados, "COMBINADO");
    }

    /**
     * Genera el CSV consolidado con todos los benchmarks.
     */
    @Test
    @Timeout(value = 1800, unit = TimeUnit.SECONDS)
    void generarCSVCompleto() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("GENERANDO CSV CONSOLIDADO DE ESCALABILIDAD");
        System.out.println("=".repeat(70));

        List<ResultadoBenchmark> todos = new ArrayList<>();

        // Ancho
        for (int ancho : new int[]{4, 8, 16, 32, 64, 128}) {
            todos.add(ejecutarComparativa("ancho", new int[]{2, ancho, 2}, 10));
        }
        // Profundidad
        for (int prof : new int[]{1, 2, 3, 4, 5, 6}) {
            int[] t = new int[prof + 2];
            t[0] = 2;
            for (int i = 1; i <= prof; i++) t[i] = 8;
            t[prof + 1] = 2;
            todos.add(ejecutarComparativa("profundidad", t, 10));
        }
        // Tamaño nube
        for (int n : new int[]{5, 10, 20, 40, 80}) {
            todos.add(ejecutarComparativa("tamano_nube", new int[]{2, 16, 8, 2}, n));
        }
        // Combinado
        for (int[][] ts : new int[][][]{
            {{2, 4, 2}}, {{2, 8, 4, 2}}, {{2, 16, 8, 2}},
            {{2, 32, 16, 8, 2}}, {{2, 64, 32, 16, 2}}, {{2, 64, 32, 16, 8, 2}}
        }) {
            todos.add(ejecutarComparativa("combinado", ts[0], 10));
        }

        escribirCSV(todos);
    }

    // ==================== Motor de benchmark ====================

    private ResultadoBenchmark ejecutarComparativa(String categoria, int[] topologia, int tamanoNube) {
        int paramsInicial = contarParametros(topologia);

        // --- Warmup ---
        for (int w = 0; w < WARMUP_RUNS; w++) {
            entrenarClasico(topologia);
            entrenarNube(topologia, tamanoNube);
        }

        // --- Medición: Clásico ---
        long[] tiemposClasico = new long[MEASURE_RUNS];
        double precisionClasica = 0;
        for (int m = 0; m < MEASURE_RUNS; m++) {
            long inicio = System.nanoTime();
            NeuralNetwork red = entrenarClasico(topologia);
            tiemposClasico[m] = (System.nanoTime() - inicio) / 1_000_000;
            if (m == MEASURE_RUNS - 1) precisionClasica = calcularPrecision(red);
        }
        long tiempoClasico = mediana(tiemposClasico);

        // --- Medición: Nube ---
        long[] tiemposNube = new long[MEASURE_RUNS];
        InformeNube ultimoInforme = null;
        for (int m = 0; m < MEASURE_RUNS; m++) {
            long inicio = System.nanoTime();
            ultimoInforme = entrenarNube(topologia, tamanoNube);
            tiemposNube[m] = (System.nanoTime() - inicio) / 1_000_000;
        }
        long tiempoNube = mediana(tiemposNube);

        int paramsFinal = ultimoInforme.exitoso() ? contarParametros(ultimoInforme.topologiaFinal()) : paramsInicial;
        double precisionNube = ultimoInforme.exitoso() ? ultimoInforme.precision() : 0.0;
        int[] topoFinal = ultimoInforme.exitoso() ? ultimoInforme.topologiaFinal() : topologia;

        return new ResultadoBenchmark(
                categoria, topologia, topoFinal, tamanoNube,
                paramsInicial, paramsFinal,
                tiempoClasico, tiempoNube,
                precisionClasica, precisionNube,
                ultimoInforme.exitoso(),
                ultimoInforme.totalReducciones()
        );
    }

    private NeuralNetwork entrenarClasico(int[] topologia) {
        NeuralNetwork red = new NeuralNetwork(topologia);
        red.setLearningRate(LR);
        for (int e = 0; e < EPOCAS; e++) {
            for (int i = 0; i < ENTRADAS.length; i++) {
                red.train(ENTRADAS[i], OBJETIVOS[i]);
            }
        }
        return red;
    }

    private InformeNube entrenarNube(int[] topologia, int tamanoNube) {
        ConfiguracionNube config = new ConfiguracionNubeBuilder()
                .tamañoNube(tamanoNube)
                .topologiaInicial(topologia)
                .umbralAcierto(0.25)
                .neuronasEliminar(1)
                .epocasRefinamiento(EPOCAS)
                .tasaAprendizaje(LR)
                .semilla(SEMILLA)
                .build();
        return new MotorNube(config, ENTRADAS, OBJETIVOS).ejecutar();
    }

    private double calcularPrecision(NeuralNetwork red) {
        int correctas = 0;
        for (int i = 0; i < ENTRADAS.length; i++) {
            if (argmax(red.feedForward(ENTRADAS[i])) == argmax(OBJETIVOS[i])) correctas++;
        }
        return (double) correctas / ENTRADAS.length;
    }

    private static int argmax(double[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }

    private static int contarParametros(int[] topologia) {
        int total = 0;
        for (int i = 0; i < topologia.length - 1; i++) {
            total += topologia[i] * topologia[i + 1] + topologia[i + 1];
        }
        return total;
    }

    private static long mediana(long[] valores) {
        java.util.Arrays.sort(valores);
        return valores[valores.length / 2];
    }

    private static String formatTopologia(int[] t) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(t[i]);
        }
        return sb.append("]").toString();
    }

    // ==================== Salida ====================

    private void imprimirTabla(List<ResultadoBenchmark> resultados) {
        System.out.println();
        System.out.println("┌──────────────────┬────────┬──────────┬──────────┬───────────┬──────────┬──────────┬──────────┐");
        System.out.println("│ Topología        │ Params │ T.Clás   │ T.Nube   │ Ratio T.  │ P.Final  │ Prec.Clá │ Prec.Nub │");
        System.out.println("│                  │ inic.  │ (ms)     │ (ms)     │ Nube/Clás │ Nube     │          │          │");
        System.out.println("├──────────────────┼────────┼──────────┼──────────┼───────────┼──────────┼──────────┼──────────┤");

        for (ResultadoBenchmark r : resultados) {
            double ratioTiempo = r.tiempoClasico > 0 ? (double) r.tiempoNube / r.tiempoClasico : 0;
            System.out.printf("│ %-16s │ %6d │ %8d │ %8d │ %9.2f │ %8d │ %7.1f%% │ %7.1f%% │%n",
                    formatTopologia(r.topologiaInicial),
                    r.paramsInicial,
                    r.tiempoClasico,
                    r.tiempoNube,
                    ratioTiempo,
                    r.paramsFinal,
                    r.precisionClasica * 100,
                    r.precisionNube * 100);
        }

        System.out.println("└──────────────────┴────────┴──────────┴──────────┴───────────┴──────────┴──────────┴──────────┘");
    }

    private void imprimirDatosGrafica(List<ResultadoBenchmark> resultados, String titulo) {
        System.out.println("\n--- Datos para gráfica: " + titulo + " ---");
        System.out.println("params_inicial,params_final_nube,tiempo_clasico_ms,tiempo_nube_ms,ratio_tiempo,ratio_params,precision_clasica,precision_nube,topologia,reducciones");
        for (ResultadoBenchmark r : resultados) {
            double ratioTiempo = r.tiempoClasico > 0 ? (double) r.tiempoNube / r.tiempoClasico : 0;
            double ratioParams = r.paramsInicial > 0 ? (double) r.paramsFinal / r.paramsInicial : 0;
            System.out.printf("%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%s,%d%n",
                    r.paramsInicial, r.paramsFinal,
                    r.tiempoClasico, r.tiempoNube,
                    ratioTiempo, ratioParams,
                    r.precisionClasica, r.precisionNube,
                    formatTopologia(r.topologiaInicial),
                    r.reducciones);
        }
    }

    private void escribirCSV(List<ResultadoBenchmark> resultados) {
        Path csvPath = Path.of("target", "benchmark_escalabilidad.csv");
        try {
            Files.createDirectories(csvPath.getParent());
            try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(csvPath))) {
                pw.println("categoria,topologia_inicial,topologia_final,tamano_nube,params_inicial,params_final,"
                        + "tiempo_clasico_ms,tiempo_nube_ms,ratio_tiempo,ratio_params,"
                        + "precision_clasica,precision_nube,exitoso,reducciones");
                for (ResultadoBenchmark r : resultados) {
                    double ratioTiempo = r.tiempoClasico > 0 ? (double) r.tiempoNube / r.tiempoClasico : 0;
                    double ratioParams = r.paramsInicial > 0 ? (double) r.paramsFinal / r.paramsInicial : 0;
                    pw.printf("%s,%s,%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%b,%d%n",
                            r.categoria,
                            formatTopologia(r.topologiaInicial),
                            formatTopologia(r.topoFinal),
                            r.tamanoNube,
                            r.paramsInicial, r.paramsFinal,
                            r.tiempoClasico, r.tiempoNube,
                            ratioTiempo, ratioParams,
                            r.precisionClasica, r.precisionNube,
                            r.exitoso, r.reducciones);
                }
            }
            System.out.println("\n✓ CSV generado en: " + csvPath.toAbsolutePath());
        } catch (IOException e) {
            System.err.println("Error escribiendo CSV: " + e.getMessage());
        }
    }

    // ==================== Record de resultados ====================

    private record ResultadoBenchmark(
            String categoria,
            int[] topologiaInicial,
            int[] topoFinal,
            int tamanoNube,
            int paramsInicial,
            int paramsFinal,
            long tiempoClasico,
            long tiempoNube,
            double precisionClasica,
            double precisionNube,
            boolean exitoso,
            int reducciones
    ) {}
}
