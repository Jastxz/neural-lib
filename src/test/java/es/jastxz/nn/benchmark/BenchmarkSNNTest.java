package es.jastxz.nn.benchmark;

import org.junit.jupiter.api.*;

import java.nio.file.Path;
import java.time.Duration;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

/**
 * Test principal de la suite de benchmarks para la SNN.
 *
 * <p>Genera dinámicamente la matriz completa de configuraciones
 * (topología × nivel de complejidad) y ejecuta cada benchmark
 * como un {@link DynamicTest} individual con timeout de 120 segundos.</p>
 *
 * <p>Al finalizar todos los tests, genera un informe comparativo
 * por consola y gráficas de visualización en {@code target/benchmark-graficas/}.</p>
 *
 * @since 1.1
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class BenchmarkSNNTest {

    /** Resultados acumulados de todos los benchmarks ejecutados. */
    private static final List<ResultadoBenchmark> resultados =
            Collections.synchronizedList(new ArrayList<>());

    // --- Parámetros de benchmark ---
    private static final int EPOCAS = 10;
    private static final int DURACION_TIMESTEPS = 50;
    private static final int REPETICIONES = 1;
    private static final long SEMILLA = 42L;
    private static final Duration TIMEOUT = Duration.ofSeconds(120);

    private final RecolectorMetricas recolector = new RecolectorMetricas();

    // =========================================================================
    // Task 11.1 — @TestFactory: genera la matriz completa de benchmarks
    // =========================================================================

    /**
     * Genera un {@link DynamicTest} por cada combinación de
     * {@link NivelComplejidad} × topología generada por
     * {@link GeneradorTopologias}.
     *
     * <p>Cada test individual:</p>
     * <ol>
     *   <li>Obtiene los datos de entrenamiento del nivel</li>
     *   <li>Ejecuta {@link RecolectorMetricas#ejecutarYRecolectar}</li>
     *   <li>Aplica {@link DetectorLimites#clasificar}</li>
     *   <li>Almacena el resultado para el informe final</li>
     * </ol>
     *
     * @return colección de tests dinámicos (uno por configuración)
     */
    @TestFactory
    Collection<DynamicTest> benchmarkMatriz() {
        List<DynamicTest> tests = new ArrayList<>();

        for (NivelComplejidad nivel : NivelComplejidad.values()) {
            double[][] inputs = obtenerInputs(nivel);
            double[][] targets = obtenerTargets(nivel);
            List<int[]> topologias = GeneradorTopologias.generar(
                    nivel.getDimensionEntrada(), nivel.getDimensionSalida());

            for (int[] topologia : topologias) {
                ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                        nivel, topologia, EPOCAS, DURACION_TIMESTEPS,
                        REPETICIONES, SEMILLA);

                String nombre = config.etiqueta();

                tests.add(dynamicTest(nombre, () ->
                    assertTimeoutPreemptively(TIMEOUT, () -> {
                        ResultadoBenchmark resultado =
                                recolector.ejecutarYRecolectar(config, inputs, targets);

                        // Apply limit detection
                        String clasificacion = DetectorLimites.clasificar(resultado);
                        if (clasificacion != null) {
                            resultado = new ResultadoBenchmark(
                                    resultado.configuracion(),
                                    resultado.precisionFinal(),
                                    resultado.errorMSEPorEpoca(),
                                    resultado.tiempoEntrenamientoMs(),
                                    resultado.totalSpikes(),
                                    resultado.tasaDisparoPromedio(),
                                    resultado.costoEnergetico(),
                                    resultado.dispersionActividad(),
                                    resultado.neuronasActivas(),
                                    resultado.neuronasTotal(),
                                    clasificacion
                            );
                        }

                        resultados.add(resultado);
                    })
                ));
            }
        }

        return tests;
    }

    // =========================================================================
    // Task 11.2 — @AfterAll: informe y gráficas
    // =========================================================================

    /**
     * Genera el informe comparativo por consola y las gráficas de
     * visualización una vez completados todos los benchmarks.
     */
    @AfterAll
    void generarInforme() {
        if (resultados.isEmpty()) {
            System.out.println("No hay resultados de benchmark para generar informe.");
            return;
        }

        // 1. Informe por consola
        GeneradorInforme.generar(resultados);

        // 2. Calcular fronteras de capacidad
        Map<String, List<String>> fronteras =
                GeneradorInforme.detectarFronterasCapacidad(resultados);

        // 3. Generar gráficas (PNGs + índice HTML)
        Path dirGraficas = Path.of("target", "benchmark-graficas");
        GeneradorGraficas generadorGraficas = new GeneradorGraficas(dirGraficas);
        generadorGraficas.generarTodas(resultados, fronteras);
    }

    // =========================================================================
    // Task 11.3 — Proveedores de datos de problemas
    // =========================================================================

    /**
     * Devuelve los datos de entrada para el nivel de complejidad dado.
     *
     * @param nivel nivel de complejidad
     * @return array de vectores de entrada normalizados a [0,1]
     */
    private static double[][] obtenerInputs(NivelComplejidad nivel) {
        return switch (nivel) {
            case TRIVIAL -> inputsPuertasLogicas();
            case BAJO    -> inputs3enRaya();
            case MEDIO   -> inputsGatos();
            case ALTO    -> inputsDamas();
        };
    }

    /**
     * Devuelve los datos objetivo para el nivel de complejidad dado.
     *
     * @param nivel nivel de complejidad
     * @return array de vectores objetivo
     */
    private static double[][] obtenerTargets(NivelComplejidad nivel) {
        return switch (nivel) {
            case TRIVIAL -> targetsPuertasLogicas();
            case BAJO    -> targets3enRaya();
            case MEDIO   -> targetsGatos();
            case ALTO    -> targetsDamas();
        };
    }

    // --- TRIVIAL: Puertas Lógicas (entrada=2, salida=1) ---

    /**
     * Inputs para puertas lógicas: las 4 combinaciones de 2 bits.
     * Se repiten para AND, OR y XOR (12 muestras total).
     */
    private static double[][] inputsPuertasLogicas() {
        return new double[][] {
            // AND
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            // OR
            {0, 0}, {0, 1}, {1, 0}, {1, 1},
            // XOR
            {0, 0}, {0, 1}, {1, 0}, {1, 1}
        };
    }

    /**
     * Targets para puertas lógicas: AND, OR y XOR concatenados.
     */
    private static double[][] targetsPuertasLogicas() {
        return new double[][] {
            // AND
            {0}, {0}, {0}, {1},
            // OR
            {0}, {1}, {1}, {1},
            // XOR
            {0}, {1}, {1}, {0}
        };
    }

    // --- BAJO: 3 en Raya (entrada=10, salida=9) ---

    /**
     * Inputs simplificados para 3 en Raya.
     * Cada vector tiene 10 componentes: 9 casillas + turno.
     * Valores normalizados a [0,1]: vacío=0, jugador1=0.5, jugador2=1.0, turno=0/1.
     */
    private static double[][] inputs3enRaya() {
        return new double[][] {
            // Tablero vacío, turno jugador 1
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
            // Centro ocupado por J1, turno J2
            {0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1},
            // Esquina ocupada por J1, turno J2
            {0.5, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            // J1 amenaza fila superior, turno J2
            {0.5, 0.5, 0, 0, 1.0, 0, 0, 0, 0, 1},
            // J2 amenaza columna izquierda, turno J1
            {1.0, 0, 0, 1.0, 0.5, 0, 0, 0, 0, 0},
            // Tablero casi lleno
            {0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0, 0.5, 1.0, 0},
            // J1 puede ganar en diagonal
            {0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0},
            // J2 bloquea centro
            {0.5, 0, 0, 0, 1.0, 0, 0, 0, 0, 1},
        };
    }

    /**
     * Targets para 3 en Raya: distribución de probabilidad sobre 9 casillas.
     * Cada vector indica la mejor jugada como one-hot o distribución suave.
     */
    private static double[][] targets3enRaya() {
        return new double[][] {
            // Tablero vacío → centro (posición 4)
            {0, 0, 0, 0, 1, 0, 0, 0, 0},
            // Centro ocupado → esquina (posición 0)
            {1, 0, 0, 0, 0, 0, 0, 0, 0},
            // Esquina ocupada → centro (posición 4)
            {0, 0, 0, 0, 1, 0, 0, 0, 0},
            // Bloquear fila superior → posición 2
            {0, 0, 1, 0, 0, 0, 0, 0, 0},
            // Bloquear columna izquierda → posición 6
            {0, 0, 0, 0, 0, 0, 1, 0, 0},
            // Tablero casi lleno → posición 6
            {0, 0, 0, 0, 0, 0, 1, 0, 0},
            // Ganar en diagonal → posición 8
            {0, 0, 0, 0, 0, 0, 0, 0, 1},
            // Bloquear → posición 0
            {1, 0, 0, 0, 0, 0, 0, 0, 0},
        };
    }

    // --- MEDIO: Gatos (entrada=25, salida=25) ---

    /**
     * Inputs simplificados para Gatos (Ratón vs Gatos en tablero 5×5).
     * 25 componentes representando el tablero 5×5 normalizado.
     * vacío=0, ratón=0.5, gato=1.0.
     */
    private static double[][] inputsGatos() {
        double[][] inputs = new double[8][25];
        Random rng = new Random(SEMILLA);

        // Posición inicial: ratón en centro, gatos en esquinas
        inputs[0][12] = 0.5; // ratón en (2,2)
        inputs[0][0] = 1.0;  // gato en (0,0)
        inputs[0][4] = 1.0;  // gato en (0,4)
        inputs[0][20] = 1.0; // gato en (4,0)
        inputs[0][24] = 1.0; // gato en (4,4)

        // Variaciones con posiciones aleatorias pero reproducibles
        for (int i = 1; i < inputs.length; i++) {
            int ratonPos = rng.nextInt(25);
            inputs[i][ratonPos] = 0.5;
            for (int g = 0; g < 4; g++) {
                int gatoPos;
                do {
                    gatoPos = rng.nextInt(25);
                } while (inputs[i][gatoPos] != 0);
                inputs[i][gatoPos] = 1.0;
            }
        }
        return inputs;
    }

    /**
     * Targets para Gatos: distribución de probabilidad sobre 25 posiciones.
     * Indica la mejor casilla destino para el movimiento.
     */
    private static double[][] targetsGatos() {
        double[][] targets = new double[8][25];
        Random rng = new Random(SEMILLA + 1);

        for (int i = 0; i < targets.length; i++) {
            int mejorPos = rng.nextInt(25);
            targets[i][mejorPos] = 1.0;
        }
        return targets;
    }

    // --- ALTO: Damas (entrada=32, salida=32) ---

    /**
     * Inputs simplificados para Damas.
     * 32 componentes representando las 32 casillas jugables del tablero 8×8.
     * vacío=0, pieza_propia=0.5, pieza_rival=1.0.
     */
    private static double[][] inputsDamas() {
        double[][] inputs = new double[8][32];
        Random rng = new Random(SEMILLA);

        // Posición inicial estándar de damas (simplificada)
        // Piezas propias en primeras 12 casillas, rivales en últimas 12
        for (int i = 0; i < 12; i++) {
            inputs[0][i] = 0.5;
        }
        for (int i = 20; i < 32; i++) {
            inputs[0][i] = 1.0;
        }

        // Variaciones con posiciones aleatorias pero reproducibles
        for (int i = 1; i < inputs.length; i++) {
            int numPropias = 3 + rng.nextInt(10);
            int numRivales = 3 + rng.nextInt(10);
            for (int p = 0; p < numPropias; p++) {
                inputs[i][rng.nextInt(32)] = 0.5;
            }
            for (int r = 0; r < numRivales; r++) {
                int pos;
                do {
                    pos = rng.nextInt(32);
                } while (inputs[i][pos] != 0);
                inputs[i][pos] = 1.0;
            }
        }
        return inputs;
    }

    /**
     * Targets para Damas: distribución de probabilidad sobre 32 casillas.
     * Indica la mejor casilla destino para el movimiento.
     */
    private static double[][] targetsDamas() {
        double[][] targets = new double[8][32];
        Random rng = new Random(SEMILLA + 1);

        for (int i = 0; i < targets.length; i++) {
            int mejorPos = rng.nextInt(32);
            targets[i][mejorPos] = 1.0;
        }
        return targets;
    }
}
