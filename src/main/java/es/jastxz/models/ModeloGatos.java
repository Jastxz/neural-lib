package es.jastxz.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

import org.javig.engine.FuncionesGato;
import org.javig.engine.Minimax;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.Util;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.util.DataContainer;
import es.jastxz.util.ModelManager;
import es.jastxz.util.NeuralNetworkTrainer;

public class ModeloGatos {
    private NeuralNetwork cerebro;
    private String nombreModelo = "modeloGatos.nn";
    private Mundo mundo;
    private Posicion posMouseInicial = new Posicion(0, 2); // Posición inicial típica del ratón si no se define otra
    private Tablero tablero = FuncionesGato.inicialGato(posMouseInicial);
    private Movimiento movimientoInicial = new Movimiento(tablero, posMouseInicial);
    private String juego = Util.juegoGato;
    private int dificultad = 4;
    private int profundidad = 10;
    private int marcaRatón = FuncionesGato.nombreRaton;
    private int turno = 1;
    private boolean esMaquina = true;
    private List<Movimiento> posiblesEstados = new ArrayList<>();
    private int maxEstados = 200; // Safety limit

    public ModeloGatos() {
        // Input: 8x8 + 1 (turno) = 65
        // Hidden: 64+64+1(turno) (arbitrario, suficiente para capturar lógica básica)
        // Output: 8x8 = 64 (probabilidad de mover a cada casilla)
        cerebro = new NeuralNetwork(65, 129, 64);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public ModeloGatos(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public ModeloGatos(List<Movimiento> posiblesEstados) {
        cerebro = new NeuralNetwork(65, 129, 64);
        this.posiblesEstados = posiblesEstados;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public void entrenar() {
        System.out.println("Generando datos de entrenamiento via simulación Minimax...");
        long start = System.currentTimeMillis();
        // Generamos datos de entrenamiento
        DataContainer data = generateTrainingData();
        // Generamos un número de partidas para cubrir estados diversos
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        int epocas = 500;
        int registro = epocas / 10;

        // Entrenar
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, epocas, registro);

        // Guardar
        ModelManager.saveModel(cerebro, nombreModelo);

        // Verificar carga
        cerebro = ModelManager.loadModel(nombreModelo);

        System.out.println("\nEntrenamiento completado.");
        long end = System.currentTimeMillis();
        long total = end - start;
        System.out.println("Tiempo de entrenamiento: " + total / 60000.0 + " minutos, " + total / 3600000.0 + " horas");
    }

    /**
     * Genera datos simulando partidas completas usando Minimax para ambos bandos.
     */
    /**
     * Genera datos simulando 100 partidas completas.
     * En cada paso, evalúa TODOS los movimientos posibles con Minimax para
     * identificar
     * múltiples jugadas óptimas (Multi-Target Training).
     */
    public DataContainer generateTrainingData() {
        System.out.println("Iniciando entrenamiento Multi-Target con simulación de partidas...");

        List<TrainingPair> trainingPairs = new ArrayList<>();
        AtomicInteger gamesPlayed = new AtomicInteger(0);
        int totalPartidas = 100;

        // Ejecutamos las partidas en paralelo para acelerar
        List<TrainingPair> collectedPairs = java.util.stream.IntStream.range(0, totalPartidas)
                .parallel()
                .mapToObj(gameIdx -> {
                    List<TrainingPair> gamePairs = new ArrayList<>();
                    Mundo mundoSim = new Mundo(this.mundo);
                    // Alternar quien empieza para variedad, aunque en Gatos suele empezar Ratón (1)
                    // o Gatos (2)
                    // Vamos a mantener la config standard pero quizás aleatorizar un poco la
                    // apertura si fuera necesario.
                    // Por ahora, setup standard.

                    // Reset tablero
                    Tablero tableroActual = FuncionesGato.inicialGato(posMouseInicial);
                    mundoSim.getMovimiento().setTablero(tableroActual);
                    // Profundidad 4 para acelerar el proceso y que no busque a profundidad 8.
                    mundoSim.setProfundidad(4);

                    // Determinar turno inicial (usualmente 1=Ratón)
                    int currentTurn = 1;
                    int currentBand = FuncionesGato.nombresGatos[0]; // Oponente (para Minimax)

                    boolean isGameOver = false;
                    int moves = 0;

                    while (!isGameOver && moves < 200) {
                        // 1. Generar todos los movimientos legales desde este estado
                        // Necesitamos saber de quién es el turno para generar sus movimientos.
                        // Si turno=1 (Ratón), genera movs de Ratón.
                        // Si turno=2 (Gatos), genera movs de Gatos.

                        // Vamos a usar la lógica interna 'generarMovimientosLegales' replicada o
                        // adaptada.
                        List<Movimiento> legalMoves = generarMovimientosLegales(tableroActual, currentTurn);

                        if (legalMoves.isEmpty())
                            break;

                        // 2. Evaluar cada rama
                        // Para cada movimiento candidato, simulamos que ocurre y preguntamos a Minimax
                        // "¿Cuánto vale este futuro?"
                        double[] branchScores = new double[legalMoves.size()];
                        Tablero[] outcomes = new Tablero[legalMoves.size()];

                        // Pre-calcular scores
                        for (int i = 0; i < legalMoves.size(); i++) {
                            Movimiento cand = legalMoves.get(i);
                            Tablero nextState = cand.getTablero();

                            // Preparar mundo para Minimax (preguntar valor del estado 'nextState')
                            Mundo mCopy = new Mundo(mundoSim);
                            mCopy.getMovimiento().setTablero(nextState);
                            // El turno pasa al oponente
                            int nextTurn = (currentTurn == 1) ? 2 : 1;
                            int nextBand = FuncionesGato.marcaMaquinaGato(currentBand);
                            mCopy.setTurno(nextTurn);
                            mCopy.setMarca(nextBand); // La marca que 'juega' en Minimax es la del turno activo

                            // Si el juego ya acabó en nextState, evaluamos directo
                            if (FuncionesGato.finGato(nextState)) {
                                branchScores[i] = evaluarEstadoFinal(nextState);
                                outcomes[i] = nextState;
                            } else {
                                // Ejecutar Minimax desde nextState
                                // Minimax devuelve el MEJOR estado final alcanzable desde ahí.
                                Tablero bestUltimateState = Minimax.negamax(mCopy);
                                if (bestUltimateState != null) {
                                    branchScores[i] = evaluarEstadoFinal(bestUltimateState);
                                    outcomes[i] = bestUltimateState;
                                } else {
                                    branchScores[i] = -9999; // Error/Unknown
                                }
                            }
                        }

                        // 3. Identificar el MEJOR score para el jugador actual
                        // Ratón (1) quiere MAXIMIZAR score (ej. llegar a fila 0, scores altos).
                        // Gatos (2) quieren MINIMIZAR score (encerrar ratón, scores bajos o negativos).

                        double bestScore;
                        if (currentTurn == 1) { // Ratón
                            bestScore = -Double.MAX_VALUE;
                            for (double s : branchScores)
                                if (s > bestScore)
                                    bestScore = s;
                        } else { // Gatos
                            bestScore = Double.MAX_VALUE;
                            for (double s : branchScores)
                                if (s < bestScore)
                                    bestScore = s;
                        }

                        // 4. Construir Target Vector
                        double[] target = new double[64];
                        boolean anyBestFound = false;
                        List<Integer> bestMoveIndices = new ArrayList<>();

                        for (int i = 0; i < legalMoves.size(); i++) {
                            // Usamos una pequeña tolerancia para comparar doubles
                            if (Math.abs(branchScores[i] - bestScore) < 0.001) {
                                Posicion dest = legalMoves.get(i).getPos();
                                int idx = dest.getFila() * 8 + dest.getColumna();
                                target[idx] = 1.0;
                                bestMoveIndices.add(i);
                                anyBestFound = true;
                            }
                        }

                        if (anyBestFound) {
                            gamePairs.add(new TrainingPair(tabularToInput(tableroActual, currentTurn), target));
                        }

                        // 5. Avanzar la partida
                        // Elegimos uno de los mejores movimientos al azar para explorar variantes
                        // óptimas
                        // O a veces uno sub-óptimo (epsilon-greedy) para robusticidad?
                        // El usuario pidió "aprender de 100 partidas", asumimos partidas 'buenas'.
                        // Elijamos uno de los mejores al azar.
                        if (!bestMoveIndices.isEmpty()) {
                            int chosenIdx = bestMoveIndices.get((int) (Math.random() * bestMoveIndices.size()));
                            tableroActual = legalMoves.get(chosenIdx).getTablero();
                            isGameOver = FuncionesGato.finGato(tableroActual);

                            // Switch turns
                            currentBand = FuncionesGato.marcaMaquinaGato(currentBand);
                            currentTurn = (currentTurn == 1) ? 2 : 1;
                            moves++;
                        } else {
                            break; // No moves?
                        }
                    }

                    System.out.println("Partidas simuladas: " + gamesPlayed.incrementAndGet());
                    System.out.println(
                            "Partida " + gameIdx + " simulada. Movimientos: " + moves + " Pares: " + gamePairs.size());
                    return gamePairs.stream();
                })
                .flatMap(java.util.function.Function.identity())
                .collect(Collectors.toList());

        trainingPairs.addAll(collectedPairs);

        System.out.println("Generación completada. Total de pares de entrenamiento: " + trainingPairs.size());

        // Convertir a arrays
        double[][] inputs = new double[trainingPairs.size()][];
        double[][] outputs = new double[trainingPairs.size()][];
        for (int i = 0; i < trainingPairs.size(); i++) {
            inputs[i] = trainingPairs.get(i).input;
            outputs[i] = trainingPairs.get(i).output;
        }

        return new DataContainer(inputs, outputs);
    }

    // Helper para generar movimientos (adaptado de lógica existente)
    private List<Movimiento> generarMovimientosLegales(Tablero t, int turno) {
        List<Movimiento> lista = new ArrayList<>();
        // Recorrer tablero
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int val = t.getValor(f, c);
                boolean esMio = false;
                if (turno == 1 && val == FuncionesGato.nombreRaton)
                    esMio = true;
                else if (turno == 2 && val != 0 && val != FuncionesGato.nombreRaton)
                    esMio = true; // Gato

                if (esMio) {
                    // Generar adyacentes
                    int[][] dirs;
                    if (turno == 1)
                        dirs = new int[][] { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } }; // Ratón 4 direcc
                    else
                        dirs = new int[][] { { -1, -1 }, { -1, 1 } }; // Gatos solo suben (fila disminuye? Dep.
                                                                      // orientación)
                    // NOTA: En este proyecto, fila 0 es arriba, 7 abajo.
                    // Ratón empieza en 0 (arriba) y quiere bajar? O al revés?
                    // FuncionesGato.inicialGato pone Ratón en (0,2).
                    // Gatos en filas 5,6,7?
                    // Revisando setup: posMouseInicial = (0, 2).
                    // Usualmente Ratón quiere llegar al otro lado (fila 7).
                    // Gatos suben (fila decrece).
                    // Si Ratón está en 0, quiere ir a 7 -> dirs {1,-1}, {1,1} (y hacia atrás tb?)
                    // "Ratón mueve en diagonal todas direcciones". Sí.

                    // Ajuste Gatos: dirs {{-1,-1}, {-1,1}} -> Fila disminuye (van hacia 0).
                    // Ajuste Ratón: dirs todas.

                    for (int[] d : dirs) {
                        int nf = f + d[0];
                        int nc = c + d[1];
                        if (nf >= 0 && nf < 8 && nc >= 0 && nc < 8 && (nf + nc) % 2 == 0) { // Casilla válida tablero
                            if (t.getValor(nf, nc) == 0) {
                                Tablero nuevo = new Tablero(t.getMatrix().copia());
                                nuevo.setValue(nf, nc, val);
                                nuevo.setValue(f, c, 0);
                                lista.add(new Movimiento(nuevo, new Posicion(nf, nc)));
                            }
                        }
                    }
                }
            }
        }
        return lista;
    }

    // Evalúa un estado final desde la perspectiva numérica
    // Alto = Bueno para Ratón (Ganar, avanzar)
    // Bajo = Bueno para Gatos (Encerrar, ganar)
    private double evaluarEstadoFinal(Tablero t) {
        // Usamos FuncionesGato.evalua si es posible, o heurística simple
        // Aquí necesito acceso a la heurística de 'Minimax'.
        // Si no la tengo, uso una simplificada:

        // 1. Ganó Ratón? (Llegó a fila 7)
        for (int c = 0; c < 8; c++)
            if (t.getValor(7, c) == FuncionesGato.nombreRaton)
                return 1000.0;

        // 2. Ganó Gatos? (Ratón encerrado).
        // finGato devuelve true si ratón llegó o encerrado.
        // Si finGato es true y ratón no está en fila 7 -> Gatos ganaron.
        if (FuncionesGato.finGato(t))
            return -1000.0;

        // 3. Juego no terminado (Minimax depth limit reached?)
        // Heurística: Fila del ratón (cuanto más alta mejor para él)
        Posicion pR = encontrarRaton(t);
        if (pR != null)
            return pR.getFila() * 10; // 0..70

        return 0;
    }

    public static double[] tabularToInput(Tablero tablero, int turno) {
        SmallMatrix matrix = tablero.getMatrix();
        double[] inputs = new double[65];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double val = matrix.get(i, j);
                int idx = i * 8 + j;

                // Normalización:
                // 0 (Vacío) -> 0.0
                // 9 (Ratón) -> 1.0
                // 1,3,5,7 (Gatos) -> -1.0
                if (val == 0) {
                    inputs[idx] = 0.0;
                } else if (val == FuncionesGato.nombreRaton) {
                    inputs[idx] = 1.0;
                } else {
                    // Asumimos que cualquier otro valor positivo es un gato
                    inputs[idx] = -1.0;
                }
            }
        }
        inputs[64] = turno;
        return inputs;
    }

    /**
     * Deduce la posición de destino comparando el tablero antes y después.
     */
    public Posicion obtenerMovimiento(Tablero inicial, Tablero fin, int marcaQueMovio) {
        // Determinar si el que movió fue el ratón o los gatos
        boolean movioRaton = (marcaQueMovio == FuncionesGato.nombreRaton);

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int vIni = inicial.getValor(i, j);
                int vFin = fin.getValor(i, j);

                // Si antes estaba vacío y ahora tiene una pieza
                if (vIni == 0 && vFin != 0) {
                    // Verificar si la pieza que apareció pertenece al bando correcto
                    boolean esRaton = (vFin == FuncionesGato.nombreRaton);
                    boolean esGato = Arrays.stream(FuncionesGato.nombresGatos).anyMatch(x -> x == vFin);

                    if ((movioRaton && esRaton) || (!movioRaton && esGato)) {
                        return new Posicion(i, j);
                    }
                }
            }
        }

        return null;
    }

    public List<Movimiento> generarTodosLosEstados() {
        Set<String> visitados = new HashSet<>();
        List<Movimiento> estados = new ArrayList<>();

        // Estado inicial
        Tablero inicial = FuncionesGato.inicialGato(posMouseInicial);

        Queue<Tablero> cola = new LinkedList<>();
        cola.add(inicial);
        visitados.add(matrizAString(inicial.getMatrix().getData()));
        estados.add(new Movimiento(inicial, posMouseInicial));

        int cont = 0;

        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        long startCpuTime = bean.getCurrentThreadCpuTime();

        while (!cola.isEmpty()) {
            if (estados.size() >= maxEstados) {
                System.out.println("Límite de estados alcanzado (" + maxEstados + "). Deteniendo generación.");
                break;
            }

            Tablero actual = cola.poll();
            cont++;

            if (cont % 10000 == 0) {
                long currentCpuTime = bean.getCurrentThreadCpuTime();
                long cpuTimeUsed = currentCpuTime - startCpuTime;
                double cpuSeconds = cpuTimeUsed / 1_000_000_000.0;

                long memoryUsed = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                double memoryMB = memoryUsed / (1024.0 * 1024.0);

                System.out.println(
                        String.format("Estados explorados: %d, Estados únicos: %d | CPU: %.4f s | Memoria: %.2f MB",
                                cont, estados.size(), cpuSeconds, memoryMB));
            }

            // Generar movimientos
            for (int f = 0; f < 8; f++) {
                for (int c = 0; c < 8; c++) {
                    if (actual.getValor(f, c) == FuncionesGato.nombreRaton) {
                        generarMovimientos(actual, f, c, true, cola, visitados, estados);
                    } else {
                        boolean esGato = false;
                        int v = actual.getValor(f, c);
                        for (int gId : FuncionesGato.nombresGatos) {
                            if (gId == v) {
                                esGato = true;
                                break;
                            }
                        }
                        if (esGato) {
                            generarMovimientos(actual, f, c, false, cola, visitados, estados);
                        }
                    }
                }
            }
        }
        System.out.println("Generación finalizada. Total estados: " + estados.size());
        return estados;
    }

    private void generarMovimientos(Tablero tablero, int fila, int col, boolean esRoja,
            Queue<Tablero> cola, Set<String> visitados, List<Movimiento> estados) {
        int[][] direcciones;

        if (esRoja) {
            // Roja: todas las diagonales
            direcciones = new int[][] { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
        } else {
            // Negra: solo hacia arriba (fila disminuye)
            direcciones = new int[][] { { -1, -1 }, { -1, 1 } };
        }

        for (int[] dir : direcciones) {
            int nuevaFila = fila + dir[0];
            int nuevaCol = col + dir[1];
            Posicion nuevaPosicion = new Posicion(nuevaFila, nuevaCol);

            // Verificar límites y casilla válida (blancas solamente)
            if (nuevaFila >= 0 && nuevaFila < 8 && nuevaCol >= 0 && nuevaCol < 8
                    && esCasillaBlanca(nuevaFila, nuevaCol)
                    && tablero.getValor(nuevaFila, nuevaCol) == 0) {

                Tablero nuevo = new Tablero(tablero.getMatrix().copia());
                int valorPieza = tablero.getValor(fila, col);
                nuevo.setValue(nuevaFila, nuevaCol, valorPieza);
                nuevo.setValue(fila, col, 0);

                String hash = matrizAString(nuevo.getMatrix().getData());
                if (!visitados.contains(hash) && !FuncionesGato.finGato(nuevo)) {
                    visitados.add(hash);
                    estados.add(new Movimiento(nuevo, nuevaPosicion));
                    cola.add(nuevo);
                }
            }
        }
    }

    private boolean esCasillaBlanca(int fila, int col) {
        // (0,0) es blanca, patrón de tablero ajedrez
        return (fila + col) % 2 == 0;
    }

    /**
     * Encuentra la posición del ratón en el tablero
     */
    private Posicion encontrarRaton(Tablero tablero) {
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                if (tablero.getValor(f, c) == FuncionesGato.nombreRaton) {
                    return new Posicion(f, c);
                }
            }
        }
        return null;
    }

    private String matrizAString(int[][] matriz) {
        StringBuilder sb = new StringBuilder();
        for (int[] fila : matriz) {
            for (int val : fila) {
                sb.append(val).append(",");
            }
        }
        return sb.toString();
    }

    /**
     * Helper class to hold training data pairs during parallel processing
     */
    private static class TrainingPair {
        final double[] input;
        final double[] output;

        TrainingPair(double[] input, double[] output) {
            this.input = input;
            this.output = output;
        }
    }

    public Mundo getMundo() {
        return mundo;
    }

    public void setMundo(Mundo mundo) {
        this.mundo = mundo;
    }

    public String getNombreModelo() {
        return nombreModelo;
    }

    public int getProfundidad() {
        return profundidad;
    }

    public void setProfundidad(int profundidad) {
        this.profundidad = profundidad;
    }

    public List<Movimiento> getPosiblesEstados() {
        return posiblesEstados;
    }
}
