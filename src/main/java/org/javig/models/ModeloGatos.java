package org.javig.models;

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
import org.javig.nn.NeuralNetwork;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.DataContainer;
import org.javig.util.ModelManager;
import org.javig.util.NeuralNetworkTrainer;
import org.javig.util.Util;

public class ModeloGatos {
    private NeuralNetwork cerebro;
    private String nombreModeloBasico = "modeloGatosBasico.nn";
    private String nombreModeloNormal = "modeloGatosNormal.nn";
    private String nombreModeloAvanzado = "modeloGatosAvanzado.nn";
    private String nombreModeloExperto = "modeloGatosExperto.nn";
    private Mundo mundo;
    private Posicion posMouseInicial = new Posicion(0, 2); // Posición inicial típica del ratón si no se define otra
    private Tablero tablero = FuncionesGato.inicialGato(posMouseInicial);
    private Movimiento movimientoInicial = new Movimiento(tablero, posMouseInicial);
    private String juego = Util.juegoGato;
    private int dificultad = 4; // Ajustado para un tiempo razonable en Gatos
    private int profundidadBasico = 2;
    private int profundidad = 4;
    private int profundidadAvanzado = 6;
    private int profundidadExperto = 8;
    private int marcaRatón = FuncionesGato.nombreRaton;
    private int turno = 1;
    private boolean esMaquina = true;
    private List<Movimiento> posiblesEstados = new ArrayList<>();
    private int maxEstados = 10000; // Safety limit

    public ModeloGatos() {
        // Input: 8x8 + 1 (turno) = 65
        // Hidden: 128 (arbitrario, suficiente para capturar lógica básica)
        // Output: 8x8 = 64 (probabilidad de mover a cada casilla)
        cerebro = new NeuralNetwork(65, 128, 64);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public ModeloGatos(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public ModeloGatos(List<Movimiento> posiblesEstados) {
        cerebro = new NeuralNetwork(65, 128, 64);
        this.posiblesEstados = posiblesEstados;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public void entrenar(String nombreModelo) {
        System.out.println("Generando datos de entrenamiento via simulación Minimax...");
        long start = System.currentTimeMillis();
        // Generamos datos de entrenamiento
        DataContainer data = generateTrainingData();
        // Generamos un número de partidas para cubrir estados diversos
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        // Entrenar
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, 100, 10);

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
     * 
     * @param numGames Número de partidas a simular.
     */
    public DataContainer generateTrainingData() {
        if (posiblesEstados.size() == 0) {
            System.out.println("Generando todos los estados posibles para el entrenamiento...");
            posiblesEstados = generarTodosLosEstados();
        }
        int totalEstados = posiblesEstados.size();

        System.out.println("Generando datos de entrenamiento para " + totalEstados + " estados iniciales...");
        long startSimulationTime = System.currentTimeMillis();

        // Contador atómico para progreso
        AtomicInteger procesados = new AtomicInteger(0);

        // Procesar en paralelo y recolectar pares de entrenamiento
        List<TrainingPair> trainingPairs = posiblesEstados.parallelStream()
                .flatMap(movimiento -> {
                    List<TrainingPair> pairs = new ArrayList<>(40);

                    Mundo mundo = new Mundo(this.mundo);
                    Tablero estado = movimiento.getTablero();

                    // Encontrar la posición del ratón en el estado actual
                    Posicion posRaton = encontrarRaton(estado);
                    if (posRaton == null) {
                        System.err.println("ADVERTENCIA: No se encontró el ratón en el estado inicial");
                        return pairs.stream();
                    }

                    // Determinar quién mueve en este turno
                    // Empezamos asumiendo que el ratón mueve primero
                    int currentTurn = 1; // Ratón mueve primero
                    // IMPORTANTE: Minimax necesita la marca del OPONENTE
                    // Si el ratón va a mover, pasamos la marca de un gato a Minimax
                    int currentBand = FuncionesGato.nombresGatos[0]; // Oponente del ratón

                    // Límite de movimientos para evitar bucles infinitos
                    int moves = 0;
                    boolean isGameOver = FuncionesGato.finGato(estado);

                    if (isGameOver) {
                        // Este estado ya es final, no generar pares
                        return pairs.stream();
                    }

                    while (!isGameOver && moves < 200) {
                        // Actualizar posición del ratón
                        posRaton = encontrarRaton(estado);
                        if (posRaton == null) {
                            break;
                        }

                        mundo.setMovimiento(new Movimiento(estado, posRaton));
                        mundo.setMarca(currentBand);
                        mundo.setTurno(currentTurn);

                        Tablero bestNextState = Minimax.negamax(mundo);
                        if (bestNextState == null)
                            break;

                        // Determinar quién realmente movió basándose en el resultado
                        // Si currentTurn es 1 (ratón), entonces el ratón movió
                        int quienMovio = (currentTurn == 1) ? FuncionesGato.nombreRaton : FuncionesGato.nombresGatos[0];

                        Posicion moveDest = obtenerMovimiento(estado, bestNextState, quienMovio);
                        if (moveDest != null) {
                            int turnoModelo = currentTurn;
                            double[] inputVec = tabularToInput(estado, turnoModelo);

                            double[] target = new double[64];
                            int idx = moveDest.getFila() * 8 + moveDest.getColumna();
                            target[idx] = 1.0;

                            pairs.add(new TrainingPair(inputVec, target));
                        }

                        estado = bestNextState;
                        isGameOver = FuncionesGato.finGato(estado);
                        currentBand = FuncionesGato.marcaMaquinaGato(currentBand);
                        currentTurn = (currentTurn == 1) ? 2 : 1;
                        moves++;
                    }

                    // Actualizar progreso
                    int count = procesados.incrementAndGet();
                    if (count % 100 == 0) {
                        long elapsed = System.currentTimeMillis() - startSimulationTime;
                        System.out.println(String.format(
                                "Progreso: %d/%d estados | Tiempo: %.2f min | Pares generados: %d",
                                count, totalEstados, elapsed / 60000.0, pairs.size()));
                    }

                    return pairs.stream();
                })
                .collect(Collectors.toList());

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

    public double[] tabularToInput(Tablero tablero, int turno) {
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

    public String getNombreModeloBasico() {
        return nombreModeloBasico;
    }

    public String getNombreModeloNormal() {
        return nombreModeloNormal;
    }

    public String getNombreModeloAvanzado() {
        return nombreModeloAvanzado;
    }

    public String getNombreModeloExperto() {
        return nombreModeloExperto;
    }

    public int getProfundidadBasico() {
        return profundidadBasico;
    }

    public int getProfundidadNormal() {
        return profundidad;
    }

    public int getProfundidadAvanzado() {
        return profundidadAvanzado;
    }

    public int getProfundidadExperto() {
        return profundidadExperto;
    }

    public void setProfundidad(int profundidad) {
        this.profundidad = profundidad;
    }

    public List<Movimiento> getPosiblesEstados() {
        return posiblesEstados;
    }
}
