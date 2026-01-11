package es.jastxz.models;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import es.jastxz.engine.Funciones3enRaya;
import es.jastxz.engine.Minimax;
import es.jastxz.tipos.Mundo;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import es.jastxz.util.Util;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.util.DataContainer;
import es.jastxz.util.ModelManager;
import es.jastxz.util.NeuralNetworkTrainer;

public class Modelo3enRaya {

    private NeuralNetwork cerebro;
    private String nombreModelo = "modelo3enRaya.nn";
    private Mundo mundo;
    private Posicion posInicial = new Posicion(0, 0);
    private Tablero tablero = Funciones3enRaya.inicial3enRaya();
    private Movimiento movimientoInicial = new Movimiento(tablero, posInicial);
    private String juego = Util.juego3enRaya;
    private int dificultad = 4;
    private int profundidad = 9;
    private int marca = 1;
    private int turno = 1;
    private boolean esMaquina = true;

    private Map<String, Integer> solverCache = new HashMap<>();

    public Modelo3enRaya() {
        cerebro = new NeuralNetwork(10, 30, 9);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public Modelo3enRaya(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public void entrenar(int epocas) {
        // 1. Preparar datos - Generar datos dinámicamente
        System.out.println("Generando datos de entrenamiento exhaustivos...");
        DataContainer data = generateTrainingData(mundo);
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        // 2. Entrenar usando la Clase de Utilidad
        int intervalo = epocas / 10;
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, epocas, intervalo);

        // 3. Guardar el Modelo
        ModelManager.saveModel(cerebro, nombreModelo);
    }

    public DataContainer generateTrainingData(Mundo mundo) {
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        List<int[][]> todasLasPosibilidades = new ArrayList<>(4519);
        Set<String> estadosVistos = new HashSet<>(4519);

        // Add initial state explicitly
        Tablero tableroInicial = new Tablero(new SmallMatrix(new int[3][3]));
        estadosVistos.add(tableroInicial.toString());
        todasLasPosibilidades.add(tableroInicial.getMatrix().getData());

        generarTodasLasPosibilidades(tableroInicial, 1, todasLasPosibilidades,
                estadosVistos);
        System.out.println("Total de posibilidades: " + todasLasPosibilidades.size());

        // Pre-solve everything (optional step, but good for cache warming if we were
        // doing recursion from scratch,
        // here we just iterate).
        // Actually, we can just solve on demand.

        for (int[][] estado : todasLasPosibilidades) {
            Tablero tablero = new Tablero(new SmallMatrix(estado));

            // Determinar turno
            int count1 = 0;
            int count2 = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (estado[i][j] == 1)
                        count1++;
                    else if (estado[i][j] == 2)
                        count2++;
                }
            }
            int currentTurn = (count1 == count2) ? 1 : 2;

            // Generate all legal moves from this state
            List<Movimiento> moves = Funciones3enRaya.movs3enRaya(tablero, currentTurn);

            if (moves.isEmpty())
                continue; // Terminal state already handling? No, generated list might include terminal
                          // states?
            // generarTodasLasPosibilidades adds states IF !fin3enRaya. So these are
            // non-terminal.

            // Evaluate all moves
            double[] target = new double[9];
            int bestOutcome = -2; // -1 (loss), 0 (tie), 1 (win). Starts lower than loss.

            // We need to map the solver's view (1=P1 win, -1=P2 win) to "Current Player
            // Value"
            // If Turn=1, wants max (1). If Turn=2, wants min (-1).

            // Actually, let's normalize 'solveState' validation:
            // solveState returns: 1 (P1 wins), -1 (P2 wins), 0 (Draw).

            // First pass: find the BEST reachable outcome for current player
            List<Integer> moveOutcomes = new ArrayList<>();

            for (Movimiento mov : moves) {
                int val = solveState(mov.getTablero()); // Value of the resulting state
                moveOutcomes.add(val);

                // Update best outcome for current player
                if (currentTurn == 1) {
                    // P1 wants Max (1)
                    if (val > bestOutcome || bestOutcome == -2)
                        bestOutcome = val;
                } else {
                    // P2 wants Min (-1)
                    if (val < bestOutcome || bestOutcome == -2)
                        bestOutcome = val;
                }
            }

            // Second pass: Mark all moves that achieve bestOutcome as 1.0
            boolean anyMoveAdded = false;
            for (int k = 0; k < moves.size(); k++) {
                if (moveOutcomes.get(k) == bestOutcome) {
                    Posicion pos = moves.get(k).getPos();
                    int index = pos.getFila() * 3 + pos.getColumna();
                    target[index] = 1.0;
                    anyMoveAdded = true;
                }
            }

            if (anyMoveAdded) {
                inputs.add(tabularToInput(tablero, currentTurn));
                outputs.add(target);
            }
        }

        double[][] training_inputs = inputs.toArray(new double[inputs.size()][]);
        double[][] training_outputs = outputs.toArray(new double[outputs.size()][]);

        return new DataContainer(training_inputs, training_outputs);
    }

    // Returns 1 if P1 wins, -1 if P2 wins, 0 if Draw.
    private int solveState(Tablero t) {
        String key = t.toString();
        if (solverCache.containsKey(key)) {
            return solverCache.get(key);
        }

        if (Funciones3enRaya.fin3enRaya(t)) {
            if (Funciones3enRaya.hay3EnRaya(t)) {
                // Who won? The last mover.
                // If it's P1 turn now, it means P2 just moved and won.
                // We need to check who has the pieces.
                // Actually, logic is easier: analyze board.
                // But Funciones3enRaya.hay3EnRaya doesn't say WHO won.
                // Heuristic: Count pieces. If P1==P2, P2 just moved. If P1==P2+1, P1 just
                // moved.
                // Wait, simply checking:
                // We can assume valid gameplay.
                // Iterate through lines to find winner? OR pass turn?
                // Let's implement simple line check or trust inference.
                // Simpler: pass 'turn' to solveState?
                // But we want to cache by board state. Board state implicitly contains turn
                // count.

                int c1 = 0, c2 = 0;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) {
                        if (t.getValor(i, j) == 1)
                            c1++;
                        else if (t.getValor(i, j) == 2)
                            c2++;
                    }

                // If 3-in-row exists:
                // If c1 == c2, P2 made the last move -> P2 won -> -1
                // If c1 > c2, P1 made the last move -> P1 won -> 1
                return (c1 == c2) ? -1 : 1;
            }
            return 0; // Draw
        }

        // Determinar turno based on counts
        int c1 = 0, c2 = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                if (t.getValor(i, j) == 1)
                    c1++;
                else if (t.getValor(i, j) == 2)
                    c2++;
            }
        int turn = (c1 == c2) ? 1 : 2;

        List<Movimiento> moves = Funciones3enRaya.movs3enRaya(t, turn);

        int bestVal = (turn == 1) ? -2 : 2; // P1 starts low, P2 starts high

        for (Movimiento m : moves) {
            int val = solveState(m.getTablero());
            if (turn == 1) {
                if (val > bestVal)
                    bestVal = val;
            } else {
                if (val < bestVal)
                    bestVal = val;
            }
        }

        solverCache.put(key, bestVal);
        return bestVal;
    }

    public static double[] tabularToInput(Tablero tablero, int turno) {
        SmallMatrix tableroSmallMatrix = tablero.getMatrix();
        double[] inputs = new double[10];
        for (int i = 0; i < tableroSmallMatrix.getRows(); i++) {
            for (int j = 0; j < tableroSmallMatrix.getCols(); j++) {
                double valor = tableroSmallMatrix.get(i, j);
                int index = i * tableroSmallMatrix.getCols() + j;

                // Normalizar entradas para la Red Neuronal (0, 1, -1)
                // Asumiendo que las entradas del Tablero son standard 0, 1, 2
                if (valor == 0)
                    inputs[index] = 0.0;
                else if (valor == 1)
                    inputs[index] = 1.0;
                else if (valor == 2)
                    inputs[index] = -1.0; // J2 es -1
            }
        }

        // Agregar turno al final
        inputs[9] = (turno == 1) ? 1.0 : -1.0;

        return inputs;
    }

    private void generarTodasLasPosibilidades(Tablero tablero, int turno, List<int[][]> resultado,
            Set<String> estadosVistos) {
        List<Movimiento> movimientosPosibles = Funciones3enRaya.movs3enRaya(tablero, turno);
        for (Movimiento movimiento : movimientosPosibles) {
            Tablero tableroNuevo = movimiento.getTablero();
            int[][] matriz = tableroNuevo.getMatrix().getData();
            if (!Funciones3enRaya.fin3enRaya(tableroNuevo) && !estadosVistos.contains(tableroNuevo.toString())) {
                estadosVistos.add(tableroNuevo.toString());
                resultado.add(matriz);
                generarTodasLasPosibilidades(tableroNuevo, turno == 1 ? 2 : 1, resultado, estadosVistos);
            }
        }
    }

    public Mundo getMundo() {
        return mundo;
    }

    public String getNombreModelo() {
        return nombreModelo;
    }

    public void setNombreModelo(String nombreModelo) {
        this.nombreModelo = nombreModelo;
    }

    public Posicion obtenerMovimiento(Tablero inicial, Tablero fin) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (inicial.getValor(i, j) == 0 && fin.getValor(i, j) != 0) {
                    return new Posicion(i, j);
                }
            }
        }
        return null;
    }

}
