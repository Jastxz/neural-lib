package org.javig.models;

import java.util.ArrayList;
import java.util.List;

import org.javig.engine.Funciones3enRaya;
import org.javig.engine.Minimax;
import org.javig.tipos.Mundo;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.Util;
import org.javig.nn.NeuralNetwork;
import org.javig.util.DataContainer;
import org.javig.util.FormatClassToString;
import org.javig.util.ModelManager;
import org.javig.util.NeuralNetworkTrainer;

public class Modelo3enRaya {

    private NeuralNetwork cerebro;
    private String nombreModelo = "modelo3enRaya.nn";
    private Mundo mundo;

    public Modelo3enRaya() {
        cerebro = new NeuralNetwork(9, 18, 9);
        Posicion posInicial = new Posicion(0, 0);
        Tablero tablero = Funciones3enRaya.inicial3enRaya();
        Movimiento movimientoInicial = new Movimiento(tablero, posInicial);
        String juego = Util.juego3enRaya;
        int dificultad = 2;
        int profundidad = 9;
        int marca = 1;
        int turno = 1;
        boolean esMaquina = true;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public void entrenar() {
        // 2. Prepare Data - Generar datos dinámicamente
        System.out.println("Generando datos de entrenamiento exhaustivos...");
        DataContainer data = generateTrainingData(mundo);
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        // 3. Train using the Utility Class
        // Trains for 400 epochs, logging progress every 50 epochs
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, 400, 50);

        // 4. Save the Model
        ModelManager.saveModel(cerebro, nombreModelo);

        // 5. Load the Model (Simulator)
        cerebro = ModelManager.loadModel(nombreModelo);

        // 6. Predict using the loaded model
        System.out.println("\nPredictions from loaded model:");
        for (int i = 0; i < training_inputs.length; i += 2000) {
            List<Double> output = cerebro.feedForward(training_inputs[i]);
            System.out.println(
                    "Input: " + FormatClassToString.formatDoubleArray(training_inputs[i]) + " -> Output: "
                            + ModelManager.getMejorMovimiento(output));
        }
    }

    public DataContainer generateTrainingData(Mundo mundo) {
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        List<int[][]> todasLasPosibilidades = generarTodasLasPosibilidades();

        for (int[][] estado : todasLasPosibilidades) {
            Tablero tablero = new Tablero(new SmallMatrix(estado));
            mundo.setMovimiento(new Movimiento(tablero, new Posicion(0, 0)));

            if (Funciones3enRaya.fin3enRaya(tablero)) {
                continue;
                // double[] input = tabularToInput(tablero);
                // inputs.add(input);
                // outputs.add(input);
            } else {
                while (!Funciones3enRaya.fin3enRaya(tablero)) {
                    // 1. Get the IDEAL move from Minimax (Teacher)
                    // We must ensure 'mundo' has the current board state for Minimax to calculate
                    // correctly
                    mundo.getMovimiento().setTablero(tablero);

                    // Note: Minimax.negamax usually returns the Resulting Board, not the move.
                    // It calculates best move for whoever turn it is in 'mundo'
                    Tablero bestNextState = Minimax.negamax(mundo);

                    if (bestNextState == null)
                        break;

                    // 2. Record Training Data: (CurrentState -> IdealMove)
                    Posicion idealMove = obtenerMovimiento(tablero, bestNextState);
                    if (idealMove != null) {
                        double[] input = tabularToInput(tablero);
                        inputs.add(input);

                        double[] target = new double[9];
                        int index = idealMove.getFila() * 3 + idealMove.getColumna();
                        target[index] = 1.0;
                        outputs.add(target);

                        mundo.setMovimiento(new Movimiento(tablero, idealMove));
                        tablero = bestNextState;
                    } else {
                        break;
                    }
                }
            }

        }

        System.out.println("Total de muestras: " + inputs.size());
        double[][] training_inputs = inputs.toArray(new double[inputs.size()][]);
        double[][] training_outputs = outputs.toArray(new double[outputs.size()][]);

        return new DataContainer(training_inputs, training_outputs);
    }

    public double[] tabularToInput(Tablero tablero) {
        SmallMatrix tableroSmallMatrix = tablero.getMatrix();
        double[] inputs = new double[9];

        int count1 = 0;
        int count2 = 0;

        // First pass: count pieces to determine turn
        for (int i = 0; i < tableroSmallMatrix.getRows(); i++) {
            for (int j = 0; j < tableroSmallMatrix.getCols(); j++) {
                double valor = tableroSmallMatrix.get(i, j);
                if (valor == 1)
                    count1++;
                else if (valor == 2)
                    count2++;
            }
        }

        // Assuming 1 starts: if counts are equal, it's 1's turn. If 1 has more, it's
        // 2's turn.
        int currentTurn = (count1 == count2) ? 1 : 2;

        // Current Player -> 1.0
        // Opponent -> -1.0
        // Empty -> 0.0
        for (int i = 0; i < tableroSmallMatrix.getRows(); i++) {
            for (int j = 0; j < tableroSmallMatrix.getCols(); j++) {
                double valor = tableroSmallMatrix.get(i, j);
                int index = i * tableroSmallMatrix.getCols() + j;

                if (valor == 0) {
                    inputs[index] = 0.0;
                } else if (valor == currentTurn) {
                    inputs[index] = 1.0; // Me
                } else {
                    inputs[index] = -1.0; // Enemy
                }
            }
        }

        return inputs;
    }

    public List<int[][]> generarTodasLasPosibilidades() {
        List<int[][]> resultado = new ArrayList<>();
        int tamano = 3;
        int totalCeldas = tamano * tamano;
        int totalCombinaciones = (int) Math.pow(3, totalCeldas);
        System.out.println("Total de combinaciones de tablero para 3 en raya: " + totalCombinaciones);

        for (int i = 0; i < totalCombinaciones; i++) {
            int[][] matriz = new int[tamano][tamano];
            int valor = i;

            for (int fila = 0; fila < tamano; fila++) {
                for (int col = 0; col < tamano; col++) {
                    int resto = valor % 3;
                    matriz[fila][col] = resto - 1; // Convierte 0,1,2 a -1,0,1
                    valor /= 3;
                }
            }

            resultado.add(matriz);
        }

        return resultado;
    }

    private Posicion obtenerMovimiento(Tablero inicial, Tablero fin) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (inicial.getValor(i, j) == 0 && fin.getValor(i, j) != 0) {
                    return new Posicion(i, j);
                }
            }
        }
        return null;
    }

    private List<Posicion> getMovimientosValidos(Tablero t) {
        List<Posicion> moves = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (t.getValor(i, j) == 0)
                    moves.add(new Posicion(i, j));
            }
        }
        return moves;
    }

}
