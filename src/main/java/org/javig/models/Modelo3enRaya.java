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
    private Posicion posInicial = new Posicion(0, 0);
    private Tablero tablero = Funciones3enRaya.inicial3enRaya();
    private Movimiento movimientoInicial = new Movimiento(tablero, posInicial);
    private String juego = Util.juego3enRaya;
    private int dificultad = 2;
    private int profundidad = 9;
    private int marca = 1;
    private int turno = 1;
    private boolean esMaquina = true;

    public Modelo3enRaya() {
        cerebro = new NeuralNetwork(9, 15, 9);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public Modelo3enRaya(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
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
        // Trains for 100 epochs, logging progress every 10 epochs
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, 100, 10);

        // 4. Save the Model
        ModelManager.saveModel(cerebro, nombreModelo);

        // 5. Load the Model (Simulator)
        cerebro = ModelManager.loadModel(nombreModelo);

        // 6. Predict using the loaded model
        System.out.println("\nPredictions from loaded model:");
        for (int i = 0; i < training_inputs.length; i += 200) {
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

            // 1. Get the IDEAL move from Minimax (Teacher)
            // We must ensure 'mundo' has the current board state for Minimax to calculate
            // correctly
            mundo.getMovimiento().setTablero(tablero);

            // Note: Minimax.negamax usually returns the Resulting Board, not the move.
            // It calculates best move for whoever turn it is in 'mundo'
            Tablero bestNextState = Minimax.negamax(mundo);

            if (bestNextState == null) {
                break;
            }

            // 2. Record Training Data: (CurrentState -> IdealMove)
            Posicion idealMove = obtenerMovimiento(tablero, bestNextState);
            if (idealMove != null) {
                double[] input = tabularToInput(tablero);
                inputs.add(input);

                double[] target = new double[9];
                int index = idealMove.getFila() * 3 + idealMove.getColumna();
                // FIX: Target should be 1.0 (High activation) so predictIndex (Max) picks it.
                target[index] = 1.0;
                outputs.add(target);
            }
        }

        double[][] training_inputs = inputs.toArray(new double[inputs.size()][]);
        double[][] training_outputs = outputs.toArray(new double[outputs.size()][]);

        return new DataContainer(training_inputs, training_outputs);
    }

    public double[] tabularToInput(Tablero tablero) {
        SmallMatrix tableroSmallMatrix = tablero.getMatrix();
        double[] inputs = new double[9];
        for (int i = 0; i < tableroSmallMatrix.getRows(); i++) {
            for (int j = 0; j < tableroSmallMatrix.getCols(); j++) {
                double valor = tableroSmallMatrix.get(i, j);
                int index = i * tableroSmallMatrix.getCols() + j;

                // FIX: Normalize inputs for Neural Network (0, 1, -1)
                // Assuming Tablero inputs are standard 0, 1, 2
                if (valor == 0)
                    inputs[index] = 0.0;
                else if (valor == 1)
                    inputs[index] = 1.0;
                else if (valor == 2)
                    inputs[index] = -1.0; // P2 is -1
            }
        }

        return inputs;
    }

    public List<int[][]> generarTodasLasPosibilidades() {
        List<int[][]> resultado = new ArrayList<>();
        int tamano = 3;
        int totalCeldas = tamano * tamano; // 9
        int totalCombinaciones = (int) Math.pow(3, totalCeldas);

        for (int i = 0; i < totalCombinaciones; i++) {
            int[][] matriz = new int[tamano][tamano];
            int valor = i;
            int count1 = 0;
            int count2 = 0;

            for (int fila = 0; fila < tamano; fila++) {
                for (int col = 0; col < tamano; col++) {
                    int resto = valor % 3;
                    // FIX: Use standard game values 0, 1, 2
                    matriz[fila][col] = resto;

                    if (matriz[fila][col] == 1)
                        count1++;
                    else if (matriz[fila][col] == 2)
                        count2++;

                    valor /= 3;
                }
            }

            // Valid state check
            Tablero tablero = new Tablero(new SmallMatrix(matriz));
            // We want states where it is P2's turn? The user said "use -1".
            // Since we mapped 2 -> -1 in tabularToInput, P2 is the "-1 player".
            // P2 moves when P1 has 1 more piece (assuming P1 starts).
            // So count1 == count2 + 1.
            if (count1 == count2 + 1 && !Funciones3enRaya.fin3enRaya(tablero)) {
                resultado.add(matriz);
            }
        }

        return resultado;
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

    public Tablero obtenerTablero(Tablero tablero, Posicion pos) {
        Tablero nuevoTablero = new Tablero(tablero.getMatrix());
        nuevoTablero.setValue(pos, mundo.getMarca());
        return nuevoTablero;
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
