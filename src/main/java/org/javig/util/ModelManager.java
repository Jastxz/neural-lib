package org.javig.util;

import org.javig.nn.NeuralNetwork;

import java.io.IOException;
import java.util.List;

import org.javig.tipos.Posicion;

public class ModelManager {

    private static final String resourcesPath = "/home/javier/Repos/neural-lib/src/main/resources/";

    public static void saveModel(NeuralNetwork nn, String path) {
        path = resourcesPath + path;
        try {
            nn.save(path);
            System.out.println("Model saved successfully to: " + path);
        } catch (IOException e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static NeuralNetwork loadModel(String path) {
        path = resourcesPath + path;
        try {
            NeuralNetwork nn = NeuralNetwork.load(path);
            System.out.println("Model loaded successfully from: " + path);
            return nn;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Returns the index of the neuron with the highest activation value.
     * Useful for classification problems (e.g. choosing a move in Tic-Tac-Toe).
     */
    public static int predictIndex(NeuralNetwork nn, double[] input) {
        List<Double> output = nn.feedForward(input);
        int maxIndex = -1;
        double maxValue = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < output.size(); i++) {
            if (output.get(i) > maxValue) {
                maxValue = output.get(i);
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static Posicion getMejorMovimiento(List<Double> output) {
        int mejorIndice = 0;
        double maxValor = -1.0;

        for (int i = 0; i < output.size(); i++) {
            if (output.get(i) > maxValor) {
                maxValor = output.get(i);
                mejorIndice = i;
            }
        }

        int fila = mejorIndice / 3;
        int columna = mejorIndice % 3;

        return new Posicion(fila, columna);
    }

    public static Posicion getMejorMovimiento(int mejorIndice) {
        int fila = mejorIndice / 3;
        int columna = mejorIndice % 3;

        return new Posicion(fila, columna);
    }
}