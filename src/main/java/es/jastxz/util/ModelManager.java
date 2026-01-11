package es.jastxz.util;

import java.io.IOException;
import java.util.List;

import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.Tablero;

import es.jastxz.nn.NeuralNetwork;

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
     * Devuelve el índice de la neurona con el valor de activación más alto.
     * Útil para problemas de clasificación (ej. elegir un movimiento en 3 en Raya).
     */
    public static int predictIndex(NeuralNetwork nn, double[] input) {
        double[] output = nn.feedForward(input);
        int maxIndex = -1;
        double maxValue = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static Posicion getMejorMovimiento(double[] output) {
        return getMejorMovimiento(output, 3);
    }

    public static Posicion getMejorMovimiento(double[] output, int columnas) {
        int mejorIndice = 0;
        double maxValor = -1.0;

        for (int i = 0; i < output.length; i++) {
            if (output[i] > maxValor) {
                maxValor = output[i];
                mejorIndice = i;
            }
        }

        int fila = mejorIndice / columnas;
        int columna = mejorIndice % columnas;

        return new Posicion(fila, columna);
    }

    /*
     * Método heredado para compatibilidad si alguien pasa List, aunque se
     * recomienda usar double[]
     */
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
        return getMejorMovimiento(mejorIndice, 3);
    }

    public static Posicion getMejorMovimiento(int mejorIndice, int columnas) {
        int fila = mejorIndice / columnas;
        int columna = mejorIndice % columnas;

        return new Posicion(fila, columna);
    }

    public static Tablero obtenerTablero(Tablero tablero, Posicion pos, int marca) {
        Tablero nuevoTablero = new Tablero(tablero.getMatrix());
        nuevoTablero.setValue(pos, marca);
        return nuevoTablero;
    }
}