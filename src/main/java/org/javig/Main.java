package org.javig;

import org.javig.nn.NeuralNetwork;
import org.javig.util.ModelManager;
import org.javig.util.NeuralNetworkTrainer;

public class Main {
    public static void main(String[] args) {
        System.out.println("---------------------neural-lib main---------------------");
        System.out.println("Demostración de la API de Alto Nivel con el Problema XOR");

        // 1. Crear la red
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);

        // 2. Preparar datos
        double[][] training_inputs = {
                { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }
        };
        double[][] training_outputs = {
                { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 }
        };

        // 3. Entrenar usando la Clase de Utilidad
        // Entrena por 50,000 épocas, registrando progreso cada 10,000 épocas
        NeuralNetworkTrainer.train(nn, training_inputs, training_outputs, 50000, 10000);

        // 4. Guardar el Modelo
        ModelManager.saveModel(nn, "xor_model.nn");

        // 5. Cargar el Modelo (Simulador)
        NeuralNetwork loadedModel = ModelManager.loadModel("xor_model.nn");

        // 6. Predecir usando el modelo cargado
        System.out.println("\nPredicciones del modelo cargado:");
        for (double[] input : training_inputs) {
            double[] output = loadedModel.feedForward(input);
            System.out.printf("Entrada: [%.0f, %.0f] -> Salida: %.4f%n",
                    input[0], input[1], output[0]);
        }
    }
}