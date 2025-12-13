package org.javig;

import org.javig.nn.NeuralNetwork;
import org.javig.util.ModelManager;
import org.javig.util.NeuralNetworkTrainer;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("---------------------neural-lib main---------------------");
        System.out.println("Demonstrating High-Level API with XOR Problem");

        // 1. Create a Network
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);

        // 2. Prepare Data
        double[][] training_inputs = {
                { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 }
        };
        double[][] training_outputs = {
                { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 }
        };

        // 3. Train using the Utility Class
        // Trains for 50,000 epochs, logging progress every 10,000 epochs
        NeuralNetworkTrainer.train(nn, training_inputs, training_outputs, 50000, 10000);

        // 4. Save the Model
        ModelManager.saveModel(nn, "xor_model.nn");

        // 5. Load the Model (Simulator)
        NeuralNetwork loadedModel = ModelManager.loadModel("xor_model.nn");

        // 6. Predict using the loaded model
        System.out.println("\nPredictions from loaded model:");
        for (double[] input : training_inputs) {
            List<Double> output = loadedModel.feedForward(input);
            System.out.printf("Input: [%.0f, %.0f] -> Output: %.4f%n",
                    input[0], input[1], output.get(0));
        }
    }
}