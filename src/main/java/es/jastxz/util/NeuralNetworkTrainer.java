package es.jastxz.util;

import es.jastxz.nn.NeuralNetwork;

public class NeuralNetworkTrainer {

    public static void train(NeuralNetwork nn, double[][] inputs, double[][] targets, int epochs, int logInterval) {
        System.out.println("Starting training for " + epochs + " epochs...");
        long startTime = System.currentTimeMillis();

        for (int i = 1; i <= epochs; i++) {
            // Shuffle could be implemented here for better SGD performance

            for (int j = 0; j < inputs.length; j++) {
                nn.train(inputs[j], targets[j]);
            }

            if (i % logInterval == 0 || i == epochs) {
                double mse = calculateMSE(nn, inputs, targets);
                System.out.printf("Epoch %d/%d - Error (MSE): %.6f%n", i, epochs, mse);
                long milis = (System.currentTimeMillis() - startTime);
                System.out.println("Time taken: " + milis + "ms, " + milis / 1000 + "s, " + milis / 60000 + "min");
            }
        }
    }

    private static double calculateMSE(NeuralNetwork nn, double[][] inputs, double[][] targets) {
        double sumError = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] output = nn.feedForward(inputs[i]);
            double[] target = targets[i];

            for (int k = 0; k < target.length; k++) {
                double error = target[k] - output[k];
                sumError += error * error;
            }
        }
        return sumError / (inputs.length * targets[0].length);
    }
}
