package org.javig.util;

public class TrainingPair {
    private double[] input;
    private double[] output;

    public TrainingPair(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public double[] getOutput() {
        return output;
    }
}
