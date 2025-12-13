package org.javig.util;

public class DataContainer {
    double[][] inputs;
    double[][] outputs;

    public DataContainer(double[][] inputs, double[][] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getOutputs() {
        return outputs;
    }
}
