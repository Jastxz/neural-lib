package org.javig.util;

public class DataContainer {
    double[][] inputs;
    double[][] outputs;
    boolean initialized;

    public DataContainer() {
        this.initialized = false;
    }

    public DataContainer(double[][] inputs, double[][] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.initialized = true;
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getOutputs() {
        return outputs;
    }

    public boolean isInitialized() {
        return initialized;
    }

    @Override
    public String toString() {
        return "DataContainer [inputs=" + inputs.length + ", outputs=" + outputs.length + ", initialized=" + initialized
                + "]";
    }
}
