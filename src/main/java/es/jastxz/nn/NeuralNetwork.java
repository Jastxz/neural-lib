package es.jastxz.nn;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import es.jastxz.math.Matrix;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
    private int[] topology;
    private List<Matrix> weights;
    private List<Matrix> biases;
    private double learningRate = 0.1;
    private Activation activation = Activation.SIGMOID;

    // Pre-allocated buffers for train() — eliminates per-call allocation
    private transient Matrix[] layerOutputs;
    private transient Matrix[] gradients;
    private transient Matrix[] transposedPrev;
    private transient Matrix[] deltaWeights;
    private transient Matrix[] transposedWeights;
    private transient Matrix[] errors;
    private transient boolean buffersInitialized = false;

    public NeuralNetwork(int... topology) {
        this.topology = topology;
        this.weights = new ArrayList<>(topology.length - 1);
        this.biases = new ArrayList<>(topology.length - 1);

        for (int i = 0; i < topology.length - 1; i++) {
            Matrix w = new Matrix(topology[i + 1], topology[i]);
            w.randomize();
            weights.add(w);

            Matrix b = new Matrix(topology[i + 1], 1);
            b.randomize();
            biases.add(b);
        }
    }

    /**
     * Creates a neural network with pre-existing weights and biases.
     * Useful for reconstructing networks with reduced topologies.
     *
     * @param topology the network topology (layer sizes)
     * @param weights  the weight matrices (one per layer transition)
     * @param biases   the bias matrices (one per layer transition)
     * @throws IllegalArgumentException if dimensions are inconsistent
     */
    public NeuralNetwork(int[] topology, List<Matrix> weights, List<Matrix> biases) {
        int expected = topology.length - 1;
        if (weights.size() != expected) {
            throw new IllegalArgumentException(
                    "weights.size() debe ser " + expected + " (valor: " + weights.size() + ")");
        }
        if (biases.size() != expected) {
            throw new IllegalArgumentException(
                    "biases.size() debe ser " + expected + " (valor: " + biases.size() + ")");
        }
        for (int i = 0; i < expected; i++) {
            Matrix w = weights.get(i);
            if (w.getRows() != topology[i + 1] || w.getCols() != topology[i]) {
                throw new IllegalArgumentException(
                        "weights[" + i + "] debe ser " + topology[i + 1] + "x" + topology[i]
                                + " (actual: " + w.getRows() + "x" + w.getCols() + ")");
            }
            Matrix b = biases.get(i);
            if (b.getRows() != topology[i + 1] || b.getCols() != 1) {
                throw new IllegalArgumentException(
                        "biases[" + i + "] debe ser " + topology[i + 1] + "x1"
                                + " (actual: " + b.getRows() + "x" + b.getCols() + ")");
            }
        }

        this.topology = Arrays.copyOf(topology, topology.length);
        this.weights = new ArrayList<>(expected);
        this.biases = new ArrayList<>(expected);
        for (int i = 0; i < expected; i++) {
            this.weights.add(deepCopyMatrix(weights.get(i)));
            this.biases.add(deepCopyMatrix(biases.get(i)));
        }
    }

    private static Matrix deepCopyMatrix(Matrix src) {
        double[][] srcData = src.getData();
        double[][] copy = new double[src.getRows()][src.getCols()];
        for (int r = 0; r < src.getRows(); r++) {
            System.arraycopy(srcData[r], 0, copy[r], 0, src.getCols());
        }
        return new Matrix(copy);
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    /**
     * Returns a defensive copy of the network topology.
     */
    public int[] getTopology() {
        return java.util.Arrays.copyOf(topology, topology.length);
    }

    /**
     * Returns an immutable copy of the weight matrices.
     */
    public List<Matrix> getWeights() {
        return List.copyOf(weights);
    }

    /**
     * Returns an immutable copy of the bias matrices.
     */
    public List<Matrix> getBiases() {
        return List.copyOf(biases);
    }


    /**
     * Lazily initializes all reusable buffers for training.
     * Called once, then reused across all train() calls — zero GC pressure in steady state.
     */
    private void ensureBuffers() {
        if (buffersInitialized) return;

        int numLayers = topology.length;
        int numWeights = weights.size();

        // Layer outputs: one per layer (input + hidden + output)
        layerOutputs = new Matrix[numLayers];
        for (int i = 0; i < numLayers; i++) {
            layerOutputs[i] = new Matrix(topology[i], 1);
        }

        // Gradients: one per weight layer
        gradients = new Matrix[numWeights];
        for (int i = 0; i < numWeights; i++) {
            gradients[i] = new Matrix(topology[i + 1], 1);
        }

        // Transposed previous layer outputs (for delta weight calculation)
        transposedPrev = new Matrix[numWeights];
        for (int i = 0; i < numWeights; i++) {
            transposedPrev[i] = new Matrix(1, topology[i]);
        }

        // Delta weights: same dimensions as weights
        deltaWeights = new Matrix[numWeights];
        for (int i = 0; i < numWeights; i++) {
            deltaWeights[i] = new Matrix(topology[i + 1], topology[i]);
        }

        // Transposed weights (for error backpropagation)
        transposedWeights = new Matrix[numWeights];
        for (int i = 0; i < numWeights; i++) {
            transposedWeights[i] = new Matrix(topology[i], topology[i + 1]);
        }

        // Errors: one per layer (reused across backprop steps)
        // Max size needed is the largest layer
        errors = new Matrix[numWeights];
        for (int i = 0; i < numWeights; i++) {
            errors[i] = new Matrix(topology[i + 1], 1);
        }

        buffersInitialized = true;
    }

    public double[] feedForward(double[] inputArray) {
        Matrix current = Matrix.fromArray(inputArray);

        for (int i = 0; i < weights.size(); i++) {
            current = Matrix.multiply(weights.get(i), current);
            current.add(biases.get(i));
            current.map(activation.function);
        }

        return current.toArray();
    }

    public void train(double[] inputArray, double[] targetArray) {
        ensureBuffers();

        int numWeights = weights.size();

        // --- Forward pass (storing outputs in pre-allocated buffers) ---
        Matrix.fromArrayInto(inputArray, layerOutputs[0]);

        for (int i = 0; i < numWeights; i++) {
            Matrix.multiplyInto(weights.get(i), layerOutputs[i], layerOutputs[i + 1]);
            layerOutputs[i + 1].add(biases.get(i));
            layerOutputs[i + 1].map(activation.function);
        }

        // --- Backpropagation ---
        // Initial error = targets - output (into errors[last])
        Matrix lastError = errors[numWeights - 1];
        Matrix output = layerOutputs[numWeights];
        double[][] lastErrData = lastError.getData();
        double[][] outData = output.getData();
        for (int i = 0; i < lastError.getRows(); i++) {
            lastErrData[i][0] = targetArray[i] - outData[i][0];
        }

        Matrix currentError = lastError;

        for (int i = numWeights - 1; i >= 0; i--) {
            Matrix nextLayer = layerOutputs[i + 1];
            Matrix prevLayer = layerOutputs[i];

            // Gradient = derivative(nextLayer) * error * learningRate
            Matrix gradient = gradients[i];
            Matrix.mapInto(nextLayer, activation.derivative, gradient);
            gradient.multiply(currentError);
            gradient.multiply(learningRate);

            // Delta weights = gradient * transpose(prevLayer)
            Matrix.transposeInto(prevLayer, transposedPrev[i]);
            Matrix.multiplyInto(gradient, transposedPrev[i], deltaWeights[i]);

            // Update weights and biases
            weights.get(i).add(deltaWeights[i]);
            biases.get(i).add(gradient);

            // Propagate error to previous layer
            if (i > 0) {
                Matrix.transposeInto(weights.get(i), transposedWeights[i]);
                Matrix.multiplyInto(transposedWeights[i], currentError, errors[i - 1]);
                currentError = errors[i - 1];
            }
        }
    }

    public void save(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(path)))) {
            oos.writeObject(this);
        }
    }

    public static NeuralNetwork load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(path)))) {
            return (NeuralNetwork) ois.readObject();
        }
    }

    /**
     * Re-initializes transient buffers after deserialization.
     */
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        this.buffersInitialized = false;
    }
}
