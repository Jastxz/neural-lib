package org.javig.nn;

import org.javig.math.Matrix;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;
    private int[] topology;
    private List<Matrix> weights;
    private List<Matrix> biases;
    private double learningRate = 0.1;
    private Activation activation = Activation.SIGMOID;

    public NeuralNetwork(int... topology) {
        this.topology = topology;
        this.weights = new ArrayList<>();
        this.biases = new ArrayList<>();

        for (int i = 0; i < topology.length - 1; i++) {
            // Weights connecting layer i to i+1
            // Rows = neurons in i+1, Cols = neurons in i
            Matrix w = new Matrix(topology[i + 1], topology[i]);
            w.randomize();
            weights.add(w);

            // Biases for layer i+1
            Matrix b = new Matrix(topology[i + 1], 1);
            b.randomize();
            biases.add(b);
        }
    }

    public void setLearningRate(double lr) {
        this.learningRate = lr;
    }

    public void setActivation(Activation activation) {
        this.activation = activation;
    }

    public List<Double> feedForward(double[] inputArray) {
        Matrix current = Matrix.fromArray(inputArray);

        for (int i = 0; i < weights.size(); i++) {
            current = Matrix.multiply(weights.get(i), current);
            current.add(biases.get(i));
            current.map(activation.function);
        }

        double[] outArr = current.toArray();
        List<Double> output = new ArrayList<>();
        for (double d : outArr)
            output.add(d);
        return output;
    }

    public void train(double[] inputArray, double[] targetArray) {
        // --- Feed Forward (Storing Hidden States) ---
        Matrix inputs = Matrix.fromArray(inputArray);
        List<Matrix> layers = new ArrayList<>();
        layers.add(inputs);

        Matrix current = inputs;
        for (int i = 0; i < weights.size(); i++) {
            current = Matrix.multiply(weights.get(i), current);
            current.add(biases.get(i));
            current.map(activation.function);
            layers.add(current);
        }

        // --- Backpropagation ---
        Matrix targets = Matrix.fromArray(targetArray);
        Matrix output = layers.get(layers.size() - 1);

        // Error = Targets - Output
        Matrix error = Matrix.subtract(targets, output);

        // Propagate backwards
        for (int i = weights.size() - 1; i >= 0; i--) {
            // Gradient = learning_rate * error * derivative(output)
            // Note: In simple SGD, derivative is of the Activated value

            Matrix nextLayer = layers.get(i + 1); // Output of current layer
            Matrix prevLayer = layers.get(i); // Input to current layer (Output of prev)

            // Calculate Gradient
            Matrix gradient = Matrix.map(nextLayer, activation.derivative);
            gradient.multiply(error);
            gradient.multiply(learningRate);

            // Calculate Deltas: Gradient * Transpose(prevLayer)
            Matrix prevT = Matrix.transpose(prevLayer);
            Matrix deltaWeights = Matrix.multiply(gradient, prevT);

            // Adjust Weights and Biases
            weights.get(i).add(deltaWeights);
            biases.get(i).add(gradient);

            // Calculate Error for next iteration (previous layer)
            // error_prev = Transpose(weights) * error_current
            Matrix weightsT = Matrix.transpose(weights.get(i));
            error = Matrix.multiply(weightsT, error);
        }
    }

    public void save(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        }
    }

    public static NeuralNetwork load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (NeuralNetwork) ois.readObject();
        }
    }
}
