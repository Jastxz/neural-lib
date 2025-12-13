package org.javig.nn;

import java.util.function.DoubleUnaryOperator;

public enum Activation {
    SIGMOID(
            x -> 1 / (1 + Math.exp(-x)),
            y -> y * (1 - y)),
    TANH(
            Math::tanh,
            y -> 1 - (y * y)),
    RELU(
            x -> Math.max(0, x),
            y -> y > 0 ? 1 : 0);

    public final DoubleUnaryOperator function;
    public final DoubleUnaryOperator derivative;

    Activation(DoubleUnaryOperator function, DoubleUnaryOperator derivative) {
        this.function = function;
        this.derivative = derivative;
    }
}
