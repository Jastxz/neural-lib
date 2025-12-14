package org.javig;

import org.javig.nn.NeuralNetwork;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class XORTest {
    @Test
    void testXORLearning() {
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
        double[][] inputs = { { 0.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 0.0 }, { 1.0, 1.0 } };
        double[][] targets = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };

        // Train heavy to ensure convergence for test
        for (int i = 0; i < 50000; i++) {
            for (int j = 0; j < inputs.length; j++) {
                nn.train(inputs[j], targets[j]);
            }
        }

        // Validation with relaxed tolerance
        // 0,0 -> 0
        assertTrue(nn.feedForward(inputs[0]).get(0) < 0.1);
        // 0,1 -> 1
        assertTrue(nn.feedForward(inputs[1]).get(0) > 0.9);
        // 1,0 -> 1
        assertTrue(nn.feedForward(inputs[2]).get(0) > 0.9);
        // 1,1 -> 0
        assertTrue(nn.feedForward(inputs[3]).get(0) < 0.1);
    }
}
