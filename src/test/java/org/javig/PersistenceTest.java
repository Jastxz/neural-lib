package org.javig;

import org.javig.nn.NeuralNetwork;
import org.junit.jupiter.api.Test;
import java.io.File;
import java.io.IOException;
import static org.junit.jupiter.api.Assertions.*;

public class PersistenceTest {
    @Test
    void testSaveAndLoad() throws IOException, ClassNotFoundException {
        NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
        double[] input = { 1.0, 0.5 };

        // Get initial prediction/state
        double predictionBefore = nn.feedForward(input)[0];

        String filename = "test_model.nn";
        nn.save(filename);

        NeuralNetwork loadedNN = NeuralNetwork.load(filename);
        double predictionAfter = loadedNN.feedForward(input)[0];

        assertEquals(predictionBefore, predictionAfter, 0.0000001, "Prediction should be identical after loading");

        // Cleanup
        new File(filename).delete();
    }
}
