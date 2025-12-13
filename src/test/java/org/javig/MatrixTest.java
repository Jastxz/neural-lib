package org.javig;

import org.javig.math.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {
    @Test
    void testAdd() {
        Matrix m1 = new Matrix(2, 2);
        // Default init is 0.0
        m1.add(1.0);

        Matrix m2 = new Matrix(2, 2);
        m2.add(2.0);

        m1.add(m2);
        assertEquals(3.0, m1.getData()[0][0]);
    }

    @Test
    void testMultiply() {
        Matrix m1 = new Matrix(2, 3);
        m1.randomize();
        Matrix m2 = new Matrix(3, 2);
        m2.randomize();

        Matrix result = Matrix.multiply(m1, m2);
        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
    }

    @Test
    void testTranspose() {
        Matrix m = new Matrix(2, 3); // 2 rows, 3 cols
        Matrix t = Matrix.transpose(m);
        assertEquals(3, t.getRows());
        assertEquals(2, t.getCols());
    }
}
