package es.jastxz.math;

import java.io.Serializable;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.DoubleUnaryOperator;

public class Matrix implements Serializable {
    private static final long serialVersionUID = 1L;
    private double[][] data;
    private int rows;
    private int cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.data = data;
        this.rows = data.length;
        this.cols = data[0].length;
    }

    // ── In-place operations (zero allocation) ──

    public void add(Matrix other) {
        for (int i = 0; i < rows; i++) {
            double[] thisRow = this.data[i];
            double[] otherRow = other.data[i];
            for (int j = 0; j < cols; j++) {
                thisRow[j] += otherRow[j];
            }
        }
    }

    public void add(double scaler) {
        for (int i = 0; i < rows; i++) {
            double[] row = this.data[i];
            for (int j = 0; j < cols; j++) {
                row[j] += scaler;
            }
        }
    }

    public void multiply(Matrix other) {
        for (int i = 0; i < rows; i++) {
            double[] thisRow = this.data[i];
            double[] otherRow = other.data[i];
            for (int j = 0; j < cols; j++) {
                thisRow[j] *= otherRow[j];
            }
        }
    }

    public void multiply(double scaler) {
        for (int i = 0; i < rows; i++) {
            double[] row = this.data[i];
            for (int j = 0; j < cols; j++) {
                row[j] *= scaler;
            }
        }
    }

    public void map(DoubleUnaryOperator func) {
        for (int i = 0; i < rows; i++) {
            double[] row = this.data[i];
            for (int j = 0; j < cols; j++) {
                row[j] = func.applyAsDouble(row[j]);
            }
        }
    }

    // ── Static operations ──

    public static Matrix subtract(Matrix a, Matrix b) {
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            double[] aRow = a.data[i];
            double[] bRow = b.data[i];
            double[] rRow = result.data[i];
            for (int j = 0; j < a.cols; j++) {
                rRow[j] = aRow[j] - bRow[j];
            }
        }
        return result;
    }

    /**
     * Subtract into a pre-allocated target matrix (zero allocation).
     */
    public static void subtractInto(Matrix a, Matrix b, Matrix target) {
        for (int i = 0; i < a.rows; i++) {
            double[] aRow = a.data[i];
            double[] bRow = b.data[i];
            double[] tRow = target.data[i];
            for (int j = 0; j < a.cols; j++) {
                tRow[j] = aRow[j] - bRow[j];
            }
        }
    }

    /**
     * Cache-friendly matrix multiplication using loop reordering (i-k-j).
     * Accesses b.data[k] row-sequentially instead of column-striding.
     */
    public static Matrix multiply(Matrix a, Matrix b) {
        int aRows = a.rows, aCols = a.cols, bCols = b.cols;
        Matrix result = new Matrix(aRows, bCols);
        double[][] ad = a.data, bd = b.data, rd = result.data;

        for (int i = 0; i < aRows; i++) {
            double[] aRow = ad[i];
            double[] rRow = rd[i];
            for (int k = 0; k < aCols; k++) {
                double aik = aRow[k];
                double[] bRow = bd[k];
                for (int j = 0; j < bCols; j++) {
                    rRow[j] += aik * bRow[j];
                }
            }
        }
        return result;
    }

    /**
     * Multiply into a pre-allocated target matrix (zero allocation).
     * Target is zeroed before accumulation.
     */
    public static void multiplyInto(Matrix a, Matrix b, Matrix target) {
        int aRows = a.rows, aCols = a.cols, bCols = b.cols;
        double[][] ad = a.data, bd = b.data, td = target.data;

        // Zero target
        for (int i = 0; i < aRows; i++) {
            Arrays.fill(td[i], 0.0);
        }

        for (int i = 0; i < aRows; i++) {
            double[] aRow = ad[i];
            double[] tRow = td[i];
            for (int k = 0; k < aCols; k++) {
                double aik = aRow[k];
                double[] bRow = bd[k];
                for (int j = 0; j < bCols; j++) {
                    tRow[j] += aik * bRow[j];
                }
            }
        }
    }

    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            for (int j = 0; j < m.cols; j++) {
                result.data[j][i] = mRow[j];
            }
        }
        return result;
    }

    /**
     * Transpose into a pre-allocated target matrix (zero allocation).
     */
    public static void transposeInto(Matrix m, Matrix target) {
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            for (int j = 0; j < m.cols; j++) {
                target.data[j][i] = mRow[j];
            }
        }
    }

    public static Matrix map(Matrix m, DoubleUnaryOperator func) {
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            double[] rRow = result.data[i];
            for (int j = 0; j < m.cols; j++) {
                rRow[j] = func.applyAsDouble(mRow[j]);
            }
        }
        return result;
    }

    /**
     * Map into a pre-allocated target matrix (zero allocation).
     */
    public static void mapInto(Matrix m, DoubleUnaryOperator func, Matrix target) {
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            double[] tRow = target.data[i];
            for (int j = 0; j < m.cols; j++) {
                tRow[j] = func.applyAsDouble(mRow[j]);
            }
        }
    }

    // ── Conversion ──

    public static Matrix fromArray(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    /**
     * Fill a pre-allocated column matrix from an array (zero allocation).
     */
    public static void fromArrayInto(double[] arr, Matrix target) {
        for (int i = 0; i < arr.length; i++) {
            target.data[i][0] = arr[i];
        }
    }

    public double[] toArray() {
        double[] arr = new double[rows * cols];
        int k = 0;
        for (int i = 0; i < rows; i++) {
            double[] row = data[i];
            for (int j = 0; j < cols; j++) {
                arr[k++] = row[j];
            }
        }
        return arr;
    }

    public void randomize() {
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < rows; i++) {
            double[] row = this.data[i];
            for (int j = 0; j < cols; j++) {
                row[j] = rng.nextDouble(-1.0, 1.0);
            }
        }
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double[][] getData() {
        return data;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (double[] row : data) {
            sb.append(Arrays.toString(row)).append("\n");
        }
        return sb.toString();
    }
}
