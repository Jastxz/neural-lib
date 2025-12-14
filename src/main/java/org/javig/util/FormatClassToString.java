package org.javig.util;

public class FormatClassToString {

    public static String formatDoubleDimensionArray(double[][] d) {
        String s = "[";
        for (double[] row : d) {
            s += "[";
            for (double value : row) {
                s += String.format("%.4f", value) + ", ";
            }
            s += "]";
        }
        s += "]";
        return s;
    }

    public static String formatDoubleArray(double[] d) {
        String s = "[";
        int rows = (int) Math.sqrt(d.length);
        int acum = 0;
        for (double value : d) {
            if (acum == 0) {
                s += "[";
            }
            s += String.format("%.4f", value) + ", ";
            if (acum == rows - 1) {
                s += "]";
                acum = 0;
            } else {
                acum++;
            }
        }
        s += "]";
        return s;
    }

    public static String formatIntDoubleDimensionArray(int[][] d) {
        String s = "[";
        for (int[] row : d) {
            s += "[";
            for (int value : row) {
                s += String.format("%d", value) + ", ";
            }
            s += "]";
        }
        s += "]";
        return s;
    }

}
