package org.javig.util;

public class FormatClassToString {

    public static String formatDoubleDimensionArray(double[][] d) {
        String s = "";
        for (double[] row : d) {
            s += "[";
            for (double value : row) {
                s += String.format("%.4f", value) + ", ";
            }
            s += "]\n";
        }
        return s;
    }

    public static String formatDoubleArray(double[] d) {
        String s = "[";
        for (double value : d) {
            s += String.format("%.4f", value) + ", ";
        }
        s += "]";
        return s;
    }

}
