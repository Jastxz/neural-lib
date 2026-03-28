// Quick diagnostic - run inside the test to see actual precision
package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.*;
import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import java.util.List;

class DiagnosticoTest {

    private static final double[][] INPUTS = {{0,0},{0,1},{1,0},{1,1}};
    private static final double[][] TARGETS_OR  = {{0},{1},{1},{1}};
    private static final double[][] TARGETS_AND = {{0},{0},{0},{1}};
    private static final double[][] TARGETS_XOR = {{0},{1},{1},{0}};
    private static final int LIMITE = 64;
    private static final long SEMILLA = 42L;

    @Test
    void diagnostico() {
        diagnosticarPuerta("OR", TARGETS_OR);
        diagnosticarPuerta("AND", TARGETS_AND);
        diagnosticarPuerta("XOR", TARGETS_XOR);
    }

    void diagnosticarPuerta(String nombre, double[][] targets) {
        System.out.println("\n===== " + nombre + " =====");

        // Probar con más épocas y diferentes topologías
        int[][] topologias = {{2,4,1},{2,8,1},{2,4,4,1},{2,8,4,1}};
        int[] epocasArr = {10, 30, 50};

        for (int[] topo : topologias) {
            for (int epocas : epocasArr) {
                try {
                    ConfiguracionBenchmark cb = new ConfiguracionBenchmark(
                        NivelComplejidad.TRIVIAL, topo, epocas, 1, 1, SEMILLA);
                    RecolectorMetricas rec = new RecolectorMetricas();
                    ResultadoBenchmark res = rec.ejecutarYRecolectar(cb, INPUTS, targets);
                    System.out.printf("  Topo=%s epocas=%d -> precision=%.2f clasificacion=%s%n",
                        java.util.Arrays.toString(topo), epocas, res.precisionFinal(), res.clasificacion());
                } catch (Exception e) {
                    System.out.printf("  Topo=%s epocas=%d -> ERROR: %s%n",
                        java.util.Arrays.toString(topo), epocas, e.getMessage());
                }
            }
        }
    }
}
