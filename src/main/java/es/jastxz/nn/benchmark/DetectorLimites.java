package es.jastxz.nn.benchmark;

/**
 * Detector de límites operativos de la SNN basado en métricas de benchmark.
 *
 * <p>Aplica reglas de clasificación para identificar configuraciones que
 * alcanzan los límites de la red: precisión insuficiente, convergencia
 * estancada, infrautilización de neuronas o ineficiencia energética.</p>
 *
 * @since 1.1
 */
public class DetectorLimites {

    private DetectorLimites() {
        // Clase utilitaria, no instanciable
    }

    /**
     * Clasifica un resultado de benchmark según las reglas de detección de límites.
     *
     * <p>Las reglas se evalúan en orden de prioridad:</p>
     * <ol>
     *   <li>Precisión final &lt; 0.6 → {@code "limite_no_superado"}</li>
     *   <li>MSE no disminuye &gt;1% en 3 épocas consecutivas → {@code "convergencia_estancada"}</li>
     *   <li>Neuronas activas &lt; 20% del total → {@code "red_infrautilizada"}</li>
     * </ol>
     *
     * @param resultado resultado de benchmark a clasificar
     * @return etiqueta de clasificación, o {@code null} si no se detecta límite
     */
    public static String clasificar(ResultadoBenchmark resultado) {
        // Regla 1: Precisión < 60%
        if (resultado.precisionFinal() < 0.6) {
            return "limite_no_superado";
        }

        // Regla 2: MSE no baja >1% en 3 épocas consecutivas
        if (detectarEstancamiento(resultado.errorMSEPorEpoca())) {
            return "convergencia_estancada";
        }

        // Regla 3: Neuronas activas < 20% del total
        if (resultado.neuronasTotal() > 0
                && (double) resultado.neuronasActivas() / resultado.neuronasTotal() < 0.20) {
            return "red_infrautilizada";
        }

        return null;
    }

    /**
     * Clasifica un resultado como ineficiencia energética si su costo
     * supera 10 veces el costo mínimo proporcionado.
     *
     * @param resultado   resultado de benchmark a evaluar
     * @param costoMinimo costo energético mínimo de referencia
     * @return {@code "ineficiencia_energetica"} si el costo supera 10× el mínimo,
     *         {@code null} en caso contrario
     */
    public static String clasificarIneficiencia(ResultadoBenchmark resultado, double costoMinimo) {
        if (resultado.costoEnergetico() > costoMinimo * 10) {
            return "ineficiencia_energetica";
        }
        return null;
    }

    /**
     * Detecta si el MSE está estancado: no disminuye más de un 1% entre
     * épocas consecutivas durante 3 pares consecutivos.
     * Requiere al menos 4 épocas (3 pares consecutivos).
     */
    private static boolean detectarEstancamiento(double[] mse) {
        if (mse == null || mse.length < 4) {
            return false;
        }

        int paresEstancados = 0;
        for (int i = 0; i < mse.length - 1; i++) {
            double mejora = mse[i] > 0 ? (mse[i] - mse[i + 1]) / mse[i] : 0;
            if (mejora <= 0.01) {
                paresEstancados++;
                if (paresEstancados >= 3) {
                    return true;
                }
            } else {
                paresEstancados = 0;
            }
        }
        return false;
    }
}
