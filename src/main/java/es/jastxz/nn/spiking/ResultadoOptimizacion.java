package es.jastxz.nn.spiking;

import es.jastxz.nn.genetico.InformeEvolucion;

/**
 * Resultado inmutable del proceso de optimización de una SNN.
 *
 * <p>Contiene la red entrenada lista para usar, la configuración óptima
 * encontrada, las métricas de rendimiento y el informe completo de la
 * evolución genética.</p>
 *
 * <h3>Uso típico:</h3>
 * <pre>{@code
 * ResultadoOptimizacion resultado = OptimizadorSNN.optimizar(inputs, targets);
 *
 * // Usar la red directamente
 * double[] salida = resultado.red().procesar(entrada, 100);
 *
 * // Consultar métricas
 * System.out.println("Precisión: " + resultado.precision());
 *
 * // Ver informe de evolución
 * resultado.informeEvolucion().imprimir();
 * }</pre>
 *
 * @param red                red neuronal entrenada con la mejor configuración
 * @param configuracionRed   configuración de red óptima encontrada
 * @param precision          precisión final sobre los datos de entrenamiento (0.0–1.0)
 * @param fitness            valor de fitness del mejor individuo
 * @param informeEvolucion   informe completo del proceso evolutivo
 * @since 1.2
 */
public record ResultadoOptimizacion(
    RedNeuralSpiking red,
    ConfiguracionRed configuracionRed,
    double precision,
    double fitness,
    InformeEvolucion informeEvolucion
) {

    /**
     * Imprime un resumen del resultado de la optimización.
     */
    public void imprimir() {
        System.out.println("=== Resultado de Optimización SNN ===");
        System.out.printf("  Precisión:    %.1f%%%n", precision * 100);
        System.out.printf("  Fitness:      %.6f%n", fitness);
        System.out.printf("  Topología:    %s%n",
                java.util.Arrays.toString(configuracionRed.getTopologia()));
        System.out.printf("  Generaciones: %d (parada: %s)%n",
                informeEvolucion.totalGeneraciones(),
                informeEvolucion.motivoParada());
    }
}
