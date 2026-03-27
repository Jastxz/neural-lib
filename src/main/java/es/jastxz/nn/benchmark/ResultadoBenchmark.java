package es.jastxz.nn.benchmark;

/**
 * Resultado inmutable de una ejecución individual de benchmark de la SNN.
 *
 * <p>Almacena todas las métricas recopiladas durante el entrenamiento y
 * evaluación de la red, asociadas a la configuración que las produjo.
 * El campo {@code clasificacion} puede ser {@code null} si no se detectó
 * ningún límite, o contener una etiqueta como {@code "limite_no_superado"},
 * {@code "convergencia_estancada"}, {@code "red_infrautilizada"}, etc.</p>
 *
 * @param configuracion       configuración de benchmark usada
 * @param precisionFinal      porcentaje de aciertos (0.0–1.0)
 * @param errorMSEPorEpoca    evolución del MSE por época
 * @param tiempoEntrenamientoMs tiempo total de entrenamiento en milisegundos
 * @param totalSpikes         número total de spikes generados
 * @param tasaDisparoPromedio tasa de disparo promedio global
 * @param costoEnergetico     costo energético (proporcional a spikes)
 * @param dispersionActividad desviación estándar de spikes entre neuronas
 * @param neuronasActivas     neuronas que dispararon al menos un spike
 * @param neuronasTotal       total de neuronas en la red
 * @param clasificacion       clasificación de límite detectado, o {@code null}
 * @since 1.1
 */
public record ResultadoBenchmark(
    ConfiguracionBenchmark configuracion,
    double precisionFinal,
    double[] errorMSEPorEpoca,
    long tiempoEntrenamientoMs,
    long totalSpikes,
    double tasaDisparoPromedio,
    double costoEnergetico,
    double dispersionActividad,
    int neuronasActivas,
    int neuronasTotal,
    String clasificacion
) {

    /**
     * Constructor compacto que realiza una copia defensiva del array
     * {@code errorMSEPorEpoca} para preservar la inmutabilidad del record.
     */
    public ResultadoBenchmark {
        errorMSEPorEpoca = errorMSEPorEpoca == null
            ? new double[0]
            : errorMSEPorEpoca.clone();
    }
}
