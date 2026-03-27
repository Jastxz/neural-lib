package es.jastxz.nn.benchmark;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

/**
 * Resultado agregado de múltiples repeticiones de un benchmark.
 *
 * <p>Calcula media y desviación estándar de las métricas principales
 * (precisión, tiempo de entrenamiento y costo energético) a partir de
 * una lista de {@link ResultadoBenchmark} con la misma configuración.
 * También determina la clasificación mayoritaria y el ratio de eficiencia
 * (precisión / costo energético).</p>
 *
 * @param configuracion           configuración de benchmark compartida
 * @param mediaPrecision          media de precisión entre repeticiones
 * @param desvPrecision           desviación estándar de precisión
 * @param mediaTiempo             media de tiempo de entrenamiento (ms)
 * @param desvTiempo              desviación estándar de tiempo
 * @param mediaCostoEnergetico    media de costo energético
 * @param desvCostoEnergetico     desviación estándar de costo energético
 * @param ratioEficiencia         precisión / costo energético
 * @param clasificacionMayoritaria clasificación más frecuente, o {@code null}
 * @since 1.1
 */
public record ResultadoAgregado(
    ConfiguracionBenchmark configuracion,
    double mediaPrecision,
    double desvPrecision,
    double mediaTiempo,
    double desvTiempo,
    double mediaCostoEnergetico,
    double desvCostoEnergetico,
    double ratioEficiencia,
    String clasificacionMayoritaria
) {

    /**
     * Agrega una lista de resultados de benchmark calculando media y
     * desviación estándar de precisión, tiempo y costo energético.
     *
     * <p>Para N=1, la desviación estándar es 0. El ratio de eficiencia
     * se calcula como {@code mediaPrecision / mediaCostoEnergetico};
     * si el costo medio es 0, se usa {@link Double#POSITIVE_INFINITY}.</p>
     *
     * @param resultados lista no vacía de resultados con la misma configuración
     * @return resultado agregado con estadísticas calculadas
     * @throws IllegalArgumentException si la lista es nula o vacía
     */
    public static ResultadoAgregado agregar(List<ResultadoBenchmark> resultados) {
        if (resultados == null || resultados.isEmpty()) {
            throw new IllegalArgumentException(
                "La lista de resultados no puede ser nula ni vacía");
        }

        ConfiguracionBenchmark config = resultados.get(0).configuracion();

        double mediaPrecision = media(resultados, ResultadoBenchmark::precisionFinal);
        double desvPrecision = desviacion(resultados, ResultadoBenchmark::precisionFinal);

        double mediaTiempo = media(resultados, r -> r.tiempoEntrenamientoMs());
        double desvTiempo = desviacion(resultados, r -> r.tiempoEntrenamientoMs());

        double mediaCosto = media(resultados, ResultadoBenchmark::costoEnergetico);
        double desvCosto = desviacion(resultados, ResultadoBenchmark::costoEnergetico);

        double ratio = mediaCosto == 0.0
            ? Double.POSITIVE_INFINITY
            : mediaPrecision / mediaCosto;

        String clasificacion = clasificacionMayoritaria(resultados);

        return new ResultadoAgregado(
            config, mediaPrecision, desvPrecision,
            mediaTiempo, desvTiempo,
            mediaCosto, desvCosto,
            ratio, clasificacion);
    }

    private static double media(List<ResultadoBenchmark> resultados,
                                ToDoubleFunction<ResultadoBenchmark> extractor) {
        return resultados.stream()
            .mapToDouble(extractor)
            .average()
            .orElse(0.0);
    }

    private static double desviacion(List<ResultadoBenchmark> resultados,
                                     ToDoubleFunction<ResultadoBenchmark> extractor) {
        int n = resultados.size();
        if (n <= 1) {
            return 0.0;
        }
        double media = media(resultados, extractor);
        double sumaCuadrados = resultados.stream()
            .mapToDouble(extractor)
            .map(v -> (v - media) * (v - media))
            .sum();
        return Math.sqrt(sumaCuadrados / (n - 1));
    }

    private static String clasificacionMayoritaria(List<ResultadoBenchmark> resultados) {
        Map<String, Long> frecuencias = resultados.stream()
            .map(ResultadoBenchmark::clasificacion)
            .filter(Objects::nonNull)
            .collect(Collectors.groupingBy(c -> c, Collectors.counting()));

        if (frecuencias.isEmpty()) {
            return null;
        }

        return frecuencias.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse(null);
    }
}
