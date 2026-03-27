package es.jastxz.nn.benchmark;

import java.io.PrintStream;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Generador de informes comparativos de benchmarks de la SNN.
 *
 * <p>Produce un informe estructurado por consola con las siguientes secciones:</p>
 * <ol>
 *   <li>Tabla resumen agrupada por nivel de complejidad</li>
 *   <li>Hallazgos de límites detectados</li>
 *   <li>Topología óptima por nivel (mayor ratio Precisión/CostoEnergético)</li>
 *   <li>Análisis de eficiencia por configuración</li>
 *   <li>Fronteras de capacidad detectadas</li>
 * </ol>
 *
 * @since 1.1
 */
public class GeneradorInforme {

    private GeneradorInforme() {
        // Clase utilitaria, no instanciable
    }

    /**
     * Genera el informe completo por {@code System.out}.
     *
     * @param resultados lista de resultados de benchmark
     */
    public static void generar(List<ResultadoBenchmark> resultados) {
        generar(resultados, System.out);
    }

    /**
     * Genera el informe completo por el {@code PrintStream} proporcionado.
     *
     * @param resultados lista de resultados de benchmark
     * @param out        flujo de salida para el informe
     */
    public static void generar(List<ResultadoBenchmark> resultados, PrintStream out) {
        if (resultados == null || resultados.isEmpty()) {
            out.println("No hay resultados de benchmark disponibles.");
            return;
        }

        out.println("=== INFORME DE BENCHMARKS SNN ===");
        out.println();

        // 1. Tabla resumen agrupada por NivelComplejidad
        Map<NivelComplejidad, List<ResultadoBenchmark>> porNivel = resultados.stream()
                .collect(Collectors.groupingBy(
                        r -> r.configuracion().nivel(),
                        LinkedHashMap::new,
                        Collectors.toList()));

        for (Map.Entry<NivelComplejidad, List<ResultadoBenchmark>> entry : porNivel.entrySet()) {
            NivelComplejidad nivel = entry.getKey();
            List<ResultadoBenchmark> grupo = entry.getValue();

            out.println("--- " + nivel.getNombreProblema()
                    + " (Espacio de estados: " + nivel.getEspacioEstados() + ") ---");
            out.println();
            imprimirTablaResumen(grupo, out);
            out.println();
        }

        // 2. Hallazgos de límites
        imprimirHallazgosLimites(resultados, out);

        // 3. Topología óptima por nivel
        imprimirTopologiaOptima(porNivel, out);

        // 4. Análisis de eficiencia
        imprimirAnalisisEficiencia(resultados, out);

        // 5. Fronteras de capacidad
        imprimirFronterasCapacidad(resultados, out);
    }

    private static void imprimirTablaResumen(List<ResultadoBenchmark> grupo, PrintStream out) {
        out.printf("%-30s | %9s | %9s | %10s | %8s | %12s | %16s | %11s | %16s | %s%n",
                "Topología", "Precisión", "MSE Final", "Tiempo(ms)", "Spikes",
                "Tasa Disparo", "Costo Energético", "Dispersión",
                "Neuronas Activas", "Clasificación");
        out.println("-".repeat(170));

        for (ResultadoBenchmark r : grupo) {
            String topologia = Arrays.toString(r.configuracion().topologia());
            double mseFinal = r.errorMSEPorEpoca().length > 0
                    ? r.errorMSEPorEpoca()[r.errorMSEPorEpoca().length - 1]
                    : 0.0;
            String clasificacion = r.clasificacion() != null ? r.clasificacion() : "-";

            out.printf("%-30s | %9.4f | %9.4f | %10d | %8d | %12.6f | %16.6f | %11.6f | %8d/%-7d | %s%n",
                    topologia,
                    r.precisionFinal(),
                    mseFinal,
                    r.tiempoEntrenamientoMs(),
                    r.totalSpikes(),
                    r.tasaDisparoPromedio(),
                    r.costoEnergetico(),
                    r.dispersionActividad(),
                    r.neuronasActivas(),
                    r.neuronasTotal(),
                    clasificacion);
        }
    }

    private static void imprimirHallazgosLimites(List<ResultadoBenchmark> resultados, PrintStream out) {
        List<ResultadoBenchmark> conClasificacion = resultados.stream()
                .filter(r -> r.clasificacion() != null)
                .toList();

        out.println("=== HALLAZGOS DE LÍMITES ===");
        out.println();

        if (conClasificacion.isEmpty()) {
            out.println("No se detectaron límites en ninguna configuración.");
        } else {
            for (ResultadoBenchmark r : conClasificacion) {
                out.printf("  [%s] %s - Precisión: %.4f, Costo: %.6f%n",
                        r.clasificacion(),
                        r.configuracion().etiqueta(),
                        r.precisionFinal(),
                        r.costoEnergetico());
            }
        }
        out.println();
    }

    private static void imprimirTopologiaOptima(
            Map<NivelComplejidad, List<ResultadoBenchmark>> porNivel, PrintStream out) {
        out.println("=== TOPOLOGÍA ÓPTIMA POR NIVEL ===");
        out.println();

        for (Map.Entry<NivelComplejidad, List<ResultadoBenchmark>> entry : porNivel.entrySet()) {
            NivelComplejidad nivel = entry.getKey();
            List<ResultadoBenchmark> grupo = entry.getValue();

            ResultadoBenchmark optimo = grupo.stream()
                    .max(Comparator.comparingDouble(r -> calcularRatioEficiencia(r)))
                    .orElse(null);

            if (optimo != null) {
                double ratio = calcularRatioEficiencia(optimo);
                out.printf("  %s: %s (Precisión: %.4f, Costo: %.6f, Ratio: %.4f)%n",
                        nivel.getNombreProblema(),
                        Arrays.toString(optimo.configuracion().topologia()),
                        optimo.precisionFinal(),
                        optimo.costoEnergetico(),
                        ratio);
            }
        }
        out.println();
    }

    private static void imprimirAnalisisEficiencia(List<ResultadoBenchmark> resultados, PrintStream out) {
        out.println("=== ANÁLISIS DE EFICIENCIA ===");
        out.println();

        for (ResultadoBenchmark r : resultados) {
            double ratio = calcularRatioEficiencia(r);
            out.printf("  %s - Ratio Precisión/Costo: %.4f%n",
                    r.configuracion().etiqueta(),
                    ratio);
        }
        out.println();
    }

    private static void imprimirFronterasCapacidad(List<ResultadoBenchmark> resultados, PrintStream out) {
        out.println("=== FRONTERAS DE CAPACIDAD ===");
        out.println();

        Map<String, List<String>> fronteras = detectarFronterasCapacidad(resultados);

        if (fronteras.isEmpty()) {
            out.println("No se detectaron fronteras de capacidad.");
        } else {
            for (Map.Entry<String, List<String>> entry : fronteras.entrySet()) {
                out.printf("  Topología %s:%n", entry.getKey());
                for (String transicion : entry.getValue()) {
                    out.printf("    - Frontera: %s%n", transicion);
                }
            }
        }
        out.println();
    }

    /**
     * Detecta fronteras de capacidad: transiciones donde una topología mantiene
     * precisión &gt; 0.7 en un nivel de complejidad pero cae a &lt; 0.5 en el siguiente.
     *
     * @param resultados lista de resultados de benchmark
     * @return mapa de topología (como string) a lista de descripciones de transición
     */
    public static Map<String, List<String>> detectarFronterasCapacidad(
            List<ResultadoBenchmark> resultados) {
        if (resultados == null || resultados.isEmpty()) {
            return Collections.emptyMap();
        }

        NivelComplejidad[] niveles = NivelComplejidad.values();

        // Agrupar por topología string
        Map<String, Map<NivelComplejidad, ResultadoBenchmark>> porTopologia = new LinkedHashMap<>();
        for (ResultadoBenchmark r : resultados) {
            String topoKey = Arrays.toString(r.configuracion().topologia());
            porTopologia
                    .computeIfAbsent(topoKey, k -> new EnumMap<>(NivelComplejidad.class))
                    .put(r.configuracion().nivel(), r);
        }

        Map<String, List<String>> fronteras = new LinkedHashMap<>();

        for (Map.Entry<String, Map<NivelComplejidad, ResultadoBenchmark>> entry : porTopologia.entrySet()) {
            String topoKey = entry.getKey();
            Map<NivelComplejidad, ResultadoBenchmark> porNivel = entry.getValue();

            for (int i = 0; i < niveles.length - 1; i++) {
                ResultadoBenchmark actual = porNivel.get(niveles[i]);
                ResultadoBenchmark siguiente = porNivel.get(niveles[i + 1]);

                if (actual != null && siguiente != null
                        && actual.precisionFinal() > 0.7
                        && siguiente.precisionFinal() < 0.5) {
                    fronteras
                            .computeIfAbsent(topoKey, k -> new ArrayList<>())
                            .add(niveles[i].getNombreProblema()
                                    + " → " + niveles[i + 1].getNombreProblema());
                }
            }
        }

        return fronteras;
    }

    /**
     * Calcula la ratio de eficiencia Precisión/CostoEnergético.
     * Si el costo es 0, retorna {@code Double.POSITIVE_INFINITY}.
     */
    private static double calcularRatioEficiencia(ResultadoBenchmark r) {
        if (r.costoEnergetico() == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        return r.precisionFinal() / r.costoEnergetico();
    }
}
