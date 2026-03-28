package es.jastxz.nn.benchmark;

import org.knowm.xchart.*;
import org.knowm.xchart.internal.chartpart.Chart;
import org.knowm.xchart.style.Styler;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Generador de gráficas de visualización para resultados de benchmarks de la SNN.
 *
 * <p>Produce gráficas PNG y un archivo HTML índice a partir de los resultados
 * de benchmark. Cada método de generación captura {@link IOException} internamente,
 * imprime un aviso en {@code System.err} y continúa con las demás gráficas.</p>
 *
 * @since 1.1
 */
public class GeneradorGraficas {

    private final Path directorioSalida;
    private final List<String> archivosGenerados = new ArrayList<>();
    private final List<Chart<?, ?>> graficasGeneradas = new ArrayList<>();

    /**
     * Crea un generador de gráficas con el directorio de salida indicado.
     * Si el directorio no existe, se crea automáticamente.
     *
     * @param directorioSalida ruta del directorio donde se guardarán las gráficas
     */
    public GeneradorGraficas(Path directorioSalida) {
        this.directorioSalida = directorioSalida;
        try {
            Files.createDirectories(directorioSalida);
        } catch (IOException e) {
            System.err.println("No se pudo crear el directorio de salida: " + e.getMessage());
        }
    }

    /**
     * Genera todas las gráficas y el índice HTML.
     *
     * @param resultados          lista de resultados de benchmark
     * @param fronterasCapacidad  mapa de topología a lista de transiciones de frontera
     */
    public void generarTodas(List<ResultadoBenchmark> resultados,
                             Map<String, List<String>> fronterasCapacidad) {
        if (resultados == null || resultados.isEmpty()) {
            System.out.println("No hay resultados para generar gráficas.");
            return;
        }
        generarGraficasPrecisionPorNivel(resultados);
        generarGraficasMSEPorConfiguracion(resultados);
        generarGraficasCostoEnergetico(resultados);
        generarHistogramaTasaDisparo(resultados);
        generarGraficaFronterasCapacidad(resultados, fronterasCapacidad);
        generarIndiceHTML();
    }

    /**
     * Genera gráficas de barras de precisión por topología, una por NivelComplejidad.
     */
    void generarGraficasPrecisionPorNivel(List<ResultadoBenchmark> resultados) {
        Map<NivelComplejidad, List<ResultadoBenchmark>> porNivel = agruparPorNivel(resultados);

        for (Map.Entry<NivelComplejidad, List<ResultadoBenchmark>> entry : porNivel.entrySet()) {
            NivelComplejidad nivel = entry.getKey();
            List<ResultadoBenchmark> grupo = entry.getValue();

            try {
                CategoryChart chart = new CategoryChartBuilder()
                        .width(1000).height(700)
                        .title("Precisión por Topología - " + nivel.getNombreProblema())
                        .xAxisTitle("Topología")
                        .yAxisTitle("Precisión")
                        .build();

                chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);

                List<String> categorias = new ArrayList<>();
                List<Number> valores = new ArrayList<>();
                for (ResultadoBenchmark r : grupo) {
                    categorias.add(Arrays.toString(r.configuracion().topologia()));
                    valores.add(r.precisionFinal());
                }

                chart.addSeries("Precisión", categorias, valores);

                String nombreArchivo = "precision_" + nivel.name().toLowerCase() + ".png";
                guardarGrafica(chart, nombreArchivo);
            } catch (Exception e) {
                System.err.println("Error generando gráfica de precisión para "
                        + nivel.getNombreProblema() + ": " + e.getMessage());
            }
        }
    }

    /**
     * Genera gráficas de líneas de evolución MSE por época, una por configuración.
     */
    void generarGraficasMSEPorConfiguracion(List<ResultadoBenchmark> resultados) {
        for (ResultadoBenchmark r : resultados) {
            double[] mse = r.errorMSEPorEpoca();
            if (mse.length == 0) {
                continue;
            }

            try {
                XYChart chart = new XYChartBuilder()
                        .width(1000).height(700)
                        .title("Evolución MSE - " + r.configuracion().etiqueta())
                        .xAxisTitle("Época")
                        .yAxisTitle("MSE")
                        .build();

                chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);

                double[] epocas = new double[mse.length];
                for (int i = 0; i < mse.length; i++) {
                    epocas[i] = i + 1;
                }

                chart.addSeries("MSE", epocas, mse);

                String nombreArchivo = "mse_" + sanitizar(r.configuracion().etiqueta()) + ".png";
                guardarGrafica(chart, nombreArchivo);
            } catch (Exception e) {
                System.err.println("Error generando gráfica MSE para "
                        + r.configuracion().etiqueta() + ": " + e.getMessage());
            }
        }
    }

    /**
     * Genera gráficas de barras de costo energético por topología, una por NivelComplejidad.
     */
    void generarGraficasCostoEnergetico(List<ResultadoBenchmark> resultados) {
        Map<NivelComplejidad, List<ResultadoBenchmark>> porNivel = agruparPorNivel(resultados);

        for (Map.Entry<NivelComplejidad, List<ResultadoBenchmark>> entry : porNivel.entrySet()) {
            NivelComplejidad nivel = entry.getKey();
            List<ResultadoBenchmark> grupo = entry.getValue();

            try {
                CategoryChart chart = new CategoryChartBuilder()
                        .width(1000).height(700)
                        .title("Costo Energético por Topología - " + nivel.getNombreProblema())
                        .xAxisTitle("Topología")
                        .yAxisTitle("Costo Energético")
                        .build();

                chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);

                List<String> categorias = new ArrayList<>();
                List<Number> valores = new ArrayList<>();
                for (ResultadoBenchmark r : grupo) {
                    categorias.add(Arrays.toString(r.configuracion().topologia()));
                    valores.add(r.costoEnergetico());
                }

                chart.addSeries("Costo Energético", categorias, valores);

                String nombreArchivo = "costo_" + nivel.name().toLowerCase() + ".png";
                guardarGrafica(chart, nombreArchivo);
            } catch (Exception e) {
                System.err.println("Error generando gráfica de costo energético para "
                        + nivel.getNombreProblema() + ": " + e.getMessage());
            }
        }
    }

    /**
     * Genera histograma de distribución de tasa de disparo agrupado por topología.
     */
    void generarHistogramaTasaDisparo(List<ResultadoBenchmark> resultados) {
        try {
            CategoryChart chart = new CategoryChartBuilder()
                    .width(1000).height(700)
                    .title("Distribución de Tasa de Disparo por Topología")
                    .xAxisTitle("Topología")
                    .yAxisTitle("Tasa de Disparo Promedio")
                    .build();

            chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);

            // Agrupar por topología y calcular valores
            Map<String, List<Double>> porTopologia = new LinkedHashMap<>();
            for (ResultadoBenchmark r : resultados) {
                String topoKey = Arrays.toString(r.configuracion().topologia());
                porTopologia.computeIfAbsent(topoKey, k -> new ArrayList<>())
                        .add(r.tasaDisparoPromedio());
            }

            List<String> categorias = new ArrayList<>();
            List<Number> valores = new ArrayList<>();
            for (Map.Entry<String, List<Double>> entry : porTopologia.entrySet()) {
                categorias.add(entry.getKey());
                double media = entry.getValue().stream()
                        .mapToDouble(Double::doubleValue).average().orElse(0.0);
                valores.add(media);
            }

            chart.addSeries("Tasa de Disparo", categorias, valores);

            String nombreArchivo = "histograma_tasa_disparo.png";
            guardarGrafica(chart, nombreArchivo);
        } catch (Exception e) {
            System.err.println("Error generando histograma de tasa de disparo: " + e.getMessage());
        }
    }

    /**
     * Genera gráfica de precisión vs nivel de espacio de estados por topología,
     * con marcadores en las fronteras de capacidad.
     */
    void generarGraficaFronterasCapacidad(List<ResultadoBenchmark> resultados,
                                          Map<String, List<String>> fronterasCapacidad) {
        try {
            XYChart chart = new XYChartBuilder()
                    .width(1000).height(700)
                    .title("Fronteras de Capacidad - Precisión vs Nivel de Complejidad")
                    .xAxisTitle("Nivel de Complejidad (índice)")
                    .yAxisTitle("Precisión")
                    .build();

            chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNE);

            NivelComplejidad[] niveles = NivelComplejidad.values();

            // Agrupar por topología
            Map<String, Map<NivelComplejidad, Double>> porTopologia = new LinkedHashMap<>();
            for (ResultadoBenchmark r : resultados) {
                String topoKey = Arrays.toString(r.configuracion().topologia());
                porTopologia.computeIfAbsent(topoKey, k -> new EnumMap<>(NivelComplejidad.class))
                        .put(r.configuracion().nivel(), r.precisionFinal());
            }

            // Serie principal por cada topología
            for (Map.Entry<String, Map<NivelComplejidad, Double>> entry : porTopologia.entrySet()) {
                String topoKey = entry.getKey();
                Map<NivelComplejidad, Double> porNivel = entry.getValue();

                List<Double> xData = new ArrayList<>();
                List<Double> yData = new ArrayList<>();
                for (int i = 0; i < niveles.length; i++) {
                    Double precision = porNivel.get(niveles[i]);
                    if (precision != null) {
                        xData.add((double) i);
                        yData.add(precision);
                    }
                }

                if (!xData.isEmpty()) {
                    chart.addSeries(topoKey, xData, yData);
                }
            }

            // Añadir marcadores de frontera como series adicionales
            if (fronterasCapacidad != null && !fronterasCapacidad.isEmpty()) {
                int marcadorIdx = 0;
                for (Map.Entry<String, List<String>> entry : fronterasCapacidad.entrySet()) {
                    String topoKey = entry.getKey();
                    Map<NivelComplejidad, Double> porNivel = porTopologia.get(topoKey);
                    if (porNivel == null) {
                        continue;
                    }

                    for (String transicion : entry.getValue()) {
                        // Encontrar el índice del nivel de la transición
                        for (int i = 0; i < niveles.length - 1; i++) {
                            String esperada = niveles[i].getNombreProblema()
                                    + " → " + niveles[i + 1].getNombreProblema();
                            if (transicion.equals(esperada)) {
                                Double precisionActual = porNivel.get(niveles[i]);
                                if (precisionActual != null) {
                                    chart.addSeries("Frontera " + marcadorIdx,
                                            List.of((double) i),
                                            List.of(precisionActual));
                                    marcadorIdx++;
                                }
                                break;
                            }
                        }
                    }
                }
            }

            String nombreArchivo = "fronteras_capacidad.png";
            guardarGrafica(chart, nombreArchivo);
        } catch (Exception e) {
            System.err.println("Error generando gráfica de fronteras de capacidad: "
                    + e.getMessage());
        }
    }

    /**
     * Genera un archivo HTML índice con enlaces a todas las gráficas por sección,
     * incluyendo descripciones explicativas, navegación y diseño responsive.
     */
    void generarIndiceHTML() {
        Path htmlPath = directorioSalida.resolve("index.html");
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(htmlPath))) {
            // Clasificar archivos por sección
            List<String> precision = new ArrayList<>();
            List<String> mse = new ArrayList<>();
            List<String> costo = new ArrayList<>();
            List<String> disparo = new ArrayList<>();
            List<String> fronteras = new ArrayList<>();

            for (String archivo : archivosGenerados) {
                if (archivo.startsWith("precision_")) {
                    precision.add(archivo);
                } else if (archivo.startsWith("mse_")) {
                    mse.add(archivo);
                } else if (archivo.startsWith("costo_")) {
                    costo.add(archivo);
                } else if (archivo.startsWith("histograma_")) {
                    disparo.add(archivo);
                } else if (archivo.startsWith("fronteras_")) {
                    fronteras.add(archivo);
                }
            }

            int totalGraficas = archivosGenerados.size();

            writer.println("<!DOCTYPE html>");
            writer.println("<html lang=\"es\">");
            writer.println("<head>");
            writer.println("  <meta charset=\"UTF-8\">");
            writer.println("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
            writer.println("  <title>Benchmark SNN — Resultados</title>");
            writer.println("  <style>");
            writer.println("    :root { --accent: #2563eb; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; --muted: #64748b; --border: #e2e8f0; }");
            writer.println("    * { box-sizing: border-box; margin: 0; padding: 0; }");
            writer.println("    body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }");
            writer.println("    header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 2.5rem 2rem; }");
            writer.println("    header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem; }");
            writer.println("    header p { color: #94a3b8; font-size: 0.95rem; }");
            writer.println("    .stats { display: flex; gap: 1.5rem; margin-top: 1.2rem; flex-wrap: wrap; }");
            writer.println("    .stat { background: rgba(255,255,255,0.1); border-radius: 8px; padding: 0.6rem 1rem; }");
            writer.println("    .stat-value { font-size: 1.3rem; font-weight: 700; }");
            writer.println("    .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }");
            writer.println("    nav { background: var(--card); border-bottom: 1px solid var(--border); padding: 0.8rem 2rem; position: sticky; top: 0; z-index: 10; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }");
            writer.println("    nav ul { list-style: none; display: flex; gap: 0.5rem; flex-wrap: wrap; }");
            writer.println("    nav a { text-decoration: none; color: var(--muted); padding: 0.4rem 0.8rem; border-radius: 6px; font-size: 0.85rem; transition: all 0.2s; }");
            writer.println("    nav a:hover, nav a.active { background: var(--accent); color: white; }");
            writer.println("    main { max-width: 1200px; margin: 0 auto; padding: 2rem; }");
            writer.println("    .seccion { margin-bottom: 3rem; }");
            writer.println("    .seccion-header { margin-bottom: 1rem; }");
            writer.println("    .seccion-header h2 { font-size: 1.4rem; color: var(--text); margin-bottom: 0.3rem; }");
            writer.println("    .seccion-header p { color: var(--muted); font-size: 0.9rem; }");
            writer.println("    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 1.5rem; }");
            writer.println("    .grid-full { grid-template-columns: 1fr; }");
            writer.println("    .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: box-shadow 0.2s; }");
            writer.println("    .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }");
            writer.println("    .card img { width: 100%; height: auto; display: block; cursor: pointer; }");
            writer.println("    .card-footer { padding: 0.6rem 1rem; border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }");
            writer.println("    .card-footer span { font-size: 0.8rem; color: var(--muted); }");
            writer.println("    .card-footer a { font-size: 0.8rem; color: var(--accent); text-decoration: none; }");
            writer.println("    .card-footer a:hover { text-decoration: underline; }");
            writer.println("    .collapsible { cursor: pointer; user-select: none; }");
            writer.println("    .collapsible::before { content: '▸ '; transition: transform 0.2s; }");
            writer.println("    .collapsible.open::before { content: '▾ '; }");
            writer.println("    .collapse-content { display: none; }");
            writer.println("    .collapse-content.show { display: grid; }");
            writer.println("    footer { text-align: center; padding: 2rem; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); }");
            writer.println("    @media (max-width: 600px) { .grid { grid-template-columns: 1fr; } header h1 { font-size: 1.3rem; } }");
            writer.println("  </style>");
            writer.println("</head>");
            writer.println("<body>");

            // Header
            writer.println("  <header>");
            writer.println("    <h1>&#x1F9E0; Benchmark Suite — Red Neuronal Spiking</h1>");
            writer.println("    <p>Evaluación paramétrica de rendimiento variando topologías y niveles de complejidad de problemas.</p>");
            writer.println("    <div class=\"stats\">");
            writer.println("      <div class=\"stat\"><div class=\"stat-value\">" + totalGraficas + "</div><div class=\"stat-label\">Gráficas generadas</div></div>");
            writer.println("      <div class=\"stat\"><div class=\"stat-value\">" + precision.size() + "</div><div class=\"stat-label\">Niveles evaluados</div></div>");
            writer.println("      <div class=\"stat\"><div class=\"stat-value\">" + mse.size() + "</div><div class=\"stat-label\">Curvas MSE</div></div>");
            writer.println("    </div>");
            writer.println("  </header>");

            // Navigation
            writer.println("  <nav>");
            writer.println("    <ul>");
            if (!precision.isEmpty()) writer.println("      <li><a href=\"#precision\">Precisión</a></li>");
            if (!mse.isEmpty()) writer.println("      <li><a href=\"#mse\">Evolución MSE</a></li>");
            if (!costo.isEmpty()) writer.println("      <li><a href=\"#costo\">Costo Energético</a></li>");
            if (!disparo.isEmpty()) writer.println("      <li><a href=\"#disparo\">Tasa de Disparo</a></li>");
            if (!fronteras.isEmpty()) writer.println("      <li><a href=\"#fronteras\">Fronteras de Capacidad</a></li>");
            writer.println("    </ul>");
            writer.println("  </nav>");

            writer.println("  <main>");

            // Sección: Precisión
            escribirSeccionHTML(writer, "precision", "Precisión por Nivel de Complejidad",
                    "Compara la precisión final alcanzada por cada topología de red dentro de un mismo nivel de complejidad. "
                            + "Cada barra representa una topología distinta (número de capas ocultas × factor de neuronas). "
                            + "Valores más altos indican mejor capacidad de aprendizaje para ese problema.",
                    precision, false);

            // Sección: MSE (colapsable por la cantidad de gráficas)
            escribirSeccionHTML(writer, "mse", "Evolución del Error (MSE) por Configuración",
                    "Muestra cómo evoluciona el error cuadrático medio (MSE) a lo largo de las épocas de entrenamiento "
                            + "para cada combinación de topología y problema. Una curva descendente indica convergencia; "
                            + "una curva plana sugiere estancamiento del aprendizaje.",
                    mse, mse.size() > 8);

            // Sección: Costo Energético
            escribirSeccionHTML(writer, "costo", "Costo Energético por Nivel de Complejidad",
                    "Compara el costo energético (proporcional al total de spikes generados) de cada topología. "
                            + "Topologías más grandes tienden a generar más spikes. Un costo desproporcionadamente alto "
                            + "respecto a la precisión obtenida indica ineficiencia energética.",
                    costo, false);

            // Sección: Tasa de Disparo
            escribirSeccionHTML(writer, "disparo", "Distribución de Tasa de Disparo",
                    "Muestra la tasa de disparo promedio agrupada por topología. Tasas muy bajas pueden indicar "
                            + "neuronas inactivas (red infrautilizada), mientras que tasas muy altas pueden señalar "
                            + "saturación de la red.",
                    disparo, false);

            // Sección: Fronteras de Capacidad
            escribirSeccionHTML(writer, "fronteras", "Fronteras de Capacidad",
                    "Visualiza la precisión de cada topología a medida que aumenta la complejidad del problema. "
                            + "Los puntos donde la precisión cae abruptamente (de >70% a <50%) entre niveles consecutivos "
                            + "representan las fronteras de capacidad de esa topología: el límite de complejidad que puede manejar.",
                    fronteras, false);

            writer.println("  </main>");

            // Footer
            writer.println("  <footer>");
            writer.println("    Generado automáticamente por la Suite de Benchmarks SNN &mdash; " + java.time.LocalDate.now());
            writer.println("  </footer>");

            // Script para secciones colapsables
            writer.println("  <script>");
            writer.println("    document.querySelectorAll('.collapsible').forEach(el => {");
            writer.println("      el.addEventListener('click', () => {");
            writer.println("        el.classList.toggle('open');");
            writer.println("        const content = el.nextElementSibling;");
            writer.println("        content.classList.toggle('show');");
            writer.println("      });");
            writer.println("    });");
            writer.println("  </script>");

            writer.println("</body>");
            writer.println("</html>");
        } catch (IOException e) {
            System.err.println("Error generando índice HTML: " + e.getMessage());
        }
    }

    // --- Métodos auxiliares ---

    /**
     * Escribe una sección del informe HTML con título, descripción, y tarjetas de gráficas.
     *
     * @param writer      escritor HTML
     * @param id          identificador para ancla de navegación
     * @param titulo      título visible de la sección
     * @param descripcion texto explicativo de la sección
     * @param archivos    lista de archivos PNG de la sección
     * @param colapsable  si true, el grid de imágenes se muestra colapsado inicialmente
     */
    private void escribirSeccionHTML(PrintWriter writer, String id, String titulo,
                                     String descripcion, List<String> archivos, boolean colapsable) {
        if (archivos.isEmpty()) {
            return;
        }
        writer.println("    <section class=\"seccion\" id=\"" + id + "\">");
        writer.println("      <div class=\"seccion-header\">");
        if (colapsable) {
            writer.println("        <h2 class=\"collapsible\">" + titulo + " (" + archivos.size() + " gráficas)</h2>");
        } else {
            writer.println("        <h2>" + titulo + "</h2>");
        }
        writer.println("        <p>" + descripcion + "</p>");
        writer.println("      </div>");

        String gridClass = archivos.size() == 1 ? "grid grid-full" : "grid";
        String collapseClass = colapsable ? "collapse-content" : "collapse-content show";
        writer.println("      <div class=\"" + gridClass + " " + collapseClass + "\">");
        for (String archivo : archivos) {
            String etiquetaLegible = extraerEtiquetaLegible(archivo);
            writer.println("        <div class=\"card\">");
            writer.println("          <a href=\"" + archivo + "\" target=\"_blank\">");
            writer.println("            <img src=\"" + archivo + "\" alt=\"" + etiquetaLegible + "\" loading=\"lazy\">");
            writer.println("          </a>");
            writer.println("          <div class=\"card-footer\">");
            writer.println("            <span>" + etiquetaLegible + "</span>");
            writer.println("            <a href=\"" + archivo + "\" download>Descargar</a>");
            writer.println("          </div>");
            writer.println("        </div>");
        }
        writer.println("      </div>");
        writer.println("    </section>");
    }

    /**
     * Extrae una etiqueta legible a partir del nombre de archivo PNG.
     * Ejemplo: "precision_trivial.png" → "Precisión — Trivial"
     *          "mse_puertas_l_gicas_2_4_1_.png" → "MSE — Puertas Lógicas [2, 4, 1]"
     */
    private String extraerEtiquetaLegible(String archivo) {
        String sinExt = archivo.replace(".png", "");
        if (sinExt.startsWith("precision_")) {
            return "Precisión — " + capitalizarNivel(sinExt.substring("precision_".length()));
        } else if (sinExt.startsWith("costo_")) {
            return "Costo Energético — " + capitalizarNivel(sinExt.substring("costo_".length()));
        } else if (sinExt.startsWith("mse_")) {
            return "MSE — " + formatearEtiquetaMSE(sinExt.substring("mse_".length()));
        } else if (sinExt.startsWith("histograma_")) {
            return "Histograma — Tasa de Disparo";
        } else if (sinExt.startsWith("fronteras_")) {
            return "Fronteras de Capacidad";
        }
        return sinExt;
    }

    private String capitalizarNivel(String nivel) {
        return switch (nivel.toLowerCase()) {
            case "trivial" -> "Puertas Lógicas (Trivial)";
            case "bajo" -> "3 en Raya (Bajo)";
            case "medio" -> "Gatos (Medio)";
            case "alto" -> "Damas (Alto)";
            default -> nivel.substring(0, 1).toUpperCase() + nivel.substring(1);
        };
    }

    private String formatearEtiquetaMSE(String etiqueta) {
        // Intentar extraer el nombre del problema y la topología
        // Formato: "puertas_l_gicas_2_4_1_" o "3_en_raya_10_20_9_"
        for (NivelComplejidad nivel : NivelComplejidad.values()) {
            String nombreSanitizado = sanitizar(nivel.getNombreProblema());
            if (etiqueta.startsWith(nombreSanitizado + "_")) {
                String topoPart = etiqueta.substring(nombreSanitizado.length() + 1);
                // Reconstruir topología: quitar trailing underscore, split por _
                topoPart = topoPart.replaceAll("_$", "");
                String topologia = "[" + topoPart.replace("_", ", ") + "]";
                return nivel.getNombreProblema() + " " + topologia;
            }
        }
        return etiqueta.replace("_", " ");
    }

    private void guardarGrafica(CategoryChart chart, String nombreArchivo) {
        try {
            String rutaSinExtension = directorioSalida.resolve(
                    nombreArchivo.replace(".png", "")).toString();
            BitmapEncoder.saveBitmap(chart, rutaSinExtension, BitmapEncoder.BitmapFormat.PNG);
            archivosGenerados.add(nombreArchivo);
            graficasGeneradas.add(chart);
        } catch (IOException e) {
            System.err.println("Error guardando gráfica " + nombreArchivo + ": " + e.getMessage());
        }
    }

    private void guardarGrafica(XYChart chart, String nombreArchivo) {
        try {
            String rutaSinExtension = directorioSalida.resolve(
                    nombreArchivo.replace(".png", "")).toString();
            BitmapEncoder.saveBitmap(chart, rutaSinExtension, BitmapEncoder.BitmapFormat.PNG);
            archivosGenerados.add(nombreArchivo);
            graficasGeneradas.add(chart);
        } catch (IOException e) {
            System.err.println("Error guardando gráfica " + nombreArchivo + ": " + e.getMessage());
        }
    }

    private Map<NivelComplejidad, List<ResultadoBenchmark>> agruparPorNivel(
            List<ResultadoBenchmark> resultados) {
        return resultados.stream()
                .collect(Collectors.groupingBy(
                        r -> r.configuracion().nivel(),
                        LinkedHashMap::new,
                        Collectors.toList()));
    }

    private String sanitizar(String texto) {
        return texto.replaceAll("[^a-zA-Z0-9_-]", "_")
                .replaceAll("_+", "_")
                .toLowerCase();
    }

    /**
     * Retorna la lista de nombres de archivos generados.
     *
     * @return lista inmutable de nombres de archivos generados
     */
    List<String> getArchivosGenerados() {
        return Collections.unmodifiableList(archivosGenerados);
    }

    /**
     * Retorna el directorio de salida configurado.
     *
     * @return ruta del directorio de salida
     */
    Path getDirectorioSalida() {
        return directorioSalida;
    }

    /**
     * Retorna la lista de objetos de gráficas generados (para inspección en tests).
     *
     * @return lista inmutable de objetos Chart generados
     */
    List<Chart<?, ?>> getGraficasGeneradas() {
        return Collections.unmodifiableList(graficasGeneradas);
    }
}
