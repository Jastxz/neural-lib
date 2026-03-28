package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import org.knowm.xchart.internal.chartpart.Chart;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para la completitud de gráficas de barras
 * por nivel de complejidad generadas por {@link GeneradorGraficas}.
 *
 * <p><b>Validates: Requisitos 8.1, 8.3</b></p>
 *
 * @since 1.1
 */
class GeneradorGraficasPropertyTest {

    // Feature: snn-benchmark-suite, Property 12: Completitud de gráficas de barras por nivel de complejidad

    /** Configuraciones fijas por nivel, compatibles con las dimensiones de cada NivelComplejidad. */
    private static final Map<NivelComplejidad, ConfiguracionBenchmark> CONFIGS_POR_NIVEL;

    static {
        CONFIGS_POR_NIVEL = new EnumMap<>(NivelComplejidad.class);
        CONFIGS_POR_NIVEL.put(NivelComplejidad.TRIVIAL,
                new ConfiguracionBenchmark(NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 50, 1, 42L));
        CONFIGS_POR_NIVEL.put(NivelComplejidad.BAJO,
                new ConfiguracionBenchmark(NivelComplejidad.BAJO, new int[]{10, 20, 9}, 10, 50, 1, 42L));
        CONFIGS_POR_NIVEL.put(NivelComplejidad.MEDIO,
                new ConfiguracionBenchmark(NivelComplejidad.MEDIO, new int[]{25, 50, 25}, 10, 50, 1, 42L));
        CONFIGS_POR_NIVEL.put(NivelComplejidad.ALTO,
                new ConfiguracionBenchmark(NivelComplejidad.ALTO, new int[]{32, 64, 32}, 10, 50, 1, 42L));
    }

    /** Topologías adicionales por nivel para generar múltiples barras por gráfica. */
    private static final Map<NivelComplejidad, int[][]> TOPOLOGIAS_EXTRA;

    static {
        TOPOLOGIAS_EXTRA = new EnumMap<>(NivelComplejidad.class);
        TOPOLOGIAS_EXTRA.put(NivelComplejidad.TRIVIAL, new int[][]{
                {2, 2, 1}, {2, 4, 1}, {2, 8, 1}
        });
        TOPOLOGIAS_EXTRA.put(NivelComplejidad.BAJO, new int[][]{
                {10, 10, 9}, {10, 20, 9}, {10, 40, 9}
        });
        TOPOLOGIAS_EXTRA.put(NivelComplejidad.MEDIO, new int[][]{
                {25, 25, 25}, {25, 50, 25}, {25, 100, 25}
        });
        TOPOLOGIAS_EXTRA.put(NivelComplejidad.ALTO, new int[][]{
                {32, 32, 32}, {32, 64, 32}, {32, 128, 32}
        });
    }

    // --- Propiedad 12a: Exactamente un PNG de precisión por NivelComplejidad presente ---

    /**
     * Para cualquier lista de ResultadoBenchmark con al menos un resultado por nivel
     * de complejidad, generarGraficasPrecisionPorNivel debe producir exactamente un
     * archivo PNG por cada nivel presente en los resultados.
     */
    @Property(tries = 20)
    void unPNGDePrecisionPorCadaNivelPresente(
            @ForAll("listaResultadosConNivelesVariados") List<ResultadoBenchmark> resultados) throws IOException {

        Path tempDir = Files.createTempDirectory("graficas_precision_test_");
        try {
            GeneradorGraficas generador = new GeneradorGraficas(tempDir);
            generador.generarGraficasPrecisionPorNivel(resultados);

            Set<NivelComplejidad> nivelesPresentes = resultados.stream()
                    .map(r -> r.configuracion().nivel())
                    .collect(Collectors.toSet());

            List<String> archivos = generador.getArchivosGenerados();

            // Exactamente un PNG por nivel presente
            assertEquals(nivelesPresentes.size(), archivos.size(),
                    "Debe haber exactamente un PNG de precisión por nivel presente. "
                            + "Niveles: " + nivelesPresentes + ", Archivos: " + archivos);

            // Cada archivo existe en disco
            for (String archivo : archivos) {
                Path archivoPath = tempDir.resolve(archivo);
                assertTrue(Files.exists(archivoPath),
                        "El archivo " + archivo + " debe existir en disco");
            }

            // Cada nivel presente tiene su archivo correspondiente
            for (NivelComplejidad nivel : nivelesPresentes) {
                String esperado = "precision_" + nivel.name().toLowerCase() + ".png";
                assertTrue(archivos.contains(esperado),
                        "Debe existir el archivo " + esperado + " para el nivel " + nivel);
            }
        } finally {
            limpiarDirectorio(tempDir);
        }
    }

    // --- Propiedad 12b: Exactamente un PNG de costo energético por NivelComplejidad presente ---

    /**
     * Para cualquier lista de ResultadoBenchmark con al menos un resultado por nivel
     * de complejidad, generarGraficasCostoEnergetico debe producir exactamente un
     * archivo PNG por cada nivel presente en los resultados.
     */
    @Property(tries = 20)
    void unPNGDeCostoPorCadaNivelPresente(
            @ForAll("listaResultadosConNivelesVariados") List<ResultadoBenchmark> resultados) throws IOException {

        Path tempDir = Files.createTempDirectory("graficas_costo_test_");
        try {
            GeneradorGraficas generador = new GeneradorGraficas(tempDir);
            generador.generarGraficasCostoEnergetico(resultados);

            Set<NivelComplejidad> nivelesPresentes = resultados.stream()
                    .map(r -> r.configuracion().nivel())
                    .collect(Collectors.toSet());

            List<String> archivos = generador.getArchivosGenerados();

            // Exactamente un PNG por nivel presente
            assertEquals(nivelesPresentes.size(), archivos.size(),
                    "Debe haber exactamente un PNG de costo por nivel presente. "
                            + "Niveles: " + nivelesPresentes + ", Archivos: " + archivos);

            // Cada archivo existe en disco
            for (String archivo : archivos) {
                Path archivoPath = tempDir.resolve(archivo);
                assertTrue(Files.exists(archivoPath),
                        "El archivo " + archivo + " debe existir en disco");
            }

            // Cada nivel presente tiene su archivo correspondiente
            for (NivelComplejidad nivel : nivelesPresentes) {
                String esperado = "costo_" + nivel.name().toLowerCase() + ".png";
                assertTrue(archivos.contains(esperado),
                        "Debe existir el archivo " + esperado + " para el nivel " + nivel);
            }
        } finally {
            limpiarDirectorio(tempDir);
        }
    }

    // --- Propiedad 13: Correspondencia de datos en gráficas de evolución MSE ---
    // Feature: snn-benchmark-suite, Property 13: Correspondencia de datos en gráficas de evolución MSE

    /**
     * Para cualquier lista de ResultadoBenchmark con MSE arrays de longitud variable (1-20 épocas),
     * generarGraficasMSEPorConfiguracion debe producir exactamente un archivo PNG por cada resultado
     * con MSE no vacío, y los resultados con MSE vacío no deben generar gráficas.
     *
     * <p><b>Validates: Requisito 8.2</b></p>
     */
    @Property(tries = 20)
    void unPNGDeMSEPorCadaResultadoConMSENoVacio(
            @ForAll("listaResultadosConMSEVariado") List<ResultadoBenchmark> resultados) throws IOException {

        Path tempDir = Files.createTempDirectory("graficas_mse_test_");
        try {
            GeneradorGraficas generador = new GeneradorGraficas(tempDir);
            generador.generarGraficasMSEPorConfiguracion(resultados);

            long resultadosConMSE = resultados.stream()
                    .filter(r -> r.errorMSEPorEpoca().length > 0)
                    .count();

            List<String> archivos = generador.getArchivosGenerados();

            // Exactamente un PNG por resultado con MSE no vacío
            assertEquals(resultadosConMSE, archivos.size(),
                    "Debe haber exactamente un PNG de MSE por resultado con MSE no vacío. "
                            + "Resultados con MSE: " + resultadosConMSE + ", Archivos: " + archivos);

            // Cada archivo existe en disco y es no vacío
            for (String archivo : archivos) {
                Path archivoPath = tempDir.resolve(archivo);
                assertTrue(Files.exists(archivoPath),
                        "El archivo " + archivo + " debe existir en disco");
                assertTrue(Files.size(archivoPath) > 0,
                        "El archivo " + archivo + " no debe estar vacío");
            }

            // Resultados con MSE vacío no generan archivos
            for (ResultadoBenchmark r : resultados) {
                String nombreEsperado = "mse_" + sanitizarEtiqueta(r.configuracion().etiqueta()) + ".png";
                if (r.errorMSEPorEpoca().length == 0) {
                    assertFalse(archivos.contains(nombreEsperado),
                            "No debe existir archivo MSE para resultado con MSE vacío: " + nombreEsperado);
                } else {
                    assertTrue(archivos.contains(nombreEsperado),
                            "Debe existir archivo MSE para resultado con MSE no vacío: " + nombreEsperado);
                }
            }
        } finally {
            limpiarDirectorio(tempDir);
        }
    }

    // --- Propiedad 14: Visualización correcta de fronteras de capacidad ---
    // Feature: snn-benchmark-suite, Property 14: Visualización correcta de fronteras de capacidad

    /**
     * Para cualquier conjunto de resultados (con o sin fronteras de capacidad detectadas),
     * generarGraficaFronterasCapacidad debe producir exactamente un archivo PNG
     * (fronteras_capacidad.png) que exista en disco y sea no vacío.
     *
     * <p><b>Validates: Requisito 8.5</b></p>
     */
    @Property(tries = 20)
    void graficaFronterasCapacidadSiempreGeneraPNG(
            @ForAll("listaResultadosConFronterasPosibles") List<ResultadoBenchmark> resultados,
            @ForAll("fronterasCapacidadArbitrarias") Map<String, List<String>> fronteras) throws IOException {

        Path tempDir = Files.createTempDirectory("graficas_fronteras_test_");
        try {
            GeneradorGraficas generador = new GeneradorGraficas(tempDir);
            generador.generarGraficaFronterasCapacidad(resultados, fronteras);

            List<String> archivos = generador.getArchivosGenerados();

            // Exactamente un archivo generado: fronteras_capacidad.png
            assertEquals(1, archivos.size(),
                    "Debe generarse exactamente un archivo PNG de fronteras de capacidad. "
                            + "Archivos: " + archivos);
            assertEquals("fronteras_capacidad.png", archivos.get(0),
                    "El archivo generado debe llamarse fronteras_capacidad.png");

            // El archivo existe en disco y es no vacío
            Path archivoPath = tempDir.resolve("fronteras_capacidad.png");
            assertTrue(Files.exists(archivoPath),
                    "El archivo fronteras_capacidad.png debe existir en disco");
            assertTrue(Files.size(archivoPath) > 0,
                    "El archivo fronteras_capacidad.png no debe estar vacío");
        } finally {
            limpiarDirectorio(tempDir);
        }
    }


    // --- Propiedad 15: Corrección de metadatos y formato de salida de gráficas ---
    // Feature: snn-benchmark-suite, Property 15: Corrección de metadatos y formato de salida de gráficas

    /**
     * Para cualquier gráfica generada por el GeneradorGraficas, el archivo de salida
     * debe existir en el directorio configurado en formato PNG, y el objeto de gráfica
     * debe contener un título no vacío, etiquetas en ambos ejes, y leyenda cuando la
     * gráfica incluye más de una serie de datos.
     *
     * <p><b>Validates: Requisitos 8.6, 8.7</b></p>
     */
    @Property(tries = 20)
    void todasLasGraficasTienenMetadatosYFormatoCorrecto(
            @ForAll("listaResultadosConNivelesVariados") List<ResultadoBenchmark> resultados,
            @ForAll("fronterasCapacidadArbitrarias") Map<String, List<String>> fronteras) throws IOException {

        Path tempDir = Files.createTempDirectory("graficas_metadatos_test_");
        try {
            GeneradorGraficas generador = new GeneradorGraficas(tempDir);
            generador.generarTodas(resultados, fronteras);

            List<String> archivos = generador.getArchivosGenerados();
            List<Chart<?, ?>> graficas = generador.getGraficasGeneradas();

            // Debe haber la misma cantidad de archivos que de objetos de gráfica
            assertEquals(archivos.size(), graficas.size(),
                    "La cantidad de archivos generados debe coincidir con la cantidad de objetos de gráfica");

            for (int i = 0; i < archivos.size(); i++) {
                String archivo = archivos.get(i);
                Chart<?, ?> chart = graficas.get(i);

                // 1. El archivo PNG debe existir en el directorio configurado
                Path archivoPath = tempDir.resolve(archivo);
                assertTrue(Files.exists(archivoPath),
                        "El archivo " + archivo + " debe existir en el directorio de salida");
                assertTrue(archivo.endsWith(".png"),
                        "El archivo " + archivo + " debe tener extensión .png");
                assertTrue(Files.size(archivoPath) > 0,
                        "El archivo " + archivo + " no debe estar vacío");

                // 2. Título no vacío
                assertNotNull(chart.getTitle(),
                        "La gráfica " + archivo + " debe tener un título no nulo");
                assertFalse(chart.getTitle().isBlank(),
                        "La gráfica " + archivo + " debe tener un título no vacío");

                // 3. Etiquetas en ambos ejes
                assertNotNull(chart.getXAxisTitle(),
                        "La gráfica " + archivo + " debe tener etiqueta en el eje X");
                assertFalse(chart.getXAxisTitle().isBlank(),
                        "La gráfica " + archivo + " debe tener etiqueta no vacía en el eje X");
                assertNotNull(chart.getYAxisTitle(),
                        "La gráfica " + archivo + " debe tener etiqueta en el eje Y");
                assertFalse(chart.getYAxisTitle().isBlank(),
                        "La gráfica " + archivo + " debe tener etiqueta no vacía en el eje Y");

                // 4. Leyenda visible cuando hay más de una serie de datos
                int numSeries = chart.getSeriesMap().size();
                if (numSeries > 1) {
                    assertTrue(chart.getStyler().isLegendVisible(),
                            "La gráfica " + archivo + " con " + numSeries
                                    + " series debe tener la leyenda visible");
                }
            }
        } finally {
            limpiarDirectorio(tempDir);
        }
    }


    /** Replica la lógica de sanitización de GeneradorGraficas para predecir nombres de archivo. */
    private String sanitizarEtiqueta(String texto) {
        return texto.replaceAll("[^a-zA-Z0-9_-]", "_")
                .replaceAll("_+", "_")
                .toLowerCase();
    }


    // --- Proveedores (Arbitraries) ---

    /**
     * Genera una lista no vacía de ResultadoBenchmark con niveles de complejidad
     * variados y múltiples topologías por nivel, garantizando al menos un resultado
     * por cada nivel seleccionado.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultadosConNivelesVariados() {
        // Seleccionar un subconjunto no vacío de niveles
        return Arbitraries.of(NivelComplejidad.values()).set().ofMinSize(1).ofMaxSize(4)
                .flatMap(niveles -> {
                    // Para cada nivel, generar entre 1 y 3 resultados con topologías distintas
                    List<Arbitrary<List<ResultadoBenchmark>>> porNivel = new ArrayList<>();
                    for (NivelComplejidad nivel : niveles) {
                        porNivel.add(resultadosParaNivel(nivel));
                    }
                    // Combinar todas las listas en una sola
                    return combinarListas(porNivel);
                });
    }

    /**
     * Genera una lista de ResultadoBenchmark con arrays MSE de longitud variable (0-20 épocas),
     * incluyendo algunos con MSE vacío para verificar que no generan gráficas.
     * Cada resultado usa una topología distinta para evitar colisiones de nombres de archivo.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultadosConMSEVariado() {
        return Arbitraries.integers().between(1, 5).flatMap(cantidad -> {
            List<Arbitrary<ResultadoBenchmark>> arbitrarios = new ArrayList<>();
            for (int i = 0; i < cantidad; i++) {
                final int index = i;
                arbitrarios.add(resultadoConMSEVariado(index));
            }
            return combinarResultados(arbitrarios);
        });
    }

    /**
     * Genera una lista de ResultadoBenchmark con resultados en múltiples niveles de complejidad
     * y topologías variadas, adecuada para la gráfica de fronteras de capacidad.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultadosConFronterasPosibles() {
        // Generar resultados que cubran al menos 2 niveles con la misma topología
        return Arbitraries.of(NivelComplejidad.values()).set().ofMinSize(2).ofMaxSize(4)
                .flatMap(niveles -> {
                    List<Arbitrary<List<ResultadoBenchmark>>> porNivel = new ArrayList<>();
                    for (NivelComplejidad nivel : niveles) {
                        porNivel.add(resultadosParaNivel(nivel));
                    }
                    return combinarListas(porNivel);
                });
    }

    /**
     * Genera un mapa de fronteras de capacidad arbitrario. Las claves son representaciones
     * de topologías (como Arrays.toString) y los valores son listas de transiciones
     * en formato "NombreNivel → NombreNivelSiguiente".
     */
    @Provide
    Arbitrary<Map<String, List<String>>> fronterasCapacidadArbitrarias() {
        NivelComplejidad[] niveles = NivelComplejidad.values();
        // Generar entre 0 y 2 entradas de frontera
        return Arbitraries.integers().between(0, 2).flatMap(numEntradas -> {
            if (numEntradas == 0) {
                return Arbitraries.just(Collections.<String, List<String>>emptyMap());
            }
            // Usar topologías del mapa TOPOLOGIAS_EXTRA para que coincidan con los resultados
            return Arbitraries.of(NivelComplejidad.values()).flatMap(nivel -> {
                int[][] topologias = TOPOLOGIAS_EXTRA.get(nivel);
                int[] topologia = topologias[0];
                String topoKey = Arrays.toString(topologia);

                // Generar transiciones válidas entre niveles consecutivos
                return Arbitraries.integers().between(0, niveles.length - 2).map(idx -> {
                    String transicion = niveles[idx].getNombreProblema()
                            + " → " + niveles[idx + 1].getNombreProblema();
                    Map<String, List<String>> mapa = new HashMap<>();
                    mapa.put(topoKey, List.of(transicion));
                    return mapa;
                });
            });
        });
    }


    /**
     * Genera un ResultadoBenchmark con un array MSE de longitud aleatoria (0-20),
     * usando una topología única basada en el índice para evitar colisiones de nombre.
     */
    private Arbitrary<ResultadoBenchmark> resultadoConMSEVariado(int index) {
        // Use a large multiplier to guarantee unique hidden layer sizes across all indices,
        // avoiding etiqueta collisions after sanitization (e.g., index 1 and 3 previously
        // both produced hidden=5 when using topologia[1]+index with wrapping).
        int hiddenSize = (index + 1) * 100;
        int[] topologiaUnica = new int[]{2, hiddenSize, 1};
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, topologiaUnica, 10, 50, 1, 42L);

        return Arbitraries.integers().between(0, 20).flatMap(numEpocas -> {
            if (numEpocas == 0) {
                return Arbitraries.just(new ResultadoBenchmark(
                        config, 0.5, new double[0],
                        100L, 50L, 0.5, 1.0, 0.1, 10, 20, null));
            }
            return Arbitraries.doubles().between(0.01, 5.0).list().ofSize(numEpocas)
                    .map(mseList -> {
                        double[] mseArray = mseList.stream().mapToDouble(Double::doubleValue).toArray();
                        return new ResultadoBenchmark(
                                config, 0.5, mseArray,
                                100L, 50L, 0.5, 1.0, 0.1, 10, 20, null);
                    });
        });
    }


    /**
     * Genera entre 1 y 3 ResultadoBenchmark para un nivel dado, usando topologías
     * distintas del mapa TOPOLOGIAS_EXTRA.
     */
    private Arbitrary<List<ResultadoBenchmark>> resultadosParaNivel(NivelComplejidad nivel) {
        int[][] topologias = TOPOLOGIAS_EXTRA.get(nivel);
        return Arbitraries.integers().between(1, topologias.length).flatMap(cantidad -> {
            List<Arbitrary<ResultadoBenchmark>> arbitrarios = new ArrayList<>();
            for (int i = 0; i < cantidad; i++) {
                int[] topologia = topologias[i];
                ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                        nivel, topologia, 10, 50, 1, 42L);
                arbitrarios.add(resultadoArbitrario(config));
            }
            return combinarResultados(arbitrarios);
        });
    }

    /** Genera un ResultadoBenchmark con métricas aleatorias para la configuración dada. */
    private Arbitrary<ResultadoBenchmark> resultadoArbitrario(ConfiguracionBenchmark config) {
        return Combinators.combine(
                Arbitraries.doubles().between(0.0, 1.0),       // precisionFinal
                Arbitraries.doubles().between(0.01, 10.0)      // costoEnergetico
        ).as((precision, costo) ->
                new ResultadoBenchmark(
                        config, precision, new double[]{1.0, 0.5, 0.3},
                        100L, 50L, 0.5, costo, 0.1, 10, 20, null)
        );
    }

    /** Combina una lista de Arbitrary<ResultadoBenchmark> en un Arbitrary<List<ResultadoBenchmark>>. */
    private Arbitrary<List<ResultadoBenchmark>> combinarResultados(
            List<Arbitrary<ResultadoBenchmark>> arbitrarios) {
        if (arbitrarios.size() == 1) {
            return arbitrarios.get(0).map(List::of);
        }
        Arbitrary<List<ResultadoBenchmark>> resultado = arbitrarios.get(0).map(r -> {
            List<ResultadoBenchmark> lista = new ArrayList<>();
            lista.add(r);
            return lista;
        });
        for (int i = 1; i < arbitrarios.size(); i++) {
            resultado = Combinators.combine(resultado, arbitrarios.get(i))
                    .as((lista, r) -> {
                        lista.add(r);
                        return lista;
                    });
        }
        return resultado;
    }

    /** Combina múltiples Arbitrary<List<ResultadoBenchmark>> en una sola lista aplanada. */
    private Arbitrary<List<ResultadoBenchmark>> combinarListas(
            List<Arbitrary<List<ResultadoBenchmark>>> listas) {
        if (listas.size() == 1) {
            return listas.get(0);
        }
        Arbitrary<List<ResultadoBenchmark>> resultado = listas.get(0).map(ArrayList::new);
        for (int i = 1; i < listas.size(); i++) {
            resultado = Combinators.combine(resultado, listas.get(i))
                    .as((acumulada, nueva) -> {
                        acumulada.addAll(nueva);
                        return acumulada;
                    });
        }
        return resultado;
    }

    // --- Utilidades ---

    /** Limpia un directorio temporal y todos sus contenidos. */
    private static void limpiarDirectorio(Path dir) {
        try {
            if (Files.exists(dir)) {
                Files.walk(dir)
                        .sorted(Comparator.reverseOrder())
                        .forEach(path -> {
                            try {
                                Files.deleteIfExists(path);
                            } catch (IOException ignored) {
                                // Ignorar errores de limpieza
                            }
                        });
            }
        } catch (IOException ignored) {
            // Ignorar errores de limpieza
        }
    }
}
