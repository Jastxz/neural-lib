package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para la completitud del informe generado
 * por {@link GeneradorInforme}.
 *
 * <p><b>Validates: Requisitos 5.1, 5.2, 5.3</b></p>
 *
 * @since 1.1
 */
class GeneradorInformePropertyTest {

    // Feature: snn-benchmark-suite, Property 8: Completitud del informe

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

    // --- Propiedad 8a: Cada topología de configuración aparece en la salida ---

    /**
     * Para cualquier lista no vacía de ResultadoBenchmark, el informe generado
     * debe contener la representación de la topología de cada configuración.
     */
    @Property(tries = 100)
    void cadaTopologiaApareceEnElInforme(
            @ForAll("listaResultados") List<ResultadoBenchmark> resultados) {

        String salida = generarInforme(resultados);

        for (ResultadoBenchmark r : resultados) {
            String topologia = Arrays.toString(r.configuracion().topologia());
            assertTrue(salida.contains(topologia),
                    "La topología " + topologia + " debería aparecer en el informe");
        }
    }

    // --- Propiedad 8b: Resultados agrupados por NivelComplejidad ---

    /**
     * Para cualquier lista de ResultadoBenchmark con múltiples niveles de complejidad,
     * el informe debe contener una cabecera de sección para cada nivel presente.
     */
    @Property(tries = 100)
    void resultadosAgrupadosPorNivelDeComplejidad(
            @ForAll("listaResultados") List<ResultadoBenchmark> resultados) {

        String salida = generarInforme(resultados);

        Set<NivelComplejidad> nivelesPresentes = new HashSet<>();
        for (ResultadoBenchmark r : resultados) {
            nivelesPresentes.add(r.configuracion().nivel());
        }

        for (NivelComplejidad nivel : nivelesPresentes) {
            assertTrue(salida.contains(nivel.getNombreProblema()),
                    "El nivel " + nivel.getNombreProblema()
                            + " debería tener una cabecera en el informe");
        }
    }

    // --- Propiedad 8c: Configuraciones con clasificación no nula en hallazgos de límites ---

    /**
     * Para cualquier lista de ResultadoBenchmark donde al menos uno tiene clasificación
     * no nula, todas las configuraciones clasificadas deben aparecer en la sección
     * "HALLAZGOS DE LÍMITES".
     */
    @Property(tries = 100)
    void configuracionesClasificadasAparecenEnHallazgos(
            @ForAll("listaResultadosConClasificacion") List<ResultadoBenchmark> resultados) {

        String salida = generarInforme(resultados);

        // Extraer la sección de hallazgos de límites
        int inicioHallazgos = salida.indexOf("=== HALLAZGOS DE LÍMITES ===");
        assertTrue(inicioHallazgos >= 0, "Debe existir la sección HALLAZGOS DE LÍMITES");

        String seccionHallazgos = salida.substring(inicioHallazgos);

        for (ResultadoBenchmark r : resultados) {
            if (r.clasificacion() != null) {
                assertTrue(seccionHallazgos.contains(r.clasificacion()),
                        "La clasificación '" + r.clasificacion()
                                + "' debería aparecer en la sección de hallazgos");
                assertTrue(seccionHallazgos.contains(r.configuracion().etiqueta()),
                        "La etiqueta '" + r.configuracion().etiqueta()
                                + "' debería aparecer en la sección de hallazgos");
            }
        }
    }

    // Feature: snn-benchmark-suite, Property 9: Selección de topología óptima

    /** Topologías fijas para el nivel TRIVIAL usadas en la propiedad 9. */
    private static final int[][] TOPOLOGIAS_TRIVIAL = {
        {2, 2, 1}, {2, 4, 1}, {2, 8, 1}, {2, 16, 1}
    };

    /**
     * Para cualquier conjunto de resultados para el nivel TRIVIAL con diferentes
     * topologías y valores aleatorios de precisión y costo, la topología óptima
     * seleccionada en el informe debe ser aquella con la mayor ratio
     * Precisión/CostoEnergético.
     *
     * <p><b>Validates: Requisitos 5.5, 7.1</b></p>
     */
    @Property(tries = 100)
    void topologiaOptimaEsLaDeMayorRatioPrecisionCosto(
            @ForAll("listaResultadosTrivialMultiTopologia") List<ResultadoBenchmark> resultados) {

        String salida = generarInforme(resultados);

        // Calcular la topología con mayor ratio Precisión/CostoEnergético
        double mejorRatio = Double.NEGATIVE_INFINITY;
        List<String> topologiasOptimas = new ArrayList<>();

        for (ResultadoBenchmark r : resultados) {
            double ratio = r.costoEnergetico() == 0.0
                    ? Double.POSITIVE_INFINITY
                    : r.precisionFinal() / r.costoEnergetico();
            if (ratio > mejorRatio) {
                mejorRatio = ratio;
                topologiasOptimas.clear();
                topologiasOptimas.add(Arrays.toString(r.configuracion().topologia()));
            } else if (ratio == mejorRatio) {
                topologiasOptimas.add(Arrays.toString(r.configuracion().topologia()));
            }
        }

        // Extraer la sección de topología óptima por nivel
        int inicioSeccion = salida.indexOf("=== TOPOLOGÍA ÓPTIMA POR NIVEL ===");
        assertTrue(inicioSeccion >= 0, "Debe existir la sección TOPOLOGÍA ÓPTIMA POR NIVEL");

        // Buscar hasta la siguiente sección (===)
        int finSeccion = salida.indexOf("===", inicioSeccion + 34);
        String seccionOptima = finSeccion >= 0
                ? salida.substring(inicioSeccion, finSeccion)
                : salida.substring(inicioSeccion);

        // Verificar que al menos una de las topologías óptimas aparece en la sección
        boolean algunaPresente = topologiasOptimas.stream()
                .anyMatch(seccionOptima::contains);
        assertTrue(algunaPresente,
                "La sección TOPOLOGÍA ÓPTIMA POR NIVEL debería contener alguna de las "
                        + "topologías con mayor ratio Precisión/Costo: " + topologiasOptimas
                        + ". Sección: " + seccionOptima);
    }

    // Feature: snn-benchmark-suite, Property 10: Detección de frontera de capacidad

    /**
     * Para cualquier par de resultados en niveles de complejidad consecutivos donde
     * las topologías tienen representaciones string distintas (lo cual siempre ocurre
     * con el enum actual, ya que cada nivel tiene dimensiones de entrada/salida únicas),
     * el método {@code detectarFronterasCapacidad} no debe detectar fronteras,
     * independientemente de los valores de precisión.
     *
     * <p>Esto verifica que el método agrupa correctamente por topología string y
     * solo compara resultados con la misma representación de topología.</p>
     *
     * <p><b>Validates: Requisito 7.4</b></p>
     */
    @Property(tries = 100)
    void sinFronteraCuandoTopologiasDifierenEntreNivelesConsecutivos(
            @ForAll("parResultadosNivelesConsecutivosTopologiasDistintas") List<ResultadoBenchmark> resultados) {

        Map<String, List<String>> fronteras =
                GeneradorInforme.detectarFronterasCapacidad(resultados);

        // Verificar que cada topología string es única por nivel (precondición)
        Map<String, Set<NivelComplejidad>> topologiasANiveles = new HashMap<>();
        for (ResultadoBenchmark r : resultados) {
            String topoKey = Arrays.toString(r.configuracion().topologia());
            topologiasANiveles
                    .computeIfAbsent(topoKey, k -> new HashSet<>())
                    .add(r.configuracion().nivel());
        }

        // Cada topología string solo aparece en un nivel, por lo que no puede haber
        // fronteras entre niveles consecutivos
        for (Map.Entry<String, Set<NivelComplejidad>> entry : topologiasANiveles.entrySet()) {
            if (entry.getValue().size() <= 1) {
                assertFalse(fronteras.containsKey(entry.getKey()),
                        "No debería detectarse frontera para topología " + entry.getKey()
                                + " que solo aparece en un nivel");
            }
        }
    }

    /**
     * Para cualquier conjunto de resultados donde ninguna topología tiene precisión
     * &gt; 0.7 en un nivel y &lt; 0.5 en el siguiente consecutivo, no se deben
     * detectar fronteras de capacidad. Se usa precisión uniforme en el rango [0.5, 0.7]
     * para todos los niveles, lo que impide cumplir ambas condiciones simultáneamente.
     *
     * <p><b>Validates: Requisito 7.4</b></p>
     */
    @Property(tries = 100)
    void sinFronteraCuandoPrecisionNoCaeEntreNiveles(
            @ForAll("resultadosSinCondicionFrontera") List<ResultadoBenchmark> resultados) {

        Map<String, List<String>> fronteras =
                GeneradorInforme.detectarFronterasCapacidad(resultados);

        assertTrue(fronteras.isEmpty(),
                "No deberían detectarse fronteras de capacidad cuando la condición "
                        + "de precisión no se cumple. Fronteras detectadas: " + fronteras);
    }

    /**
     * Genera pares de resultados para niveles consecutivos con topologías distintas
     * (dimensiones diferentes por nivel) y precisiones aleatorias que podrían cumplir
     * la condición de frontera, pero no la cumplen porque las topologías string difieren.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> parResultadosNivelesConsecutivosTopologiasDistintas() {
        return Arbitraries.integers().between(0, 2).flatMap(indiceNivel -> {
            NivelComplejidad nivelInferior = NivelComplejidad.values()[indiceNivel];
            NivelComplejidad nivelSuperior = NivelComplejidad.values()[indiceNivel + 1];

            ConfiguracionBenchmark configInferior = CONFIGS_POR_NIVEL.get(nivelInferior);
            ConfiguracionBenchmark configSuperior = CONFIGS_POR_NIVEL.get(nivelSuperior);

            return Combinators.combine(
                    Arbitraries.doubles().between(0.71, 1.0),   // precisionAlta > 0.7
                    Arbitraries.doubles().between(0.0, 0.49)    // precisionBaja < 0.5
            ).as((precAlta, precBaja) -> {
                ResultadoBenchmark rInferior = new ResultadoBenchmark(
                        configInferior, precAlta, new double[]{1.0, 0.5, 0.3},
                        100L, 50L, 0.5, 1.0, 0.1, 10, 20, null);
                ResultadoBenchmark rSuperior = new ResultadoBenchmark(
                        configSuperior, precBaja, new double[]{1.0, 0.8, 0.7},
                        200L, 80L, 0.6, 2.0, 0.2, 8, 20, null);
                return List.of(rInferior, rSuperior);
            });
        });
    }

    /**
     * Genera una lista de resultados donde ninguna topología cumple la condición
     * de frontera de capacidad. Se usa precisión uniforme en [0.5, 0.7] para todos
     * los niveles, lo que impide que se cumpla simultáneamente &gt; 0.7 y &lt; 0.5.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> resultadosSinCondicionFrontera() {
        return Arbitraries.doubles().between(0.5, 0.7).flatMap(precisionUniforme -> {
            List<ResultadoBenchmark> resultados = new ArrayList<>();
            for (NivelComplejidad nivel : NivelComplejidad.values()) {
                ConfiguracionBenchmark config = CONFIGS_POR_NIVEL.get(nivel);
                resultados.add(new ResultadoBenchmark(
                        config, precisionUniforme, new double[]{1.0, 0.5, 0.3},
                        100L, 50L, 0.5, 1.0, 0.1, 10, 20, null));
            }
            return Arbitraries.just(resultados);
        });
    }


    /**
     * Genera una lista de 4 ResultadoBenchmark para el nivel TRIVIAL, cada uno
     * con una topología diferente y valores aleatorios de precisión y costo.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultadosTrivialMultiTopologia() {
        return Combinators.combine(
                resultadoParaTopologiaTrivial(TOPOLOGIAS_TRIVIAL[0]),
                resultadoParaTopologiaTrivial(TOPOLOGIAS_TRIVIAL[1]),
                resultadoParaTopologiaTrivial(TOPOLOGIAS_TRIVIAL[2]),
                resultadoParaTopologiaTrivial(TOPOLOGIAS_TRIVIAL[3])
        ).as((r0, r1, r2, r3) -> List.of(r0, r1, r2, r3));
    }

    private Arbitrary<ResultadoBenchmark> resultadoParaTopologiaTrivial(int[] topologia) {
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, topologia, 10, 50, 1, 42L);
        return Combinators.combine(
                Arbitraries.doubles().between(0.0, 1.0),    // precisionFinal
                Arbitraries.doubles().between(0.01, 10.0)   // costoEnergetico
        ).as((precision, costo) ->
                new ResultadoBenchmark(
                        config, precision, new double[]{1.0, 0.5, 0.3},
                        100L, 50L, 0.5, costo, 0.1, 10, 20, null)
        );
    }


    // --- Proveedores (Arbitraries) ---

    /**
     * Genera una lista no vacía de ResultadoBenchmark con niveles de complejidad
     * variados y clasificaciones opcionales (algunas null, algunas no null).
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultados() {
        return nivelArbitrario().flatMap(nivel -> {
            ConfiguracionBenchmark config = CONFIGS_POR_NIVEL.get(nivel);
            return resultadoArbitrario(config);
        }).list().ofMinSize(1).ofMaxSize(8);
    }

    /**
     * Genera una lista de ResultadoBenchmark garantizando que al menos uno
     * tiene clasificación no nula.
     */
    @Provide
    Arbitrary<List<ResultadoBenchmark>> listaResultadosConClasificacion() {
        return Combinators.combine(
                resultadoConClasificacion(),
                listaResultados()
        ).as((conClasif, resto) -> {
            List<ResultadoBenchmark> combinada = new ArrayList<>();
            combinada.add(conClasif);
            combinada.addAll(resto);
            return combinada;
        });
    }

    /** Genera un NivelComplejidad aleatorio. */
    private Arbitrary<NivelComplejidad> nivelArbitrario() {
        return Arbitraries.of(NivelComplejidad.values());
    }

    /** Genera un ResultadoBenchmark con métricas aleatorias para la configuración dada. */
    private Arbitrary<ResultadoBenchmark> resultadoArbitrario(ConfiguracionBenchmark config) {
        return Combinators.combine(
                Arbitraries.doubles().between(0.0, 1.0),       // precisionFinal
                Arbitraries.doubles().between(0.01, 10.0),     // costoEnergetico
                clasificacionArbitraria()                       // clasificacion
        ).as((precision, costo, clasificacion) ->
                new ResultadoBenchmark(
                        config, precision, new double[]{1.0, 0.5, 0.3},
                        100L, 50L, 0.5, costo, 0.1, 10, 20, clasificacion)
        );
    }

    /** Genera un ResultadoBenchmark que siempre tiene clasificación no nula. */
    private Arbitrary<ResultadoBenchmark> resultadoConClasificacion() {
        return nivelArbitrario().flatMap(nivel -> {
            ConfiguracionBenchmark config = CONFIGS_POR_NIVEL.get(nivel);
            return Combinators.combine(
                    Arbitraries.doubles().between(0.0, 1.0),
                    Arbitraries.doubles().between(0.01, 10.0),
                    clasificacionNoNula()
            ).as((precision, costo, clasificacion) ->
                    new ResultadoBenchmark(
                            config, precision, new double[]{1.0, 0.5, 0.3},
                            100L, 50L, 0.5, costo, 0.1, 10, 20, clasificacion)
            );
        });
    }

    /** Genera una clasificación aleatoria: null o una de las clasificaciones válidas. */
    private Arbitrary<String> clasificacionArbitraria() {
        return Arbitraries.of(
                null,
                "limite_no_superado",
                "convergencia_estancada",
                "red_infrautilizada",
                "ineficiencia_energetica"
        );
    }

    /** Genera una clasificación no nula. */
    private Arbitrary<String> clasificacionNoNula() {
        return Arbitraries.of(
                "limite_no_superado",
                "convergencia_estancada",
                "red_infrautilizada",
                "ineficiencia_energetica"
        );
    }

    // --- Utilidades ---

    private static String generarInforme(List<ResultadoBenchmark> resultados) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
        GeneradorInforme.generar(resultados, ps);
        return baos.toString(StandardCharsets.UTF_8);
    }
}
