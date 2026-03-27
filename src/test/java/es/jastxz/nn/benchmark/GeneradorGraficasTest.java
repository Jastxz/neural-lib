package es.jastxz.nn.benchmark;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link GeneradorGraficas}.
 */
class GeneradorGraficasTest {

    @TempDir
    Path tempDir;

    private GeneradorGraficas generador;

    @BeforeEach
    void setUp() {
        generador = new GeneradorGraficas(tempDir);
    }

    @Test
    void constructorCreaDirectorioSiNoExiste() {
        Path subDir = tempDir.resolve("sub/dir/graficas");
        new GeneradorGraficas(subDir);
        assertTrue(Files.isDirectory(subDir));
    }

    @Test
    void generarTodasConListaVaciaNoGeneraArchivos() {
        generador.generarTodas(List.of(), Map.of());
        assertTrue(generador.getArchivosGenerados().isEmpty());
    }

    @Test
    void generarTodasConNullNoGeneraArchivos() {
        generador.generarTodas(null, Map.of());
        assertTrue(generador.getArchivosGenerados().isEmpty());
    }

    @Test
    void generarGraficasPrecisionPorNivelCreaUnPNGPorNivel() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarGraficasPrecisionPorNivel(resultados);

        List<String> archivos = generador.getArchivosGenerados();
        assertEquals(2, archivos.size());
        assertTrue(archivos.stream().anyMatch(a -> a.contains("trivial")));
        assertTrue(archivos.stream().anyMatch(a -> a.contains("bajo")));

        for (String archivo : archivos) {
            assertTrue(Files.exists(tempDir.resolve(archivo)));
        }
    }

    @Test
    void generarGraficasMSEPorConfiguracionCreaUnPNGPorResultado() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarGraficasMSEPorConfiguracion(resultados);

        List<String> archivos = generador.getArchivosGenerados();
        assertEquals(2, archivos.size());
        for (String archivo : archivos) {
            assertTrue(archivo.startsWith("mse_"));
            assertTrue(Files.exists(tempDir.resolve(archivo)));
        }
    }

    @Test
    void generarGraficasCostoEnergeticoCreaUnPNGPorNivel() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarGraficasCostoEnergetico(resultados);

        List<String> archivos = generador.getArchivosGenerados();
        assertEquals(2, archivos.size());
        assertTrue(archivos.stream().anyMatch(a -> a.contains("costo_trivial")));
        assertTrue(archivos.stream().anyMatch(a -> a.contains("costo_bajo")));
    }

    @Test
    void generarHistogramaTasaDisparoCreaUnPNG() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarHistogramaTasaDisparo(resultados);

        List<String> archivos = generador.getArchivosGenerados();
        assertEquals(1, archivos.size());
        assertEquals("histograma_tasa_disparo.png", archivos.get(0));
        assertTrue(Files.exists(tempDir.resolve(archivos.get(0))));
    }

    @Test
    void generarGraficaFronterasCapacidadCreaUnPNG() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        Map<String, List<String>> fronteras = Map.of(
                Arrays.toString(new int[]{2, 4, 1}),
                List.of("Puertas Lógicas → 3 en Raya"));

        generador.generarGraficaFronterasCapacidad(resultados, fronteras);

        List<String> archivos = generador.getArchivosGenerados();
        assertEquals(1, archivos.size());
        assertEquals("fronteras_capacidad.png", archivos.get(0));
        assertTrue(Files.exists(tempDir.resolve(archivos.get(0))));
    }

    @Test
    void generarIndiceHTMLCreaArchivoConEnlaces() throws Exception {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarTodas(resultados, Map.of());

        Path htmlPath = tempDir.resolve("index.html");
        assertTrue(Files.exists(htmlPath));

        String contenido = Files.readString(htmlPath);
        assertTrue(contenido.contains("Benchmark SNN"));
        assertTrue(contenido.contains("Precisión por Nivel"));
        assertTrue(contenido.contains("Evolución MSE"));
        assertTrue(contenido.contains("Costo Energético"));
        assertTrue(contenido.contains("Tasa de Disparo"));

        // Verificar que contiene enlaces a los archivos generados
        for (String archivo : generador.getArchivosGenerados()) {
            assertTrue(contenido.contains(archivo),
                    "El HTML debe contener enlace a " + archivo);
        }
    }

    @Test
    void generarTodasGeneraTodasLasGraficas() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarTodas(resultados, Map.of());

        List<String> archivos = generador.getArchivosGenerados();
        // 2 precisión + 2 MSE + 2 costo + 1 histograma + 1 fronteras = 8
        assertEquals(8, archivos.size());
        assertTrue(Files.exists(tempDir.resolve("index.html")));
    }

    @Test
    void generarGraficaFronterasCapacidadConFronterasNull() {
        List<ResultadoBenchmark> resultados = crearResultadosPorNivel();
        generador.generarGraficaFronterasCapacidad(resultados, null);

        List<String> archivos = generador.getArchivosGenerados();
        assertEquals(1, archivos.size());
    }

    @Test
    void generarGraficasMSEOmiteResultadosSinMSE() {
        ConfiguracionBenchmark config = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 50, 1, 42L);
        ResultadoBenchmark resultado = new ResultadoBenchmark(
                config, 0.8, new double[0], 100, 50, 0.5, 1.0, 0.1, 3, 7, null);

        generador.generarGraficasMSEPorConfiguracion(List.of(resultado));
        assertTrue(generador.getArchivosGenerados().isEmpty());
    }

    // --- Helpers ---

    private List<ResultadoBenchmark> crearResultadosPorNivel() {
        List<ResultadoBenchmark> resultados = new ArrayList<>();

        ConfiguracionBenchmark configTrivial = new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, new int[]{2, 4, 1}, 10, 50, 1, 42L);
        resultados.add(new ResultadoBenchmark(
                configTrivial, 0.85, new double[]{0.5, 0.3, 0.2}, 100, 50, 0.5, 1.0,
                0.1, 5, 7, null));

        ConfiguracionBenchmark configBajo = new ConfiguracionBenchmark(
                NivelComplejidad.BAJO, new int[]{10, 20, 9}, 10, 50, 1, 42L);
        resultados.add(new ResultadoBenchmark(
                configBajo, 0.65, new double[]{0.8, 0.6, 0.4}, 200, 100, 0.3, 2.5,
                0.2, 15, 39, null));

        return resultados;
    }
}
