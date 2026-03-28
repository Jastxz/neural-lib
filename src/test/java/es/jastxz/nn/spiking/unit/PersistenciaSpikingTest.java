package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para persistencia binaria de RedNeuralSpiking.
 * Feature: spiking-neural-network, Task 21.1
 * Requisitos: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
 */
class PersistenciaSpikingTest {

    @TempDir
    Path tempDir;

    @Test
    void guardarYCargarPreservaArquitectura() throws Exception {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 4, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, j), 0.3, 1);

        String archivo = tempDir.resolve("red_test.snn").toString();
        red.guardar(archivo);

        RedNeuralSpiking cargada = RedNeuralSpiking.cargar(archivo);

        assertArrayEquals(red.getTopologia(), cargada.getTopologia());
        assertEquals(red.getSinapsis().size(), cargada.getSinapsis().size());
    }

    @Test
    void guardarYCargarPreservaPesos() throws Exception {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), 0.75, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 1), 0.25, 1);

        String archivo = tempDir.resolve("pesos_test.snn").toString();
        red.guardar(archivo);

        RedNeuralSpiking cargada = RedNeuralSpiking.cargar(archivo);

        assertEquals(2, cargada.getSinapsis().size());
        // Verificar que los pesos se preservan
        for (int i = 0; i < red.getSinapsis().size(); i++) {
            assertEquals(red.getSinapsis().get(i).getPeso(),
                cargada.getSinapsis().get(i).getPeso(), 0.0001);
        }
    }

    @Test
    void cargarReseteaEstadoTemporal() throws Exception {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), 0.5, 1);

        // Avanzar simulación para generar estado temporal
        red.getNeurona(0, 0).recibirSeñal(20.0);
        for (int i = 0; i < 10; i++) red.avanzarTimestep();

        assertTrue(red.getTimestepActual() > 0);

        String archivo = tempDir.resolve("reset_test.snn").toString();
        red.guardar(archivo);

        RedNeuralSpiking cargada = RedNeuralSpiking.cargar(archivo);

        // Estado temporal debe estar reseteado (Requisito 12.7)
        assertEquals(0, cargada.getTimestepActual());
    }

    @Test
    void redCargadaFuncionaCorrectamente() throws Exception {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, j), 0.5, 1);

        String archivo = tempDir.resolve("funcional_test.snn").toString();
        red.guardar(archivo);

        RedNeuralSpiking cargada = RedNeuralSpiking.cargar(archivo);

        // La red cargada debe poder procesar patrones
        double[] salida = cargada.procesar(new double[]{0.8, 0.3}, 100);
        assertEquals(2, salida.length);
        for (double v : salida) {
            assertTrue(v >= 0.0 && v <= 1.0);
        }
    }

    @Test
    void guardarConFilenameNullLanzaExcepcion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        assertThrows(IllegalArgumentException.class, () -> red.guardar(null));
        assertThrows(IllegalArgumentException.class, () -> red.guardar(""));
    }

    @Test
    void cargarConFilenameNullLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () -> RedNeuralSpiking.cargar(null));
        assertThrows(IllegalArgumentException.class, () -> RedNeuralSpiking.cargar(""));
    }

    @Test
    void cargarArchivoInexistenteLanzaExcepcion() {
        assertThrows(IOException.class, () -> RedNeuralSpiking.cargar("/tmp/no_existe_12345.snn"));
    }
}
