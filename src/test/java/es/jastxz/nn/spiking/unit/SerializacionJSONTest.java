package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para serialización JSON de configuración.
 * Feature: spiking-neural-network, Task 21.3
 * Requisitos: 20.1, 20.2, 20.3, 20.5, 20.6
 */
class SerializacionJSONTest {

    @Test
    void toJSONContieneTopologia() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 4, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        String json = red.toJSON();
        assertTrue(json.contains("\"topologia\": [3, 4, 2]"));
        assertTrue(json.contains("\"umbralDisparo\": -55.0"));
        assertTrue(json.contains("\"potencialReposo\": -70.0"));
    }

    @Test
    void roundTripJSONPreservaConfiguracion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .homeostasis(true, 10.0, 0.01)
            .inhibicionLateral(true, 2, 0.5)
            .build();
        RedNeuralSpiking original = new RedNeuralSpiking(config);

        String json = original.toJSON();
        RedNeuralSpiking reconstruida = RedNeuralSpiking.desdeJSON(json);

        assertArrayEquals(original.getTopologia(), reconstruida.getTopologia());
    }

    @Test
    void desdeJSONConTopologiaMinima() {
        String json = "{ \"topologia\": [2, 1] }";
        RedNeuralSpiking red = RedNeuralSpiking.desdeJSON(json);

        assertArrayEquals(new int[]{2, 1}, red.getTopologia());
    }

    @Test
    void desdeJSONSinTopologiaLanzaExcepcion() {
        String json = "{ \"umbralDisparo\": -55.0 }";
        assertThrows(IllegalArgumentException.class, () -> RedNeuralSpiking.desdeJSON(json));
    }

    @Test
    void desdeJSONNullLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () -> RedNeuralSpiking.desdeJSON(null));
        assertThrows(IllegalArgumentException.class, () -> RedNeuralSpiking.desdeJSON(""));
    }

    @Test
    void redDesdeJSONFuncionaCorrectamente() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking original = new RedNeuralSpiking(config);
        String json = original.toJSON();

        RedNeuralSpiking reconstruida = RedNeuralSpiking.desdeJSON(json);

        // Crear conexiones y procesar
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                reconstruida.crearConexion(reconstruida.getNeurona(0, i), reconstruida.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                reconstruida.crearConexion(reconstruida.getNeurona(1, i), reconstruida.getNeurona(2, j), 0.5, 1);

        double[] salida = reconstruida.procesar(new double[]{0.8, 0.3}, 100);
        assertEquals(2, salida.length);
        for (double v : salida) {
            assertTrue(v >= 0.0 && v <= 1.0);
        }
    }

    @Test
    void toJSONContieneParametrosAvanzados() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .homeostasis(true, 15.0, 0.05)
            .inhibicionLateral(true, 3, 0.8)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        String json = red.toJSON();
        assertTrue(json.contains("\"homeostasisActiva\": true"));
        assertTrue(json.contains("\"tasaDisparoObjetivo\": 15.0"));
        assertTrue(json.contains("\"inhibicionLateralActiva\": true"));
        assertTrue(json.contains("\"radioInhibicion\": 3"));
        assertTrue(json.contains("\"fuerzaInhibicion\": 0.8"));
    }
}
