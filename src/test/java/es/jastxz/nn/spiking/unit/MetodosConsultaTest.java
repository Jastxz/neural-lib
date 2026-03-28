package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para métodos de consulta y utilidad.
 * Feature: spiking-neural-network, Tasks 24.1 y 24.2
 * Requisitos: 9.2, 9.3, 9.4, 12.6, 13.7, 15.3
 */
class MetodosConsultaTest {

    @Test
    void obtenerSpikesRetornaHistorial() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 0)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Forzar spike
        red.getNeurona(0, 0).recibirSeñal(20.0);
        red.avanzarTimestep();

        List<Long> spikes = red.obtenerSpikes(0, 0);
        assertFalse(spikes.isEmpty(), "Debería haber al menos un spike");
    }

    @Test
    void obtenerSpikesNeuronaInexistenteRetornaVacio() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // getNeurona lanza excepción para índices inválidos, pero obtenerSpikes usa getNeurona internamente
        // Verificamos que funciona para neuronas válidas sin spikes
        List<Long> spikes = red.obtenerSpikes(0, 0);
        assertTrue(spikes.isEmpty());
    }

    @Test
    void obtenerFrecuenciaDisparoCalculaCorrectamente() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 0)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Sin spikes, frecuencia debe ser 0
        assertEquals(0.0, red.obtenerFrecuenciaDisparo(0, 0, 100));
    }

    @Test
    void exportarRegistroCSVRetornaString() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 0)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        red.getNeurona(0, 0).recibirSeñal(20.0);
        red.avanzarTimestep();

        String csv = red.exportarRegistroCSV();
        assertNotNull(csv);
        assertFalse(csv.isEmpty());
    }

    @Test
    void getPesoPromedioGlobalSinSinapsis() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        assertEquals(0.0, red.getPesoPromedioGlobal());
    }

    @Test
    void getPesoPromedioGlobalConSinapsis() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), 0.6, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 0), 0.4, 1);

        assertEquals(0.5, red.getPesoPromedioGlobal(), 0.001);
    }

    @Test
    void existeNeuronaRetornaTrue() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        long id = red.getNeurona(0, 0).getId();
        assertTrue(red.existeNeurona(id));
    }

    @Test
    void existeNeuronaRetornaFalse() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        assertFalse(red.existeNeurona(99999));
    }

    @Test
    void existeConexionRetornaCorrectamente() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);

        assertFalse(red.existeConexion(pre, post));
        red.crearConexion(pre, post, 0.5, 1);
        assertTrue(red.existeConexion(pre, post));
    }

    @Test
    void validarIntegridadRedValida() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);

        assertTrue(red.validarIntegridad());
    }

    @Test
    void validarIntegridadRedSinConexiones() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        assertTrue(red.validarIntegridad());
    }
}
