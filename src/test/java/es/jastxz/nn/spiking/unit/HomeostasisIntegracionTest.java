package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para verificar la integración de homeostasis en la simulación.
 * Feature: spiking-neural-network, Task 19.3
 * Requisito: 18.1
 */
class HomeostasisIntegracionTest {

    @Test
    void homeostasisAjustaUmbralDuranteSimulacion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .homeostasis(true, 50.0, 0.1)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking neurona = red.getNeurona(0, 0);
        double umbralInicial = neurona.getUmbralDisparoAjustado();

        // Simular 200 timesteps sin spikes (tasa baja -> umbral debería bajar)
        for (int i = 0; i < 200; i++) {
            red.avanzarTimestep();
        }

        double umbralFinal = neurona.getUmbralDisparoAjustado();
        // Con tasa 0 y objetivo 50Hz, el umbral debería decrementarse
        assertTrue(umbralFinal < umbralInicial,
            "Umbral debería decrementarse cuando tasa es menor que objetivo. " +
            "Inicial: " + umbralInicial + ", Final: " + umbralFinal);
    }

    @Test
    void sinHomeostasisUmbralNoSeAjusta() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build(); // homeostasis desactivada por defecto

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking neurona = red.getNeurona(0, 0);
        double umbralInicial = neurona.getUmbralDisparoAjustado();

        for (int i = 0; i < 200; i++) {
            red.avanzarTimestep();
        }

        double umbralFinal = neurona.getUmbralDisparoAjustado();
        assertEquals(umbralInicial, umbralFinal, 0.001,
            "Umbral no debería cambiar sin homeostasis activa");
    }

    @Test
    void homeostasisNoRompeSimulacionBasica() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .homeostasis(true, 10.0, 0.01)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, j), 0.5, 1);

        // Procesar patrón con homeostasis activa
        double[] salida = red.procesar(new double[]{0.8, 0.3}, 150);

        assertEquals(2, salida.length);
        for (double v : salida) {
            assertTrue(v >= 0.0 && v <= 1.0);
        }
    }
}
