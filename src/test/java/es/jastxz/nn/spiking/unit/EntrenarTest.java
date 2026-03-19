package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para el método entrenar() de RedNeuralSpiking.
 * Feature: spiking-neural-network, Task 17.2
 * Requisitos: 5.1, 5.2, 5.3
 */
class EntrenarTest {

    private RedNeuralSpiking crearRedSimple() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
            .inicializacionPesos(TipoInicializacion.UNIFORME, 0.3, 0.7)
            .retardos(1, 3)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, j), 0.5, 1);

        return red;
    }

    @Test
    void entrenarRetornaError() {
        RedNeuralSpiking red = crearRedSimple();
        double[][] inputs = {{0.8, 0.3}, {0.2, 0.9}};
        double[][] targets = {{1.0, 0.0}, {0.0, 1.0}};

        double error = red.entrenar(inputs, targets, 100);
        assertTrue(error >= 0.0, "Error debe ser no negativo");
    }

    @Test
    void entrenarCambiaPesos() {
        RedNeuralSpiking red = crearRedSimple();

        // Recopilar pesos iniciales de todas las sinapsis
        double sumaPesosInicial = 0;
        for (SinapsisSpiking s : red.getSinapsis()) {
            sumaPesosInicial += s.getPeso();
        }

        double[][] inputs = {{0.9, 0.1}, {0.1, 0.9}};
        double[][] targets = {{1.0, 0.0}, {0.0, 1.0}};

        for (int epoca = 0; epoca < 20; epoca++) {
            red.entrenar(inputs, targets, 150);
        }

        double sumaPesosFinal = 0;
        for (SinapsisSpiking s : red.getSinapsis()) {
            sumaPesosFinal += s.getPeso();
        }

        assertNotEquals(sumaPesosInicial, sumaPesosFinal, 0.001,
            "La suma de pesos debe cambiar tras entrenamiento");
    }

    @Test
    void entrenarConInputsNullFalla() {
        RedNeuralSpiking red = crearRedSimple();
        assertThrows(IllegalArgumentException.class,
            () -> red.entrenar(null, new double[][]{{1.0, 0.0}}, 100));
    }

    @Test
    void entrenarConTargetsNullFalla() {
        RedNeuralSpiking red = crearRedSimple();
        assertThrows(IllegalArgumentException.class,
            () -> red.entrenar(new double[][]{{0.5, 0.5}}, null, 100));
    }

    @Test
    void entrenarConTamañosDistintosFalla() {
        RedNeuralSpiking red = crearRedSimple();
        double[][] inputs = {{0.5, 0.5}, {0.3, 0.7}};
        double[][] targets = {{1.0, 0.0}};

        assertThrows(IllegalArgumentException.class,
            () -> red.entrenar(inputs, targets, 100));
    }

    @Test
    void entrenarConDuracionCeroFalla() {
        RedNeuralSpiking red = crearRedSimple();
        assertThrows(IllegalArgumentException.class,
            () -> red.entrenar(new double[][]{{0.5, 0.5}}, new double[][]{{1.0, 0.0}}, 0));
    }

    @Test
    void entrenarRestauraModoEntrenamiento() {
        RedNeuralSpiking red = crearRedSimple();
        assertFalse(red.isModoEntrenamiento());

        red.entrenar(new double[][]{{0.5, 0.5}}, new double[][]{{1.0, 0.0}}, 50);

        // Debe restaurar el modo anterior (false)
        assertFalse(red.isModoEntrenamiento());
    }

    @Test
    void entrenarConTargetIncorrectoFalla() {
        RedNeuralSpiking red = crearRedSimple();
        // Target con 3 valores pero red tiene 2 salidas
        assertThrows(IllegalArgumentException.class,
            () -> red.entrenar(new double[][]{{0.5, 0.5}}, new double[][]{{1.0, 0.0, 0.5}}, 100));
    }
}
