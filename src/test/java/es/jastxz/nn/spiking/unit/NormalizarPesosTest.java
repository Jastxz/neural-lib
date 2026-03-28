package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para normalización de pesos.
 * Feature: spiking-neural-network, Task 18.1
 * Requisitos: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6
 */
class NormalizarPesosTest {

    private RedNeuralSpiking crearRedConConexiones() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones con pesos variados
        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 0), 0.6, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 0), 0.4, 1);
        red.crearConexion(red.getNeurona(0, 2), red.getNeurona(1, 0), 0.8, 1);
        red.crearConexion(red.getNeurona(0, 0), red.getNeurona(1, 1), 0.3, 1);
        red.crearConexion(red.getNeurona(0, 1), red.getNeurona(1, 1), 0.7, 1);

        return red;
    }

    @Test
    void normalizarL1SumaAbsolutosIgualA1() {
        RedNeuralSpiking red = crearRedConConexiones();

        red.normalizarPesos(TipoNormalizacion.L1, new int[]{1});

        // Verificar que la suma de abs de pesos entrantes a neurona (1,0) ≈ 1.0
        double sumaAbs = 0;
        for (SinapsisSpiking s : red.getSinapsis()) {
            if (s.getPostsinaptica().getCapa() == 1 && s.getPostsinaptica().getIndice() == 0) {
                sumaAbs += Math.abs(s.getPeso());
            }
        }
        assertEquals(1.0, sumaAbs, 0.01, "Suma L1 de pesos entrantes debe ser ~1.0");
    }

    @Test
    void normalizarL2SumaCuadradosIgualA1() {
        RedNeuralSpiking red = crearRedConConexiones();

        red.normalizarPesos(TipoNormalizacion.L2, new int[]{1});

        // Verificar que la suma de cuadrados de pesos entrantes a neurona (1,0) ≈ 1.0
        double sumaCuadrados = 0;
        for (SinapsisSpiking s : red.getSinapsis()) {
            if (s.getPostsinaptica().getCapa() == 1 && s.getPostsinaptica().getIndice() == 0) {
                sumaCuadrados += s.getPeso() * s.getPeso();
            }
        }
        assertEquals(1.0, sumaCuadrados, 0.01, "Suma L2 de cuadrados debe ser ~1.0");
    }

    @Test
    void normalizarPreservaSigno() {
        // Crear red con pesos negativos no es posible con pesoMin=0, así que verificamos
        // que los pesos positivos se mantienen positivos
        RedNeuralSpiking red = crearRedConConexiones();

        red.normalizarPesos(TipoNormalizacion.L1, new int[]{1});

        for (SinapsisSpiking s : red.getSinapsis()) {
            assertTrue(s.getPeso() >= 0, "Pesos positivos deben mantenerse positivos");
        }
    }

    @Test
    void normalizarConTipoNullFalla() {
        RedNeuralSpiking red = crearRedConConexiones();
        assertThrows(IllegalArgumentException.class,
            () -> red.normalizarPesos(null, new int[]{1}));
    }

    @Test
    void normalizarConCapasNullFalla() {
        RedNeuralSpiking red = crearRedConConexiones();
        assertThrows(IllegalArgumentException.class,
            () -> red.normalizarPesos(TipoNormalizacion.L1, null));
    }

    @Test
    void normalizarConCapaInvalidaFalla() {
        RedNeuralSpiking red = crearRedConConexiones();
        assertThrows(IllegalArgumentException.class,
            () -> red.normalizarPesos(TipoNormalizacion.L1, new int[]{5}));
    }

    @Test
    void normalizarCapaSinConexionesNoFalla() {
        RedNeuralSpiking red = crearRedConConexiones();
        // Capa 0 no tiene conexiones entrantes
        assertDoesNotThrow(() -> red.normalizarPesos(TipoNormalizacion.L1, new int[]{0}));
    }
}
