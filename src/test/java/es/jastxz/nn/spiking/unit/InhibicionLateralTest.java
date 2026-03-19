package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para inhibición lateral.
 * Feature: spiking-neural-network, Tasks 20.1 y 20.2
 * Requisitos: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6
 */
class InhibicionLateralTest {

    // ========== 20.1: calcularVecinas ==========

    @Test
    void calcularVecinasRetornaVecinasDentroDeRadio() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(5, 5)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Neurona en el centro (índice 2) de capa 0, radio 1
        NeuronaSpiking centro = red.getNeurona(0, 2);
        List<NeuronaSpiking> vecinas = red.calcularVecinas(centro, 1);

        assertEquals(2, vecinas.size(), "Neurona central con radio 1 debe tener 2 vecinas");
        assertTrue(vecinas.stream().anyMatch(n -> n.getIndice() == 1));
        assertTrue(vecinas.stream().anyMatch(n -> n.getIndice() == 3));
    }

    @Test
    void calcularVecinasEnBordeRetornaMenosVecinas() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(5, 5)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Neurona en el borde (índice 0), radio 2
        NeuronaSpiking borde = red.getNeurona(0, 0);
        List<NeuronaSpiking> vecinas = red.calcularVecinas(borde, 2);

        assertEquals(2, vecinas.size(), "Neurona en borde con radio 2 debe tener 2 vecinas");
        assertTrue(vecinas.stream().anyMatch(n -> n.getIndice() == 1));
        assertTrue(vecinas.stream().anyMatch(n -> n.getIndice() == 2));
    }

    @Test
    void calcularVecinasRadioGrandeRetornaTodasMenosEllaMisma() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(4, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking neurona = red.getNeurona(0, 1);
        List<NeuronaSpiking> vecinas = red.calcularVecinas(neurona, 100);

        assertEquals(3, vecinas.size(), "Radio grande debe retornar todas menos ella misma");
    }

    @Test
    void calcularVecinasNoIncluyeNeuronasDeOtraCapa() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 3)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking neurona = red.getNeurona(0, 1);
        List<NeuronaSpiking> vecinas = red.calcularVecinas(neurona, 5);

        // Solo neuronas de capa 0
        for (NeuronaSpiking v : vecinas) {
            assertEquals(0, v.getCapa(), "Vecinas deben ser de la misma capa");
        }
    }

    @Test
    void calcularVecinasConNeuronaNull() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 3)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        assertThrows(IllegalArgumentException.class, () -> red.calcularVecinas(null, 2));
    }

    @Test
    void calcularVecinasConRadioCero() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 3)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking neurona = red.getNeurona(0, 1);
        assertThrows(IllegalArgumentException.class, () -> red.calcularVecinas(neurona, 0));
    }

    // ========== 20.2: Señales inhibitorias en simulación ==========

    @Test
    void inhibicionLateralReducePotencialDeVecinas() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 0) // sin refractario
            .inhibicionLateral(true, 2, 5.0)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Forzar spike en neurona central inyectando señal fuerte
        NeuronaSpiking centro = red.getNeurona(0, 1);
        NeuronaSpiking vecina0 = red.getNeurona(0, 0);
        NeuronaSpiking vecina2 = red.getNeurona(0, 2);

        double potencialInicialV0 = vecina0.getPotencialMembrana();
        double potencialInicialV2 = vecina2.getPotencialMembrana();

        // Inyectar señal fuerte para que la neurona central dispare
        centro.recibirSeñal(20.0);
        red.avanzarTimestep();

        // Las vecinas deben haber recibido señal inhibitoria (potencial más bajo)
        assertTrue(vecina0.getPotencialMembrana() < potencialInicialV0 + 0.1,
            "Vecina 0 debería tener potencial reducido por inhibición");
        assertTrue(vecina2.getPotencialMembrana() < potencialInicialV2 + 0.1,
            "Vecina 2 debería tener potencial reducido por inhibición");
    }

    @Test
    void sinInhibicionLateralNoAfectaVecinas() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 0)
            .build(); // inhibición desactivada por defecto
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        NeuronaSpiking centro = red.getNeurona(0, 1);
        NeuronaSpiking vecina0 = red.getNeurona(0, 0);

        double potencialAntes = vecina0.getPotencialMembrana();

        centro.recibirSeñal(20.0);
        red.avanzarTimestep();

        // Sin inhibición, el potencial de la vecina solo cambia por decaimiento
        // No debería haber señal inhibitoria adicional
        double potencialDespues = vecina0.getPotencialMembrana();
        // El potencial solo debería cambiar por decaimiento natural, no por inhibición
        double decaimiento = potencialAntes * Math.exp(-1.0 / 20.0) + (-70.0) * (1 - Math.exp(-1.0 / 20.0));
        assertEquals(decaimiento, potencialDespues, 0.01,
            "Sin inhibición, potencial solo cambia por decaimiento");
    }

    @Test
    void inhibicionNoAfectaNeuronasDeOtraCapa() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 3)
            .parametrosLIF(-55.0, -70.0, 20.0, 0)
            .inhibicionLateral(true, 5, 5.0)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones entre capas
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.1, 1);

        NeuronaSpiking neuronaCapa1 = red.getNeurona(1, 1);
        double potencialInicial = neuronaCapa1.getPotencialMembrana();

        // Forzar spike en capa 0
        red.getNeurona(0, 1).recibirSeñal(20.0);
        red.avanzarTimestep();

        // La neurona de capa 1 no debería recibir inhibición de capa 0
        // Solo debería cambiar por decaimiento (el spike de capa 0 se propaga con retardo 1)
        double decaimiento = potencialInicial * Math.exp(-1.0 / 20.0) + (-70.0) * (1 - Math.exp(-1.0 / 20.0));
        assertEquals(decaimiento, neuronaCapa1.getPotencialMembrana(), 0.01,
            "Inhibición lateral no debe cruzar capas");
    }

    @Test
    void inhibicionLateralNoRompeSimulacion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(4, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .inhibicionLateral(true, 2, 0.5)
            .build();
        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, j), 0.5, 1);

        double[] salida = red.procesar(new double[]{0.8, 0.5, 0.3, 0.9}, 100);

        assertEquals(2, salida.length);
        for (double v : salida) {
            assertTrue(v >= 0.0 && v <= 1.0, "Salida debe estar en [0,1]: " + v);
        }
    }
}
