package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para el método resetearEstadoTemporal() de RedNeuralSpiking.
 * 
 * Feature: spiking-neural-network
 * Task: 14.3 Implementar reset de estado temporal
 */
class ResetEstadoTemporalTest {

    /**
     * Crea una configuración básica para tests.
     */
    private ConfiguracionRed crearConfiguracionBasica() {
        return new ConfiguracionRedBuilder()
            .topologia(3, 5, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset resetea timestep actual a 0
    void resetResetearTimestepActualACero() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Avanzar algunos timesteps
        red.avanzarTimestep();
        red.avanzarTimestep();
        red.avanzarTimestep();
        
        assertEquals(3, red.getTimestepActual(), "Timestep debe ser 3 después de 3 avances");
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        assertEquals(0, red.getTimestepActual(), "Timestep debe ser 0 después de reset");
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset limpia potenciales de neuronas
    void resetLimpiaPotencialesDeNeuronas() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Obtener una neurona y modificar su potencial
        NeuronaSpiking neurona = red.getNeurona(0, 0);
        double potencialReposo = neurona.getPotencialReposo();
        
        // Modificar el potencial
        neurona.recibirSeñal(10.0);
        assertNotEquals(potencialReposo, neurona.getPotencialMembrana(), 
            "Potencial debe cambiar después de recibir señal");
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        assertEquals(potencialReposo, neurona.getPotencialMembrana(), 0.001,
            "Potencial debe volver al reposo después de reset");
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset limpia historiales de spikes
    void resetLimpiaHistorialesDeSpikes() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Obtener una neurona y generar algunos spikes
        NeuronaSpiking neurona = red.getNeurona(0, 0);
        neurona.generarSpike(10);
        neurona.generarSpike(20);
        neurona.generarSpike(30);
        
        assertEquals(3, neurona.getHistorialSpikes().size(), 
            "Neurona debe tener 3 spikes en historial");
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        assertTrue(neurona.getHistorialSpikes().isEmpty(), 
            "Historial de spikes debe estar vacío después de reset");
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset limpia timestamps de último spike
    void resetLimpiaTimestampsDeUltimoSpike() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Obtener una neurona y generar un spike
        NeuronaSpiking neurona = red.getNeurona(0, 0);
        neurona.generarSpike(100);
        
        assertEquals(100, neurona.getTimestampUltimoSpike(), 
            "Timestamp de último spike debe ser 100");
        assertEquals(0, neurona.getTimestepsDesdeUltimoSpike(),
            "Timesteps desde último spike debe ser 0");
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        assertEquals(-1, neurona.getTimestampUltimoSpike(), 
            "Timestamp de último spike debe ser -1 después de reset");
        assertEquals(Integer.MAX_VALUE, neurona.getTimestepsDesdeUltimoSpike(),
            "Timesteps desde último spike debe ser MAX_VALUE después de reset");
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset limpia todas las neuronas de todas las capas
    void resetLimpiaTodasLasNeuronas() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Modificar estado de neuronas en diferentes capas
        red.getNeurona(0, 0).generarSpike(10);
        red.getNeurona(0, 1).generarSpike(15);
        red.getNeurona(1, 0).generarSpike(20);
        red.getNeurona(1, 2).generarSpike(25);
        red.getNeurona(2, 0).generarSpike(30);
        
        // Verificar que todas tienen spikes
        assertFalse(red.getNeurona(0, 0).getHistorialSpikes().isEmpty());
        assertFalse(red.getNeurona(0, 1).getHistorialSpikes().isEmpty());
        assertFalse(red.getNeurona(1, 0).getHistorialSpikes().isEmpty());
        assertFalse(red.getNeurona(1, 2).getHistorialSpikes().isEmpty());
        assertFalse(red.getNeurona(2, 0).getHistorialSpikes().isEmpty());
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        // Verificar que todas están limpias
        assertTrue(red.getNeurona(0, 0).getHistorialSpikes().isEmpty());
        assertTrue(red.getNeurona(0, 1).getHistorialSpikes().isEmpty());
        assertTrue(red.getNeurona(1, 0).getHistorialSpikes().isEmpty());
        assertTrue(red.getNeurona(1, 2).getHistorialSpikes().isEmpty());
        assertTrue(red.getNeurona(2, 0).getHistorialSpikes().isEmpty());
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset no modifica arquitectura de red
    void resetNoModificaArquitectura() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Guardar topología original
        int[] topologiaOriginal = red.getTopologia();
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        // Verificar que la topología no cambió
        assertArrayEquals(topologiaOriginal, red.getTopologia(),
            "Topología no debe cambiar después de reset");
        
        // Verificar que las neuronas siguen existiendo
        assertNotNull(red.getNeurona(0, 0));
        assertNotNull(red.getNeurona(1, 2));
        assertNotNull(red.getNeurona(2, 1));
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset no modifica pesos sinápticos
    void resetNoModificaPesosSinapticos() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Crear algunas conexiones
        NeuronaSpiking n1 = red.getNeurona(0, 0);
        NeuronaSpiking n2 = red.getNeurona(1, 0);
        red.crearConexion(n1, n2, 0.75, 1);
        
        // Inicializar pesos con distribución uniforme
        red.inicializarPesos(TipoInicializacion.UNIFORME, 0.3, 0.7);
        
        // Guardar peso de la primera sinapsis
        double pesoOriginal = red.getSinapsis().get(0).getPeso();
        
        // Resetear estado temporal
        red.resetearEstadoTemporal();
        
        // Verificar que el peso no cambió
        assertEquals(pesoOriginal, red.getSinapsis().get(0).getPeso(), 0.001,
            "Peso sináptico no debe cambiar después de reset");
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset permite procesar nuevo patrón con estado limpio
    void resetPermiteProcesarNuevoPatronConEstadoLimpio() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Simular procesamiento de primer patrón
        NeuronaSpiking neurona = red.getNeurona(0, 0);
        neurona.generarSpike(10);
        neurona.generarSpike(20);
        red.avanzarTimestep();
        red.avanzarTimestep();
        
        // Verificar que hay estado temporal
        assertFalse(neurona.getHistorialSpikes().isEmpty());
        assertTrue(red.getTimestepActual() > 0);
        
        // Resetear para procesar nuevo patrón
        red.resetearEstadoTemporal();
        
        // Verificar que el estado está limpio para el nuevo patrón
        assertTrue(neurona.getHistorialSpikes().isEmpty());
        assertEquals(0, red.getTimestepActual());
        assertEquals(neurona.getPotencialReposo(), neurona.getPotencialMembrana(), 0.001);
    }

    @Test
    // Feature: spiking-neural-network, Example: Reset múltiples veces mantiene consistencia
    void resetMultiplesVecesMantieneConsistencia() {
        RedNeuralSpiking red = new RedNeuralSpiking(crearConfiguracionBasica());
        
        // Primer ciclo: modificar estado y resetear
        red.getNeurona(0, 0).generarSpike(10);
        red.avanzarTimestep();
        red.resetearEstadoTemporal();
        
        assertEquals(0, red.getTimestepActual());
        assertTrue(red.getNeurona(0, 0).getHistorialSpikes().isEmpty());
        
        // Segundo ciclo: modificar estado y resetear
        red.getNeurona(1, 0).generarSpike(20);
        red.avanzarTimestep();
        red.avanzarTimestep();
        red.resetearEstadoTemporal();
        
        assertEquals(0, red.getTimestepActual());
        assertTrue(red.getNeurona(1, 0).getHistorialSpikes().isEmpty());
        
        // Tercer ciclo: modificar estado y resetear
        red.getNeurona(2, 0).generarSpike(30);
        red.avanzarTimestep();
        red.resetearEstadoTemporal();
        
        assertEquals(0, red.getTimestepActual());
        assertTrue(red.getNeurona(2, 0).getHistorialSpikes().isEmpty());
    }
}
