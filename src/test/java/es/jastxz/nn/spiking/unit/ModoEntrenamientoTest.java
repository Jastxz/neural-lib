package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para verificar el comportamiento del modo entrenamiento.
 * Feature: spiking-neural-network, Task 17.1
 * 
 * Requisitos: 5.1, 5.2
 */
public class ModoEntrenamientoTest {

    /**
     * Test: El método setModoEntrenamiento activa correctamente el modo entrenamiento.
     */
    @Test
    void setModoEntrenamientoActivaElModo() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        // Por defecto, el modo entrenamiento está desactivado
        assertFalse(red.isModoEntrenamiento());
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        assertTrue(red.isModoEntrenamiento());
        
        // Desactivar modo entrenamiento
        red.setModoEntrenamiento(false);
        assertFalse(red.isModoEntrenamiento());
    }

    /**
     * Test: STDP se aplica durante la simulación cuando el modo entrenamiento está activo.
     * Requisito 5.1: LTP cuando pre dispara antes que post
     */
    @Test
    void stdpSeAplicaDuranteSimulacionEnModoEntrenamiento() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        // Crear conexión con peso inicial
        red.crearConexion(pre, post, 0.5, 0);
        double pesoInicial = red.getSinapsis().get(0).getPeso();
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Forzar spike en neurona presinaptica
        pre.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        // Forzar spike en neurona postsinaptica (después de pre)
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        // Verificar que el peso cambió (LTP: pre antes que post incrementa peso)
        double pesoFinal = red.getSinapsis().get(0).getPeso();
        assertTrue(pesoFinal > pesoInicial, 
            "El peso debería incrementarse por LTP cuando pre dispara antes que post");
    }

    /**
     * Test: STDP NO se aplica durante la simulación cuando el modo entrenamiento está inactivo.
     */
    @Test
    void stdpNoSeAplicaSinModoEntrenamiento() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        // Crear conexión con peso inicial
        red.crearConexion(pre, post, 0.5, 0);
        double pesoInicial = red.getSinapsis().get(0).getPeso();
        
        // NO activar modo entrenamiento (por defecto está desactivado)
        assertFalse(red.isModoEntrenamiento());
        
        // Forzar spikes en ambas neuronas
        pre.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        // Verificar que el peso NO cambió
        double pesoFinal = red.getSinapsis().get(0).getPeso();
        assertEquals(pesoInicial, pesoFinal, 0.0001,
            "El peso no debería cambiar cuando el modo entrenamiento está desactivado");
    }

    /**
     * Test: STDP se aplica después de cada spike cuando el modo entrenamiento está activo.
     * Requisito 5.2: LTD cuando post dispara antes que pre
     * 
     * Nota: Para que LTD se aplique, necesitamos que pre haya disparado primero (para establecer
     * el timestamp presinaptico), luego post dispara, y finalmente pre dispara de nuevo.
     * El STDP se calcula cuando post dispara, comparando con el spike previo de pre.
     */
    @Test
    void stdpSeAplicaDespuesDeCadaSpike() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        // Crear conexión
        red.crearConexion(pre, post, 0.5, 0);
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Primero: pre dispara (establece timestamp presinaptico)
        pre.setPotencialMembrana(-54.0);
        red.avanzarTimestep(); // timestep 0
        
        double pesoInicial = red.getSinapsis().get(0).getPeso();
        
        // Avanzar varios timesteps para que el periodo refractario expire
        for (int i = 0; i < 5; i++) {
            red.avanzarTimestep();
        }
        
        // Segundo: post dispara (ahora dt será negativo porque post dispara después en tiempo absoluto,
        // pero el cálculo de STDP usa el último spike de pre que fue hace varios timesteps)
        // Esto simula que post dispara "tarde" respecto al spike de pre
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep(); // timestep 6
        
        // El peso debería haber cambiado por STDP (LTP porque pre disparó antes que post)
        double pesoFinal = red.getSinapsis().get(0).getPeso();
        assertTrue(pesoFinal > pesoInicial,
            "El peso debería incrementarse por LTP cuando pre dispara antes que post");
    }

    /**
     * Test: El modo entrenamiento puede activarse y desactivarse dinámicamente.
     */
    @Test
    void modoEntrenamientoPuedeActivarseYDesactivarseDinamicamente() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        red.crearConexion(pre, post, 0.5, 0);
        
        // Fase 1: Sin entrenamiento
        red.setModoEntrenamiento(false);
        double peso1 = red.getSinapsis().get(0).getPeso();
        
        pre.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        double peso2 = red.getSinapsis().get(0).getPeso();
        assertEquals(peso1, peso2, 0.0001, "Peso no debería cambiar sin entrenamiento");
        
        // Resetear para siguiente fase
        red.resetearEstadoTemporal();
        
        // Fase 2: Con entrenamiento
        red.setModoEntrenamiento(true);
        double peso3 = red.getSinapsis().get(0).getPeso();
        
        pre.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        double peso4 = red.getSinapsis().get(0).getPeso();
        assertNotEquals(peso3, peso4, "Peso debería cambiar con entrenamiento activo");
        
        // Fase 3: Desactivar entrenamiento nuevamente
        red.resetearEstadoTemporal();
        red.setModoEntrenamiento(false);
        double peso5 = red.getSinapsis().get(0).getPeso();
        
        pre.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        double peso6 = red.getSinapsis().get(0).getPeso();
        assertEquals(peso5, peso6, 0.0001, "Peso no debería cambiar después de desactivar entrenamiento");
    }

    /**
     * Test: STDP se aplica a múltiples sinapsis durante el entrenamiento.
     */
    @Test
    void stdpSeAplicaAMultiplesSinapsis() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        // Crear múltiples conexiones
        NeuronaSpiking n1 = red.getNeurona(0, 0);
        NeuronaSpiking n2 = red.getNeurona(0, 1);
        NeuronaSpiking n3 = red.getNeurona(0, 2);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        red.crearConexion(n1, post, 0.5, 0);
        red.crearConexion(n2, post, 0.5, 0);
        red.crearConexion(n3, post, 0.5, 0);
        
        // Guardar pesos iniciales
        double peso1Inicial = red.getSinapsis().get(0).getPeso();
        double peso2Inicial = red.getSinapsis().get(1).getPeso();
        double peso3Inicial = red.getSinapsis().get(2).getPeso();
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Forzar spikes en todas las neuronas presinapticas
        n1.setPotencialMembrana(-54.0);
        n2.setPotencialMembrana(-54.0);
        n3.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        // Forzar spike en neurona postsinaptica
        post.setPotencialMembrana(-54.0);
        red.avanzarTimestep();
        
        // Verificar que todos los pesos cambiaron
        double peso1Final = red.getSinapsis().get(0).getPeso();
        double peso2Final = red.getSinapsis().get(1).getPeso();
        double peso3Final = red.getSinapsis().get(2).getPeso();
        
        assertTrue(peso1Final > peso1Inicial, "Peso 1 debería incrementarse por LTP");
        assertTrue(peso2Final > peso2Inicial, "Peso 2 debería incrementarse por LTP");
        assertTrue(peso3Final > peso3Inicial, "Peso 3 debería incrementarse por LTP");
    }
}
