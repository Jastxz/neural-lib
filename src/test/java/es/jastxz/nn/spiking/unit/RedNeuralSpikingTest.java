package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para RedNeuralSpiking.
 * 
 * <p>Verifica el constructor y la validación de arquitectura.</p>
 */
class RedNeuralSpikingTest {
    
    // ========== Tests del Constructor (Task 12.2) ==========
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor básico
    void constructorCreaRedConConfiguracionValida() {
        // Crear configuración válida
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 20, 10)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        // Construir red
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        // Verificar que la red se construyó correctamente
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Validación de configuración null
    void constructorFallaConConfiguracionNull() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, 
            () -> new RedNeuralSpiking(null));
        assertTrue(exception.getMessage().contains("configuración no puede ser null"));
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2, Requisito 15.1: Validación de arquitectura vacía
    void constructorFallaConTopologiaVacia() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            ConfiguracionRed config = new ConfiguracionRedBuilder()
                .topologia() // Topología vacía
                .build();
            new RedNeuralSpiking(config);
        });
        assertTrue(exception.getMessage().contains("al menos una capa"));
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2, Requisito 15.2: Validación de capa vacía
    void constructorFallaConCapaVacia() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            // Crear configuración con capa de 0 neuronas
            ConfiguracionRed config = new ConfiguracionRed(
                new int[]{10, 0, 5}, // Capa 1 tiene 0 neuronas
                -55.0, -70.0, 20.0, 2,
                0.01, 0.012, 20.0, 20.0,
                100.0, ModoCodificacion.POISSON, 50,
                TipoInicializacion.UNIFORME, 0.0, 1.0,
                TipoNormalizacion.L1, 1.0,
                1, 5,
                false, 10.0, 0.01,
                false, 2, 0.5,
                1.0,
                false, false, false, 0, 2.0, 0.1
            );
            new RedNeuralSpiking(config);
        });
        assertTrue(exception.getMessage().contains("al menos una neurona"));
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con red simple
    void constructorCreaRedSimple() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(5, 5)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con red multicapa
    void constructorCreaRedMulticapa() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 20, 15, 10, 5)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con homeostasis activa
    void constructorCreaRedConHomeostasis() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 10)
            .homeostasis(true, 15.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con inhibición lateral activa
    void constructorCreaRedConInhibicionLateral() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 10)
            .inhibicionLateral(true, 2, 0.5)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con todas las características
    void constructorCreaRedConTodasLasCaracteristicas() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 20, 10)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
            .inicializacionPesos(TipoInicializacion.UNIFORME, 0.0, 1.0)
            .retardos(1, 5)
            .homeostasis(true, 10.0)
            .inhibicionLateral(true, 2, 0.5)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con red de una sola capa
    void constructorCreaRedDeUnaCapaValida() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con diferentes modos de codificación
    void constructorCreaRedConCodificacionRegular() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 10)
            .parametrosCodificacion(100.0, ModoCodificacion.REGULAR, 50)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
    
    @Test
    // Feature: spiking-neural-network, Task 12.2: Constructor con codificación burst
    void constructorCreaRedConCodificacionBurst() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(10, 10)
            .parametrosCodificacion(100.0, ModoCodificacion.BURST, 50)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        assertNotNull(red);
    }
}
