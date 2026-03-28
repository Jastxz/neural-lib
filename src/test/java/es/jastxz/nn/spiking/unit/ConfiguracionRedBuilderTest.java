package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para ConfiguracionRedBuilder.
 * 
 * Feature: spiking-neural-network
 */
class ConfiguracionRedBuilderTest {
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con valores por defecto
    void builderConValoresPorDefectoConstruyeConfiguracionValida() {
        ConfiguracionRed config = new ConfiguracionRedBuilder().build();
        
        assertNotNull(config);
        assertArrayEquals(new int[]{10, 20, 10}, config.topologia);
        assertEquals(-55.0, config.umbralDisparo, 0.001);
        assertEquals(-70.0, config.potencialReposo, 0.001);
        assertEquals(20.0, config.constanteDecaimiento, 0.001);
        assertEquals(2, config.duracionRefractario);
        assertEquals(0.01, config.amplitudLTP, 0.001);
        assertEquals(0.012, config.amplitudLTD, 0.001);
        assertEquals(20.0, config.tauLTP, 0.001);
        assertEquals(20.0, config.tauLTD, 0.001);
        assertEquals(100.0, config.frecuenciaMaxima, 0.001);
        assertEquals(ModoCodificacion.POISSON, config.modoCodificacion);
        assertEquals(50, config.ventanaDecodificacion);
        assertEquals(TipoInicializacion.UNIFORME, config.tipoInicializacion);
        assertEquals(0.0, config.pesoMin, 0.001);
        // pesoMax se autoconfigura basándose en la brecha umbral-reposo:
        // brecha = -55 - (-70) = 15 mV, pesoRelay = 15 * 1.1 = 16.5, pesoMax = 16.5 * 2 = 33.0
        assertEquals(33.0, config.pesoMax, 0.001);
        assertEquals(TipoNormalizacion.L1, config.tipoNormalizacion);
        assertEquals(1.0, config.valorObjetivoNormalizacion, 0.001);
        assertEquals(1, config.retardoMin);
        assertEquals(5, config.retardoMax);
        assertFalse(config.homeostasisActiva);
        assertEquals(10.0, config.tasaDisparoObjetivo, 0.001);
        assertEquals(0.01, config.tasaAjusteHomeostasis, 0.001);
        assertFalse(config.inhibicionLateralActiva);
        assertEquals(2, config.radioInhibicion);
        assertEquals(0.5, config.fuerzaInhibicion, 0.001);
        assertEquals(1.0, config.duracionTimestep, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con topología personalizada
    void builderPermiteConfigurarTopologia() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(784, 100, 10)
            .build();
        
        assertArrayEquals(new int[]{784, 100, 10}, config.topologia);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con parámetros LIF personalizados
    void builderPermiteConfigurarParametrosLIF() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .parametrosLIF(-50.0, -65.0, 15.0, 3)
            .build();
        
        assertEquals(-50.0, config.umbralDisparo, 0.001);
        assertEquals(-65.0, config.potencialReposo, 0.001);
        assertEquals(15.0, config.constanteDecaimiento, 0.001);
        assertEquals(3, config.duracionRefractario);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con parámetros STDP personalizados
    void builderPermiteConfigurarParametrosSTDP() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .parametrosSTDP(0.02, 0.025, 25.0, 25.0)
            .build();
        
        assertEquals(0.02, config.amplitudLTP, 0.001);
        assertEquals(0.025, config.amplitudLTD, 0.001);
        assertEquals(25.0, config.tauLTP, 0.001);
        assertEquals(25.0, config.tauLTD, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con parámetros de codificación personalizados
    void builderPermiteConfigurarParametrosCodificacion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .parametrosCodificacion(200.0, ModoCodificacion.REGULAR, 100)
            .build();
        
        assertEquals(200.0, config.frecuenciaMaxima, 0.001);
        assertEquals(ModoCodificacion.REGULAR, config.modoCodificacion);
        assertEquals(100, config.ventanaDecodificacion);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con inicialización de pesos personalizada
    void builderPermiteConfigurarInicializacionPesos() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .inicializacionPesos(TipoInicializacion.NORMAL, -0.5, 0.5)
            .build();
        
        assertEquals(TipoInicializacion.NORMAL, config.tipoInicializacion);
        assertEquals(-0.5, config.pesoMin, 0.001);
        assertEquals(0.5, config.pesoMax, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con homeostasis activada
    void builderPermiteActivarHomeostasis() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .homeostasis(true, 15.0, 0.02)
            .build();
        
        assertTrue(config.homeostasisActiva);
        assertEquals(15.0, config.tasaDisparoObjetivo, 0.001);
        assertEquals(0.02, config.tasaAjusteHomeostasis, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con homeostasis usando método conveniente
    void builderPermiteActivarHomeostasisConMetodoConveniente() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .conHomeostasis()
            .build();
        
        assertTrue(config.homeostasisActiva);
        assertEquals(10.0, config.tasaDisparoObjetivo, 0.001); // Valor por defecto
        assertEquals(0.01, config.tasaAjusteHomeostasis, 0.001); // Valor por defecto
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con inhibición lateral activada
    void builderPermiteActivarInhibicionLateral() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .inhibicionLateral(true, 3, 0.7)
            .build();
        
        assertTrue(config.inhibicionLateralActiva);
        assertEquals(3, config.radioInhibicion);
        assertEquals(0.7, config.fuerzaInhibicion, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con inhibición lateral usando método conveniente
    void builderPermiteActivarInhibicionLateralConMetodoConveniente() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .conInhibicionLateral()
            .build();
        
        assertTrue(config.inhibicionLateralActiva);
        assertEquals(2, config.radioInhibicion); // Valor por defecto
        assertEquals(0.5, config.fuerzaInhibicion, 0.001); // Valor por defecto
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Builder con encadenamiento fluido completo
    void builderPermiteEncadenamientoFluido() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(100, 50, 25)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
            .inicializacionPesos(TipoInicializacion.UNIFORME, 0.0, 1.0)
            .parametrosNormalizacion(TipoNormalizacion.L1, 1.0)
            .retardos(1, 5)
            .conHomeostasis()
            .conInhibicionLateral()
            .duracionTimestep(1.0)
            .build();
        
        assertNotNull(config);
        assertArrayEquals(new int[]{100, 50, 25}, config.topologia);
        assertTrue(config.homeostasisActiva);
        assertTrue(config.inhibicionLateralActiva);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de topología vacía falla
    void validacionDeTopologiaVaciaFalla() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new ConfiguracionRedBuilder()
                .topologia()
                .build();
        });
        assertTrue(exception.getMessage().contains("al menos una capa"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de topología con capa vacía falla
    void validacionDeTopologiaConCapaVaciaFalla() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new ConfiguracionRedBuilder()
                .topologia(10, 0, 5)
                .build();
        });
        assertTrue(exception.getMessage().contains("al menos una neurona"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de umbral <= reposo falla
    void validacionDeUmbralMenorOIgualQueReposoFalla() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new ConfiguracionRedBuilder()
                .parametrosLIF(-70.0, -70.0, 20.0, 2)
                .build();
        });
        assertTrue(exception.getMessage().contains("Umbral de disparo"));
        assertTrue(exception.getMessage().contains("debe ser mayor que potencial de reposo"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de tau <= 0 falla
    void validacionDeTauNoPositivaFalla() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new ConfiguracionRedBuilder()
                .constanteDecaimiento(0.0)
                .build();
        });
        assertTrue(exception.getMessage().contains("Constante de decaimiento debe ser positiva"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de peso min >= max falla
    void validacionDePesoMinMayorOIgualQueMaxFalla() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new ConfiguracionRedBuilder()
                .rangoPesos(1.0, 0.5)
                .build();
        });
        assertTrue(exception.getMessage().contains("peso_min"));
        assertTrue(exception.getMessage().contains("debe ser menor que peso_max"));
    }
}
