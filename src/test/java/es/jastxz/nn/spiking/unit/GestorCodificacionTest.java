package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.GestorCodificacion;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase GestorCodificacion.
 * 
 * <p>Verifica la construcción correcta del gestor, validación de parámetros,
 * y casos edge de codificación.</p>
 * 
 * Feature: spiking-neural-network
 */
class GestorCodificacionTest {
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción válida
    void construccionConParametrosValidosExitosa() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);
        
        assertNotNull(codificador);
        assertEquals(100.0, codificador.getFrecuenciaMaxima(), 0.001);
        assertEquals(ModoCodificacion.POISSON, codificador.getModo());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción con semilla
    void construccionConSemillaExitosa() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.REGULAR, 42L);
        
        assertNotNull(codificador);
        assertEquals(100.0, codificador.getFrecuenciaMaxima(), 0.001);
        assertEquals(ModoCodificacion.REGULAR, codificador.getModo());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de frecuencia negativa
    void construccionConFrecuenciaNegativaFalla() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new GestorCodificacion(-10.0, ModoCodificacion.POISSON);
        });
        assertTrue(exception.getMessage().contains("frecuencia máxima debe ser positiva"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de frecuencia cero
    void construccionConFrecuenciaCeroFalla() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new GestorCodificacion(0.0, ModoCodificacion.REGULAR);
        });
        assertTrue(exception.getMessage().contains("frecuencia máxima debe ser positiva"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de modo null
    void construccionConModoNullFalla() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new GestorCodificacion(100.0, null);
        });
        assertTrue(exception.getMessage().contains("modo de codificación no puede ser null"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción con diferentes modos
    void construccionConDiferentesModos() {
        GestorCodificacion poissonCodificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);
        GestorCodificacion regularCodificador = new GestorCodificacion(100.0, ModoCodificacion.REGULAR);
        GestorCodificacion burstCodificador = new GestorCodificacion(100.0, ModoCodificacion.BURST);
        
        assertEquals(ModoCodificacion.POISSON, poissonCodificador.getModo());
        assertEquals(ModoCodificacion.REGULAR, regularCodificador.getModo());
        assertEquals(ModoCodificacion.BURST, burstCodificador.getModo());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Frecuencias máximas diferentes
    void construccionConDiferentesFrecuencias() {
        GestorCodificacion codificador50Hz = new GestorCodificacion(50.0, ModoCodificacion.POISSON);
        GestorCodificacion codificador100Hz = new GestorCodificacion(100.0, ModoCodificacion.POISSON);
        GestorCodificacion codificador200Hz = new GestorCodificacion(200.0, ModoCodificacion.POISSON);
        
        assertEquals(50.0, codificador50Hz.getFrecuenciaMaxima(), 0.001);
        assertEquals(100.0, codificador100Hz.getFrecuenciaMaxima(), 0.001);
        assertEquals(200.0, codificador200Hz.getFrecuenciaMaxima(), 0.001);
    }


    @Test
    // Feature: spiking-neural-network, Example: Codificación de valor 0
    void codificacionDeValorCeroNoGeneraSpikes() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes = codificador.codificar(new double[]{0.0}, 100);
        assertTrue(spikes.isEmpty(), "Valor 0 no debe generar spikes");
    }

    @Test
    // Feature: spiking-neural-network, Example: Codificación de valor 1 con modo REGULAR
    void codificacionDeValorUnoGeneraFrecuenciaMaxima() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.REGULAR, 42L);
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes = codificador.codificar(new double[]{1.0}, 1000); // 1 segundo

        double frecuencia = spikes.size() / 1.0; // Hz
        assertEquals(100.0, frecuencia, 5.0, "Frecuencia debe estar cerca de la máxima (100 Hz)");
    }

    @Test
    // Feature: spiking-neural-network, Example: Modo REGULAR produce intervalos uniformes
    void modoRegularProduceIntervalosUniformes() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.REGULAR);
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes = codificador.codificarRegular(0.5, 0, 200);

        // Con valor 0.5 y frecuencia máxima 100 Hz, esperamos 50 Hz
        // Intervalo = 1000ms / 50Hz = 20ms
        if (spikes.size() > 1) {
            long intervaloEsperado = 20; // timesteps
            for (int i = 1; i < spikes.size(); i++) {
                long intervalo = spikes.get(i).getTimestamp() - spikes.get(i - 1).getTimestamp();
                assertEquals(intervaloEsperado, intervalo, 1, "Intervalos deben ser uniformes");
            }
        }
    }

    @Test
    // Feature: spiking-neural-network, Example: Codificación Poisson con semilla reproducible
    void codificacionPoissonConSemillaEsReproducible() {
        GestorCodificacion codificador1 = new GestorCodificacion(100.0, ModoCodificacion.POISSON, 42L);
        GestorCodificacion codificador2 = new GestorCodificacion(100.0, ModoCodificacion.POISSON, 42L);

        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes1 = codificador1.codificar(new double[]{0.7}, 100);
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes2 = codificador2.codificar(new double[]{0.7}, 100);

        assertEquals(spikes1.size(), spikes2.size(), "Misma semilla debe producir mismo número de spikes");

        for (int i = 0; i < spikes1.size(); i++) {
            assertEquals(spikes1.get(i).getTimestamp(), spikes2.get(i).getTimestamp(),
                "Misma semilla debe producir spikes en mismos timestamps");
        }
    }

    @Test
    // Feature: spiking-neural-network, Example: Codificación burst genera ráfagas
    void codificacionBurstGeneraRafagas() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.BURST);
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes = codificador.codificarBurst(0.5, 0, 300);

        assertFalse(spikes.isEmpty(), "Debe generar spikes");

        // Verificar que hay spikes consecutivos (característica de burst)
        boolean hayConsecutivos = false;
        for (int i = 1; i < spikes.size(); i++) {
            if (spikes.get(i).getTimestamp() - spikes.get(i - 1).getTimestamp() == 1) {
                hayConsecutivos = true;
                break;
            }
        }
        assertTrue(hayConsecutivos, "Modo burst debe generar spikes consecutivos");
    }

    @Test
    // Feature: spiking-neural-network, Example: Codificación de múltiples valores
    void codificacionDeMultiplesValores() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.REGULAR);
        double[] valores = {0.8, 0.3, 0.5};
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes = codificador.codificar(valores, 100);

        // Verificar que se generaron spikes para cada neurona
        java.util.Set<Long> neuronasConSpikes = new java.util.HashSet<>();
        for (es.jastxz.nn.spiking.EventoSpike spike : spikes) {
            neuronasConSpikes.add(spike.getNeuronaId());
        }

        assertTrue(neuronasConSpikes.contains(0L), "Neurona 0 debe tener spikes");
        assertTrue(neuronasConSpikes.contains(1L), "Neurona 1 debe tener spikes");
        assertTrue(neuronasConSpikes.contains(2L), "Neurona 2 debe tener spikes");
    }

    @Test
    // Feature: spiking-neural-network, Example: Spikes ordenados por timestamp
    void spikesOrdenadosPorTimestamp() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON, 42L);
        double[] valores = {0.7, 0.6, 0.5};
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes = codificador.codificar(valores, 100);

        // Verificar que están ordenados
        for (int i = 1; i < spikes.size(); i++) {
            assertTrue(spikes.get(i).getTimestamp() >= spikes.get(i - 1).getTimestamp(),
                "Spikes deben estar ordenados por timestamp");
        }
    }

    @Test
    // Feature: spiking-neural-network, Example: Validación de array vacío
    void codificacionConArrayVacioFalla() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            codificador.codificar(new double[]{}, 100);
        });
        assertTrue(exception.getMessage().contains("no puede ser null o vacío"));
    }

    @Test
    // Feature: spiking-neural-network, Example: Validación de array null
    void codificacionConArrayNullFalla() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            codificador.codificar(null, 100);
        });
        assertTrue(exception.getMessage().contains("no puede ser null o vacío"));
    }

    @Test
    // Feature: spiking-neural-network, Example: Validación de duración negativa
    void codificacionConDuracionNegativaFalla() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            codificador.codificar(new double[]{0.5}, -10);
        });
        assertTrue(exception.getMessage().contains("duración debe ser positiva"));
    }

    @Test
    // Feature: spiking-neural-network, Example: Validación de duración cero
    void codificacionConDuracionCeroFalla() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            codificador.codificar(new double[]{0.5}, 0);
        });
        assertTrue(exception.getMessage().contains("duración debe ser positiva"));
    }

    @Test
    // Feature: spiking-neural-network, Example: Valores fuera de rango se clampean
    void valoresFueraDeRangoSeClampean() {
        GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.REGULAR);

        // Valor > 1 debe tratarse como 1
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes1 = codificador.codificarRegular(1.5, 0, 1000);
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes2 = codificador.codificarRegular(1.0, 0, 1000);

        assertEquals(spikes1.size(), spikes2.size(), "Valor > 1 debe tratarse como 1");

        // Valor < 0 debe tratarse como 0
        java.util.List<es.jastxz.nn.spiking.EventoSpike> spikes3 = codificador.codificarRegular(-0.5, 0, 1000);
        assertTrue(spikes3.isEmpty(), "Valor < 0 debe tratarse como 0");
    }

}
