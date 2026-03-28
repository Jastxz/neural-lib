package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.GestorMetricas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para GestorMetricas.
 * 
 * @author jastxz
 * @version 1.0
 * @since 1.0
 */
class GestorMetricasTest {
    
    private GestorMetricas gestor;
    
    @BeforeEach
    void setUp() {
        gestor = new GestorMetricas();
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Inicialización de métricas
    void gestorInicializaConCero() {
        assertEquals(0, gestor.getTotalSpikes());
        assertEquals(0.0, gestor.getTasaPromedioGlobal(), 0.0001);
        assertEquals(0.0, gestor.calcularCostoEnergetico(), 0.0001);
        assertEquals(0.0, gestor.calcularDispersionActividad(), 0.0001);
        assertEquals(0, gestor.getNeuronasActivas());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registro de spike único
    void registrarSpikeIncrementaContadores() {
        gestor.registrarSpike(1L, 100L);
        
        assertEquals(1, gestor.getTotalSpikes());
        assertEquals(1, gestor.getSpikesPorNeurona(1L));
        assertEquals(1, gestor.getNeuronasActivas());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registro de múltiples spikes
    void registrarMultiplesSpikesAcumula() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 110L);
        gestor.registrarSpike(1L, 120L);
        
        assertEquals(3, gestor.getTotalSpikes());
        assertEquals(3, gestor.getSpikesPorNeurona(1L));
        assertEquals(1, gestor.getNeuronasActivas());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registro de spikes de múltiples neuronas
    void registrarSpikesDeVariasNeuronas() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(3L, 110L);
        gestor.registrarSpike(1L, 115L);
        
        assertEquals(4, gestor.getTotalSpikes());
        assertEquals(2, gestor.getSpikesPorNeurona(1L));
        assertEquals(1, gestor.getSpikesPorNeurona(2L));
        assertEquals(1, gestor.getSpikesPorNeurona(3L));
        assertEquals(3, gestor.getNeuronasActivas());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Tasa promedio global
    void tasaPromedioGlobalCalculaCorrectamente() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 110L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(3L, 110L);
        gestor.registrarSpike(3L, 115L);
        gestor.registrarSpike(3L, 120L);
        
        // Neurona 1: 2 spikes, Neurona 2: 1 spike, Neurona 3: 3 spikes
        // Promedio: (2 + 1 + 3) / 3 = 2.0
        assertEquals(2.0, gestor.getTasaPromedioGlobal(), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Tasa promedio por neurona
    void tasaPromedioPorNeuronaCalculaCorrectamente() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 110L);
        gestor.registrarSpike(1L, 120L);
        
        // 3 spikes en 20 timesteps (120 - 100)
        // Tasa = 3 / 20 = 0.15
        assertEquals(0.15, gestor.getTasaPromedioPorNeurona(1L), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Tasa por neurona sin spikes
    void tasaPorNeuronaSinSpikesRetornaCero() {
        assertEquals(0.0, gestor.getTasaPromedioPorNeurona(999L), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Tasa por neurona con un solo spike
    void tasaPorNeuronaConUnSoloSpike() {
        gestor.registrarSpike(1L, 100L);
        
        // Un solo spike retorna tasa mínima de 1.0
        assertEquals(1.0, gestor.getTasaPromedioPorNeurona(1L), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Costo energético proporcional a spikes
    void costoEnergeticoEsProporcionalASpikes() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(3L, 110L);
        
        // Costo = número de spikes (asumiendo costo unitario de 1.0)
        assertEquals(3.0, gestor.calcularCostoEnergetico(), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Dispersión de actividad uniforme
    void dispersionConActividadUniforme() {
        // Todas las neuronas con el mismo número de spikes
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 110L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(2L, 115L);
        gestor.registrarSpike(3L, 108L);
        gestor.registrarSpike(3L, 118L);
        
        // Dispersión debe ser 0 (todas tienen 2 spikes)
        assertEquals(0.0, gestor.calcularDispersionActividad(), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Dispersión de actividad variada
    void dispersionConActividadVariada() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(2L, 110L);
        gestor.registrarSpike(2L, 115L);
        gestor.registrarSpike(3L, 108L);
        gestor.registrarSpike(3L, 118L);
        
        // Neurona 1: 1 spike, Neurona 2: 3 spikes, Neurona 3: 2 spikes
        // Media: 2.0
        // Varianza: ((1-2)^2 + (3-2)^2 + (2-2)^2) / 3 = (1 + 1 + 0) / 3 = 0.666...
        // Desviación estándar: sqrt(0.666...) ≈ 0.8165
        assertEquals(0.8165, gestor.calcularDispersionActividad(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Dispersión sin neuronas activas
    void dispersionSinNeuronasActivasRetornaCero() {
        assertEquals(0.0, gestor.calcularDispersionActividad(), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Exportar métricas
    void exportarMetricasIncluyeTodosCampos() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(1L, 110L);
        
        Map<String, Object> metricas = gestor.exportarMetricas();
        
        assertTrue(metricas.containsKey("totalSpikes"));
        assertTrue(metricas.containsKey("tasaPromedioGlobal"));
        assertTrue(metricas.containsKey("costoEnergetico"));
        assertTrue(metricas.containsKey("dispersionActividad"));
        assertTrue(metricas.containsKey("neuronasActivas"));
        assertTrue(metricas.containsKey("spikesPorNeurona"));
        
        assertEquals(3L, metricas.get("totalSpikes"));
        assertEquals(2, metricas.get("neuronasActivas"));
        assertEquals(1.5, (Double) metricas.get("tasaPromedioGlobal"), 0.0001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Resetear métricas
    void resetearLimpiaTodosContadores() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(2L, 105L);
        gestor.registrarSpike(3L, 110L);
        
        gestor.resetear();
        
        assertEquals(0, gestor.getTotalSpikes());
        assertEquals(0.0, gestor.getTasaPromedioGlobal(), 0.0001);
        assertEquals(0.0, gestor.calcularCostoEnergetico(), 0.0001);
        assertEquals(0.0, gestor.calcularDispersionActividad(), 0.0001);
        assertEquals(0, gestor.getNeuronasActivas());
        assertEquals(0, gestor.getSpikesPorNeurona(1L));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener timestamps por neurona
    void obtenerTimestampsPorNeurona() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 110L);
        gestor.registrarSpike(1L, 120L);
        
        List<Long> timestamps = gestor.getTimestampsPorNeurona(1L);
        
        assertEquals(3, timestamps.size());
        assertEquals(100L, timestamps.get(0));
        assertEquals(110L, timestamps.get(1));
        assertEquals(120L, timestamps.get(2));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Timestamps de neurona sin spikes
    void timestampsDeNeuronaInexistenteRetornaListaVacia() {
        List<Long> timestamps = gestor.getTimestampsPorNeurona(999L);
        
        assertTrue(timestamps.isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Spikes de neurona inexistente
    void spikesDeNeuronaInexistenteRetornaCero() {
        assertEquals(0, gestor.getSpikesPorNeurona(999L));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Múltiples spikes en mismo timestamp
    void multiplesSpikesEnMismoTimestamp() {
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 100L);
        gestor.registrarSpike(1L, 100L);
        
        assertEquals(3, gestor.getTotalSpikes());
        assertEquals(3, gestor.getSpikesPorNeurona(1L));
        
        // Con múltiples spikes en el mismo timestamp, la tasa debe reflejar esto
        double tasa = gestor.getTasaPromedioPorNeurona(1L);
        assertEquals(3.0, tasa, 0.0001); // 3 spikes en duración 0
    }
}
