package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.EventoSpike;
import es.jastxz.nn.spiking.NeuronaSpiking;
import es.jastxz.nn.spiking.SinapsisSpiking;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase SinapsisSpiking.
 * 
 * <p>Verifica la correcta implementación de:</p>
 * <ul>
 *   <li>Validación de parámetros en construcción</li>
 *   <li>Propagación de spikes con retardo</li>
 *   <li>Ajuste de peso con límites</li>
 *   <li>Registro de timestamps para STDP</li>
 * </ul>
 */
class SinapsisSpikingTest {
    
    /**
     * Crea una neurona de prueba con parámetros por defecto.
     */
    private NeuronaSpiking crearNeurona(long id, int capa, int indice) {
        return new NeuronaSpiking(
            id, capa, indice,
            -55.0,  // umbral
            -70.0,  // reposo
            20.0,   // tau
            2       // refractario
        );
    }
    
    // ========== Tests de Construcción y Validación ==========
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción válida de sinapsis
    void construccionValidaCreaSinapsis() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        
        assertEquals(pre, sinapsis.getPresinaptica());
        assertEquals(post, sinapsis.getPostsinaptica());
        assertEquals(0.5, sinapsis.getPeso());
        assertEquals(2, sinapsis.getRetardo());
        assertEquals(0.0, sinapsis.getPesoMin());
        assertEquals(1.0, sinapsis.getPesoMax());
        assertNull(sinapsis.getTimestampUltimoSpikePresinaptico());
        assertNull(sinapsis.getTimestampUltimoSpikePostsinaptico());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de neurona presinaptica null
    void construccionConPresinapticaNullFalla() {
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new SinapsisSpiking(null, post, 0.5, 2, 0.0, 1.0);
        });
        assertTrue(exception.getMessage().contains("presinaptica"));
        assertTrue(exception.getMessage().contains("null"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de neurona postsinaptica null
    void construccionConPostsinapticaNullFalla() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new SinapsisSpiking(pre, null, 0.5, 2, 0.0, 1.0);
        });
        assertTrue(exception.getMessage().contains("postsinaptica"));
        assertTrue(exception.getMessage().contains("null"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de retardo negativo
    void construccionConRetardoNegativoFalla() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new SinapsisSpiking(pre, post, 0.5, -1, 0.0, 1.0);
        });
        assertTrue(exception.getMessage().contains("retardo"));
        assertTrue(exception.getMessage().contains("negativo"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de límites de peso inválidos
    void construccionConLimitesInvalidosFalla() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            new SinapsisSpiking(pre, post, 0.5, 2, 1.0, 0.0); // min > max
        });
        assertTrue(exception.getMessage().contains("peso_min"));
        assertTrue(exception.getMessage().contains("peso_max"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de peso fuera de límites
    void construccionConPesoFueraDeRangoFalla() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        // Peso menor que pesoMin
        IllegalArgumentException exception1 = assertThrows(IllegalArgumentException.class, () -> {
            new SinapsisSpiking(pre, post, -0.1, 2, 0.0, 1.0);
        });
        assertTrue(exception1.getMessage().contains("peso inicial"));
        assertTrue(exception1.getMessage().contains("rango"));
        
        // Peso mayor que pesoMax
        IllegalArgumentException exception2 = assertThrows(IllegalArgumentException.class, () -> {
            new SinapsisSpiking(pre, post, 1.5, 2, 0.0, 1.0);
        });
        assertTrue(exception2.getMessage().contains("peso inicial"));
        assertTrue(exception2.getMessage().contains("rango"));
    }
    
    // ========== Tests de Propagación de Spikes ==========
    
    @Test
    // Feature: spiking-neural-network, Example: Retardo cero transmite instantáneamente
    void retardoCeroTransmiteInstantaneamente() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        EventoSpike evento = sinapsis.propagarSpike(100);
        
        assertEquals(100, evento.getTimestamp());
        assertEquals(post.getId(), evento.getNeuronaId());
        assertEquals(post.getCapa(), evento.getCapa());
        assertEquals(post.getIndice(), evento.getIndice());
        assertEquals(0.5, evento.getPotencialMembrana()); // El peso se transmite
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Propagación con retardo
    void propagacionConRetardoCalculaTimestampCorrectamente() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.7, 5, 0.0, 1.0);
        
        EventoSpike evento = sinapsis.propagarSpike(100);
        
        assertEquals(105, evento.getTimestamp()); // 100 + 5
        assertEquals(post.getId(), evento.getNeuronaId());
        assertEquals(0.7, evento.getPotencialMembrana());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Propagación actualiza timestamp presinaptico
    void propagacionActualizaTimestampPresinaptico() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        
        assertNull(sinapsis.getTimestampUltimoSpikePresinaptico());
        
        sinapsis.propagarSpike(100);
        
        assertEquals(100L, sinapsis.getTimestampUltimoSpikePresinaptico());
    }
    
    // ========== Tests de Ajuste de Peso ==========
    
    @Test
    // Feature: spiking-neural-network, Example: Ajuste de peso respeta límites
    void ajusteDePesoRespetaLimites() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        
        // Ajuste positivo dentro de límites
        sinapsis.ajustarPeso(0.2);
        assertEquals(0.7, sinapsis.getPeso());
        
        // Ajuste que excede pesoMax
        sinapsis.ajustarPeso(0.5);
        assertEquals(1.0, sinapsis.getPeso()); // Clamped a pesoMax
        
        // Ajuste negativo dentro de límites
        sinapsis.ajustarPeso(-0.3);
        assertEquals(0.7, sinapsis.getPeso());
        
        // Ajuste que excede pesoMin
        sinapsis.ajustarPeso(-1.0);
        assertEquals(0.0, sinapsis.getPeso()); // Clamped a pesoMin
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Ajuste de peso con delta cero
    void ajusteDePesoConDeltaCeroNoModificaPeso() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        
        sinapsis.ajustarPeso(0.0);
        
        assertEquals(0.5, sinapsis.getPeso());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Múltiples ajustes de peso
    void multiplesAjustesDePesoSeAcumulan() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        
        sinapsis.ajustarPeso(0.1);
        sinapsis.ajustarPeso(0.1);
        sinapsis.ajustarPeso(0.1);
        
        assertEquals(0.8, sinapsis.getPeso(), 0.0001);
    }
    
    // ========== Tests de Timestamps para STDP ==========
    
    @Test
    // Feature: spiking-neural-network, Example: Setters de timestamps STDP
    void settersDeTimestampsSTDPFuncionan() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        
        sinapsis.setTimestampUltimoSpikePresinaptico(100L);
        sinapsis.setTimestampUltimoSpikePostsinaptico(105L);
        
        assertEquals(100L, sinapsis.getTimestampUltimoSpikePresinaptico());
        assertEquals(105L, sinapsis.getTimestampUltimoSpikePostsinaptico());
    }
    
    // ========== Tests de Límites de Peso Negativos ==========
    
    @Test
    // Feature: spiking-neural-network, Example: Sinapsis con pesos negativos (inhibitorios)
    void sinapsisConPesosNegativosPermitida() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        // Sinapsis inhibitoria con peso negativo
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, -0.5, 2, -1.0, 0.0);
        
        assertEquals(-0.5, sinapsis.getPeso());
        assertEquals(-1.0, sinapsis.getPesoMin());
        assertEquals(0.0, sinapsis.getPesoMax());
        
        // Ajustar peso negativo
        sinapsis.ajustarPeso(-0.3);
        assertEquals(-0.8, sinapsis.getPeso());
        
        // Verificar clamp a límites negativos
        sinapsis.ajustarPeso(-0.5);
        assertEquals(-1.0, sinapsis.getPeso()); // Clamped a pesoMin
    }
    
    // ========== Tests de Equals y HashCode ==========
    
    @Test
    // Feature: spiking-neural-network, Example: Equals compara por IDs de neuronas
    void equalsComparaPorIDsDeNeuronas() {
        NeuronaSpiking pre = crearNeurona(1L, 0, 0);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        SinapsisSpiking sinapsis1 = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
        SinapsisSpiking sinapsis2 = new SinapsisSpiking(pre, post, 0.7, 3, 0.0, 1.0);
        
        // Mismas neuronas pre y post -> iguales
        assertEquals(sinapsis1, sinapsis2);
        assertEquals(sinapsis1.hashCode(), sinapsis2.hashCode());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Equals distingue diferentes conexiones
    void equalsDistingueDiferentesConexiones() {
        NeuronaSpiking pre1 = crearNeurona(1L, 0, 0);
        NeuronaSpiking pre2 = crearNeurona(3L, 0, 1);
        NeuronaSpiking post = crearNeurona(2L, 1, 0);
        
        SinapsisSpiking sinapsis1 = new SinapsisSpiking(pre1, post, 0.5, 2, 0.0, 1.0);
        SinapsisSpiking sinapsis2 = new SinapsisSpiking(pre2, post, 0.5, 2, 0.0, 1.0);
        
        // Diferentes neuronas presinapticas -> no iguales
        assertNotEquals(sinapsis1, sinapsis2);
    }
}
