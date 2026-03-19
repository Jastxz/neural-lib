package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.GestorSTDP;
import es.jastxz.nn.spiking.NeuronaSpiking;
import es.jastxz.nn.spiking.SinapsisSpiking;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase GestorSTDP.
 * 
 * @author jastxz
 * @version 1.0
 */
class GestorSTDPTest {
    
    // Feature: spiking-neural-network, Task 8.1: Crear clase GestorSTDP
    
    @Test
    void constructorConParametrosValidosCreaCorrectamente() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        assertEquals(0.01, gestor.getAmplitudLTP(), 0.0001);
        assertEquals(0.012, gestor.getAmplitudLTD(), 0.0001);
        assertEquals(20.0, gestor.getTauLTP(), 0.0001);
        assertEquals(20.0, gestor.getTauLTD(), 0.0001);
    }
    
    @Test
    void constructorConAmplitudLTPCeroFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.0, 0.012, 20.0, 20.0)
        );
        assertTrue(exception.getMessage().contains("Amplitud LTP"));
    }
    
    @Test
    void constructorConAmplitudLTPNegativaFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(-0.01, 0.012, 20.0, 20.0)
        );
        assertTrue(exception.getMessage().contains("Amplitud LTP"));
    }
    
    @Test
    void constructorConAmplitudLTDCeroFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.01, 0.0, 20.0, 20.0)
        );
        assertTrue(exception.getMessage().contains("Amplitud LTD"));
    }
    
    @Test
    void constructorConAmplitudLTDNegativaFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.01, -0.012, 20.0, 20.0)
        );
        assertTrue(exception.getMessage().contains("Amplitud LTD"));
    }
    
    @Test
    void constructorConTauLTPCeroFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.01, 0.012, 0.0, 20.0)
        );
        assertTrue(exception.getMessage().contains("Tau LTP"));
    }
    
    @Test
    void constructorConTauLTPNegativoFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.01, 0.012, -20.0, 20.0)
        );
        assertTrue(exception.getMessage().contains("Tau LTP"));
    }
    
    @Test
    void constructorConTauLTDCeroFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.01, 0.012, 20.0, 0.0)
        );
        assertTrue(exception.getMessage().contains("Tau LTD"));
    }
    
    @Test
    void constructorConTauLTDNegativoFalla() {
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> new GestorSTDP(0.01, 0.012, 20.0, -20.0)
        );
        assertTrue(exception.getMessage().contains("Tau LTD"));
    }
    
    // Feature: spiking-neural-network, Task 8.2: Implementar cálculo de cambio de peso
    
    @Test
    void calcularCambioPesoConDtCeroRetornaCero() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        double cambio = gestor.calcularCambioPeso(0, true);
        assertEquals(0.0, cambio, 0.0001);
        
        cambio = gestor.calcularCambioPeso(0, false);
        assertEquals(0.0, cambio, 0.0001);
    }
    
    @Test
    void calcularCambioPesoLTPEsPositivo() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        // dt > 0, pre antes que post -> LTP (refuerzo)
        double cambio = gestor.calcularCambioPeso(10, true);
        assertTrue(cambio > 0, "LTP debe producir cambio positivo");
    }
    
    @Test
    void calcularCambioPesoLTDEsNegativo() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        // dt < 0, post antes que pre -> LTD (debilitamiento)
        double cambio = gestor.calcularCambioPeso(-10, false);
        assertTrue(cambio < 0, "LTD debe producir cambio negativo");
    }
    
    @Test
    void calcularCambioPesoLTPSigueFormulaExponencial() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        // Fórmula LTP: dw = A_LTP * exp(-dt/tau_LTP)
        long dt = 10;
        double cambioEsperado = 0.01 * Math.exp(-10.0 / 20.0);
        double cambioObtenido = gestor.calcularCambioPeso(dt, true);
        
        assertEquals(cambioEsperado, cambioObtenido, 0.0001);
    }
    
    @Test
    void calcularCambioPesoLTDSigueFormulaExponencial() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        // Fórmula LTD: dw = -A_LTD * exp(dt/tau_LTD)
        long dt = -10;
        double cambioEsperado = -0.012 * Math.exp(-10.0 / 20.0);
        double cambioObtenido = gestor.calcularCambioPeso(dt, false);
        
        assertEquals(cambioEsperado, cambioObtenido, 0.0001);
    }
    
    @Test
    void calcularCambioPesoLTPDecreceConDtMayor() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        double cambio1 = gestor.calcularCambioPeso(5, true);
        double cambio10 = gestor.calcularCambioPeso(10, true);
        double cambio50 = gestor.calcularCambioPeso(50, true);
        
        assertTrue(cambio1 > cambio10, "Cambio debe decrecer con dt mayor");
        assertTrue(cambio10 > cambio50, "Cambio debe decrecer con dt mayor");
    }
    
    @Test
    void calcularCambioPesoLTDDecreceConDtMenor() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        double cambio5 = gestor.calcularCambioPeso(-5, false);
        double cambio10 = gestor.calcularCambioPeso(-10, false);
        double cambio50 = gestor.calcularCambioPeso(-50, false);
        
        // Los cambios son negativos, así que comparamos valores absolutos
        assertTrue(Math.abs(cambio5) > Math.abs(cambio10), "Magnitud debe decrecer con |dt| mayor");
        assertTrue(Math.abs(cambio10) > Math.abs(cambio50), "Magnitud debe decrecer con |dt| mayor");
    }
    
    // Feature: spiking-neural-network, Task 8.3: Implementar aplicación de STDP a sinapsis
    
    @Test
    void aplicarSTDPConSinapsisNullFalla() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> gestor.aplicarSTDP(null, 100)
        );
        assertTrue(exception.getMessage().contains("sinapsis"));
    }
    
    @Test
    void aplicarSTDPSinSpikesNoModificaPeso() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        double pesoInicial = sinapsis.getPeso();
        gestor.aplicarSTDP(sinapsis, 100);
        
        assertEquals(pesoInicial, sinapsis.getPeso(), 0.0001);
    }
    
    @Test
    void aplicarSTDPConSoloSpikePreNoModificaPeso() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        sinapsis.setTimestampUltimoSpikePresinaptico(100L);
        
        double pesoInicial = sinapsis.getPeso();
        gestor.aplicarSTDP(sinapsis, 150);
        
        assertEquals(pesoInicial, sinapsis.getPeso(), 0.0001);
    }
    
    @Test
    void aplicarSTDPConSoloSpikePostNoModificaPeso() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        sinapsis.setTimestampUltimoSpikePostsinaptico(100L);
        
        double pesoInicial = sinapsis.getPeso();
        gestor.aplicarSTDP(sinapsis, 150);
        
        assertEquals(pesoInicial, sinapsis.getPeso(), 0.0001);
    }
    
    @Test
    void aplicarSTDPConPreAntesQuePostIncrementaPeso() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        // Pre dispara primero
        sinapsis.setTimestampUltimoSpikePresinaptico(100L);
        // Post dispara después
        sinapsis.setTimestampUltimoSpikePostsinaptico(110L);
        
        double pesoInicial = sinapsis.getPeso();
        gestor.aplicarSTDP(sinapsis, 110);
        double pesoFinal = sinapsis.getPeso();
        
        assertTrue(pesoFinal > pesoInicial, "LTP debe incrementar el peso");
    }
    
    @Test
    void aplicarSTDPConPostAntesQuePreDecrementaPeso() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        // Post dispara primero
        sinapsis.setTimestampUltimoSpikePostsinaptico(100L);
        // Pre dispara después
        sinapsis.setTimestampUltimoSpikePresinaptico(110L);
        
        double pesoInicial = sinapsis.getPeso();
        gestor.aplicarSTDP(sinapsis, 110);
        double pesoFinal = sinapsis.getPeso();
        
        assertTrue(pesoFinal < pesoInicial, "LTD debe decrementar el peso");
    }
    
    @Test
    void aplicarSTDPRespetaLimiteSuperior() {
        GestorSTDP gestor = new GestorSTDP(0.5, 0.5, 20.0, 20.0); // Amplitudes grandes
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.95, 0, 0.0, 1.0);
        
        // Pre antes que post -> LTP
        sinapsis.setTimestampUltimoSpikePresinaptico(100L);
        sinapsis.setTimestampUltimoSpikePostsinaptico(101L);
        
        gestor.aplicarSTDP(sinapsis, 101);
        
        assertTrue(sinapsis.getPeso() <= 1.0, "Peso no debe exceder el límite superior");
    }
    
    @Test
    void aplicarSTDPRespetaLimiteInferior() {
        GestorSTDP gestor = new GestorSTDP(0.5, 0.5, 20.0, 20.0); // Amplitudes grandes
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.05, 0, 0.0, 1.0);
        
        // Post antes que pre -> LTD
        sinapsis.setTimestampUltimoSpikePostsinaptico(100L);
        sinapsis.setTimestampUltimoSpikePresinaptico(101L);
        
        gestor.aplicarSTDP(sinapsis, 101);
        
        assertTrue(sinapsis.getPeso() >= 0.0, "Peso no debe ser menor que el límite inferior");
    }
    
    @Test
    void aplicarSTDPConSpikesSimultaneosNoModificaPeso() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        NeuronaSpiking pre = crearNeurona(1L);
        NeuronaSpiking post = crearNeurona(2L);
        SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
        
        // Ambos disparan al mismo tiempo
        sinapsis.setTimestampUltimoSpikePresinaptico(100L);
        sinapsis.setTimestampUltimoSpikePostsinaptico(100L);
        
        double pesoInicial = sinapsis.getPeso();
        gestor.aplicarSTDP(sinapsis, 100);
        
        assertEquals(pesoInicial, sinapsis.getPeso(), 0.0001);
    }
    
    // Feature: spiking-neural-network, Task 8.4: Implementar aplicación de STDP a capa
    
    @Test
    void aplicarSTDPACapaConListaNullFalla() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        IllegalArgumentException exception = assertThrows(
            IllegalArgumentException.class,
            () -> gestor.aplicarSTDPACapa(null, 100)
        );
        assertTrue(exception.getMessage().contains("sinapsis"));
    }
    
    @Test
    void aplicarSTDPACapaConListaVaciaNoFalla() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        List<SinapsisSpiking> sinapsis = new ArrayList<>();
        
        assertDoesNotThrow(() -> gestor.aplicarSTDPACapa(sinapsis, 100));
    }
    
    @Test
    void aplicarSTDPACapaAplicaATodasLasSinapsis() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        List<SinapsisSpiking> sinapsis = new ArrayList<>();
        
        // Crear 3 sinapsis con LTP
        for (int i = 0; i < 3; i++) {
            NeuronaSpiking pre = crearNeurona((long) (i * 2 + 1));
            NeuronaSpiking post = crearNeurona((long) (i * 2 + 2));
            SinapsisSpiking sinap = new SinapsisSpiking(pre, post, 0.5, 0, 0.0, 1.0);
            
            // Pre antes que post -> LTP
            sinap.setTimestampUltimoSpikePresinaptico(100L);
            sinap.setTimestampUltimoSpikePostsinaptico(110L);
            
            sinapsis.add(sinap);
        }
        
        // Aplicar STDP a todas
        gestor.aplicarSTDPACapa(sinapsis, 110);
        
        // Verificar que todas aumentaron su peso
        for (SinapsisSpiking sinap : sinapsis) {
            assertTrue(sinap.getPeso() > 0.5, "Todas las sinapsis deben haber incrementado su peso");
        }
    }
    
    @Test
    void aplicarSTDPACapaAplicaLTPyLTDCorrectamente() {
        GestorSTDP gestor = new GestorSTDP(0.01, 0.012, 20.0, 20.0);
        
        List<SinapsisSpiking> sinapsis = new ArrayList<>();
        
        // Sinapsis con LTP
        NeuronaSpiking pre1 = crearNeurona(1L);
        NeuronaSpiking post1 = crearNeurona(2L);
        SinapsisSpiking sinap1 = new SinapsisSpiking(pre1, post1, 0.5, 0, 0.0, 1.0);
        sinap1.setTimestampUltimoSpikePresinaptico(100L);
        sinap1.setTimestampUltimoSpikePostsinaptico(110L);
        sinapsis.add(sinap1);
        
        // Sinapsis con LTD
        NeuronaSpiking pre2 = crearNeurona(3L);
        NeuronaSpiking post2 = crearNeurona(4L);
        SinapsisSpiking sinap2 = new SinapsisSpiking(pre2, post2, 0.5, 0, 0.0, 1.0);
        sinap2.setTimestampUltimoSpikePostsinaptico(100L);
        sinap2.setTimestampUltimoSpikePresinaptico(110L);
        sinapsis.add(sinap2);
        
        // Aplicar STDP
        gestor.aplicarSTDPACapa(sinapsis, 110);
        
        // Verificar resultados
        assertTrue(sinap1.getPeso() > 0.5, "Sinapsis 1 debe haber incrementado (LTP)");
        assertTrue(sinap2.getPeso() < 0.5, "Sinapsis 2 debe haber decrementado (LTD)");
    }
    
    // Métodos auxiliares
    
    private NeuronaSpiking crearNeurona(long id) {
        return new NeuronaSpiking(
            id,
            0,
            (int) id,
            -55.0,  // umbralDisparo
            -70.0,  // potencialReposo
            20.0,   // constanteDecaimiento
            2       // duracionRefractario
        );
    }
}
