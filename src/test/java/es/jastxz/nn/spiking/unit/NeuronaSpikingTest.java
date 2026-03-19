package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.NeuronaSpiking;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase NeuronaSpiking.
 * 
 * Feature: spiking-neural-network
 */
class NeuronaSpikingTest {
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción válida de neurona
    void construccionValidaDeNeurona() {
        NeuronaSpiking neurona = new NeuronaSpiking(
            1L,      // id
            0,       // capa
            0,       // indice
            -55.0,   // umbralDisparo
            -70.0,   // potencialReposo
            20.0,    // constanteDecaimiento
            2        // duracionRefractario
        );
        
        assertEquals(1L, neurona.getId());
        assertEquals(0, neurona.getCapa());
        assertEquals(0, neurona.getIndice());
        assertEquals(-55.0, neurona.getUmbralDisparo(), 0.001);
        assertEquals(-70.0, neurona.getPotencialReposo(), 0.001);
        assertEquals(20.0, neurona.getConstanteDecaimiento(), 0.001);
        assertEquals(2, neurona.getDuracionRefractario());
        
        // Estado inicial
        assertEquals(-70.0, neurona.getPotencialMembrana(), 0.001); // Inicia en reposo
        assertEquals(-1, neurona.getTimestampUltimoSpike());
        assertEquals(Integer.MAX_VALUE, neurona.getTimestepsDesdeUltimoSpike());
        assertTrue(neurona.getHistorialSpikes().isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de umbral <= reposo falla
    void validacionUmbralMenorOIgualQueReposoFalla() {
        // Umbral igual a reposo
        IllegalArgumentException ex1 = assertThrows(IllegalArgumentException.class, () -> {
            new NeuronaSpiking(1L, 0, 0, -70.0, -70.0, 20.0, 2);
        });
        assertTrue(ex1.getMessage().contains("Umbral de disparo"));
        assertTrue(ex1.getMessage().contains("debe ser mayor que potencial de reposo"));
        
        // Umbral menor que reposo
        IllegalArgumentException ex2 = assertThrows(IllegalArgumentException.class, () -> {
            new NeuronaSpiking(1L, 0, 0, -75.0, -70.0, 20.0, 2);
        });
        assertTrue(ex2.getMessage().contains("Umbral de disparo"));
        assertTrue(ex2.getMessage().contains("debe ser mayor que potencial de reposo"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de constante de decaimiento <= 0 falla
    void validacionConstanteDecaimientoCeroONegativaFalla() {
        // Constante cero
        IllegalArgumentException ex1 = assertThrows(IllegalArgumentException.class, () -> {
            new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 0.0, 2);
        });
        assertTrue(ex1.getMessage().contains("Constante de decaimiento debe ser positiva"));
        
        // Constante negativa
        IllegalArgumentException ex2 = assertThrows(IllegalArgumentException.class, () -> {
            new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, -5.0, 2);
        });
        assertTrue(ex2.getMessage().contains("Constante de decaimiento debe ser positiva"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de duración refractario negativa falla
    void validacionDuracionRefractarioNegativaFalla() {
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> {
            new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, -1);
        });
        assertTrue(ex.getMessage().contains("Duración de período refractario no puede ser negativa"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Duración refractario cero es válida
    void duracionRefractarioCeroEsValida() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 0);
        assertEquals(0, neurona.getDuracionRefractario());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Modificación de potencial de membrana
    void modificacionDePotencialDeMembrana() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        neurona.setPotencialMembrana(-60.0);
        assertEquals(-60.0, neurona.getPotencialMembrana(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Configuración de homeostasis
    void configuracionDeHomeostasis() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        neurona.setTasaDisparoObjetivo(10.0);
        neurona.setTasaDisparoPromedio(8.5);
        neurona.setTasaAjusteHomeostasis(0.01);
        
        assertEquals(10.0, neurona.getTasaDisparoObjetivo(), 0.001);
        assertEquals(8.5, neurona.getTasaDisparoPromedio(), 0.001);
        assertEquals(0.01, neurona.getTasaAjusteHomeostasis(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Historial de spikes es inmutable
    void historialDeSpikesEsInmutable() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        List<Long> historial = neurona.getHistorialSpikes();
        historial.add(100L); // Intentar modificar la copia
        
        // El historial original no debe cambiar
        assertTrue(neurona.getHistorialSpikes().isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: toString proporciona información útil
    void toStringProporcionaInformacionUtil() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        String str = neurona.toString();
        assertTrue(str.contains("id=1"));
        assertTrue(str.contains("capa=0"));
        assertTrue(str.contains("indice=0"));
        assertTrue(str.contains("potencialMembrana"));
        assertTrue(str.contains("umbralDisparo"));
        assertTrue(str.contains("potencialReposo"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decaimiento exponencial básico
    void decaimientoExponencialBasico() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Establecer potencial inicial diferente del reposo
        neurona.setPotencialMembrana(-60.0);
        
        // Aplicar decaimiento
        neurona.aplicarDecaimiento(1.0); // dt = 1ms
        
        // El potencial debe haber decaído hacia el reposo
        double potencialDespues = neurona.getPotencialMembrana();
        
        // Verificar que está entre el valor inicial y el reposo
        assertTrue(potencialDespues < -60.0, "El potencial debe decrecer hacia el reposo");
        assertTrue(potencialDespues > -70.0, "El potencial no debe alcanzar el reposo en un solo timestep");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decaimiento converge a reposo
    void decaimientoConvergeAReposo() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Establecer potencial inicial muy diferente del reposo
        neurona.setPotencialMembrana(-50.0);
        
        // Aplicar decaimiento muchas veces
        for (int i = 0; i < 200; i++) {
            neurona.aplicarDecaimiento(1.0); // dt = 1ms
        }
        
        // Después de muchos timesteps, debe estar muy cerca del reposo
        double potencialFinal = neurona.getPotencialMembrana();
        assertEquals(-70.0, potencialFinal, 0.1, "Después de muchos timesteps debe converger al reposo");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Fórmula de decaimiento exponencial correcta
    void formulaDeDecaimientoExponencialCorrecta() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        double potencialInicial = -60.0;
        double dt = 1.0;
        double tau = 20.0;
        double reposo = -70.0;
        
        neurona.setPotencialMembrana(potencialInicial);
        neurona.aplicarDecaimiento(dt);
        
        // Calcular el valor esperado usando la fórmula
        double factorDecaimiento = Math.exp(-dt / tau);
        double potencialEsperado = potencialInicial * factorDecaimiento + reposo * (1 - factorDecaimiento);
        
        assertEquals(potencialEsperado, neurona.getPotencialMembrana(), 0.0001,
            "El potencial debe seguir la fórmula V(t+1) = V(t) * exp(-dt/tau) + V_reposo * (1 - exp(-dt/tau))");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decaimiento con potencial en reposo no cambia
    void decaimientoConPotencialEnReposoNoCambia() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // El potencial ya está en reposo (valor inicial)
        assertEquals(-70.0, neurona.getPotencialMembrana(), 0.001);
        
        // Aplicar decaimiento
        neurona.aplicarDecaimiento(1.0);
        
        // El potencial debe permanecer en reposo
        assertEquals(-70.0, neurona.getPotencialMembrana(), 0.0001,
            "Si el potencial está en reposo, el decaimiento no debe cambiarlo");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Recepción de señal incrementa potencial
    void recepcionDeSeñalIncrementaPotencial() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        double potencialInicial = neurona.getPotencialMembrana();
        double señal = 5.0;
        
        neurona.recibirSeñal(señal);
        
        assertEquals(potencialInicial + señal, neurona.getPotencialMembrana(), 0.0001,
            "El potencial debe incrementarse exactamente por el valor de la señal");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Recepción de señal negativa decrementa potencial
    void recepcionDeSeñalNegativaDecrementaPotencial() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        double potencialInicial = neurona.getPotencialMembrana();
        double señalInhibitoria = -3.0;
        
        neurona.recibirSeñal(señalInhibitoria);
        
        assertEquals(potencialInicial + señalInhibitoria, neurona.getPotencialMembrana(), 0.0001,
            "Una señal negativa (inhibitoria) debe decrementar el potencial");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Múltiples señales se acumulan
    void multiplesSeñalesSeAcumulan() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        double potencialInicial = neurona.getPotencialMembrana();
        
        neurona.recibirSeñal(2.0);
        neurona.recibirSeñal(3.0);
        neurona.recibirSeñal(-1.0);
        
        assertEquals(potencialInicial + 4.0, neurona.getPotencialMembrana(), 0.0001,
            "Las señales múltiples deben acumularse en el potencial");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Evaluación sin alcanzar umbral no genera spike
    void evaluacionSinAlcanzarUmbralNoGeneraSpike() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Potencial por debajo del umbral
        neurona.setPotencialMembrana(-60.0);
        
        boolean disparo = neurona.evaluarActivacion(100L);
        
        assertFalse(disparo, "No debe generar spike si no alcanza el umbral");
        assertEquals(-60.0, neurona.getPotencialMembrana(), 0.001, "El potencial no debe cambiar");
        assertTrue(neurona.getHistorialSpikes().isEmpty(), "No debe haber spikes en el historial");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Evaluación alcanzando umbral genera spike
    void evaluacionAlcanzandoUmbralGeneraSpike() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Potencial alcanza el umbral
        neurona.setPotencialMembrana(-55.0);
        
        boolean disparo = neurona.evaluarActivacion(100L);
        
        assertTrue(disparo, "Debe generar spike al alcanzar el umbral");
        assertEquals(-70.0, neurona.getPotencialMembrana(), 0.001, 
            "El potencial debe resetearse al reposo después del spike");
        assertEquals(1, neurona.getHistorialSpikes().size(), "Debe haber un spike en el historial");
        assertEquals(100L, neurona.getHistorialSpikes().get(0), "El timestamp debe ser correcto");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Evaluación superando umbral genera spike
    void evaluacionSuperandoUmbralGeneraSpike() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Potencial supera el umbral
        neurona.setPotencialMembrana(-50.0);
        
        boolean disparo = neurona.evaluarActivacion(100L);
        
        assertTrue(disparo, "Debe generar spike al superar el umbral");
        assertEquals(-70.0, neurona.getPotencialMembrana(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Neurona en refractario no dispara
    void neuronaEnRefractarioNoDispara() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Generar primer spike
        neurona.setPotencialMembrana(-55.0);
        neurona.evaluarActivacion(100L);
        
        // Intentar disparar inmediatamente después (dentro del período refractario)
        neurona.setPotencialMembrana(-50.0); // Potencial muy por encima del umbral
        boolean disparo = neurona.evaluarActivacion(101L); // Solo 1 timestep después
        
        assertFalse(disparo, "No debe disparar durante el período refractario");
        assertEquals(1, neurona.getHistorialSpikes().size(), 
            "Solo debe haber un spike en el historial");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Neurona se recupera después del refractario
    void neuronaSeRecuperaDespuesDelRefractario() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Generar primer spike en timestep 100
        neurona.setPotencialMembrana(-55.0);
        neurona.evaluarActivacion(100L);
        
        // Esperar a que expire el período refractario (duración = 2 timesteps)
        // Timestep 102 está fuera del refractario
        neurona.setPotencialMembrana(-50.0);
        boolean disparo = neurona.evaluarActivacion(102L);
        
        assertTrue(disparo, "Debe poder disparar después de que expire el período refractario");
        assertEquals(2, neurona.getHistorialSpikes().size(), 
            "Debe haber dos spikes en el historial");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Verificación de estado en refractario
    void verificacionDeEstadoEnRefractario() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Antes de disparar, no está en refractario
        assertFalse(neurona.estaEnRefractario(100L));
        
        // Generar spike
        neurona.generarSpike(100L);
        
        // Inmediatamente después, está en refractario
        assertTrue(neurona.estaEnRefractario(100L));
        assertTrue(neurona.estaEnRefractario(101L));
        
        // Después de la duración del refractario, no está en refractario
        assertFalse(neurona.estaEnRefractario(102L));
        assertFalse(neurona.estaEnRefractario(103L));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Generación de spike resetea potencial
    void generacionDeSpikeReseteaPotencial() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        neurona.setPotencialMembrana(-50.0);
        neurona.generarSpike(100L);
        
        assertEquals(-70.0, neurona.getPotencialMembrana(), 0.001,
            "El potencial debe resetearse al reposo después de generar un spike");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Generación de spike registra timestamp
    void generacionDeSpikeRegistraTimestamp() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        neurona.generarSpike(100L);
        neurona.generarSpike(200L);
        neurona.generarSpike(300L);
        
        List<Long> historial = neurona.getHistorialSpikes();
        assertEquals(3, historial.size());
        assertEquals(100L, historial.get(0));
        assertEquals(200L, historial.get(1));
        assertEquals(300L, historial.get(2));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Generación de spike actualiza timestamp último spike
    void generacionDeSpikeActualizaTimestampUltimoSpike() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        assertEquals(-1, neurona.getTimestampUltimoSpike(), "Inicialmente debe ser -1");
        
        neurona.generarSpike(100L);
        assertEquals(100L, neurona.getTimestampUltimoSpike());
        
        neurona.generarSpike(250L);
        assertEquals(250L, neurona.getTimestampUltimoSpike());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Cálculo de frecuencia sin spikes
    void calculoDeFrecuenciaSinSpikes() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        double frecuencia = neurona.calcularFrecuencia(100L);
        
        assertEquals(0.0, frecuencia, 0.001, "La frecuencia debe ser 0 si no hay spikes");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Cálculo de frecuencia con spikes
    void calculoDeFrecuenciaConSpikes() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Generar 5 spikes en 100 ms
        neurona.generarSpike(0L);
        neurona.generarSpike(20L);
        neurona.generarSpike(40L);
        neurona.generarSpike(60L);
        neurona.generarSpike(80L);
        
        // Calcular frecuencia en ventana de 100 ms
        double frecuencia = neurona.calcularFrecuencia(100L);
        
        // 5 spikes en 100 ms = 50 Hz
        assertEquals(50.0, frecuencia, 0.1, "La frecuencia debe ser 50 Hz");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Cálculo de frecuencia con ventana parcial
    void calculoDeFrecuenciaConVentanaParcial() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Generar spikes distribuidos en el tiempo
        neurona.generarSpike(0L);
        neurona.generarSpike(50L);
        neurona.generarSpike(100L);
        neurona.generarSpike(150L);
        neurona.generarSpike(200L);
        
        // Calcular frecuencia solo en los últimos 50 ms
        double frecuencia = neurona.calcularFrecuencia(50L);
        
        // Solo el último spike está en la ventana [150, 200]
        // 1 spike en 50 ms = 20 Hz
        assertEquals(20.0, frecuencia, 0.1, "La frecuencia debe considerar solo la ventana especificada");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Duración refractario cero permite disparo inmediato
    void duracionRefractarioCeroPermiteDisparoInmediato() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 0);
        
        // Generar primer spike
        neurona.setPotencialMembrana(-55.0);
        neurona.evaluarActivacion(100L);
        
        // Intentar disparar inmediatamente después
        neurona.setPotencialMembrana(-55.0);
        boolean disparo = neurona.evaluarActivacion(100L); // Mismo timestep
        
        assertTrue(disparo, "Con duración refractario 0, debe poder disparar inmediatamente");
        assertEquals(2, neurona.getHistorialSpikes().size());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Homeostasis sin configurar no modifica umbral
    void homeostasisSinConfigurarNoModificaUmbral() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // No configurar homeostasis (valores por defecto son 0)
        double umbralInicial = neurona.getUmbralDisparoAjustado();
        
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        assertEquals(umbralInicial, neurona.getUmbralDisparoAjustado(), 0.0001,
            "Sin configurar homeostasis, el umbral no debe cambiar");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Homeostasis con tasa baja decrementa umbral
    void homeostasisConTasaBajaDecrementaUmbral() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Configurar homeostasis
        neurona.setTasaDisparoObjetivo(50.0); // Objetivo: 50 Hz
        neurona.setTasaAjusteHomeostasis(0.1); // Tasa de ajuste: 10%
        
        // Generar pocos spikes (tasa baja)
        neurona.generarSpike(0L);
        neurona.generarSpike(100L);
        // 2 spikes en 100 ms = 20 Hz (menor que objetivo de 50 Hz)
        
        double umbralInicial = neurona.getUmbralDisparoAjustado();
        
        // Aplicar homeostasis
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        double umbralFinal = neurona.getUmbralDisparoAjustado();
        
        assertTrue(umbralFinal < umbralInicial,
            "Con tasa de disparo baja, el umbral debe decrementarse para hacer la neurona más excitable");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Homeostasis con tasa alta incrementa umbral
    void homeostasisConTasaAltaIncrementaUmbral() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Configurar homeostasis
        neurona.setTasaDisparoObjetivo(20.0); // Objetivo: 20 Hz
        neurona.setTasaAjusteHomeostasis(0.1); // Tasa de ajuste: 10%
        
        // Generar muchos spikes (tasa alta)
        for (long t = 0; t < 100; t += 10) {
            neurona.generarSpike(t);
        }
        // 10 spikes en 100 ms = 100 Hz (mayor que objetivo de 20 Hz)
        
        double umbralInicial = neurona.getUmbralDisparoAjustado();
        
        // Aplicar homeostasis
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        double umbralFinal = neurona.getUmbralDisparoAjustado();
        
        assertTrue(umbralFinal > umbralInicial,
            "Con tasa de disparo alta, el umbral debe incrementarse para hacer la neurona menos excitable");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Homeostasis respeta límites de umbral
    void homeostasisRespetaLimitesDeUmbral() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Configurar homeostasis con ajuste muy agresivo
        neurona.setTasaDisparoObjetivo(50.0);
        neurona.setTasaAjusteHomeostasis(1.0); // 100% de ajuste (muy agresivo)
        
        // Generar muchos spikes para forzar incremento grande
        for (long t = 0; t < 100; t += 5) {
            neurona.generarSpike(t);
        }
        // 20 spikes en 100 ms = 200 Hz (mucho mayor que objetivo)
        
        // Aplicar homeostasis con límites estrictos
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        double umbralFinal = neurona.getUmbralDisparoAjustado();
        
        assertTrue(umbralFinal >= -60.0 && umbralFinal <= -50.0,
            "El umbral ajustado debe respetar los límites especificados");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Homeostasis actualiza tasa promedio
    void homeostasisActualizaTasaPromedio() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Configurar homeostasis
        neurona.setTasaDisparoObjetivo(50.0);
        neurona.setTasaAjusteHomeostasis(0.1);
        
        // Generar spikes
        neurona.generarSpike(0L);
        neurona.generarSpike(50L);
        neurona.generarSpike(100L);
        // 3 spikes en 100 ms = 30 Hz
        
        assertEquals(0.0, neurona.getTasaDisparoPromedio(), 0.001,
            "Inicialmente la tasa promedio debe ser 0");
        
        // Aplicar homeostasis
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        assertEquals(30.0, neurona.getTasaDisparoPromedio(), 0.1,
            "Después de aplicar homeostasis, la tasa promedio debe actualizarse");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Homeostasis con tasa en objetivo no cambia umbral
    void homeostasisConTasaEnObjetivoNoCambiaUmbral() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Configurar homeostasis
        neurona.setTasaDisparoObjetivo(30.0); // Objetivo: 30 Hz
        neurona.setTasaAjusteHomeostasis(0.1);
        
        // Generar spikes exactamente a la tasa objetivo
        neurona.generarSpike(0L);
        neurona.generarSpike(33L);
        neurona.generarSpike(67L);
        // 3 spikes en 100 ms = 30 Hz (igual al objetivo)
        
        double umbralInicial = neurona.getUmbralDisparoAjustado();
        
        // Aplicar homeostasis
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        double umbralFinal = neurona.getUmbralDisparoAjustado();
        
        assertEquals(umbralInicial, umbralFinal, 0.0001,
            "Si la tasa actual es igual al objetivo, el umbral no debe cambiar");
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Umbral ajustado se usa en evaluación
    void umbralAjustadoSeUsaEnEvaluacion() {
        NeuronaSpiking neurona = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
        
        // Configurar homeostasis para decrementar umbral
        neurona.setTasaDisparoObjetivo(50.0);
        neurona.setTasaAjusteHomeostasis(0.2);
        
        // Generar pocos spikes
        neurona.generarSpike(0L);
        neurona.generarSpike(100L);
        
        // Aplicar homeostasis (debería decrementar el umbral)
        neurona.aplicarHomeostasis(100L, 100L, -60.0, -50.0);
        
        double umbralAjustado = neurona.getUmbralDisparoAjustado();
        assertTrue(umbralAjustado < -55.0, "El umbral debe haberse decrementado");
        
        // Establecer potencial entre el umbral original y el ajustado
        double potencialPrueba = (umbralAjustado + -55.0) / 2.0;
        neurona.setPotencialMembrana(potencialPrueba);
        
        // Evaluar activación - debería disparar con el umbral ajustado
        boolean disparo = neurona.evaluarActivacion(200L);
        
        assertTrue(disparo, 
            "La neurona debe usar el umbral ajustado en la evaluación de activación");
    }
}
