package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.EventoSpike;
import es.jastxz.nn.spiking.GestorDecodificacion;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para GestorDecodificacion.
 * 
 * <p>Verifica la funcionalidad de decodificación de trenes de spikes a valores continuos,
 * incluyendo casos edge y validaciones.</p>
 */
class GestorDecodificacionTest {
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción válida
    void construccionConParametrosValidosExito() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        assertEquals(50, gestor.getVentanaTemporal());
        assertEquals(100.0, gestor.getFrecuenciaMaxima(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de ventana temporal
    void construccionConVentanaTemporalInvalidaFalla() {
        Exception e1 = assertThrows(IllegalArgumentException.class, 
            () -> new GestorDecodificacion(0, 100.0));
        assertTrue(e1.getMessage().contains("ventana temporal debe ser positiva"));
        
        Exception e2 = assertThrows(IllegalArgumentException.class,
            () -> new GestorDecodificacion(-10, 100.0));
        assertTrue(e2.getMessage().contains("ventana temporal debe ser positiva"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de frecuencia máxima
    void construccionConFrecuenciaMaximaInvalidaFalla() {
        Exception e1 = assertThrows(IllegalArgumentException.class,
            () -> new GestorDecodificacion(50, 0.0));
        assertTrue(e1.getMessage().contains("frecuencia máxima debe ser positiva"));
        
        Exception e2 = assertThrows(IllegalArgumentException.class,
            () -> new GestorDecodificacion(50, -50.0));
        assertTrue(e2.getMessage().contains("frecuencia máxima debe ser positiva"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación sin spikes retorna 0
    void decodificacionSinSpikesRetornaCero() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        double valor = gestor.decodificar(Collections.emptyList());
        
        assertEquals(0.0, valor, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación con lista null retorna 0
    void decodificacionConListaNullRetornaCero() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        double valor = gestor.decodificar(null);
        
        assertEquals(0.0, valor, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación de frecuencia máxima retorna 1
    void decodificacionDeFrecuenciaMaximaRetornaUno() {
        GestorDecodificacion gestor = new GestorDecodificacion(1000, 100.0);
        
        // Generar spikes a frecuencia máxima (100 Hz durante 1 segundo = 100 spikes)
        List<EventoSpike> spikes = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            spikes.add(new EventoSpike(1L, 0, 0, i * 10, -55.0));
        }
        
        double valor = gestor.decodificar(spikes);
        
        assertEquals(1.0, valor, 0.01);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación de frecuencia media
    void decodificacionDeFrecuenciaMediaRetornaValorMedio() {
        GestorDecodificacion gestor = new GestorDecodificacion(1000, 100.0);
        
        // Generar spikes a 50 Hz (mitad de la frecuencia máxima)
        List<EventoSpike> spikes = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            spikes.add(new EventoSpike(1L, 0, 0, i * 20, -55.0));
        }
        
        double valor = gestor.decodificar(spikes);
        
        assertEquals(0.5, valor, 0.01);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Valor decodificado se clampea a [0, 1]
    void valorDecodificadoSeClampea() {
        GestorDecodificacion gestor = new GestorDecodificacion(100, 100.0);
        
        // Generar más spikes de los que corresponden a frecuencia máxima
        List<EventoSpike> spikes = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            spikes.add(new EventoSpike(1L, 0, 0, i, -55.0));
        }
        
        double valor = gestor.decodificar(spikes);
        
        // Debe estar clampeado a 1.0
        assertEquals(1.0, valor, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Conteo de spikes en ventana vacía
    void conteoEnVentanaVaciaRetornaCero() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        int conteo = gestor.contarSpikesEnVentana(1L, 100);
        
        assertEquals(0, conteo);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Actualización y conteo de ventana
    void actualizacionYConteoDeVentana() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        // Agregar spikes a la ventana
        gestor.actualizarVentana(1L, 100);
        gestor.actualizarVentana(1L, 110);
        gestor.actualizarVentana(1L, 120);
        
        int conteo = gestor.contarSpikesEnVentana(1L, 130);
        
        // Todos los spikes están dentro de la ventana [80, 130]
        assertEquals(3, conteo);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Ventana deslizante limpia spikes antiguos
    void ventanaDeslizanteLimpiaSpikesAntiguos() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        // Agregar spikes
        gestor.actualizarVentana(1L, 100);
        gestor.actualizarVentana(1L, 110);
        gestor.actualizarVentana(1L, 160);
        gestor.actualizarVentana(1L, 170);
        
        // Contar en timestamp 180 (ventana [130, 180])
        int conteo = gestor.contarSpikesEnVentana(1L, 180);
        
        // Solo los spikes en 160 y 170 están en la ventana
        assertEquals(2, conteo);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Ventanas independientes por neurona
    void ventanasIndependientesPorNeurona() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        // Agregar spikes a diferentes neuronas
        gestor.actualizarVentana(1L, 100);
        gestor.actualizarVentana(1L, 110);
        gestor.actualizarVentana(2L, 100);
        
        int conteo1 = gestor.contarSpikesEnVentana(1L, 130);
        int conteo2 = gestor.contarSpikesEnVentana(2L, 130);
        
        assertEquals(2, conteo1);
        assertEquals(1, conteo2);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación de capa vacía
    void decodificacionDeCapaVaciaRetornaArrayVacio() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        double[] valores = gestor.decodificarCapa(Collections.emptyList());
        
        assertEquals(0, valores.length);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación de capa con null
    void decodificacionDeCapaConNullRetornaArrayVacio() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        double[] valores = gestor.decodificarCapa(null);
        
        assertEquals(0, valores.length);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación de capa con múltiples neuronas
    void decodificacionDeCapaConMultiplesNeuronas() {
        GestorDecodificacion gestor = new GestorDecodificacion(1000, 100.0);
        
        // Crear spikes para 3 neuronas con diferentes frecuencias
        List<List<EventoSpike>> spikesPorNeurona = new ArrayList<>();
        
        // Neurona 0: 100 spikes (frecuencia máxima)
        List<EventoSpike> spikes0 = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            spikes0.add(new EventoSpike(0L, 0, 0, i * 10, -55.0));
        }
        spikesPorNeurona.add(spikes0);
        
        // Neurona 1: 50 spikes (mitad de frecuencia)
        List<EventoSpike> spikes1 = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            spikes1.add(new EventoSpike(1L, 0, 1, i * 20, -55.0));
        }
        spikesPorNeurona.add(spikes1);
        
        // Neurona 2: 0 spikes
        spikesPorNeurona.add(new ArrayList<>());
        
        double[] valores = gestor.decodificarCapa(spikesPorNeurona);
        
        assertEquals(3, valores.length);
        assertEquals(1.0, valores[0], 0.01);
        assertEquals(0.5, valores[1], 0.01);
        assertEquals(0.0, valores[2], 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Limpieza de ventanas
    void limpiezaDeVentanas() {
        GestorDecodificacion gestor = new GestorDecodificacion(50, 100.0);
        
        // Agregar spikes
        gestor.actualizarVentana(1L, 100);
        gestor.actualizarVentana(2L, 110);
        
        // Verificar que hay spikes
        assertTrue(gestor.contarSpikesEnVentana(1L, 130) > 0);
        
        // Limpiar ventanas
        gestor.limpiarVentanas();
        
        // Verificar que las ventanas están vacías
        assertEquals(0, gestor.contarSpikesEnVentana(1L, 130));
        assertEquals(0, gestor.contarSpikesEnVentana(2L, 130));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación con ventana pequeña
    void decodificacionConVentanaPequena() {
        GestorDecodificacion gestor = new GestorDecodificacion(10, 100.0);
        
        // Generar 10 spikes en 10 timesteps (1000 Hz efectivo)
        List<EventoSpike> spikes = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            spikes.add(new EventoSpike(1L, 0, 0, i, -55.0));
        }
        
        double valor = gestor.decodificar(spikes);
        
        // 10 spikes / (100 Hz * 0.01 s) = 10 / 1 = 10, clampeado a 1.0
        assertEquals(1.0, valor, 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Decodificación con ventana grande
    void decodificacionConVentanaGrande() {
        GestorDecodificacion gestor = new GestorDecodificacion(10000, 100.0);
        
        // Generar 100 spikes en 10 segundos (10 Hz)
        List<EventoSpike> spikes = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            spikes.add(new EventoSpike(1L, 0, 0, i * 100, -55.0));
        }
        
        double valor = gestor.decodificar(spikes);
        
        // 100 spikes / (100 Hz * 10 s) = 100 / 1000 = 0.1
        assertEquals(0.1, valor, 0.01);
    }
}
