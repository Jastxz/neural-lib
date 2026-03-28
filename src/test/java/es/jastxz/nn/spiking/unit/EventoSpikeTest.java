package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.EventoSpike;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase EventoSpike.
 * 
 * Feature: spiking-neural-network
 */
class EventoSpikeTest {
    
    @Test
    // Feature: spiking-neural-network, Example: Construcción básica de EventoSpike
    void construccionBasica() {
        EventoSpike evento = new EventoSpike(1L, 0, 5, 100L, -55.0);
        
        assertEquals(1L, evento.getNeuronaId());
        assertEquals(0, evento.getCapa());
        assertEquals(5, evento.getIndice());
        assertEquals(100L, evento.getTimestamp());
        assertEquals(-55.0, evento.getPotencialMembrana(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de capa negativa
    void capaNoDebeSerNegativa() {
        assertThrows(IllegalArgumentException.class, () -> {
            new EventoSpike(1L, -1, 0, 100L, -55.0);
        });
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de índice negativo
    void indiceNoDebeSerNegativo() {
        assertThrows(IllegalArgumentException.class, () -> {
            new EventoSpike(1L, 0, -1, 100L, -55.0);
        });
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Validación de timestamp negativo
    void timestampNoDebeSerNegativo() {
        assertThrows(IllegalArgumentException.class, () -> {
            new EventoSpike(1L, 0, 0, -1L, -55.0);
        });
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Comparación por timestamp
    void comparacionPorTimestamp() {
        EventoSpike evento1 = new EventoSpike(1L, 0, 0, 100L, -55.0);
        EventoSpike evento2 = new EventoSpike(2L, 1, 3, 150L, -60.0);
        EventoSpike evento3 = new EventoSpike(3L, 0, 2, 100L, -58.0);
        
        // evento1 debe ser menor que evento2 (timestamp 100 < 150)
        assertTrue(evento1.compareTo(evento2) < 0);
        
        // evento2 debe ser mayor que evento1
        assertTrue(evento2.compareTo(evento1) > 0);
        
        // evento1 y evento3 tienen el mismo timestamp, deben ser iguales en comparación
        assertEquals(0, evento1.compareTo(evento3));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Ordenamiento de eventos por timestamp
    void ordenamientoPorTimestamp() {
        List<EventoSpike> eventos = new ArrayList<>();
        eventos.add(new EventoSpike(1L, 0, 0, 150L, -55.0));
        eventos.add(new EventoSpike(2L, 0, 1, 50L, -60.0));
        eventos.add(new EventoSpike(3L, 0, 2, 100L, -58.0));
        eventos.add(new EventoSpike(4L, 1, 0, 75L, -62.0));
        
        Collections.sort(eventos);
        
        // Verificar que están ordenados por timestamp
        assertEquals(50L, eventos.get(0).getTimestamp());
        assertEquals(75L, eventos.get(1).getTimestamp());
        assertEquals(100L, eventos.get(2).getTimestamp());
        assertEquals(150L, eventos.get(3).getTimestamp());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Igualdad de eventos
    void igualdadDeEventos() {
        EventoSpike evento1 = new EventoSpike(1L, 0, 5, 100L, -55.0);
        EventoSpike evento2 = new EventoSpike(1L, 0, 5, 100L, -55.0);
        EventoSpike evento3 = new EventoSpike(2L, 0, 5, 100L, -55.0);
        
        // Eventos con los mismos valores deben ser iguales
        assertEquals(evento1, evento2);
        assertEquals(evento1.hashCode(), evento2.hashCode());
        
        // Eventos con diferentes neuronaId no deben ser iguales
        assertNotEquals(evento1, evento3);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Serialización de EventoSpike
    void serializacion() throws IOException, ClassNotFoundException {
        EventoSpike eventoOriginal = new EventoSpike(42L, 2, 7, 1000L, -55.5);
        
        // Serializar
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(eventoOriginal);
        }
        
        // Deserializar
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        EventoSpike eventoDeserializado;
        try (ObjectInputStream ois = new ObjectInputStream(bais)) {
            eventoDeserializado = (EventoSpike) ois.readObject();
        }
        
        // Verificar que son iguales
        assertEquals(eventoOriginal, eventoDeserializado);
        assertEquals(eventoOriginal.getNeuronaId(), eventoDeserializado.getNeuronaId());
        assertEquals(eventoOriginal.getCapa(), eventoDeserializado.getCapa());
        assertEquals(eventoOriginal.getIndice(), eventoDeserializado.getIndice());
        assertEquals(eventoOriginal.getTimestamp(), eventoDeserializado.getTimestamp());
        assertEquals(eventoOriginal.getPotencialMembrana(), eventoDeserializado.getPotencialMembrana(), 0.001);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: toString produce formato legible
    void toStringProduceFormatoLegible() {
        EventoSpike evento = new EventoSpike(1L, 0, 5, 100L, -55.5);
        String str = evento.toString();
        
        // Verificar que contiene información clave
        assertTrue(str.contains("neurona=1"));
        assertTrue(str.contains("capa=0"));
        assertTrue(str.contains("indice=5"));
        assertTrue(str.contains("t=100"));
        // El formato decimal puede variar según locale (. o ,)
        assertTrue(str.contains("-55") && (str.contains("5") || str.contains("50")));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Eventos con timestamp cero
    void eventoConTimestampCero() {
        EventoSpike evento = new EventoSpike(1L, 0, 0, 0L, -70.0);
        
        assertEquals(0L, evento.getTimestamp());
        assertNotNull(evento);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Potencial de membrana puede ser cualquier valor
    void potencialMembranaAceptaCualquierValor() {
        // Potencial muy negativo
        EventoSpike evento1 = new EventoSpike(1L, 0, 0, 100L, -100.0);
        assertEquals(-100.0, evento1.getPotencialMembrana(), 0.001);
        
        // Potencial positivo (aunque inusual biológicamente)
        EventoSpike evento2 = new EventoSpike(2L, 0, 1, 100L, 50.0);
        assertEquals(50.0, evento2.getPotencialMembrana(), 0.001);
        
        // Potencial cero
        EventoSpike evento3 = new EventoSpike(3L, 0, 2, 100L, 0.0);
        assertEquals(0.0, evento3.getPotencialMembrana(), 0.001);
    }
}
