package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.ColaEventos;
import es.jastxz.nn.spiking.EventoSpike;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase ColaEventos.
 * 
 * <p>Verifica el comportamiento de la cola de eventos de spikes, incluyendo
 * encolado, extracción por timestamp, y operaciones de consulta.</p>
 */
class ColaEventosTest {
    
    private ColaEventos cola;
    
    @BeforeEach
    void setUp() {
        cola = new ColaEventos();
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Cola vacía
    void colaRecienCreadaEstaVacia() {
        assertFalse(cola.hayEventosPendientes());
        assertEquals(0, cola.tamaño());
        assertEquals(-1, cola.proximoTimestamp());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Encolar evento único
    void encolarEventoUnico() {
        EventoSpike evento = new EventoSpike(1L, 0, 0, 100, -55.0);
        cola.encolar(evento);
        
        assertTrue(cola.hayEventosPendientes());
        assertEquals(1, cola.tamaño());
        assertEquals(100, cola.proximoTimestamp());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Encolar evento null falla
    void encolarEventoNullFalla() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            cola.encolar(null);
        });
        assertTrue(exception.getMessage().contains("null"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener eventos de timestamp específico
    void obtenerEventosDeTimestampEspecifico() {
        // Encolar eventos con diferentes timestamps
        EventoSpike evento1 = new EventoSpike(1L, 0, 0, 100, -55.0);
        EventoSpike evento2 = new EventoSpike(2L, 0, 1, 102, -55.0);
        EventoSpike evento3 = new EventoSpike(3L, 1, 0, 100, -55.0);
        
        cola.encolar(evento1);
        cola.encolar(evento2);
        cola.encolar(evento3);
        
        // Obtener eventos del timestamp 100
        List<EventoSpike> eventos = cola.obtenerEventos(100);
        
        assertEquals(2, eventos.size());
        assertTrue(eventos.contains(evento1));
        assertTrue(eventos.contains(evento3));
        
        // Verificar que el evento del timestamp 102 sigue en la cola
        assertTrue(cola.hayEventosPendientes());
        assertEquals(1, cola.tamaño());
        assertEquals(102, cola.proximoTimestamp());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener eventos de timestamp sin eventos
    void obtenerEventosDeTimestampSinEventos() {
        EventoSpike evento = new EventoSpike(1L, 0, 0, 100, -55.0);
        cola.encolar(evento);
        
        // Intentar obtener eventos de un timestamp diferente
        List<EventoSpike> eventos = cola.obtenerEventos(50);
        
        assertTrue(eventos.isEmpty());
        
        // El evento original sigue en la cola
        assertTrue(cola.hayEventosPendientes());
        assertEquals(1, cola.tamaño());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Eventos se procesan en orden temporal
    void eventosSeExtraenEnOrdenTemporal() {
        // Encolar eventos en orden no temporal
        EventoSpike evento1 = new EventoSpike(1L, 0, 0, 105, -55.0);
        EventoSpike evento2 = new EventoSpike(2L, 0, 1, 100, -55.0);
        EventoSpike evento3 = new EventoSpike(3L, 1, 0, 103, -55.0);
        
        cola.encolar(evento1);
        cola.encolar(evento2);
        cola.encolar(evento3);
        
        // Verificar que el próximo timestamp es el menor
        assertEquals(100, cola.proximoTimestamp());
        
        // Extraer en orden temporal
        List<EventoSpike> eventos100 = cola.obtenerEventos(100);
        assertEquals(1, eventos100.size());
        assertEquals(evento2, eventos100.get(0));
        
        assertEquals(103, cola.proximoTimestamp());
        
        List<EventoSpike> eventos103 = cola.obtenerEventos(103);
        assertEquals(1, eventos103.size());
        assertEquals(evento3, eventos103.get(0));
        
        assertEquals(105, cola.proximoTimestamp());
        
        List<EventoSpike> eventos105 = cola.obtenerEventos(105);
        assertEquals(1, eventos105.size());
        assertEquals(evento1, eventos105.get(0));
        
        assertFalse(cola.hayEventosPendientes());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Múltiples eventos en mismo timestamp
    void multiplesEventosEnMismoTimestamp() {
        // Encolar varios eventos con el mismo timestamp
        EventoSpike evento1 = new EventoSpike(1L, 0, 0, 100, -55.0);
        EventoSpike evento2 = new EventoSpike(2L, 0, 1, 100, -56.0);
        EventoSpike evento3 = new EventoSpike(3L, 1, 0, 100, -57.0);
        EventoSpike evento4 = new EventoSpike(4L, 1, 1, 100, -58.0);
        
        cola.encolar(evento1);
        cola.encolar(evento2);
        cola.encolar(evento3);
        cola.encolar(evento4);
        
        // Extraer todos los eventos del timestamp 100
        List<EventoSpike> eventos = cola.obtenerEventos(100);
        
        assertEquals(4, eventos.size());
        assertTrue(eventos.contains(evento1));
        assertTrue(eventos.contains(evento2));
        assertTrue(eventos.contains(evento3));
        assertTrue(eventos.contains(evento4));
        assertFalse(cola.hayEventosPendientes());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Limpiar cola
    void limpiarColaEliminaTodosLosEventos() {
        // Encolar varios eventos
        cola.encolar(new EventoSpike(1L, 0, 0, 100, -55.0));
        cola.encolar(new EventoSpike(2L, 0, 1, 102, -55.0));
        cola.encolar(new EventoSpike(3L, 1, 0, 105, -55.0));
        
        assertEquals(3, cola.tamaño());
        
        // Limpiar
        cola.limpiar();
        
        assertFalse(cola.hayEventosPendientes());
        assertEquals(0, cola.tamaño());
        assertEquals(-1, cola.proximoTimestamp());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener eventos extrae de la cola
    void obtenerEventosExtraeDeLaCola() {
        EventoSpike evento = new EventoSpike(1L, 0, 0, 100, -55.0);
        cola.encolar(evento);
        
        assertEquals(1, cola.tamaño());
        
        // Obtener eventos
        List<EventoSpike> eventos = cola.obtenerEventos(100);
        
        assertEquals(1, eventos.size());
        assertEquals(0, cola.tamaño());
        assertFalse(cola.hayEventosPendientes());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Simulación de retardo sináptico
    void simulacionDeRetardoSinaptico() {
        // Simular spike en t=100 con retardo de 5 timesteps
        long timestampOrigen = 100;
        int retardo = 5;
        long timestampEntrega = timestampOrigen + retardo;
        
        EventoSpike evento = new EventoSpike(1L, 0, 0, timestampEntrega, -55.0);
        cola.encolar(evento);
        
        // En timesteps 100-104, no hay eventos
        for (long t = timestampOrigen; t < timestampEntrega; t++) {
            List<EventoSpike> eventos = cola.obtenerEventos(t);
            assertTrue(eventos.isEmpty());
        }
        
        // En timestep 105, el evento está disponible
        List<EventoSpike> eventos = cola.obtenerEventos(timestampEntrega);
        assertEquals(1, eventos.size());
        assertEquals(evento, eventos.get(0));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Cola con muchos eventos
    void colaConMuchosEventos() {
        // Encolar 1000 eventos con timestamps aleatorios
        for (int i = 0; i < 1000; i++) {
            long timestamp = (long) (Math.random() * 100);
            cola.encolar(new EventoSpike(i, 0, i, timestamp, -55.0));
        }
        
        assertEquals(1000, cola.tamaño());
        
        // Extraer todos los eventos en orden temporal
        int eventosExtraidos = 0;
        long timestampAnterior = -1;
        
        for (long t = 0; t <= 100; t++) {
            List<EventoSpike> eventos = cola.obtenerEventos(t);
            eventosExtraidos += eventos.size();
            
            // Verificar que todos los eventos tienen el timestamp correcto
            for (EventoSpike evento : eventos) {
                assertEquals(t, evento.getTimestamp());
            }
            
            // Verificar orden temporal
            if (!eventos.isEmpty()) {
                assertTrue(t >= timestampAnterior);
                timestampAnterior = t;
            }
        }
        
        assertEquals(1000, eventosExtraidos);
        assertFalse(cola.hayEventosPendientes());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: ToString proporciona información útil
    void toStringProporcionaInformacionUtil() {
        cola.encolar(new EventoSpike(1L, 0, 0, 100, -55.0));
        cola.encolar(new EventoSpike(2L, 0, 1, 105, -55.0));
        
        String str = cola.toString();
        
        assertTrue(str.contains("2")); // tamaño
        assertTrue(str.contains("100")); // próximo timestamp
    }
}
