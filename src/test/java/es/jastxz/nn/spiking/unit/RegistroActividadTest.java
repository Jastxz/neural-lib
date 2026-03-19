package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.EventoSpike;
import es.jastxz.nn.spiking.RegistroActividad;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la clase RegistroActividad.
 * 
 * Feature: spiking-neural-network
 */
class RegistroActividadTest {
    
    private RegistroActividad registro;
    
    @BeforeEach
    void setUp() {
        registro = new RegistroActividad();
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registro vacío inicial
    void registroVacioInicial() {
        assertEquals(0, registro.getTotalSpikes());
        assertEquals(0, registro.getNeuronasActivas());
        assertTrue(registro.obtenerTodosLosSpikes().isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registrar un spike
    void registrarUnSpike() {
        EventoSpike spike = new EventoSpike(1L, 0, 0, 10L, -55.0);
        
        registro.registrar(spike);
        
        assertEquals(1, registro.getTotalSpikes());
        assertEquals(1, registro.getNeuronasActivas());
        assertEquals(1, registro.obtenerTodosLosSpikes().size());
        assertEquals(spike, registro.obtenerTodosLosSpikes().get(0));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registrar múltiples spikes
    void registrarMultiplesSpikes() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        EventoSpike spike3 = new EventoSpike(1L, 0, 0, 20L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        assertEquals(3, registro.getTotalSpikes());
        assertEquals(2, registro.getNeuronasActivas());
        
        List<EventoSpike> todos = registro.obtenerTodosLosSpikes();
        assertEquals(3, todos.size());
        assertEquals(spike1, todos.get(0));
        assertEquals(spike2, todos.get(1));
        assertEquals(spike3, todos.get(2));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registrar evento null falla
    void registrarEventoNullFalla() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, 
            () -> registro.registrar(null));
        assertTrue(exception.getMessage().contains("null"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener spikes por neurona
    void obtenerSpikesPorNeurona() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        EventoSpike spike3 = new EventoSpike(1L, 0, 0, 20L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        List<EventoSpike> spikesNeurona1 = registro.obtenerSpikes(1L);
        List<EventoSpike> spikesNeurona2 = registro.obtenerSpikes(2L);
        
        assertEquals(2, spikesNeurona1.size());
        assertEquals(spike1, spikesNeurona1.get(0));
        assertEquals(spike3, spikesNeurona1.get(1));
        
        assertEquals(1, spikesNeurona2.size());
        assertEquals(spike2, spikesNeurona2.get(0));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener spikes de neurona sin actividad
    void obtenerSpikesDeNeuronaSinActividad() {
        EventoSpike spike = new EventoSpike(1L, 0, 0, 10L, -55.0);
        registro.registrar(spike);
        
        List<EventoSpike> spikes = registro.obtenerSpikes(999L);
        
        assertTrue(spikes.isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener spikes en rango temporal
    void obtenerSpikesEnRango() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        EventoSpike spike3 = new EventoSpike(3L, 0, 2, 20L, -55.0);
        EventoSpike spike4 = new EventoSpike(4L, 1, 0, 25L, -55.0);
        EventoSpike spike5 = new EventoSpike(5L, 1, 1, 30L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        registro.registrar(spike4);
        registro.registrar(spike5);
        
        List<EventoSpike> spikesEnRango = registro.obtenerSpikesEnRango(15L, 25L);
        
        assertEquals(3, spikesEnRango.size());
        assertEquals(spike2, spikesEnRango.get(0));
        assertEquals(spike3, spikesEnRango.get(1));
        assertEquals(spike4, spikesEnRango.get(2));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener spikes en rango vacío
    void obtenerSpikesEnRangoVacio() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 20L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        
        List<EventoSpike> spikesEnRango = registro.obtenerSpikesEnRango(11L, 19L);
        
        assertTrue(spikesEnRango.isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Obtener spikes con rango de un solo timestamp
    void obtenerSpikesEnRangoUnSoloTimestamp() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        EventoSpike spike3 = new EventoSpike(3L, 0, 2, 15L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        List<EventoSpike> spikesEnRango = registro.obtenerSpikesEnRango(15L, 15L);
        
        assertEquals(2, spikesEnRango.size());
        assertEquals(spike2, spikesEnRango.get(0));
        assertEquals(spike3, spikesEnRango.get(1));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Rango inválido falla
    void rangoInvalidoFalla() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
            () -> registro.obtenerSpikesEnRango(20L, 10L));
        assertTrue(exception.getMessage().contains("inicial"));
        assertTrue(exception.getMessage().contains("final"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Exportar CSV vacío
    void exportarCSVVacio() {
        String csv = registro.exportarCSV();
        
        assertEquals("timestamp,capa,indice,potencial\n", csv);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Exportar CSV con spikes
    void exportarCSVConSpikes() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -54.5);
        EventoSpike spike3 = new EventoSpike(3L, 1, 0, 20L, -56.25);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        String csv = registro.exportarCSV();
        
        String expected = "timestamp,capa,indice,potencial\n" +
                         "10,0,0,-55.00\n" +
                         "15,0,1,-54.50\n" +
                         "20,1,0,-56.25\n";
        
        assertEquals(expected, csv);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Formato CSV correcto
    void formatoCSVCorrecto() {
        EventoSpike spike = new EventoSpike(1L, 2, 3, 100L, -55.123);
        registro.registrar(spike);
        
        String csv = registro.exportarCSV();
        String[] lines = csv.split("\n");
        
        assertEquals(2, lines.length);
        assertEquals("timestamp,capa,indice,potencial", lines[0]);
        assertEquals("100,2,3,-55.12", lines[1]);
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Limpiar registro
    void limpiarRegistro() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        
        assertEquals(2, registro.getTotalSpikes());
        
        registro.limpiar();
        
        assertEquals(0, registro.getTotalSpikes());
        assertEquals(0, registro.getNeuronasActivas());
        assertTrue(registro.obtenerTodosLosSpikes().isEmpty());
        assertTrue(registro.obtenerSpikes(1L).isEmpty());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Serialización de registro
    void serializacionDeRegistro() throws IOException, ClassNotFoundException {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        
        // Serializar
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(registro);
        }
        
        // Deserializar
        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        RegistroActividad registroCargado;
        try (ObjectInputStream ois = new ObjectInputStream(bais)) {
            registroCargado = (RegistroActividad) ois.readObject();
        }
        
        // Verificar
        assertEquals(2, registroCargado.getTotalSpikes());
        assertEquals(2, registroCargado.getNeuronasActivas());
        assertEquals(1, registroCargado.obtenerSpikes(1L).size());
        assertEquals(1, registroCargado.obtenerSpikes(2L).size());
    }
    
    @Test
    // Feature: spiking-neural-network, Example: ToString informativo
    void toStringInformativo() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 15L, -55.0);
        EventoSpike spike3 = new EventoSpike(1L, 0, 0, 20L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        String str = registro.toString();
        
        assertTrue(str.contains("totalSpikes=3"));
        assertTrue(str.contains("neuronasActivas=2"));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Registro mantiene orden cronológico
    void registroMantieneOrdenCronologico() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(2L, 0, 1, 5L, -55.0);
        EventoSpike spike3 = new EventoSpike(3L, 0, 2, 15L, -55.0);
        
        // Registrar en orden no cronológico
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        List<EventoSpike> todos = registro.obtenerTodosLosSpikes();
        
        // Debe mantener el orden de inserción
        assertEquals(3, todos.size());
        assertEquals(spike1, todos.get(0));
        assertEquals(spike2, todos.get(1));
        assertEquals(spike3, todos.get(2));
    }
    
    @Test
    // Feature: spiking-neural-network, Example: Múltiples spikes de misma neurona
    void multipleSpikesDeMismaNeurona() {
        EventoSpike spike1 = new EventoSpike(1L, 0, 0, 10L, -55.0);
        EventoSpike spike2 = new EventoSpike(1L, 0, 0, 20L, -55.0);
        EventoSpike spike3 = new EventoSpike(1L, 0, 0, 30L, -55.0);
        
        registro.registrar(spike1);
        registro.registrar(spike2);
        registro.registrar(spike3);
        
        assertEquals(3, registro.getTotalSpikes());
        assertEquals(1, registro.getNeuronasActivas());
        
        List<EventoSpike> spikesNeurona = registro.obtenerSpikes(1L);
        assertEquals(3, spikesNeurona.size());
        assertEquals(spike1, spikesNeurona.get(0));
        assertEquals(spike2, spikesNeurona.get(1));
        assertEquals(spike3, spikesNeurona.get(2));
    }
}
