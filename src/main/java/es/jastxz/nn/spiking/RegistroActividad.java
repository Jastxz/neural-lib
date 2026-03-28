package es.jastxz.nn.spiking;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Registra y mantiene un historial completo de la actividad neuronal en una red de spikes.
 * 
 * <p>Esta clase mantiene un registro de todos los eventos de spike generados durante la
 * simulación, permitiendo análisis posterior, visualización y exportación de datos.
 * Los spikes se almacenan tanto en una lista cronológica completa como organizados
 * por neurona para consultas eficientes.</p>
 * 
 * <p>El registro es serializable para permitir persistencia junto con la red neuronal.</p>
 * 
 * @see EventoSpike
 * @see RedNeuralSpiking
 */
public class RegistroActividad implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * Lista cronológica de todos los spikes registrados.
     */
    private final List<EventoSpike> todosLosSpikes;
    
    /**
     * Mapa de spikes organizados por ID de neurona para consultas eficientes.
     */
    private final Map<Long, List<EventoSpike>> spikesPorNeurona;
    
    /**
     * Construye un nuevo registro de actividad vacío.
     */
    public RegistroActividad() {
        this.todosLosSpikes = new ArrayList<>();
        this.spikesPorNeurona = new HashMap<>();
    }
    
    /**
     * Registra un nuevo evento de spike en el historial.
     * 
     * <p>El spike se agrega tanto a la lista cronológica completa como al
     * historial específico de la neurona que lo generó.</p>
     * 
     * @param evento el evento de spike a registrar
     * @throws IllegalArgumentException si el evento es null
     */
    public void registrar(EventoSpike evento) {
        if (evento == null) {
            throw new IllegalArgumentException("El evento no puede ser null");
        }
        
        // Agregar a la lista completa
        todosLosSpikes.add(evento);
        
        // Agregar al historial de la neurona
        spikesPorNeurona
            .computeIfAbsent(evento.getNeuronaId(), k -> new ArrayList<>())
            .add(evento);
    }
    
    /**
     * Obtiene todos los spikes registrados de una neurona específica.
     * 
     * @param neuronaId el identificador de la neurona
     * @return lista de spikes de la neurona (vacía si no hay spikes registrados)
     */
    public List<EventoSpike> obtenerSpikes(long neuronaId) {
        return spikesPorNeurona.getOrDefault(neuronaId, new ArrayList<>());
    }
    
    /**
     * Obtiene todos los spikes registrados en el sistema.
     * 
     * @return lista cronológica de todos los spikes
     */
    public List<EventoSpike> obtenerTodosLosSpikes() {
        return new ArrayList<>(todosLosSpikes);
    }
    
    /**
     * Obtiene los spikes que ocurrieron dentro de un rango temporal específico.
     * 
     * @param timestampInicio timestamp inicial del rango (inclusivo)
     * @param timestampFin timestamp final del rango (inclusivo)
     * @return lista de spikes dentro del rango temporal
     * @throws IllegalArgumentException si timestampInicio > timestampFin
     */
    public List<EventoSpike> obtenerSpikesEnRango(long timestampInicio, long timestampFin) {
        if (timestampInicio > timestampFin) {
            throw new IllegalArgumentException(
                "El timestamp inicial (" + timestampInicio + 
                ") no puede ser mayor que el final (" + timestampFin + ")"
            );
        }
        
        return todosLosSpikes.stream()
            .filter(spike -> spike.getTimestamp() >= timestampInicio && 
                           spike.getTimestamp() <= timestampFin)
            .collect(Collectors.toList());
    }
    
    /**
     * Exporta el registro de actividad en formato CSV.
     * 
     * <p>El formato CSV incluye una línea de encabezado seguida de una línea por cada spike
     * con los campos: timestamp, capa, indice, potencial (en mV).</p>
     * 
     * <p>Ejemplo de salida:</p>
     * <pre>
     * timestamp,capa,indice,potencial
     * 10,0,0,-55.00
     * 15,0,1,-55.00
     * 20,1,0,-55.00
     * </pre>
     * 
     * @return string con el contenido CSV del registro
     */
    public String exportarCSV() {
        StringBuilder csv = new StringBuilder();
        
        // Encabezado
        csv.append("timestamp,capa,indice,potencial\n");
        
        // Datos
        for (EventoSpike spike : todosLosSpikes) {
            csv.append(spike.getTimestamp())
               .append(",")
               .append(spike.getCapa())
               .append(",")
               .append(spike.getIndice())
               .append(",")
               .append(String.format(java.util.Locale.US, "%.2f", spike.getPotencialMembrana()))
               .append("\n");
        }
        
        return csv.toString();
    }
    
    /**
     * Limpia todo el registro de actividad.
     * 
     * <p>Elimina todos los spikes registrados tanto de la lista completa como
     * de los historiales por neurona.</p>
     */
    public void limpiar() {
        todosLosSpikes.clear();
        spikesPorNeurona.clear();
    }
    
    /**
     * Retorna el número total de spikes registrados.
     * 
     * @return cantidad total de spikes
     */
    public int getTotalSpikes() {
        return todosLosSpikes.size();
    }
    
    /**
     * Retorna el número de neuronas que han generado al menos un spike.
     * 
     * @return cantidad de neuronas activas
     */
    public int getNeuronasActivas() {
        return spikesPorNeurona.size();
    }
    
    @Override
    public String toString() {
        return String.format("RegistroActividad[totalSpikes=%d, neuronasActivas=%d]",
                getTotalSpikes(), getNeuronasActivas());
    }
}
