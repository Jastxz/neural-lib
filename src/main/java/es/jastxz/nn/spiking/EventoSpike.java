package es.jastxz.nn.spiking;

import java.io.Serializable;

/**
 * Representa un evento discreto de disparo neuronal (spike) en una red neuronal de spikes.
 * 
 * <p>Un spike es un evento que ocurre cuando el potencial de membrana de una neurona
 * alcanza el umbral de disparo. Este evento incluye información sobre la neurona que
 * disparó, el momento temporal del disparo, y el estado de la neurona en ese momento.</p>
 * 
 * <p>Los eventos de spike son comparables por timestamp, lo que permite ordenarlos
 * temporalmente en colas de eventos para procesamiento eficiente de retardos sinápticos.</p>
 * 
 * @see NeuronaSpiking
 * @see ColaEventos
 */
public class EventoSpike implements Comparable<EventoSpike>, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * Identificador único de la neurona que generó el spike.
     */
    private final long neuronaId;
    
    /**
     * Índice de la capa a la que pertenece la neurona.
     */
    private final int capa;
    
    /**
     * Índice de la neurona dentro de su capa.
     */
    private final int indice;
    
    /**
     * Timestamp del evento en timesteps de simulación.
     */
    private final long timestamp;
    
    /**
     * Potencial de membrana de la neurona en el momento del disparo (en mV).
     */
    private final double potencialMembrana;
    
    /**
     * Construye un nuevo evento de spike.
     * 
     * @param neuronaId identificador único de la neurona
     * @param capa índice de la capa (debe ser >= 0)
     * @param indice índice de la neurona dentro de su capa (debe ser >= 0)
     * @param timestamp momento temporal del spike (debe ser >= 0)
     * @param potencialMembrana potencial de membrana en el momento del disparo
     * @throws IllegalArgumentException si capa, indice o timestamp son negativos
     */
    public EventoSpike(long neuronaId, int capa, int indice, long timestamp, double potencialMembrana) {
        if (capa < 0) {
            throw new IllegalArgumentException("El índice de capa no puede ser negativo: " + capa);
        }
        if (indice < 0) {
            throw new IllegalArgumentException("El índice de neurona no puede ser negativo: " + indice);
        }
        if (timestamp < 0) {
            throw new IllegalArgumentException("El timestamp no puede ser negativo: " + timestamp);
        }
        
        this.neuronaId = neuronaId;
        this.capa = capa;
        this.indice = indice;
        this.timestamp = timestamp;
        this.potencialMembrana = potencialMembrana;
    }
    
    /**
     * Compara este evento con otro basándose en el timestamp.
     * 
     * <p>Los eventos con timestamp menor se consideran "menores" y se procesarán primero
     * en una cola de prioridad. Esto permite procesar eventos en orden temporal correcto.</p>
     * 
     * @param otro el evento a comparar
     * @return un valor negativo si este evento es anterior, cero si son simultáneos,
     *         o un valor positivo si este evento es posterior
     */
    @Override
    public int compareTo(EventoSpike otro) {
        return Long.compare(this.timestamp, otro.timestamp);
    }
    
    /**
     * Retorna el identificador único de la neurona que generó el spike.
     * 
     * @return el ID de la neurona
     */
    public long getNeuronaId() {
        return neuronaId;
    }
    
    /**
     * Retorna el índice de la capa a la que pertenece la neurona.
     * 
     * @return el índice de capa
     */
    public int getCapa() {
        return capa;
    }
    
    /**
     * Retorna el índice de la neurona dentro de su capa.
     * 
     * @return el índice de la neurona
     */
    public int getIndice() {
        return indice;
    }
    
    /**
     * Retorna el timestamp del evento en timesteps de simulación.
     * 
     * @return el timestamp del spike
     */
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * Retorna el potencial de membrana de la neurona en el momento del disparo.
     * 
     * @return el potencial de membrana en mV
     */
    public double getPotencialMembrana() {
        return potencialMembrana;
    }
    
    @Override
    public String toString() {
        return String.format("EventoSpike[neurona=%d, capa=%d, indice=%d, t=%d, V=%.2f mV]",
                neuronaId, capa, indice, timestamp, potencialMembrana);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        EventoSpike otro = (EventoSpike) obj;
        return neuronaId == otro.neuronaId &&
               capa == otro.capa &&
               indice == otro.indice &&
               timestamp == otro.timestamp &&
               Double.compare(potencialMembrana, otro.potencialMembrana) == 0;
    }
    
    @Override
    public int hashCode() {
        int result = Long.hashCode(neuronaId);
        result = 31 * result + capa;
        result = 31 * result + indice;
        result = 31 * result + Long.hashCode(timestamp);
        result = 31 * result + Double.hashCode(potencialMembrana);
        return result;
    }
}
