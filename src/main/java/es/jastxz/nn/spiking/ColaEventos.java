package es.jastxz.nn.spiking;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Cola de eventos de spikes ordenada por timestamp para procesamiento eficiente
 * de retardos sinápticos en redes neuronales de spikes.
 * 
 * <p>Esta clase mantiene una cola de prioridad de eventos de spike ordenados
 * temporalmente. Los eventos se encolan con su timestamp de entrega y se extraen
 * cuando llega el momento de procesarlos. Esto permite implementar retardos
 * sinápticos de manera eficiente sin necesidad de iterar sobre todos los eventos
 * en cada timestep.</p>
 * 
 * <p>La cola utiliza internamente una {@link PriorityQueue} que mantiene los eventos
 * ordenados por timestamp, permitiendo operaciones de encolado en O(log n) y
 * extracción de eventos en O(k log n) donde k es el número de eventos en el
 * timestamp actual.</p>
 * 
 * <h3>Ejemplo de uso:</h3>
 * <pre>{@code
 * ColaEventos cola = new ColaEventos();
 * 
 * // Encolar eventos con diferentes timestamps
 * cola.encolar(new EventoSpike(1L, 0, 0, 100, -55.0));
 * cola.encolar(new EventoSpike(2L, 0, 1, 102, -55.0));
 * cola.encolar(new EventoSpike(3L, 1, 0, 100, -55.0));
 * 
 * // Extraer todos los eventos del timestamp 100
 * List<EventoSpike> eventos = cola.obtenerEventos(100);
 * // eventos contiene los spikes de neuronas 1 y 3
 * 
 * // Verificar si hay eventos pendientes
 * boolean hayMas = cola.hayEventosPendientes(); // true (queda el evento en t=102)
 * }</pre>
 * 
 * @see EventoSpike
 * @see SinapsisSpiking
 */
public class ColaEventos implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Cola de prioridad interna que mantiene los eventos ordenados por timestamp.
     * Los eventos con timestamp menor tienen mayor prioridad y se procesan primero.
     */
    private final PriorityQueue<EventoSpike> cola;
    
    /**
     * Construye una nueva cola de eventos vacía.
     * 
     * <p>La cola se inicializa con capacidad por defecto y ordenamiento natural
     * de EventoSpike (por timestamp).</p>
     */
    public ColaEventos() {
        this.cola = new PriorityQueue<>();
    }
    
    /**
     * Encola un evento de spike para procesamiento futuro.
     * 
     * <p>El evento se inserta en la cola manteniendo el orden por timestamp.
     * Esta operación tiene complejidad O(log n) donde n es el número de eventos
     * en la cola.</p>
     * 
     * @param evento el evento de spike a encolar (no puede ser null)
     * @throws IllegalArgumentException si el evento es null
     */
    public void encolar(EventoSpike evento) {
        if (evento == null) {
            throw new IllegalArgumentException("El evento no puede ser null");
        }
        cola.add(evento);
    }
    
    /**
     * Extrae y retorna todos los eventos que corresponden al timestamp especificado.
     * 
     * <p>Este método extrae de la cola todos los eventos cuyo timestamp coincide
     * exactamente con el timestamp proporcionado. Los eventos se extraen en orden
     * de inserción para eventos con el mismo timestamp.</p>
     * 
     * <p>Si no hay eventos para el timestamp especificado, se retorna una lista vacía.
     * Los eventos extraídos se eliminan de la cola.</p>
     * 
     * <p>Complejidad: O(k log n) donde k es el número de eventos en el timestamp
     * y n es el tamaño total de la cola.</p>
     * 
     * @param timestamp el timestamp para el cual extraer eventos
     * @return lista de eventos con el timestamp especificado (nunca null, puede estar vacía)
     */
    public List<EventoSpike> obtenerEventos(long timestamp) {
        List<EventoSpike> eventos = new ArrayList<>();
        
        // Extraer todos los eventos que coinciden con el timestamp
        while (!cola.isEmpty() && cola.peek().getTimestamp() == timestamp) {
            eventos.add(cola.poll());
        }
        
        return eventos;
    }
    
    /**
     * Verifica si hay eventos pendientes en la cola.
     * 
     * @return true si la cola contiene al menos un evento, false si está vacía
     */
    public boolean hayEventosPendientes() {
        return !cola.isEmpty();
    }
    
    /**
     * Elimina todos los eventos de la cola.
     * 
     * <p>Después de llamar a este método, la cola queda vacía y
     * {@link #hayEventosPendientes()} retornará false.</p>
     */
    public void limpiar() {
        cola.clear();
    }
    
    /**
     * Retorna el número de eventos actualmente en la cola.
     * 
     * @return el tamaño de la cola
     */
    public int tamaño() {
        return cola.size();
    }
    
    /**
     * Retorna el timestamp del próximo evento a procesar sin extraerlo de la cola.
     * 
     * @return el timestamp del próximo evento, o -1 si la cola está vacía
     */
    public long proximoTimestamp() {
        if (cola.isEmpty()) {
            return -1;
        }
        return cola.peek().getTimestamp();
    }
    
    @Override
    public String toString() {
        return String.format("ColaEventos[tamaño=%d, próximo=%d]", 
                tamaño(), proximoTimestamp());
    }
}
