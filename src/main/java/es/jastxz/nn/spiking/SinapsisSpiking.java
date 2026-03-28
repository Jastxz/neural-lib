package es.jastxz.nn.spiking;

import java.io.Serializable;

/**
 * Representa una conexión sináptica entre dos neuronas spiking con peso y retardo.
 * 
 * <p>Una sinapsis transmite spikes desde una neurona presinaptica a una neurona
 * postsinaptica con un peso configurable y un retardo temporal. El peso determina
 * la magnitud de la señal transmitida y puede ser modificado por mecanismos de
 * plasticidad como STDP (Spike-Timing-Dependent Plasticity).</p>
 * 
 * <p>Características principales:</p>
 * <ul>
 *   <li>Transmisión de spikes con retardo configurable</li>
 *   <li>Peso sináptico mutable (para aprendizaje STDP)</li>
 *   <li>Límites configurables para el peso (peso_min, peso_max)</li>
 *   <li>Registro de timestamps de spikes pre y post para STDP</li>
 * </ul>
 * 
 * <h3>Ejemplo de uso:</h3>
 * <pre>{@code
 * NeuronaSpiking pre = new NeuronaSpiking(1L, 0, 0, -55.0, -70.0, 20.0, 2);
 * NeuronaSpiking post = new NeuronaSpiking(2L, 1, 0, -55.0, -70.0, 20.0, 2);
 * 
 * // Crear sinapsis con peso 0.5, retardo 2 timesteps, límites [0.0, 1.0]
 * SinapsisSpiking sinapsis = new SinapsisSpiking(pre, post, 0.5, 2, 0.0, 1.0);
 * 
 * // Propagar spike desde neurona presinaptica
 * EventoSpike evento = sinapsis.propagarSpike(100);
 * // El evento se entregará en timestamp 102 (100 + retardo 2)
 * 
 * // Ajustar peso (por ejemplo, por STDP)
 * sinapsis.ajustarPeso(0.01); // Incrementa peso en 0.01
 * }</pre>
 * 
 * @see NeuronaSpiking
 * @see EventoSpike
 * @see GestorSTDP
 */
public class SinapsisSpiking implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /**
     * Neurona presinaptica (origen del spike).
     */
    private final NeuronaSpiking presinaptica;
    
    /**
     * Neurona postsinaptica (destino del spike).
     */
    private final NeuronaSpiking postsinaptica;
    
    /**
     * Peso sináptico (mutable por STDP).
     * Determina la magnitud de la señal transmitida.
     */
    private double peso;
    
    /**
     * Retardo sináptico en timesteps.
     * Tiempo que tarda un spike en propagarse desde pre a post.
     */
    private final int retardo;
    
    /**
     * Límite inferior del peso sináptico.
     */
    private final double pesoMin;
    
    /**
     * Límite superior del peso sináptico.
     */
    private final double pesoMax;
    
    /**
     * Timestamp del último spike presinaptico (para STDP).
     * Null si la neurona presinaptica no ha disparado aún.
     */
    private Long timestampUltimoSpikePresinaptico;
    
    /**
     * Timestamp del último spike postsinaptico (para STDP).
     * Null si la neurona postsinaptica no ha disparado aún.
     */
    private Long timestampUltimoSpikePostsinaptico;
    
    /**
     * Construye una nueva sinapsis spiking con los parámetros especificados.
     * 
     * @param presinaptica neurona origen del spike (no puede ser null)
     * @param postsinaptica neurona destino del spike (no puede ser null)
     * @param peso peso sináptico inicial
     * @param retardo retardo de transmisión en timesteps (debe ser >= 0)
     * @param pesoMin límite inferior del peso sináptico
     * @param pesoMax límite superior del peso sináptico
     * @throws IllegalArgumentException si presinaptica o postsinaptica son null
     * @throws IllegalArgumentException si retardo < 0
     * @throws IllegalArgumentException si pesoMin >= pesoMax
     * @throws IllegalArgumentException si peso no está en [pesoMin, pesoMax]
     */
    public SinapsisSpiking(NeuronaSpiking presinaptica, NeuronaSpiking postsinaptica,
                          double peso, int retardo, double pesoMin, double pesoMax) {
        // Validación: neuronas no pueden ser null (Requisito 15.3)
        if (presinaptica == null) {
            throw new IllegalArgumentException("La neurona presinaptica no puede ser null");
        }
        if (postsinaptica == null) {
            throw new IllegalArgumentException("La neurona postsinaptica no puede ser null");
        }
        
        // Validación: retardo >= 0 (Requisito 8.1, 8.5)
        if (retardo < 0) {
            throw new IllegalArgumentException(
                "El retardo sináptico no puede ser negativo: " + retardo
            );
        }
        
        // Validación: pesoMin < pesoMax
        if (pesoMin >= pesoMax) {
            throw new IllegalArgumentException(
                "peso_min (" + pesoMin + ") debe ser menor que peso_max (" + pesoMax + ")"
            );
        }
        
        // Validación: peso debe estar en [pesoMin, pesoMax] (Requisito 5.6)
        if (peso < pesoMin || peso > pesoMax) {
            throw new IllegalArgumentException(
                "El peso inicial (" + peso + ") debe estar en el rango [" + pesoMin + ", " + pesoMax + "]"
            );
        }
        
        // Asignar campos
        this.presinaptica = presinaptica;
        this.postsinaptica = postsinaptica;
        this.peso = peso;
        this.retardo = retardo;
        this.pesoMin = pesoMin;
        this.pesoMax = pesoMax;
        
        // Inicializar timestamps de STDP como null (no han disparado aún)
        this.timestampUltimoSpikePresinaptico = null;
        this.timestampUltimoSpikePostsinaptico = null;
    }
    
    /**
     * Retorna la neurona presinaptica (origen del spike).
     * 
     * @return la neurona presinaptica
     */
    public NeuronaSpiking getPresinaptica() {
        return presinaptica;
    }
    
    /**
     * Retorna la neurona postsinaptica (destino del spike).
     * 
     * @return la neurona postsinaptica
     */
    public NeuronaSpiking getPostsinaptica() {
        return postsinaptica;
    }
    
    /**
     * Retorna el peso sináptico actual.
     * 
     * @return el peso sináptico
     */
    public double getPeso() {
        return peso;
    }

    /**
     * Establece el peso sináptico a un valor específico.
     *
     * <p>Este método permite establecer directamente el peso sináptico,
     * asegurando que el valor esté dentro de los límites configurables
     * [peso_min, peso_max].</p>
     *
     * <p>A diferencia de {@link #ajustarPeso(double)}, que aplica un cambio
     * relativo (delta), este método establece un valor absoluto.</p>
     *
     * @param nuevoPeso el nuevo valor del peso sináptico
     * @throws IllegalArgumentException si nuevoPeso no está en [pesoMin, pesoMax]
     */
    public void setPeso(double nuevoPeso) {
        // Validar que el peso esté en los límites
        if (nuevoPeso < pesoMin || nuevoPeso > pesoMax) {
            throw new IllegalArgumentException(
                "El peso (" + nuevoPeso + ") debe estar en el rango [" + pesoMin + ", " + pesoMax + "]"
            );
        }
        this.peso = nuevoPeso;
    }

    
    /**
     * Retorna el retardo sináptico en timesteps.
     * 
     * @return el retardo en timesteps
     */
    public int getRetardo() {
        return retardo;
    }
    
    /**
     * Retorna el límite inferior del peso sináptico.
     * 
     * @return el peso mínimo
     */
    public double getPesoMin() {
        return pesoMin;
    }
    
    /**
     * Retorna el límite superior del peso sináptico.
     * 
     * @return el peso máximo
     */
    public double getPesoMax() {
        return pesoMax;
    }
    
    /**
     * Retorna el timestamp del último spike presinaptico.
     * 
     * @return el timestamp del último spike presinaptico, o null si no ha disparado
     */
    public Long getTimestampUltimoSpikePresinaptico() {
        return timestampUltimoSpikePresinaptico;
    }
    
    /**
     * Establece el timestamp del último spike presinaptico.
     * 
     * @param timestamp el timestamp del spike presinaptico
     */
    public void setTimestampUltimoSpikePresinaptico(Long timestamp) {
        this.timestampUltimoSpikePresinaptico = timestamp;
    }
    
    /**
     * Retorna el timestamp del último spike postsinaptico.
     * 
     * @return el timestamp del último spike postsinaptico, o null si no ha disparado
     */
    public Long getTimestampUltimoSpikePostsinaptico() {
        return timestampUltimoSpikePostsinaptico;
    }
    
    /**
     * Establece el timestamp del último spike postsinaptico.
     * 
     * @param timestamp el timestamp del spike postsinaptico
     */
    public void setTimestampUltimoSpikePostsinaptico(Long timestamp) {
        this.timestampUltimoSpikePostsinaptico = timestamp;
    }
    
    /**
     * Propaga un spike desde la neurona presinaptica creando un evento de entrega.
     * 
     * <p>Cuando la neurona presinaptica genera un spike, este método crea un evento
     * de spike que se entregará a la neurona postsinaptica después del retardo
     * configurado. El timestamp de entrega se calcula como:</p>
     * 
     * <pre>
     * timestamp_entrega = timestamp_origen + retardo
     * </pre>
     * 
     * <p>El evento creado incluye el peso sináptico actual, que determina la magnitud
     * de la señal que recibirá la neurona postsinaptica.</p>
     * 
     * <p>Este método también actualiza el timestamp del último spike presinaptico
     * para uso en STDP.</p>
     * 
     * <p>Implementa los requisitos 8.2 y 8.4 (retardo sináptico).</p>
     * 
     * @param timestampOrigen el timestamp en el que se generó el spike presinaptico
     * @return un EventoSpike que representa la señal a entregar a la neurona postsinaptica
     */
    public EventoSpike propagarSpike(long timestampOrigen) {
        // Requisitos: 8.2, 8.4
        
        // Actualizar timestamp del último spike presinaptico (para STDP)
        this.timestampUltimoSpikePresinaptico = timestampOrigen;
        
        // Calcular timestamp de entrega: origen + retardo
        long timestampEntrega = timestampOrigen + retardo;
        
        // Crear evento de spike para la neurona postsinaptica
        // El evento incluye el peso sináptico como "potencial" para que la neurona
        // postsinaptica lo reciba como señal
        return new EventoSpike(
            postsinaptica.getId(),
            postsinaptica.getCapa(),
            postsinaptica.getIndice(),
            timestampEntrega,
            peso // El peso se usa como la señal a transmitir
        );
    }
    
    /**
     * Ajusta el peso sináptico aplicando un cambio (delta).
     * 
     * <p>El nuevo peso se calcula como:</p>
     * <pre>
     * peso_nuevo = clamp(peso_actual + delta, peso_min, peso_max)
     * </pre>
     * 
     * <p>El peso resultante siempre se mantiene dentro de los límites configurables
     * [peso_min, peso_max], implementando el requisito 5.6.</p>
     * 
     * <p>Este método es típicamente llamado por el gestor de STDP para aplicar
     * plasticidad sináptica basada en el timing de spikes pre y post.</p>
     * 
     * @param delta el cambio a aplicar al peso (puede ser positivo o negativo)
     */
    public void ajustarPeso(double delta) {
        // Requisito: 5.6
        
        // Calcular nuevo peso
        double nuevoPeso = peso + delta;
        
        // Aplicar clamp a [pesoMin, pesoMax]
        peso = Math.max(pesoMin, Math.min(pesoMax, nuevoPeso));
    }
    
    @Override
    public String toString() {
        return String.format("SinapsisSpiking[pre=%d, post=%d, peso=%.3f, retardo=%d]",
                presinaptica.getId(), postsinaptica.getId(), peso, retardo);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        SinapsisSpiking otra = (SinapsisSpiking) obj;
        return presinaptica.getId() == otra.presinaptica.getId() &&
               postsinaptica.getId() == otra.postsinaptica.getId();
    }
    
    @Override
    public int hashCode() {
        int result = Long.hashCode(presinaptica.getId());
        result = 31 * result + Long.hashCode(postsinaptica.getId());
        return result;
    }
}
