package es.jastxz.nn.spiking;

import java.util.*;

/**
 * Gestor de decodificación que convierte trenes de spikes a valores continuos usando rate coding.
 * 
 * <p>Este gestor implementa decodificación por tasa de disparo (rate coding), donde la frecuencia
 * de spikes en una ventana temporal se convierte a un valor continuo en el rango [0, 1].
 * La decodificación se basa en contar spikes dentro de una ventana temporal deslizante y
 * normalizar el conteo según la frecuencia máxima configurada.</p>
 * 
 * <p>Fórmula de decodificación: valor = conteo_spikes / (frecuencia_maxima * duracion_ventana)</p>
 * 
 * <p>El gestor mantiene ventanas deslizantes por neurona para permitir decodificación eficiente
 * de múltiples neuronas simultáneamente.</p>
 * 
 * @see GestorCodificacion
 * @see EventoSpike
 */
public class GestorDecodificacion implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Duración de la ventana temporal en timesteps.
     */
    private final int ventanaTemporal;
    
    /**
     * Frecuencia máxima de disparo en Hz.
     */
    private final double frecuenciaMaxima;
    
    /**
     * Mapa de ventanas deslizantes por neurona.
     * La clave es el ID de la neurona, el valor es la ventana deslizante de timestamps.
     */
    private final Map<Long, VentanaDeslizante> ventanasPorNeurona;
    
    /**
     * Construye un nuevo gestor de decodificación.
     * 
     * @param ventanaTemporal duración de la ventana temporal en timesteps (debe ser > 0)
     * @param frecuenciaMaxima frecuencia máxima de disparo en Hz (debe ser > 0)
     * @throws IllegalArgumentException si ventanaTemporal o frecuenciaMaxima no son positivos
     */
    public GestorDecodificacion(int ventanaTemporal, double frecuenciaMaxima) {
        if (ventanaTemporal <= 0) {
            throw new IllegalArgumentException(
                "La ventana temporal debe ser positiva, recibido: " + ventanaTemporal
            );
        }
        if (frecuenciaMaxima <= 0) {
            throw new IllegalArgumentException(
                "La frecuencia máxima debe ser positiva, recibido: " + frecuenciaMaxima
            );
        }
        
        this.ventanaTemporal = ventanaTemporal;
        this.frecuenciaMaxima = frecuenciaMaxima;
        this.ventanasPorNeurona = new HashMap<>();
    }
    
    /**
     * Cuenta los spikes en la ventana temporal para una neurona específica.
     * 
     * <p>Este método mantiene una ventana deslizante de timestamps de spikes para la neurona.
     * Solo se cuentan los spikes que ocurrieron dentro de la ventana temporal
     * [timestampActual - ventanaTemporal, timestampActual].</p>
     * 
     * @param neuronaId identificador de la neurona
     * @param timestampActual timestamp actual de la simulación
     * @return número de spikes en la ventana temporal
     */
    public int contarSpikesEnVentana(long neuronaId, long timestampActual) {
        VentanaDeslizante ventana = ventanasPorNeurona.get(neuronaId);
        if (ventana == null) {
            return 0;
        }
        
        // Limpiar spikes fuera de la ventana
        ventana.limpiarAntiguos(timestampActual - ventanaTemporal);
        
        return ventana.contarSpikes();
    }
    
    /**
     * Actualiza la ventana deslizante de una neurona con un nuevo spike.
     * 
     * @param neuronaId identificador de la neurona
     * @param timestamp timestamp del spike
     */
    public void actualizarVentana(long neuronaId, long timestamp) {
        VentanaDeslizante ventana = ventanasPorNeurona.computeIfAbsent(
            neuronaId, 
            k -> new VentanaDeslizante()
        );
        ventana.agregarSpike(timestamp);
    }
    
    /**
     * Decodifica un tren de spikes a un valor continuo en el rango [0, 1].
     * 
     * <p>La decodificación se basa en contar todos los spikes en la lista y normalizar
     * según la frecuencia máxima y la duración de la ventana temporal.</p>
     * 
     * <p>Fórmula: valor = conteo_spikes / (frecuencia_maxima * duracion_ventana)</p>
     * 
     * <p>El valor resultante se clampea al rango [0, 1].</p>
     * 
     * @param spikes lista de eventos de spike a decodificar
     * @return valor decodificado en el rango [0, 1]
     */
    public double decodificar(List<EventoSpike> spikes) {
        if (spikes == null || spikes.isEmpty()) {
            return 0.0;
        }
        
        int conteoSpikes = spikes.size();
        
        // Convertir ventana temporal de timesteps a segundos
        // Asumiendo que cada timestep es 1ms (0.001 segundos)
        double duracionVentanaSegundos = ventanaTemporal * 0.001;
        
        // Calcular valor normalizado
        double valor = conteoSpikes / (frecuenciaMaxima * duracionVentanaSegundos);
        
        // Clampear a [0, 1]
        return Math.max(0.0, Math.min(1.0, valor));
    }
    
    /**
     * Decodifica los spikes de múltiples neuronas (típicamente una capa completa).
     * 
     * <p>Este método procesa una lista de trenes de spikes, donde cada elemento de la lista
     * representa los spikes de una neurona diferente. Retorna un array con los valores
     * decodificados para cada neurona.</p>
     * 
     * @param spikesPorNeurona lista de listas de spikes, una por neurona
     * @return array de valores decodificados, uno por neurona
     */
    public double[] decodificarCapa(List<List<EventoSpike>> spikesPorNeurona) {
        if (spikesPorNeurona == null || spikesPorNeurona.isEmpty()) {
            return new double[0];
        }
        
        double[] valores = new double[spikesPorNeurona.size()];
        
        for (int i = 0; i < spikesPorNeurona.size(); i++) {
            valores[i] = decodificar(spikesPorNeurona.get(i));
        }
        
        return valores;
    }
    
    /**
     * Retorna la duración de la ventana temporal en timesteps.
     * 
     * @return ventana temporal en timesteps
     */
    public int getVentanaTemporal() {
        return ventanaTemporal;
    }
    
    /**
     * Retorna la frecuencia máxima de disparo en Hz.
     * 
     * @return frecuencia máxima en Hz
     */
    public double getFrecuenciaMaxima() {
        return frecuenciaMaxima;
    }
    
    /**
     * Limpia todas las ventanas deslizantes mantenidas por el gestor.
     * 
     * <p>Este método es útil para resetear el estado del gestor entre diferentes
     * patrones de entrada o al comenzar una nueva simulación.</p>
     */
    public void limpiarVentanas() {
        ventanasPorNeurona.clear();
    }
    
    /**
     * Clase interna que representa una ventana deslizante de timestamps de spikes.
     * 
     * <p>Mantiene una lista ordenada de timestamps y permite limpiar eficientemente
     * los timestamps que quedan fuera de la ventana temporal.</p>
     */
    private static class VentanaDeslizante implements java.io.Serializable {
        private static final long serialVersionUID = 1L;
        
        /**
         * Lista de timestamps de spikes dentro de la ventana.
         * Se mantiene ordenada por timestamp.
         */
        private final List<Long> timestamps;
        
        /**
         * Construye una nueva ventana deslizante vacía.
         */
        public VentanaDeslizante() {
            this.timestamps = new ArrayList<>();
        }
        
        /**
         * Agrega un nuevo spike a la ventana.
         * 
         * @param timestamp timestamp del spike
         */
        public void agregarSpike(long timestamp) {
            timestamps.add(timestamp);
        }
        
        /**
         * Limpia los spikes que son anteriores al timestamp límite.
         * 
         * @param timestampLimite timestamp mínimo a mantener
         */
        public void limpiarAntiguos(long timestampLimite) {
            timestamps.removeIf(t -> t < timestampLimite);
        }
        
        /**
         * Cuenta el número de spikes en la ventana.
         * 
         * @return número de spikes
         */
        public int contarSpikes() {
            return timestamps.size();
        }
    }
}
