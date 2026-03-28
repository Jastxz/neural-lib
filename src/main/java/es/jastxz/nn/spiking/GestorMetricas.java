package es.jastxz.nn.spiking;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

/**
 * Gestor de métricas de rendimiento para redes neuronales spiking.
 * 
 * <p>Calcula y mantiene métricas sobre la actividad de la red, incluyendo:</p>
 * <ul>
 *   <li>Número total de spikes generados</li>
 *   <li>Spikes por neurona individual</li>
 *   <li>Tasa de disparo promedio global y por neurona</li>
 *   <li>Costo energético estimado (proporcional a spikes)</li>
 *   <li>Dispersión de actividad entre neuronas</li>
 * </ul>
 * 
 * <p>Las métricas pueden ser exportadas en formato estructurado (Map) para análisis.</p>
 * 
 * @author jastxz
 * @version 1.0
 * @since 1.0
 */
public class GestorMetricas implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Contador total de spikes generados en la red.
     */
    private long totalSpikes;
    
    /**
     * Mapa que almacena el número de spikes por neurona.
     * Clave: ID de neurona, Valor: número de spikes.
     */
    private final Map<Long, Integer> spikesPorNeurona;
    
    /**
     * Mapa que almacena los timestamps de spikes por neurona.
     * Clave: ID de neurona, Valor: lista de timestamps.
     */
    private final Map<Long, List<Long>> timestampsPorNeurona;
    
    /**
     * Construye un nuevo gestor de métricas con contadores inicializados a cero.
     */
    public GestorMetricas() {
        this.totalSpikes = 0;
        this.spikesPorNeurona = new HashMap<>();
        this.timestampsPorNeurona = new HashMap<>();
    }
    
    /**
     * Registra un spike generado por una neurona.
     * 
     * <p>Incrementa el contador total de spikes, actualiza el contador de la neurona
     * específica, y almacena el timestamp del spike.</p>
     * 
     * @param neuronaId ID de la neurona que generó el spike
     * @param timestamp timestamp en el que ocurrió el spike
     */
    public void registrarSpike(long neuronaId, long timestamp) {
        // Incrementar contador total
        totalSpikes++;
        
        // Incrementar contador por neurona
        spikesPorNeurona.merge(neuronaId, 1, Integer::sum);
        
        // Registrar timestamp
        timestampsPorNeurona.computeIfAbsent(neuronaId, k -> new ArrayList<>()).add(timestamp);
    }
    
    /**
     * Obtiene el número total de spikes generados en la red.
     * 
     * @return número total de spikes
     */
    public long getTotalSpikes() {
        return totalSpikes;
    }
    
    /**
     * Calcula la tasa de disparo promedio global de la red.
     * 
     * <p>La tasa se calcula como el promedio de las tasas de disparo de todas las neuronas
     * que han generado al menos un spike.</p>
     * 
     * @return tasa de disparo promedio en Hz, o 0.0 si no hay neuronas activas
     */
    public double getTasaPromedioGlobal() {
        if (spikesPorNeurona.isEmpty()) {
            return 0.0;
        }
        
        double sumaSpikes = 0.0;
        for (int spikes : spikesPorNeurona.values()) {
            sumaSpikes += spikes;
        }
        
        return sumaSpikes / spikesPorNeurona.size();
    }
    
    /**
     * Calcula la tasa de disparo promedio de una neurona específica.
     * 
     * <p>La tasa se calcula como el número de spikes dividido por la duración de la
     * ventana temporal (diferencia entre el último y primer timestamp).</p>
     * 
     * @param neuronaId ID de la neurona
     * @return tasa de disparo en Hz, o 0.0 si la neurona no ha generado spikes
     */
    public double getTasaPromedioPorNeurona(long neuronaId) {
        List<Long> timestamps = timestampsPorNeurona.get(neuronaId);
        
        if (timestamps == null || timestamps.isEmpty()) {
            return 0.0;
        }
        
        if (timestamps.size() == 1) {
            return 1.0; // Un solo spike, tasa mínima
        }
        
        // Calcular duración de la ventana temporal
        long primerTimestamp = timestamps.get(0);
        long ultimoTimestamp = timestamps.get(timestamps.size() - 1);
        long duracion = ultimoTimestamp - primerTimestamp;
        
        if (duracion == 0) {
            return timestamps.size(); // Múltiples spikes en el mismo timestamp
        }
        
        // Tasa = número de spikes / duración (en timesteps)
        return (double) timestamps.size() / duracion;
    }
    
    /**
     * Calcula el costo energético estimado de la red.
     * 
     * <p>El costo energético es proporcional al número total de spikes generados.
     * Se asume un costo unitario de 1.0 por spike.</p>
     * 
     * @return costo energético estimado
     */
    public double calcularCostoEnergetico() {
        // Costo energético proporcional al número de spikes
        // Asumimos un costo unitario de 1.0 por spike
        return (double) totalSpikes;
    }
    
    /**
     * Calcula la dispersión de actividad entre neuronas.
     * 
     * <p>La dispersión se calcula como la desviación estándar del número de spikes
     * por neurona. Una dispersión alta indica que algunas neuronas disparan mucho más
     * que otras, mientras que una dispersión baja indica actividad más uniforme.</p>
     * 
     * @return dispersión de actividad (desviación estándar), o 0.0 si no hay neuronas activas
     */
    public double calcularDispersionActividad() {
        if (spikesPorNeurona.isEmpty()) {
            return 0.0;
        }
        
        // Calcular media
        double media = getTasaPromedioGlobal();
        
        // Calcular varianza
        double sumaDesviacionesCuadradas = 0.0;
        for (int spikes : spikesPorNeurona.values()) {
            double desviacion = spikes - media;
            sumaDesviacionesCuadradas += desviacion * desviacion;
        }
        
        double varianza = sumaDesviacionesCuadradas / spikesPorNeurona.size();
        
        // Desviación estándar
        return Math.sqrt(varianza);
    }
    
    /**
     * Exporta todas las métricas en formato estructurado.
     * 
     * <p>El mapa retornado contiene las siguientes claves:</p>
     * <ul>
     *   <li>"totalSpikes": número total de spikes (Long)</li>
     *   <li>"tasaPromedioGlobal": tasa de disparo promedio global (Double)</li>
     *   <li>"costoEnergetico": costo energético estimado (Double)</li>
     *   <li>"dispersionActividad": dispersión de actividad (Double)</li>
     *   <li>"neuronasActivas": número de neuronas que han generado spikes (Integer)</li>
     *   <li>"spikesPorNeurona": mapa de spikes por neurona (Map&lt;Long, Integer&gt;)</li>
     * </ul>
     * 
     * @return mapa con todas las métricas
     */
    public Map<String, Object> exportarMetricas() {
        Map<String, Object> metricas = new HashMap<>();
        
        metricas.put("totalSpikes", totalSpikes);
        metricas.put("tasaPromedioGlobal", getTasaPromedioGlobal());
        metricas.put("costoEnergetico", calcularCostoEnergetico());
        metricas.put("dispersionActividad", calcularDispersionActividad());
        metricas.put("neuronasActivas", spikesPorNeurona.size());
        metricas.put("spikesPorNeurona", new HashMap<>(spikesPorNeurona));
        
        return metricas;
    }
    
    /**
     * Resetea todos los contadores y métricas a cero.
     * 
     * <p>Limpia el contador total de spikes, los contadores por neurona,
     * y todos los timestamps registrados.</p>
     */
    public void resetear() {
        totalSpikes = 0;
        spikesPorNeurona.clear();
        timestampsPorNeurona.clear();
    }
    
    /**
     * Obtiene el número de spikes de una neurona específica.
     * 
     * @param neuronaId ID de la neurona
     * @return número de spikes, o 0 si la neurona no ha generado spikes
     */
    public int getSpikesPorNeurona(long neuronaId) {
        return spikesPorNeurona.getOrDefault(neuronaId, 0);
    }
    
    /**
     * Obtiene los timestamps de spikes de una neurona específica.
     * 
     * @param neuronaId ID de la neurona
     * @return lista de timestamps, o lista vacía si la neurona no ha generado spikes
     */
    public List<Long> getTimestampsPorNeurona(long neuronaId) {
        return timestampsPorNeurona.getOrDefault(neuronaId, new ArrayList<>());
    }
    
    /**
     * Obtiene el número de neuronas que han generado al menos un spike.
     * 
     * @return número de neuronas activas
     */
    public int getNeuronasActivas() {
        return spikesPorNeurona.size();
    }
}
