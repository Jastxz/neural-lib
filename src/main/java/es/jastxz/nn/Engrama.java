package es.jastxz.nn;

import java.util.HashSet;
import java.util.Set;

/**
 * Representa un engrama: conjunto de neuronas y conexiones 
 * que codifican un recuerdo específico
 */
public class Engrama {
    private final String id;
    private final Set<Neurona> neuronasParticipantes;
    private final Set<Conexion> conexionesParticipantes;
    private double fuerza;  // Consolidación del engrama
    private long timestampCreacion;
    private long timestampUltimaActivacion;
    
    public Engrama(String id) {
        this.id = id;
        this.neuronasParticipantes = new HashSet<>();
        this.conexionesParticipantes = new HashSet<>();
        this.fuerza = 0.5;  // Fuerza inicial media
        this.timestampCreacion = System.currentTimeMillis();
        this.timestampUltimaActivacion = timestampCreacion;
    }
    
    /**
     * Añade neurona al engrama
     */
    public void agregarNeurona(Neurona neurona) {
        neuronasParticipantes.add(neurona);
        neurona.unirseAEngrama(this.id);
    }
    
    /**
     * Añade conexión al engrama
     */
    public void agregarConexion(Conexion conexion) {
        conexionesParticipantes.add(conexion);
        conexion.unirseAEngrama(this.id);
    }
    
    /**
     * Consolida el engrama (fortalece durante "sueño")
     */
    public void consolidar(double factorConsolidacion) {
        this.fuerza = Math.min(1.0, fuerza + factorConsolidacion);
        
        // Reforzar todas las conexiones del engrama
        for (Conexion c : conexionesParticipantes) {
            c.setPeso(Math.min(1.0, c.getPeso() + factorConsolidacion * 0.5));
        }
    }
    
    /**
     * Degrada el engrama por falta de uso (olvido)
     */
    public void degradar(double factorOlvido) {
        this.fuerza = Math.max(0.0, fuerza - factorOlvido);
        
        // Los engramas no desaparecen del todo, solo se debilitan parcialmente
        if (fuerza < 0.2) {
            for (Conexion c : conexionesParticipantes) {
                c.setPeso(Math.max(0.1, c.getPeso() - factorOlvido * 0.3));
            }
        }
    }
    
    /**
     * Verifica si el engrama está activo basándose en cuántas neuronas participantes están activas
     */
    public boolean estaActivo(double umbralActivacion) {
        long neuronasActivas = neuronasParticipantes.stream()
            .filter(Neurona::estaActiva)
            .count();
        
        double proporcionActiva = (double) neuronasActivas / neuronasParticipantes.size();
        return proporcionActiva >= umbralActivacion;  // ej: 0.3 = 30% de neuronas activas
    }
    
    /**
     * Facilita la activación del resto de neuronas (completado de patrón)
     * Solo si el engrama ya está parcialmente activo
     */
    public void completarPatron(long timestamp, double umbralActivacion) {
        if (!estaActivo(umbralActivacion)) {
            return;  // No hay suficiente activación inicial
        }
        
        this.timestampUltimaActivacion = timestamp;
        
        // Facilitar activación de neuronas inactivas del engrama
        // reduciendo temporalmente su umbral
        for (Neurona n : neuronasParticipantes) {
            if (!n.estaActiva()) {
                n.facilitarActivacion(fuerza);  // La fuerza del engrama ayuda
            }
        }
    }
    
    // Getters
    public String getId() { return id; }
    public Set<Neurona> getNeuronasParticipantes() { return new HashSet<>(neuronasParticipantes); }
    public Set<Conexion> getConexionesParticipantes() { return new HashSet<>(conexionesParticipantes); }
    public double getFuerza() { return fuerza; }
    public long getTimestampUltimaActivacion() { return timestampUltimaActivacion; }
}
