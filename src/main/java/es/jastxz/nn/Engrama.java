package es.jastxz.nn;

import java.util.HashSet;
import java.util.Set;

import java.io.Serializable;

/**
 * Representa un engrama: conjunto de neuronas y conexiones 
 * que codifican un recuerdo específico
 */
public class Engrama implements Serializable {
    private static final long serialVersionUID = 1L;
    private final String id;
    private final Set<Neurona> neuronasParticipantes;
    private final Set<Conexion> conexionesParticipantes;
    private double fuerza;  // Consolidación del engrama
    private double relevancia;  // Importancia del engrama (para consolidación)
    private int contadorActivaciones;  // Número de veces que se ha activado
    private long timestampCreacion;
    private long timestampUltimaActivacion;
    
    public Engrama(String id) {
        this.id = id;
        this.neuronasParticipantes = new HashSet<>();
        this.conexionesParticipantes = new HashSet<>();
        this.fuerza = 0.5;  // Fuerza inicial media
        this.relevancia = 1.0;  // Relevancia inicial máxima
        this.contadorActivaciones = 0;
        this.timestampCreacion = System.currentTimeMillis();
        this.timestampUltimaActivacion = timestampCreacion;
    }
    
    /**
     * Constructor con lista de neuronas participantes
     */
    public Engrama(String id, java.util.List<Neurona> neuronas, long timestamp) {
        this.id = id;
        this.neuronasParticipantes = new HashSet<>(neuronas);
        this.conexionesParticipantes = new HashSet<>();
        this.fuerza = 0.5;
        this.relevancia = 1.0;
        this.contadorActivaciones = 0;
        this.timestampCreacion = timestamp;
        this.timestampUltimaActivacion = timestamp;
    }
    
    /**
     * Añade neurona al engrama
     */
    public void agregarNeurona(Neurona neurona) {
        neuronasParticipantes.add(neurona);
        // No registrar en neurona - evitar referencias bidireccionales
    }
    
    /**
     * Añade conexión al engrama
     */
    public void agregarConexion(Conexion conexion) {
        conexionesParticipantes.add(conexion);
        // No registrar en conexión - evitar referencias bidireccionales
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
    public boolean estaActivo() {
        long neuronasActivas = neuronasParticipantes.stream()
            .filter(Neurona::estaActiva)
            .count();
        
        double proporcionActiva = (double) neuronasActivas / neuronasParticipantes.size();
        double umbralProporcion = 0.3 + Math.random()*100 % 10 / 100;
        return proporcionActiva >= umbralProporcion;  // Porcentaje mínimo para tener un recuerdo
    }
    
    /**
     * Facilita la activación del resto de neuronas (completado de patrón)
     * Solo si el engrama ya está parcialmente activo
     */
    public void completarPatron(long timestamp) {
        if (!estaActivo()) {
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
    
    /**
     * Activa el engrama, facilitando todas sus neuronas
     */
    public void activar(long timestamp) {
        this.timestampUltimaActivacion = timestamp;
        this.contadorActivaciones++;
        
        // Facilitar todas las neuronas del engrama
        for (Neurona n : neuronasParticipantes) {
            n.facilitarActivacion(fuerza);
        }
    }
    
    /**
     * Verifica si este engrama contiene un porcentaje significativo de las neuronas dadas
     * @param neuronas Lista de neuronas a verificar
     * @param umbralSolapamiento Porcentaje mínimo de solapamiento (0.0 a 1.0)
     * @return true si hay suficiente solapamiento
     */
    public boolean contieneNeuronas(java.util.List<Neurona> neuronas, double umbralSolapamiento) {
        if (neuronas == null || neuronas.isEmpty()) {
            return false;
        }
        
        long neuronasComunes = neuronas.stream()
            .filter(neuronasParticipantes::contains)
            .count();
        
        double proporcionComun = (double) neuronasComunes / neuronas.size();
        return proporcionComun >= umbralSolapamiento;
    }
    
    // Getters
    public String getId() { return id; }
    public Set<Neurona> getNeuronasParticipantes() { return new HashSet<>(neuronasParticipantes); }
    public java.util.List<Neurona> getNeuronas() { return new java.util.ArrayList<>(neuronasParticipantes); }
    public Set<Conexion> getConexionesParticipantes() { return new HashSet<>(conexionesParticipantes); }
    public double getFuerza() { return fuerza; }
    public double getRelevancia() { return relevancia; }
    public int getContadorActivaciones() { return contadorActivaciones; }
    public long getTimestampUltimaActivacion() { return timestampUltimaActivacion; }
    
    // Setters
    public void setRelevancia(double relevancia) { 
        this.relevancia = Math.max(0.0, Math.min(2.0, relevancia)); 
    }
}
