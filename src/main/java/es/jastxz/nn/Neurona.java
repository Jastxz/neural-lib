package es.jastxz.nn;

import java.util.*;

import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoNeurona;

/**
 * Representa una neurona biológica con capacidad de plasticidad,
 * competición por recursos y formación de engramas.
 */
public class Neurona {
    // Identificación
    private final long id;
    private final TipoNeurona tipo;

    // Este valor es:
    // Diferente del potencial (que es transitorio, electroquímico)
    // Persistente - representa el "conocimiento" que la neurona aporta al programa
    // Actualizable mediante aprendizaje/consolidación
    // Utilizado en cómputos cuando la red ejecuta programas aprendidos
    private double valorAlmacenado;  // "Opinión" de la neurona para programas entre -1 y 1
    
    // Estado electroquímico
    private PotencialMemoria potencial;  // Potencial de membrana actual
    private PotencialMemoria umbralActivacion;
    private boolean activa;
    
    // Conectividad
    private List<Conexion> axones;      // Conexiones salientes
    private List<Conexion> dendritas;   // Conexiones entrantes
    
    // Gestión de recursos (competición biológica)
    private double recursosAsignados;   // 0.0 a 1.0
    private double factorSupervivencia; // Basado en uso reciente
    
    // Temporalidad
    private long timestampUltimaActivacion;
    private int contadorActivaciones;
    
    // Facilitación temporal (para completado de patrón en engramas)
    private double facilitacionTemporal;
    
    // Pertenencia a engramas (memoria distribuida)
    private Set<String> engramasActivos;  // IDs de engramas a los que pertenece
    
    public Neurona(long id, TipoNeurona tipo, double valorAlmacenado, PotencialMemoria potencialInicial) {
        this.id = id;
        this.tipo = tipo;
        this.valorAlmacenado = valorAlmacenado;
        this.potencial = potencialInicial;
        this.umbralActivacion = PotencialMemoria.UMBRAL;  // mV (valor biológico típico)
        
        // Si se inicializa con potencial de pico, marcar como activa
        this.activa = (potencialInicial == PotencialMemoria.PICO);
        
        this.axones = new ArrayList<>();
        this.dendritas = new ArrayList<>();
        
        this.recursosAsignados = 1.0;  // Todas empiezan con recursos máximos
        this.factorSupervivencia = 1.0;
        
        this.timestampUltimaActivacion = 0L;
        this.contadorActivaciones = 0;
        
        this.facilitacionTemporal = 0.0;
        
        this.engramasActivos = new HashSet<>();
    }
    
    /**
     * Evalúa si la neurona debe activarse basándose en inputs de dendritas
     */
    public boolean evaluar(long timestampActual) {
        double sumaInputs = 0.0;
        boolean hayInputsActivos = false;
        
        // Sumar inputs ponderados por peso sináptico
        for (Conexion dendrita : dendritas) {
            if (dendrita.getPresinaptica().estaActiva()) {
                sumaInputs += dendrita.getPeso() * dendrita.getPresinaptica().getPotencial();
                hayInputsActivos = true;
            }
        }
        
        // Si no hay inputs activos, no puede activarse
        if (!hayInputsActivos) {
            facilitacionTemporal = Math.max(0.0, facilitacionTemporal - 0.1);
            resetear();
            return false;
        }
        
        // Umbral ajustado por facilitación (completado de patrón)
        double umbralAjustado = umbralActivacion.getValor() * (1.0 - facilitacionTemporal);
        
        // Activación todo-o-nada
        if (sumaInputs >= umbralAjustado) {
            activar(timestampActual);
            facilitacionTemporal = 0.0;  // Reset tras activación
            return true;
        }
        
        // Decaimiento gradual de la facilitación
        facilitacionTemporal = Math.max(0.0, facilitacionTemporal - 0.1);
        resetear();
        return false;
    }
    
    /**
     * Facilita temporalmente la activación reduciendo el umbral
     * Usado por engramas para completado de patrón
     */
    public void facilitarActivacion(double factor) {
        this.facilitacionTemporal = Math.min(0.5, factor * 0.3);  // Máximo 50% de reducción
    }
    
    /**
     * Activa la neurona (usado para inputs sensoriales directos)
     */
    public void activar(long timestamp) {
        this.activa = true;
        this.potencial = PotencialMemoria.PICO;  // Pico de potencial de acción (mV)
        this.timestampUltimaActivacion = timestamp;
        this.contadorActivaciones++;
        
        // Reforzar factor de supervivencia por uso
        this.factorSupervivencia = Math.min(1.0, factorSupervivencia + 0.01);
    }
    
    /**
     * Resetea la neurona a estado de reposo
     * Solo resetea estado transitorio (potencial, activación)
     * Preserva conocimiento aprendido (valorAlmacenado, conexiones, engramas)
     */
    public void resetear() {
        this.activa = false;
        this.potencial = PotencialMemoria.REPOSO;  // Potencial de reposo (mV)
    }
    
    /**
     * Degradación de recursos por falta de uso (muerte neuronal)
     */
    public void degradarPorDesuso(long timestampActual, long umbralTiempo) {
        if (timestampActual - timestampUltimaActivacion > umbralTiempo) {
            // Penalizar factor de supervivencia
            this.factorSupervivencia = Math.max(0.0, factorSupervivencia - 0.05);
            
            // Si el factor cae demasiado, reducir recursos
            if (factorSupervivencia < 0.3) {
                this.recursosAsignados = Math.max(0.0, recursosAsignados - 0.1);
            }
        }
    }
    
    /**
     * Verifica si la neurona debe ser eliminada (apoptosis)
     */
    public boolean debeSerEliminada() {
        return recursosAsignados <= 0.0 || factorSupervivencia <= 0.0;
    }
    
    /**
     * Añade esta neurona a un engrama (memoria)
     */
    public void unirseAEngrama(String engramaId) {
        engramasActivos.add(engramaId);
    }
    
    /**
     * Elimina esta neurona de un engrama
     */
    public void abandonarEngrama(String engramaId) {
        engramasActivos.remove(engramaId);
    }
    
    /**
     * Remueve esta neurona de un engrama
     */
    public void salirDeEngrama(String engramaId) {
        engramasActivos.remove(engramaId);
    }
    
    /**
     * Añade una conexión axonal (saliente)
     */
    public void agregarAxon(Conexion conexion) {
        axones.add(conexion);
    }
    
    /**
     * Añade una conexión dendrítica (entrante)
     */
    public void agregarDendrita(Conexion conexion) {
        dendritas.add(conexion);
    }
    
    // Getters y setters
    public long getId() { return id; }
    public TipoNeurona getTipo() { return tipo; }
    public double getValorAlmacenado() {return valorAlmacenado; }
    public void setValorAlmacenado(double valorAlmacenado) {this.valorAlmacenado = valorAlmacenado; }
    public double getPotencial() { return potencial.getValor(); }
    public void setPotencial(PotencialMemoria potencial) { this.potencial = potencial; }
    public boolean estaActiva() { return activa; }
    public List<Conexion> getAxones() { return axones; }
    public List<Conexion> getDendritas() { return dendritas; }
    public double getRecursosAsignados() { return recursosAsignados; }
    public void setRecursosAsignados(double recursos) { 
        this.recursosAsignados = Math.max(0.0, Math.min(1.0, recursos)); 
    }
    public double getFactorSupervivencia() { return factorSupervivencia; }
    public long getTimestampUltimaActivacion() { return timestampUltimaActivacion; }
    public int getContadorActivaciones() { return contadorActivaciones; }
    public Set<String> getEngramasActivos() { return new HashSet<>(engramasActivos); }
}
