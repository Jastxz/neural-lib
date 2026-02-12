package es.jastxz.nn;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import es.jastxz.nn.enums.TipoConexion;

/**
 * Representa una conexión sináptica entre dos neuronas
 */
public class Conexion {
    private final Neurona presináptica;
    private final List<Neurona> postsinápticas;  // Soporte para sinapsis diádicas/triádicas
    private double peso;  // Fuerza sináptica (mutable por plasticidad)
    private final TipoConexion tipo;
    
    // Gestión de recursos (competición sináptica)
    private double recursosAsignados;
    
    // Plasticidad hebiana
    private long timestampUltimaActivacion;
    private int vecesActivadaJuntas;  // Contador para "activan juntas, conectan juntas"
    private double tasaRefuerzo;  // Velocidad de cambio del peso
    private static final double TASA_REFUERZO_DEFAULT = 0.02;  // Tasa de aprendizaje
    
    // Pertenencia a engramas
    private Set<String> engramasActivos;
    
    public Conexion(Neurona presináptica, Neurona postsináptica, 
                    double pesoInicial, TipoConexion tipo) {
        this.presináptica = presináptica;
        this.postsinápticas = new ArrayList<>();
        this.postsinápticas.add(postsináptica);
        
        this.peso = pesoInicial;
        this.tipo = tipo;
        this.recursosAsignados = 1.0;
        
        this.timestampUltimaActivacion = 0L;
        this.vecesActivadaJuntas = 0;
        this.tasaRefuerzo = TASA_REFUERZO_DEFAULT;
        
        this.engramasActivos = new HashSet<>();
        
        // Registrar conexión bidireccional
        presináptica.agregarAxon(this);
        postsináptica.agregarDendrita(this);
    }
    
    /**
     * Constructor para sinapsis diádicas/triádicas
     */
    public Conexion(Neurona presináptica, List<Neurona> postsinápticas,
                    double pesoInicial, TipoConexion tipo) {
        this.presináptica = presináptica;
        this.postsinápticas = new ArrayList<>(postsinápticas);
        
        this.peso = pesoInicial;
        this.tipo = tipo;
        this.recursosAsignados = 1.0;
        
        this.timestampUltimaActivacion = 0L;
        this.vecesActivadaJuntas = 0;
        this.tasaRefuerzo = TASA_REFUERZO_DEFAULT;
        
        this.engramasActivos = new HashSet<>();
        
        // Registrar en todas las neuronas
        presináptica.agregarAxon(this);
        for (Neurona post : postsinápticas) {
            post.agregarDendrita(this);
        }
    }
    
    /**
     * Aplica plasticidad hebiana: refuerza o debilita la conexión
     * Principio: neuronas que se activan juntas, se conectan (pg. 46-47)
     */
    public void aplicarPlasticidadHebiana(long timestampActual, long ventanaTemporal) {
        boolean preActiva = presináptica.estaActiva();
        boolean algunaPostActiva = postsinápticas.stream().anyMatch(Neurona::estaActiva);
        
        // Si ambas activas en ventana temporal cercana
        if (preActiva && algunaPostActiva) {
            // Reforzar conexión (Long-Term Potentiation)
            peso = Math.min(1.0, peso + tasaRefuerzo);
            vecesActivadaJuntas++;
            timestampUltimaActivacion = timestampActual;
            
            // Reforzar recursos por uso
            recursosAsignados = Math.min(1.0, recursosAsignados + 0.005);
            
        } else if (timestampActual - timestampUltimaActivacion > ventanaTemporal) {
            // Debilitar por desuso (Long-Term Depression)
            peso = Math.max(0.0, peso - tasaRefuerzo * 0.5);
            
            // Penalizar recursos por falta de uso
            recursosAsignados = Math.max(0.0, recursosAsignados - 0.01);
        }
    }
    
    /**
     * Verifica si la conexión debe ser podada (eliminada)
     */
    public boolean debeSerPodada() {
        return peso <= 0.05 || recursosAsignados <= 0.1;
    }
    
    /**
     * Añade esta conexión a un engrama
     */
    public void unirseAEngrama(String engramaId) {
        engramasActivos.add(engramaId);
    }
    
    /**
     * Remueve esta conexión de un engrama
     */
    public void salirDeEngrama(String engramaId) {
        engramasActivos.remove(engramaId);
    }
    
    // Getters y setters
    public Neurona getPresinaptica() { return presináptica; }
    public List<Neurona> getPostsinápticas() { return new ArrayList<>(postsinápticas); }
    public double getPeso() { return peso; }
    public void setPeso(double peso) { 
        this.peso = Math.max(0.0, Math.min(1.0, peso)); 
    }
    public TipoConexion getTipo() { return tipo; }
    public double getRecursosAsignados() { return recursosAsignados; }
    public void setRecursosAsignados(double recursos) {
        this.recursosAsignados = Math.max(0.0, Math.min(1.0, recursos));
    }
    public long getTimestampUltimaActivacion() { return timestampUltimaActivacion; }
    public int getVecesActivadaJuntas() { return vecesActivadaJuntas; }
    public Set<String> getEngramasActivos() { return new HashSet<>(engramasActivos); }
}