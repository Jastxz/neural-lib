package es.jastxz.nn;

import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

import es.jastxz.nn.enums.TipoConexion;

/**
 * Representa una conexión sináptica entre neuronas.
 * 
 * Para las neuronas a las que entrega la información es una dendrita, para el otro caso es un axón.
 * Axones - Conexiones salientes
 * Dendritas - Conexiones entrantes
 * 
 * REFACTORIZACIÓN: Mantiene referencias directas a neuronas pero sin listas bidireccionales
 */
public class Conexion implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private final Neurona presinaptica;
    private final List<Neurona> postsinapticas;  // Soporte para sinapsis diádicas/triádicas
    
    private double peso;  // Fuerza sináptica (mutable por plasticidad) - puede ser negativo para inhibición
    private final TipoConexion tipo;
    
    // Gestión de recursos (competición sináptica)
    private double recursosAsignados;
    
    // Plasticidad hebiana
    private long timestampUltimaActivacion;
    private int vecesActivadaJuntas;  // Contador para "activan juntas, conectan juntas"
    private double tasaRefuerzo;  // Velocidad de cambio del peso
    private static final double TASA_REFUERZO_DEFAULT = 0.02;  // Tasa de aprendizaje
    
    /**
     * Constructor simple (1 pre → 1 post)
     */
    public Conexion(Neurona presinaptica, Neurona postsinaptica, 
                    double pesoInicial, TipoConexion tipo) {
        this.presinaptica = presinaptica;
        this.postsinapticas = new ArrayList<>();
        this.postsinapticas.add(postsinaptica);
        presinaptica.añadirVecina(postsinaptica);
        postsinaptica.añadirVecina(presinaptica);
        
        this.peso = pesoInicial;
        this.tipo = tipo;
        this.recursosAsignados = 1.0;
        
        this.timestampUltimaActivacion = 0L;
        this.vecesActivadaJuntas = 0;
        this.tasaRefuerzo = TASA_REFUERZO_DEFAULT;
    }
    
    /**
     * Constructor para sinapsis diádicas/triádicas (1 pre → N post)
     */
    public Conexion(Neurona presinaptica, List<Neurona> postsinapticas,
                    double pesoInicial, TipoConexion tipo) {
        this.presinaptica = presinaptica;
        this.postsinapticas = postsinapticas;
        presinaptica.añadirVecinas(postsinapticas);
        postsinapticas.stream().forEach(n -> n.añadirVecina(presinaptica));
        
        this.peso = pesoInicial;
        this.tipo = tipo;
        this.recursosAsignados = 1.0;
        
        this.timestampUltimaActivacion = 0L;
        this.vecesActivadaJuntas = 0;
        this.tasaRefuerzo = TASA_REFUERZO_DEFAULT;
    }
    
    /**
     * Aplica plasticidad hebiana: refuerza o debilita la conexión
     * Principio: neuronas que se activan juntas, se conectan (pg. 46-47)
     */
    public void aplicarPlasticidadHebiana(Neurona pre, List<Neurona> post, long timestampActual, long ventanaTemporal) {
        boolean preActiva = pre.estaActiva();
        boolean algunaPostActiva = post.stream().anyMatch(Neurona::estaActiva);
        
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
     * Solo podar si el peso está muy cerca de 0 (ni excitatorio ni inhibitorio)
     */
    public boolean debeSerPodada() {
        return Math.abs(peso) <= 0.05 || recursosAsignados <= 0.1;
    }
    
    // Getters y setters
    public Neurona getPresinaptica() { return presinaptica; }
    public List<Neurona> getPostsinapticas() { return postsinapticas; }
    public double getPeso() { return peso; }
    public void setPeso(double peso) { 
        // Permitir pesos negativos para inhibición (como neuronas GABAérgicas)
        this.peso = Math.max(-1.0, Math.min(1.0, peso)); 
    }
    public TipoConexion getTipo() { return tipo; }
    public double getRecursosAsignados() { return recursosAsignados; }
    public void setRecursosAsignados(double recursos) {
        this.recursosAsignados = Math.max(0.0, Math.min(1.0, recursos));
    }
    public long getTimestampUltimaActivacion() { return timestampUltimaActivacion; }
    public int getVecesActivadaJuntas() { return vecesActivadaJuntas; }
    
    /**
     * Método legacy para compatibilidad con tests.
     * Los engramas ya no mantienen referencias bidireccionales.
     * @return Siempre devuelve un Set vacío
     */
    public java.util.Set<String> getEngramasActivos() { 
        return new java.util.HashSet<>(); 
    }
    
    /**
     * Método legacy para compatibilidad con tests.
     * Los engramas ya no mantienen referencias bidireccionales.
     * Este método no hace nada.
     */
    public void unirseAEngrama(String engramaId) {
        // No hacer nada - sin referencias bidireccionales
    }
    
    /**
     * Método legacy para compatibilidad con tests.
     * Los engramas ya no mantienen referencias bidireccionales.
     * Este método no hace nada.
     */
    public void salirDeEngrama(String engramaId) {
        // No hacer nada - sin referencias bidireccionales
    }
}
