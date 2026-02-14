package es.jastxz.nn;

import java.util.*;
import java.io.Serializable;

import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoNeurona;

/**
 * Representa una neurona biológica con capacidad de plasticidad,
 * competición por recursos y formación de engramas.
 */
public class Neurona implements Serializable {
    private static final long serialVersionUID = 1L;
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
    private boolean activa;
    
    // Gestión de recursos (competición biológica)
    private double recursosAsignados;   // 0.0 a 1.0
    private double factorSupervivencia; // Basado en uso reciente
    
    // Temporalidad
    private long timestampUltimaActivacion;
    private int contadorActivaciones;
    
    // Facilitación temporal (para completado de patrón en engramas)
    private double facilitacionTemporal;

    private List<Neurona> vecinas;
    
    public Neurona(long id, TipoNeurona tipo, double valorAlmacenado, PotencialMemoria potencialInicial) {
        this.id = id;
        this.tipo = tipo;
        this.valorAlmacenado = valorAlmacenado;
        this.potencial = potencialInicial;
        
        // Si se inicializa con potencial de pico, marcar como activa
        this.activa = (potencialInicial == PotencialMemoria.PICO);
        
        this.recursosAsignados = 1.0;  // Todas empiezan con recursos máximos
        this.factorSupervivencia = 1.0;
        
        this.timestampUltimaActivacion = 0L;
        this.contadorActivaciones = 0;
        
        this.facilitacionTemporal = 0.0;

        this.vecinas = new ArrayList<>();
    }
    
    /**
     * Evalúa si la neurona debe activarse basándose en inputs de dendritas
     */
    public boolean evaluar(List<Conexion> dendritas, long timestampActual) {
        double sumaInputs = 0.0;
        boolean hayInputsActivos = false;
        
        // Sumar inputs ponderados por peso sináptico
        for (Conexion dendrita : dendritas) {
            Neurona neurona = this.vecinas.stream()
                .filter(n -> n.equals(dendrita.getPresinaptica()))
                .findFirst().orElseThrow();
            if (neurona.estaActiva()) {
                sumaInputs += dendrita.getPeso() * neurona.getPotencial();
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
        double umbralAjustado = PotencialMemoria.UMBRAL.getValor() * (1.0 - facilitacionTemporal);
        
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
    
    // Variable temporal para acumular señales durante propagación
    private transient double potencialAcumulado = 0.0;
    
    /**
     * Recibe una señal de una conexión y la acumula
     * Usado para propagación centrada en conexiones
     */
    public void recibirSeñal(double señal) {
        this.potencialAcumulado += señal;
    }
    
    /**
     * Evalúa si debe activarse basándose en el potencial acumulado
     * Usado después de recibir todas las señales
     */
    public boolean evaluarActivacion(long timestampActual) {
        // Si no hay señales acumuladas, no puede activarse
        if (potencialAcumulado == 0.0) {
            facilitacionTemporal = Math.max(0.0, facilitacionTemporal - 0.1);
            resetear();
            potencialAcumulado = 0.0;
            return false;
        }
        
        // Umbral normalizado (diferencia entre umbral y reposo en escala normalizada)
        // UMBRAL = -55mV, REPOSO = -70mV, PICO = 40mV
        // Rango total: 40 - (-70) = 110mV
        // Diferencia umbral-reposo: -55 - (-70) = 15mV
        // Umbral normalizado: 15/110 ≈ 0.136 (aproximadamente 13.6% del rango)
        double umbralNormalizado = 15.0; // Umbral en mV sobre el reposo
        double umbralAjustado = umbralNormalizado * (1.0 - facilitacionTemporal);
        
        // Activación todo-o-nada
        if (potencialAcumulado >= umbralAjustado) {
            activar(timestampActual);
            facilitacionTemporal = 0.0;  // Reset tras activación
            potencialAcumulado = 0.0;
            return true;
        }
        
        // Decaimiento gradual de la facilitación
        facilitacionTemporal = Math.max(0.0, facilitacionTemporal - 0.1);
        resetear();
        potencialAcumulado = 0.0;
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
     * Añade una vecina a esta neurona
     */
    public void añadirVecina(Neurona vecina) {
        this.vecinas.add(vecina);
    }

    /**
     * Añade varias vecinas a esta neurona
     */
    public void añadirVecinas(List<Neurona> vecinas) {
        this.vecinas.addAll(vecinas);
    }
    
    /**
     * Elimina una vecina de esta neurona
     */
    public void eliminarVecina(Neurona vecina) {
        this.vecinas.remove(vecina);
    }

    // Getters y setters
    public long getId() { return id; }
    public TipoNeurona getTipo() { return tipo; }
    public double getValorAlmacenado() {return valorAlmacenado; }
    public void setValorAlmacenado(double valorAlmacenado) {this.valorAlmacenado = valorAlmacenado; }
    public double getPotencial() { return potencial.getValor(); }
    public void setPotencial(PotencialMemoria potencial) { this.potencial = potencial; }
    public boolean estaActiva() { return activa; }
    
    public double getRecursosAsignados() { return recursosAsignados; }
    public void setRecursosAsignados(double recursos) { 
        this.recursosAsignados = Math.max(0.0, Math.min(1.0, recursos)); 
    }
    public double getFactorSupervivencia() { return factorSupervivencia; }
    public long getTimestampUltimaActivacion() { return timestampUltimaActivacion; }
    public int getContadorActivaciones() { return contadorActivaciones; }
    
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

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + (int) (id ^ (id >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Neurona other = (Neurona) obj;
        if (id != other.id)
            return false;
        return true;
    }
}
