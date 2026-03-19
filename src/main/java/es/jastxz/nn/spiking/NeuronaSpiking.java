package es.jastxz.nn.spiking;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Unidad computacional que implementa el modelo Leaky Integrate-and-Fire (LIF).
 * 
 * <p>Esta clase representa una neurona spiking que mantiene un potencial de membrana
 * que decae exponencialmente hacia el potencial de reposo, recibe señales sinápticas,
 * y genera spikes cuando el potencial alcanza el umbral de disparo.</p>
 * 
 * <p>Características principales:</p>
 * <ul>
 *   <li>Decaimiento exponencial del potencial de membrana</li>
 *   <li>Generación de spikes al alcanzar umbral</li>
 *   <li>Período refractario después de cada spike</li>
 *   <li>Registro de historial de spikes</li>
 *   <li>Soporte opcional para homeostasis sináptica</li>
 * </ul>
 * 
 * @see EventoSpike
 */
public class NeuronaSpiking implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Identificación (inmutables)
    private final long id;
    private final int capa;
    private final int indice;
    
    // Parámetros LIF (inmutables)
    private final double umbralDisparo;
    private final double potencialReposo;
    private final double constanteDecaimiento; // tau en ms
    private final int duracionRefractario; // en timesteps
    
    // Umbral ajustado por homeostasis (mutable)
    private double umbralDisparoAjustado;
    
    // Estado temporal (mutables)
    private double potencialMembrana;
    private long timestampUltimoSpike;
    private int timestepsDesdeUltimoSpike;
    
    // Historial
    private final List<Long> historialSpikes;
    
    /** Tamaño máximo del historial de spikes para evitar crecimiento ilimitado de memoria. */
    private static final int MAX_HISTORIAL_SPIKES = 10000;
    
    // Homeostasis (opcional) - Inicializar desde ConfiguracionRed si homeostasisActiva
    private double tasaDisparoObjetivo;
    private double tasaDisparoPromedio;
    private double tasaAjusteHomeostasis;

    // Cache para optimización de decaimiento
    private transient double factorDecaimientoCache = Double.NaN;
    private transient double dtCache = Double.NaN;
    
    /**
     * Construye una nueva neurona spiking con los parámetros especificados.
     * 
     * @param id identificador único de la neurona
     * @param capa índice de la capa a la que pertenece
     * @param indice índice de la neurona dentro de su capa
     * @param umbralDisparo potencial de membrana que desencadena un spike (en mV)
     * @param potencialReposo potencial base al que tiende la membrana (en mV)
     * @param constanteDecaimiento constante de tiempo tau para el decaimiento exponencial (en ms)
     * @param duracionRefractario duración del período refractario (en timesteps)
     * @throws IllegalArgumentException si umbralDisparo <= potencialReposo
     * @throws IllegalArgumentException si constanteDecaimiento <= 0
     * @throws IllegalArgumentException si duracionRefractario < 0
     */
    public NeuronaSpiking(long id, int capa, int indice, 
                         double umbralDisparo, double potencialReposo, 
                         double constanteDecaimiento, int duracionRefractario) {
        // Validación: umbralDisparo > potencialReposo (Requisito 10.6)
        if (umbralDisparo <= potencialReposo) {
            throw new IllegalArgumentException(
                "Umbral de disparo (" + umbralDisparo + ") debe ser mayor que potencial de reposo (" + potencialReposo + ")"
            );
        }
        
        // Validación: constanteDecaimiento > 0 (Requisito 10.7)
        if (constanteDecaimiento <= 0) {
            throw new IllegalArgumentException(
                "Constante de decaimiento debe ser positiva, recibido: " + constanteDecaimiento
            );
        }
        
        // Validación: duracionRefractario >= 0
        if (duracionRefractario < 0) {
            throw new IllegalArgumentException(
                "Duración de período refractario no puede ser negativa: " + duracionRefractario
            );
        }
        
        // Campos inmutables
        this.id = id;
        this.capa = capa;
        this.indice = indice;
        this.umbralDisparo = umbralDisparo;
        this.potencialReposo = potencialReposo;
        this.constanteDecaimiento = constanteDecaimiento;
        this.duracionRefractario = duracionRefractario;
        
        // Umbral ajustado inicialmente igual al umbral base
        this.umbralDisparoAjustado = umbralDisparo;
        
        // Estado temporal inicial
        this.potencialMembrana = potencialReposo;
        this.timestampUltimoSpike = -1;
        this.timestepsDesdeUltimoSpike = Integer.MAX_VALUE; // No ha disparado nunca
        
        // Historial
        this.historialSpikes = new ArrayList<>();
        
        // Homeostasis (valores por defecto, se configuran externamente si se necesita)
        this.tasaDisparoObjetivo = 0.0;
        this.tasaDisparoPromedio = 0.0;
        this.tasaAjusteHomeostasis = 0.0;
    }
    
    // Getters para campos inmutables
    
    /**
     * @return el identificador único de la neurona
     */
    public long getId() {
        return id;
    }
    
    /**
     * @return el índice de la capa a la que pertenece
     */
    public int getCapa() {
        return capa;
    }
    
    /**
     * @return el índice de la neurona dentro de su capa
     */
    public int getIndice() {
        return indice;
    }
    
    /**
     * @return el umbral de disparo en mV
     */
    public double getUmbralDisparo() {
        return umbralDisparo;
    }
    /**
     * @return el umbral de disparo ajustado por homeostasis (en mV)
     */
    public double getUmbralDisparoAjustado() {
        return umbralDisparoAjustado;
    }

    
    /**
     * @return el potencial de reposo en mV
     */
    public double getPotencialReposo() {
        return potencialReposo;
    }
    
    /**
     * @return la constante de decaimiento tau en ms
     */
    public double getConstanteDecaimiento() {
        return constanteDecaimiento;
    }
    
    /**
     * @return la duración del período refractario en timesteps
     */
    public int getDuracionRefractario() {
        return duracionRefractario;
    }
    
    // Getters y setters para estado temporal
    
    /**
     * @return el potencial de membrana actual en mV
     */
    public double getPotencialMembrana() {
        return potencialMembrana;
    }
    
    /**
     * Establece el potencial de membrana.
     * 
     * @param potencialMembrana el nuevo potencial de membrana en mV
     */
    public void setPotencialMembrana(double potencialMembrana) {
        this.potencialMembrana = potencialMembrana;
    }
    
    /**
     * @return el timestamp del último spike generado, o -1 si nunca ha disparado
     */
    public long getTimestampUltimoSpike() {
        return timestampUltimoSpike;
    }
    
    /**
     * @return el número de timesteps desde el último spike
     */
    public int getTimestepsDesdeUltimoSpike() {
        return timestepsDesdeUltimoSpike;
    }
    
    /**
     * @return una copia del historial de spikes (timestamps)
     */
    public List<Long> getHistorialSpikes() {
        return new ArrayList<>(historialSpikes);
    }
    
    // Getters y setters para homeostasis
    
    /**
     * @return la tasa de disparo objetivo para homeostasis (en Hz)
     */
    public double getTasaDisparoObjetivo() {
        return tasaDisparoObjetivo;
    }
    
    /**
     * Establece la tasa de disparo objetivo para homeostasis.
     * 
     * @param tasaDisparoObjetivo la tasa objetivo en Hz
     */
    public void setTasaDisparoObjetivo(double tasaDisparoObjetivo) {
        this.tasaDisparoObjetivo = tasaDisparoObjetivo;
    }
    
    /**
     * @return la tasa de disparo promedio reciente (en Hz)
     */
    public double getTasaDisparoPromedio() {
        return tasaDisparoPromedio;
    }
    
    /**
     * Establece la tasa de disparo promedio.
     * 
     * @param tasaDisparoPromedio la tasa promedio en Hz
     */
    public void setTasaDisparoPromedio(double tasaDisparoPromedio) {
        this.tasaDisparoPromedio = tasaDisparoPromedio;
    }
    
    /**
     * @return la tasa de ajuste para homeostasis
     */
    public double getTasaAjusteHomeostasis() {
        return tasaAjusteHomeostasis;
    }
    
    /**
     * Establece la tasa de ajuste para homeostasis.
     * 
     * @param tasaAjusteHomeostasis la tasa de ajuste
     */
    public void setTasaAjusteHomeostasis(double tasaAjusteHomeostasis) {
        this.tasaAjusteHomeostasis = tasaAjusteHomeostasis;
    }

    /**
     * Aplica decaimiento exponencial al potencial de membrana.
     *
     * <p>El potencial de membrana decae exponencialmente hacia el potencial de reposo
     * según la fórmula del modelo Leaky Integrate-and-Fire:</p>
     *
     * <pre>
     * V(t+1) = V(t) * exp(-dt/tau) + V_reposo * (1 - exp(-dt/tau))
     * </pre>
     *
     * <p>Donde:</p>
     * <ul>
     *   <li>V(t): Potencial de membrana actual</li>
     *   <li>dt: Duración del timestep (en ms)</li>
     *   <li>tau: Constante de decaimiento (en ms)</li>
     *   <li>V_reposo: Potencial de reposo (en mV)</li>
     * </ul>
     *
     * <p>Este método implementa el requisito 1.1 (decaimiento exponencial) y 7.5
     * (aplicar decaimiento antes de procesar señales).</p>
     *
     * @param dt duración del timestep en milisegundos
     */
    public void aplicarDecaimiento(double dt) {
        // Fórmula: V(t+1) = V(t) * exp(-dt/tau) + V_reposo * (1 - exp(-dt/tau))
        // Requisitos: 1.1, 7.5

        // Cache exp(-dt/tau) ya que dt y tau son constantes durante la simulación
        if (dt != dtCache || Double.isNaN(factorDecaimientoCache)) {
            factorDecaimientoCache = Math.exp(-dt / constanteDecaimiento);
            dtCache = dt;
        }
        potencialMembrana = potencialMembrana * factorDecaimientoCache +
                           potencialReposo * (1 - factorDecaimientoCache);
    }
    
    /**
     * Recibe una señal sináptica e incrementa el potencial de membrana.
     *
     * <p>Cuando la neurona recibe una señal sináptica (típicamente el peso de una sinapsis),
     * el potencial de membrana se incrementa por el valor de la señal. Las señales pueden
     * ser positivas (excitatorias) o negativas (inhibitorias).</p>
     *
     * <p>Este método implementa el requisito 1.2 (incremento de potencial por señal sináptica).</p>
     *
     * @param señal el valor de la señal sináptica (puede ser positivo o negativo)
     */
    public void recibirSeñal(double señal) {
        // Requisito: 1.2
        potencialMembrana += señal;
    }
    
    /**
     * Evalúa si la neurona debe generar un spike en el timestep actual.
     *
     * <p>La neurona genera un spike si:</p>
     * <ul>
     *   <li>El potencial de membrana alcanza o supera el umbral de disparo</li>
     *   <li>No está en período refractario</li>
     * </ul>
     *
     * <p>Si se cumplen ambas condiciones, se llama a {@link #generarSpike(long)} para
     * generar el spike y resetear el estado de la neurona.</p>
     *
     * <p>Este método implementa los requisitos 1.3 (generación de spike al alcanzar umbral)
     * y 2.2 (rechazo de spikes durante período refractario).</p>
     *
     * @param timestepActual el timestep actual de la simulación
     * @return true si se generó un spike, false en caso contrario
     */
    public boolean evaluarActivacion(long timestepActual) {
        // Requisitos: 1.3, 2.2, 2.4
        
        // Actualizar contador de timesteps desde último spike
        if (timestampUltimoSpike >= 0) {
            timestepsDesdeUltimoSpike = (int) (timestepActual - timestampUltimoSpike);
        }
        
        // Verificar si está en período refractario
        if (estaEnRefractario(timestepActual)) {
            return false;
        }
        
        // Verificar si alcanza el umbral de disparo (usar umbral ajustado por homeostasis)
        if (potencialMembrana >= umbralDisparoAjustado) {
            generarSpike(timestepActual);
            return true;
        }
        
        return false;
    }
    
    /**
     * Genera un spike y resetea el estado de la neurona.
     *
     * <p>Cuando se genera un spike:</p>
     * <ul>
     *   <li>El potencial de membrana se resetea al potencial de reposo</li>
     *   <li>Se registra el timestamp del spike en el historial</li>
     *   <li>Se actualiza el timestamp del último spike</li>
     *   <li>Se resetea el contador de timesteps desde el último spike</li>
     * </ul>
     *
     * <p>Este método implementa los requisitos 1.4 (reset de potencial después de spike)
     * y 1.6 (registro de timestamp de spike).</p>
     *
     * @param timestamp el timestamp en el que se genera el spike
     */
    public void generarSpike(long timestamp) {
        // Requisitos: 1.4, 1.6
        
        // Resetear potencial al reposo
        potencialMembrana = potencialReposo;
        
        // Registrar spike en historial (con límite de memoria)
        historialSpikes.add(timestamp);
        if (historialSpikes.size() > MAX_HISTORIAL_SPIKES) {
            historialSpikes.remove(0);
        }
        
        // Actualizar timestamp del último spike
        timestampUltimoSpike = timestamp;
        timestepsDesdeUltimoSpike = 0;
    }
    
    /**
     * Verifica si la neurona está en período refractario.
     *
     * <p>La neurona está en período refractario si el número de timesteps desde el último
     * spike es menor que la duración del período refractario configurada.</p>
     *
     * <p>Este método implementa el requisito 2.2 (rechazo de spikes durante período refractario)
     * y 2.4 (recuperación después de período refractario).</p>
     *
     * @param timestepActual el timestep actual de la simulación
     * @return true si la neurona está en período refractario, false en caso contrario
     */
    public boolean estaEnRefractario(long timestepActual) {
        // Requisitos: 2.2, 2.4
        
        // Si nunca ha disparado, no está en refractario
        if (timestampUltimoSpike < 0) {
            return false;
        }
        
        // Calcular timesteps desde último spike
        long timestepsDesde = timestepActual - timestampUltimoSpike;
        
        // Está en refractario si no ha pasado suficiente tiempo
        return timestepsDesde < duracionRefractario;
    }
    
    /**
     * Calcula la frecuencia de disparo de la neurona en una ventana temporal.
     *
     * <p>La frecuencia se calcula como el número de spikes en la ventana temporal
     * dividido por la duración de la ventana (en segundos), resultando en Hz.</p>
     *
     * <p>Este método implementa el requisito 9.3 (cálculo de frecuencia de disparo promedio).</p>
     *
     * @param ventanaTemporal duración de la ventana temporal en timesteps
     * @return la frecuencia de disparo en Hz, o 0 si no hay spikes en la ventana
     */
    public double calcularFrecuencia(long ventanaTemporal) {
        // Requisito: 9.3
        
        if (historialSpikes.isEmpty() || ventanaTemporal <= 0) {
            return 0.0;
        }
        
        // Obtener el timestamp más reciente
        long timestampActual = historialSpikes.get(historialSpikes.size() - 1);
        long timestampInicio = timestampActual - ventanaTemporal;
        
        // Contar spikes en la ventana [timestampInicio, timestampActual]
        long conteo = historialSpikes.stream()
            .filter(t -> t > timestampInicio && t <= timestampActual)
            .count();
        
        // Calcular frecuencia (asumiendo 1 timestep = 1 ms)
        // Frecuencia = spikes / (ventana_ms / 1000) = spikes * 1000 / ventana_ms
        return (conteo * 1000.0) / ventanaTemporal;
    }

    /**
     * Aplica ajuste homeostático al umbral de disparo para mantener la tasa objetivo.
     *
     * <p>La homeostasis sináptica ajusta dinámicamente el umbral de disparo:</p>
     * <ul>
     *   <li>Si la tasa actual es menor que la objetivo: decrementa el umbral (más excitable)</li>
     *   <li>Si la tasa actual es mayor que la objetivo: incrementa el umbral (menos excitable)</li>
     * </ul>
     *
     * <p>El ajuste se aplica gradualmente usando la tasa de aprendizaje homeostático:</p>
     * <pre>
     * error = tasa_objetivo - tasa_actual
     * Si error &gt; 0: umbral_nuevo = umbral_actual * (1 - tasa_ajuste * error_normalizado)
     * Si error &lt; 0: umbral_nuevo = umbral_actual * (1 + tasa_ajuste * |error_normalizado|)
     * </pre>
     *
     * @param timestampActual timestamp actual de la simulación
     * @param ventanaTemporal duración de la ventana temporal para calcular la tasa actual (en timesteps)
     * @param umbralMin umbral mínimo permitido (en mV)
     * @param umbralMax umbral máximo permitido (en mV)
     */
    public void aplicarHomeostasis(long timestampActual, long ventanaTemporal, double umbralMin, double umbralMax) {
        // Requisitos: 18.1, 18.2, 18.3, 18.4, 18.5, 18.7

        // Si homeostasis no está configurada (tasa de ajuste es 0), no hacer nada
        if (tasaAjusteHomeostasis == 0.0 || tasaDisparoObjetivo == 0.0) {
            return;
        }

        // Calcular tasa actual usando ventana deslizante (Requisito 18.7)
        // Contar spikes en la ventana [timestampActual - ventanaTemporal, timestampActual]
        long timestampInicio = timestampActual - ventanaTemporal;
        long conteo = historialSpikes.stream()
            .filter(t -> t >= timestampInicio && t <= timestampActual)
            .count();

        // Calcular frecuencia en Hz (asumiendo 1 timestep = 1 ms)
        double tasaActual = (conteo * 1000.0) / ventanaTemporal;

        // Actualizar tasa promedio (para consulta externa)
        tasaDisparoPromedio = tasaActual;

        // Calcular error: diferencia entre objetivo y actual
        double error = tasaDisparoObjetivo - tasaActual;

        // Normalizar error por la tasa objetivo para tener un factor de escala consistente
        double errorNormalizado = error / tasaDisparoObjetivo;

        // Calcular nuevo umbral basado en el error
        double nuevoUmbral = umbralDisparoAjustado;

        if (error > 0) {
            // Tasa actual < objetivo: decrementar umbral para hacer neurona más excitable (Requisito 18.3)
            // Decrementar significa hacer el umbral más negativo (más fácil de alcanzar)
            double ajuste = Math.abs(umbralDisparoAjustado) * tasaAjusteHomeostasis * errorNormalizado;
            nuevoUmbral = umbralDisparoAjustado - ajuste;
        } else if (error < 0) {
            // Tasa actual > objetivo: incrementar umbral para hacer neurona menos excitable (Requisito 18.4)
            // Incrementar significa hacer el umbral menos negativo (más difícil de alcanzar)
            double ajuste = Math.abs(umbralDisparoAjustado) * tasaAjusteHomeostasis * Math.abs(errorNormalizado);
            nuevoUmbral = umbralDisparoAjustado + ajuste;
        }

        // Aplicar límites al umbral (clamp) y actualizar el campo mutable
        umbralDisparoAjustado = Math.max(umbralMin, Math.min(umbralMax, nuevoUmbral));
    }

    /**
     * Resetea todo el estado temporal de la neurona a su estado inicial.
     *
     * <p>Este método limpia el estado temporal de la neurona, permitiendo que sea
     * reutilizada para procesar nuevos patrones sin crear una nueva instancia.
     * Es útil para procesamiento por lotes donde los patrones deben estar aislados.</p>
     *
     * <h3>Operaciones realizadas:</h3>
     * <ul>
     *   <li>Resetea el potencial de membrana al potencial de reposo</li>
     *   <li>Limpia el historial de spikes</li>
     *   <li>Resetea el timestamp del último spike a -1 (nunca ha disparado)</li>
     *   <li>Resetea el contador de timesteps desde el último spike a Integer.MAX_VALUE</li>
     * </ul>
     *
     * <p><b>Nota:</b> Este método NO modifica los parámetros inmutables de la neurona
     * (umbral, reposo, tau, refractario) ni los parámetros de homeostasis.</p>
     */
    public void resetearEstadoTemporal() {
        // Resetear potencial de membrana al potencial de reposo
        this.potencialMembrana = this.potencialReposo;

        // Limpiar historial de spikes
        this.historialSpikes.clear();

        // Resetear timestamp del último spike (indicando que no ha disparado)
        this.timestampUltimoSpike = -1;

        // Resetear contador de timesteps desde el último spike
        this.timestepsDesdeUltimoSpike = Integer.MAX_VALUE;
    }

    @Override
    public String toString() {
        return "NeuronaSpiking{" +
                "id=" + id +
                ", capa=" + capa +
                ", indice=" + indice +
                ", potencialMembrana=" + String.format("%.2f", potencialMembrana) +
                ", umbralDisparo=" + String.format("%.2f", umbralDisparo) +
                ", potencialReposo=" + String.format("%.2f", potencialReposo) +
                ", spikes=" + historialSpikes.size() +
                '}';
    }
}
