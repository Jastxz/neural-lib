package es.jastxz.nn.spiking;

/**
 * Builder para crear instancias de {@link ConfiguracionRed} con una API fluida.
 * 
 * <p>Proporciona valores por defecto razonables para todos los parámetros, permitiendo
 * crear configuraciones válidas con código mínimo mientras se mantiene la flexibilidad
 * para personalización completa.</p>
 * 
 * <p>Ejemplo de uso básico:</p>
 * <pre>{@code
 * ConfiguracionRed config = new ConfiguracionRedBuilder()
 *     .topologia(10, 20, 10)
 *     .build();
 * }</pre>
 * 
 * <p>Ejemplo con personalización completa:</p>
 * <pre>{@code
 * ConfiguracionRed config = new ConfiguracionRedBuilder()
 *     .topologia(784, 100, 10)
 *     .parametrosLIF(-55.0, -70.0, 20.0, 2)
 *     .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
 *     .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
 *     .inicializacionPesos(TipoInicializacion.UNIFORME, 0.0, 1.0)
 *     .homeostasis(true, 10.0, 0.01)
 *     .build();
 * }</pre>
 * 
 * @since 1.0
 * @see ConfiguracionRed
 */
public class ConfiguracionRedBuilder {
    
    // ==================== Valores por Defecto ====================
    
    // Arquitectura
    private int[] topologia = {10, 20, 10};
    
    // Parámetros neuronales (LIF) - Valores típicos biológicamente plausibles
    private double umbralDisparo = -55.0;           // mV
    private double potencialReposo = -70.0;         // mV
    private double constanteDecaimiento = 20.0;     // ms
    private int duracionRefractario = 2;            // timesteps
    
    // Parámetros STDP - Valores típicos de la literatura
    private double amplitudLTP = 0.01;
    private double amplitudLTD = 0.012;             // Ligeramente mayor que LTP
    private double tauLTP = 20.0;                   // ms
    private double tauLTD = 20.0;                   // ms
    
    // Codificación/Decodificación
    private double frecuenciaMaxima = 100.0;        // Hz
    private ModoCodificacion modoCodificacion = ModoCodificacion.POISSON;
    private int ventanaDecodificacion = 50;         // timesteps
    
    // Inicialización de pesos
    private TipoInicializacion tipoInicializacion = TipoInicializacion.UNIFORME;
    private double pesoMin = 0.0;
    private double pesoMax = 1.0;
    
    // Normalización
    private TipoNormalizacion tipoNormalizacion = TipoNormalizacion.L1;
    private double valorObjetivoNormalizacion = 1.0;
    
    // Retardos sinápticos
    private int retardoMin = 1;                     // timesteps
    private int retardoMax = 5;                     // timesteps
    
    // Homeostasis (desactivada por defecto)
    private boolean homeostasisActiva = false;
    private double tasaDisparoObjetivo = 10.0;      // Hz
    private double tasaAjusteHomeostasis = 0.01;
    
    // Inhibición lateral (desactivada por defecto)
    private boolean inhibicionLateralActiva = false;
    private int radioInhibicion = 2;                // neuronas
    private double fuerzaInhibicion = 0.5;
    
    // Simulación
    private double duracionTimestep = 1.0;          // ms
    
    // ==================== Métodos Builder ====================
    
    /**
     * Configura la topología de la red.
     * 
     * @param capas tamaños de cada capa (debe tener al menos una capa con al menos una neurona)
     * @return este builder para encadenamiento
     * @throws IllegalArgumentException si capas es null, vacío, o contiene valores no positivos
     */
    public ConfiguracionRedBuilder topologia(int... capas) {
        if (capas == null || capas.length == 0) {
            throw new IllegalArgumentException("La topología debe tener al menos una capa");
        }
        for (int i = 0; i < capas.length; i++) {
            if (capas[i] <= 0) {
                throw new IllegalArgumentException(
                    "Capa " + i + " debe tener al menos una neurona, recibido: " + capas[i]
                );
            }
        }
        this.topologia = capas.clone();
        return this;
    }
    
    /**
     * Configura todos los parámetros del modelo Leaky Integrate-and-Fire.
     * 
     * @param umbral umbral de disparo en mV (debe ser mayor que reposo)
     * @param reposo potencial de reposo en mV
     * @param tau constante de decaimiento en ms (debe ser > 0)
     * @param refractario duración del período refractario en timesteps (debe ser >= 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder parametrosLIF(double umbral, double reposo, double tau, int refractario) {
        this.umbralDisparo = umbral;
        this.potencialReposo = reposo;
        this.constanteDecaimiento = tau;
        this.duracionRefractario = refractario;
        return this;
    }
    
    /**
     * Configura el umbral de disparo.
     * 
     * @param umbral umbral de disparo en mV
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder umbralDisparo(double umbral) {
        this.umbralDisparo = umbral;
        return this;
    }
    
    /**
     * Configura el potencial de reposo.
     * 
     * @param reposo potencial de reposo en mV
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder potencialReposo(double reposo) {
        this.potencialReposo = reposo;
        return this;
    }
    
    /**
     * Configura la constante de decaimiento.
     * 
     * @param tau constante de decaimiento en ms (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder constanteDecaimiento(double tau) {
        this.constanteDecaimiento = tau;
        return this;
    }
    
    /**
     * Configura la duración del período refractario.
     * 
     * @param duracion duración en timesteps (debe ser >= 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder duracionRefractario(int duracion) {
        this.duracionRefractario = duracion;
        return this;
    }
    
    /**
     * Configura todos los parámetros de STDP (Spike-Timing-Dependent Plasticity).
     * 
     * @param aLTP amplitud de Long-Term Potentiation (debe ser > 0)
     * @param aLTD amplitud de Long-Term Depression (debe ser > 0)
     * @param tLTP constante de tiempo para LTP en ms (debe ser > 0)
     * @param tLTD constante de tiempo para LTD en ms (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder parametrosSTDP(double aLTP, double aLTD, double tLTP, double tLTD) {
        this.amplitudLTP = aLTP;
        this.amplitudLTD = aLTD;
        this.tauLTP = tLTP;
        this.tauLTD = tLTD;
        return this;
    }
    
    /**
     * Configura la amplitud de Long-Term Potentiation.
     * 
     * @param amplitud amplitud de LTP (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder amplitudLTP(double amplitud) {
        this.amplitudLTP = amplitud;
        return this;
    }
    
    /**
     * Configura la amplitud de Long-Term Depression.
     * 
     * @param amplitud amplitud de LTD (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder amplitudLTD(double amplitud) {
        this.amplitudLTD = amplitud;
        return this;
    }
    
    /**
     * Configura la constante de tiempo para LTP.
     * 
     * @param tau constante de tiempo en ms (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder tauLTP(double tau) {
        this.tauLTP = tau;
        return this;
    }
    
    /**
     * Configura la constante de tiempo para LTD.
     * 
     * @param tau constante de tiempo en ms (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder tauLTD(double tau) {
        this.tauLTD = tau;
        return this;
    }
    
    /**
     * Configura todos los parámetros de codificación y decodificación.
     * 
     * @param freqMax frecuencia máxima en Hz (debe ser > 0)
     * @param modo modo de codificación (no puede ser null)
     * @param ventana tamaño de ventana de decodificación en timesteps (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder parametrosCodificacion(double freqMax, ModoCodificacion modo, int ventana) {
        this.frecuenciaMaxima = freqMax;
        this.modoCodificacion = modo;
        this.ventanaDecodificacion = ventana;
        return this;
    }
    
    /**
     * Configura la frecuencia máxima de disparo.
     * 
     * @param frecuencia frecuencia máxima en Hz (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder frecuenciaMaxima(double frecuencia) {
        this.frecuenciaMaxima = frecuencia;
        return this;
    }
    
    /**
     * Configura el modo de codificación.
     * 
     * @param modo modo de codificación (no puede ser null)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder modoCodificacion(ModoCodificacion modo) {
        this.modoCodificacion = modo;
        return this;
    }
    
    /**
     * Configura el tamaño de la ventana de decodificación.
     * 
     * @param ventana tamaño en timesteps (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder ventanaDecodificacion(int ventana) {
        this.ventanaDecodificacion = ventana;
        return this;
    }
    
    /**
     * Configura los parámetros de inicialización de pesos.
     * 
     * @param tipo tipo de inicialización (no puede ser null)
     * @param min peso mínimo (debe ser < max)
     * @param max peso máximo (debe ser > min)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder inicializacionPesos(TipoInicializacion tipo, double min, double max) {
        this.tipoInicializacion = tipo;
        this.pesoMin = min;
        this.pesoMax = max;
        return this;
    }
    
    /**
     * Configura el tipo de inicialización de pesos.
     * 
     * @param tipo tipo de inicialización (no puede ser null)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder tipoInicializacion(TipoInicializacion tipo) {
        this.tipoInicializacion = tipo;
        return this;
    }
    
    /**
     * Configura el rango de pesos permitidos.
     * 
     * @param min peso mínimo (debe ser < max)
     * @param max peso máximo (debe ser > min)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder rangoPesos(double min, double max) {
        this.pesoMin = min;
        this.pesoMax = max;
        return this;
    }
    
    /**
     * Configura los parámetros de normalización de pesos.
     * 
     * @param tipo tipo de normalización (no puede ser null)
     * @param valorObjetivo valor objetivo para normalización (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder parametrosNormalizacion(TipoNormalizacion tipo, double valorObjetivo) {
        this.tipoNormalizacion = tipo;
        this.valorObjetivoNormalizacion = valorObjetivo;
        return this;
    }
    
    /**
     * Configura el tipo de normalización.
     * 
     * @param tipo tipo de normalización (no puede ser null)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder tipoNormalizacion(TipoNormalizacion tipo) {
        this.tipoNormalizacion = tipo;
        return this;
    }
    
    /**
     * Configura el valor objetivo de normalización.
     * 
     * @param valor valor objetivo (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder valorObjetivoNormalizacion(double valor) {
        this.valorObjetivoNormalizacion = valor;
        return this;
    }
    
    /**
     * Configura el rango de retardos sinápticos.
     * 
     * @param min retardo mínimo en timesteps (debe ser >= 0)
     * @param max retardo máximo en timesteps (debe ser >= min)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder retardos(int min, int max) {
        this.retardoMin = min;
        this.retardoMax = max;
        return this;
    }
    
    /**
     * Configura el retardo mínimo.
     * 
     * @param min retardo mínimo en timesteps (debe ser >= 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder retardoMin(int min) {
        this.retardoMin = min;
        return this;
    }
    
    /**
     * Configura el retardo máximo.
     * 
     * @param max retardo máximo en timesteps (debe ser >= retardoMin)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder retardoMax(int max) {
        this.retardoMax = max;
        return this;
    }
    
    /**
     * Configura la homeostasis sináptica.
     * 
     * @param activar indica si homeostasis está activa
     * @param tasaObjetivo tasa de disparo objetivo en Hz (debe ser > 0 si activar es true)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder homeostasis(boolean activar, double tasaObjetivo) {
        this.homeostasisActiva = activar;
        this.tasaDisparoObjetivo = tasaObjetivo;
        return this;
    }
    
    /**
     * Configura la homeostasis sináptica con todos los parámetros.
     * 
     * @param activar indica si homeostasis está activa
     * @param tasaObjetivo tasa de disparo objetivo en Hz (debe ser > 0 si activar es true)
     * @param tasaAjuste tasa de ajuste homeostático (debe ser > 0 si activar es true)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder homeostasis(boolean activar, double tasaObjetivo, double tasaAjuste) {
        this.homeostasisActiva = activar;
        this.tasaDisparoObjetivo = tasaObjetivo;
        this.tasaAjusteHomeostasis = tasaAjuste;
        return this;
    }
    
    /**
     * Activa la homeostasis con valores por defecto.
     * 
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder conHomeostasis() {
        this.homeostasisActiva = true;
        return this;
    }
    
    /**
     * Desactiva la homeostasis.
     * 
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder sinHomeostasis() {
        this.homeostasisActiva = false;
        return this;
    }
    
    /**
     * Configura la tasa de disparo objetivo para homeostasis.
     * 
     * @param tasa tasa objetivo en Hz (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder tasaDisparoObjetivo(double tasa) {
        this.tasaDisparoObjetivo = tasa;
        return this;
    }
    
    /**
     * Configura la tasa de ajuste homeostático.
     * 
     * @param tasa tasa de ajuste (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder tasaAjusteHomeostasis(double tasa) {
        this.tasaAjusteHomeostasis = tasa;
        return this;
    }
    
    /**
     * Configura la inhibición lateral.
     * 
     * @param activar indica si inhibición lateral está activa
     * @param radio radio de inhibición en número de neuronas (debe ser > 0 si activar es true)
     * @param fuerza fuerza de inhibición (debe ser > 0 si activar es true)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder inhibicionLateral(boolean activar, int radio, double fuerza) {
        this.inhibicionLateralActiva = activar;
        this.radioInhibicion = radio;
        this.fuerzaInhibicion = fuerza;
        return this;
    }
    
    /**
     * Activa la inhibición lateral con valores por defecto.
     * 
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder conInhibicionLateral() {
        this.inhibicionLateralActiva = true;
        return this;
    }
    
    /**
     * Desactiva la inhibición lateral.
     * 
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder sinInhibicionLateral() {
        this.inhibicionLateralActiva = false;
        return this;
    }
    
    /**
     * Configura el radio de inhibición lateral.
     * 
     * @param radio radio en número de neuronas (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder radioInhibicion(int radio) {
        this.radioInhibicion = radio;
        return this;
    }
    
    /**
     * Configura la fuerza de inhibición lateral.
     * 
     * @param fuerza fuerza de inhibición (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder fuerzaInhibicion(double fuerza) {
        this.fuerzaInhibicion = fuerza;
        return this;
    }
    
    /**
     * Configura la duración del timestep de simulación.
     * 
     * @param duracion duración en milisegundos (debe ser > 0)
     * @return este builder para encadenamiento
     */
    public ConfiguracionRedBuilder duracionTimestep(double duracion) {
        this.duracionTimestep = duracion;
        return this;
    }
    
    /**
     * Construye y retorna una instancia de {@link ConfiguracionRed} con los parámetros configurados.
     * 
     * <p>Valida todos los parámetros antes de construir. Si algún parámetro es inválido,
     * lanza {@link IllegalArgumentException} con un mensaje descriptivo.</p>
     * 
     * @return nueva instancia de ConfiguracionRed
     * @throws IllegalArgumentException si algún parámetro es inválido
     */
    public ConfiguracionRed build() {
        return new ConfiguracionRed(
            topologia,
            umbralDisparo,
            potencialReposo,
            constanteDecaimiento,
            duracionRefractario,
            amplitudLTP,
            amplitudLTD,
            tauLTP,
            tauLTD,
            frecuenciaMaxima,
            modoCodificacion,
            ventanaDecodificacion,
            tipoInicializacion,
            pesoMin,
            pesoMax,
            tipoNormalizacion,
            valorObjetivoNormalizacion,
            retardoMin,
            retardoMax,
            homeostasisActiva,
            tasaDisparoObjetivo,
            tasaAjusteHomeostasis,
            inhibicionLateralActiva,
            radioInhibicion,
            fuerzaInhibicion,
            duracionTimestep
        );
    }
}
