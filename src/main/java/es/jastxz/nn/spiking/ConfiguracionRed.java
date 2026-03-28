package es.jastxz.nn.spiking;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Configuración inmutable para una red neuronal de spikes.
 * 
 * <p>Contiene todos los parámetros necesarios para construir y configurar una
 * {@link RedNeuralSpiking}, incluyendo arquitectura de red, parámetros del modelo
 * LIF, configuración de STDP, esquemas de codificación/decodificación, y características
 * avanzadas como homeostasis e inhibición lateral.</p>
 * 
 * <p>Todos los campos son inmutables después de la construcción. Use
 * {@link ConfiguracionRedBuilder} para crear instancias con una API fluida.</p>
 * 
 * @since 1.0
 * @see ConfiguracionRedBuilder
 * @see RedNeuralSpiking
 */
public class ConfiguracionRed implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // ==================== Arquitectura ====================
    
    /**
     * Topología de la red como array de tamaños de capa.
     * Por ejemplo, [10, 20, 10] representa una red con capa de entrada de 10 neuronas,
     * capa oculta de 20 neuronas, y capa de salida de 10 neuronas.
     */
    public final int[] topologia;
    
    // ==================== Parámetros Neuronales (LIF) ====================
    
    /**
     * Umbral de disparo en milivoltios (mV).
     * Cuando el potencial de membrana alcanza o supera este valor, la neurona genera un spike.
     * Típicamente -55.0 mV.
     */
    public final double umbralDisparo;
    
    /**
     * Potencial de reposo en milivoltios (mV).
     * Valor base al que tiende el potencial de membrana por decaimiento exponencial.
     * Típicamente -70.0 mV.
     */
    public final double potencialReposo;
    
    /**
     * Constante de decaimiento (tau) en milisegundos (ms).
     * Controla la velocidad de decaimiento exponencial del potencial hacia el reposo.
     * Valores mayores producen decaimiento más lento. Típicamente 20.0 ms.
     */
    public final double constanteDecaimiento;
    
    /**
     * Duración del período refractario en timesteps.
     * Número de timesteps durante los cuales la neurona no puede disparar después de un spike.
     * Típicamente 2 timesteps.
     */
    public final int duracionRefractario;
    
    // ==================== Parámetros STDP ====================
    
    /**
     * Amplitud de Long-Term Potentiation (LTP).
     * Factor de escala para el refuerzo sináptico cuando el spike presinaptico precede al postsinaptico.
     * Típicamente 0.01.
     */
    public final double amplitudLTP;
    
    /**
     * Amplitud de Long-Term Depression (LTD).
     * Factor de escala para el debilitamiento sináptico cuando el spike postsinaptico precede al presinaptico.
     * Típicamente 0.012.
     */
    public final double amplitudLTD;
    
    /**
     * Constante de tiempo para LTP en milisegundos (ms).
     * Controla la ventana temporal exponencial para el refuerzo sináptico.
     * Típicamente 20.0 ms.
     */
    public final double tauLTP;
    
    /**
     * Constante de tiempo para LTD en milisegundos (ms).
     * Controla la ventana temporal exponencial para el debilitamiento sináptico.
     * Típicamente 20.0 ms.
     */
    public final double tauLTD;
    
    // ==================== Parámetros de Codificación/Decodificación ====================
    
    /**
     * Frecuencia máxima de disparo en Hertz (Hz).
     * Frecuencia de spikes correspondiente a un valor de entrada de 1.0.
     * Típicamente 100.0 Hz.
     */
    public final double frecuenciaMaxima;
    
    /**
     * Modo de codificación para convertir valores continuos en trenes de spikes.
     * Opciones: POISSON (probabilístico), REGULAR (determinístico), BURST (ráfagas).
     */
    public final ModoCodificacion modoCodificacion;
    
    /**
     * Tamaño de la ventana de decodificación en timesteps.
     * Número de timesteps usados para contar spikes y calcular frecuencias de salida.
     * Típicamente 50 timesteps.
     */
    public final int ventanaDecodificacion;
    
    // ==================== Inicialización de Pesos ====================
    
    /**
     * Tipo de inicialización para pesos sinápticos.
     * Opciones: UNIFORME, NORMAL, CONSTANTE, DESDE_ARRAY.
     */
    public final TipoInicializacion tipoInicializacion;
    
    /**
     * Peso mínimo permitido para sinapsis.
     * Límite inferior para pesos sinápticos durante inicialización y aprendizaje.
     * Típicamente 0.0.
     */
    public final double pesoMin;
    
    /**
     * Peso máximo permitido para sinapsis.
     * Límite superior para pesos sinápticos durante inicialización y aprendizaje.
     * Típicamente 1.0.
     */
    public final double pesoMax;
    
    // ==================== Parámetros de Normalización ====================
    
    /**
     * Tipo de normalización para pesos sinápticos.
     * Opciones: L1 (suma de valores absolutos), L2 (suma de cuadrados).
     */
    public final TipoNormalizacion tipoNormalizacion;
    
    /**
     * Valor objetivo para normalización de pesos.
     * Valor al que se escalan los pesos entrantes a cada neurona durante normalización.
     * Típicamente 1.0.
     */
    public final double valorObjetivoNormalizacion;
    
    // ==================== Retardos Sinápticos ====================
    
    /**
     * Retardo mínimo en timesteps para sinapsis.
     * Retardo mínimo de transmisión sináptica. Típicamente 1 timestep.
     */
    public final int retardoMin;
    
    /**
     * Retardo máximo en timesteps para sinapsis.
     * Retardo máximo de transmisión sináptica. Típicamente 5 timesteps.
     */
    public final int retardoMax;
    
    // ==================== Homeostasis (Opcional) ====================
    
    /**
     * Indica si la homeostasis sináptica está activa.
     * Cuando está activa, las neuronas ajustan su umbral de disparo para mantener
     * una tasa de disparo objetivo.
     */
    public final boolean homeostasisActiva;
    
    /**
     * Tasa de disparo objetivo en Hertz (Hz) para homeostasis.
     * Frecuencia de disparo que las neuronas intentan mantener mediante ajuste homeostático.
     * Típicamente 10.0 Hz.
     */
    public final double tasaDisparoObjetivo;
    
    /**
     * Tasa de ajuste para homeostasis.
     * Factor de aprendizaje para ajustes homeostáticos del umbral de disparo.
     * Valores menores producen ajustes más lentos y estables. Típicamente 0.01.
     */
    public final double tasaAjusteHomeostasis;
    
    // ==================== Inhibición Lateral (Opcional) ====================
    
    /**
     * Indica si la inhibición lateral está activa.
     * Cuando está activa, las neuronas que disparan envían señales inhibitorias
     * a neuronas vecinas en la misma capa.
     */
    public final boolean inhibicionLateralActiva;
    
    /**
     * Radio de inhibición lateral en número de neuronas.
     * Distancia máxima (en topología espacial) para aplicar inhibición lateral.
     * Típicamente 2 neuronas.
     */
    public final int radioInhibicion;
    
    /**
     * Fuerza de inhibición lateral.
     * Magnitud de la señal inhibitoria enviada a neuronas vecinas.
     * Típicamente 0.5.
     */
    public final double fuerzaInhibicion;
    
    // ==================== Simulación ====================
    
    /**
     * Duración de cada timestep en milisegundos (ms).
     * Resolución temporal de la simulación. Típicamente 1.0 ms.
     */
    public final double duracionTimestep;

    // ==================== Winner-Take-All (Opcional) ====================

    /**
     * Indica si el mecanismo Winner-Take-All está activo.
     * Cuando está activo, solo la neurona con mayor actividad en un grupo
     * mantiene su potencial; las demás son suprimidas.
     */
    public final boolean wtaActivo;

    /**
     * Indica si WTA se aplica en la capa de salida.
     * Fuerza que una sola neurona de salida "gane", mejorando la
     * discriminación en problemas de clasificación multi-clase.
     */
    public final boolean wtaCapaSalida;

    /**
     * Indica si WTA se aplica en capas ocultas.
     * Fuerza representaciones sparse donde solo unas pocas neuronas
     * se activan por patrón.
     */
    public final boolean wtaCapasOcultas;

    /**
     * Radio de competición WTA en número de neuronas.
     * Define el tamaño del grupo de neuronas que compiten entre sí.
     * Un valor de 0 significa competición global (toda la capa).
     * Típicamente 0 (global) o 3-5 (local).
     */
    public final int radioWTA;

    /**
     * Fuerza de supresión WTA.
     * Magnitud de la señal inhibitoria aplicada a las neuronas perdedoras.
     * Valores mayores producen supresión más agresiva.
     * Típicamente 1.0-5.0.
     */
    public final double fuerzaWTA;

    /**
     * Umbral mínimo de activación para participar en la competición WTA.
     * Neuronas con potencial de membrana por debajo de este umbral
     * (relativo al potencial de reposo) no participan en la competición.
     * Típicamente 0.0-0.5 (fracción de la brecha umbral-reposo).
     */
    public final double umbralActivacionWTA;
    
    /**
     * Constructor que valida todos los parámetros.
     * 
     * <p>Se recomienda usar {@link ConfiguracionRedBuilder} en lugar de este constructor
     * directamente para una API más conveniente.</p>
     * 
     * @param topologia array de tamaños de capa (debe tener al menos una capa)
     * @param umbralDisparo umbral de disparo en mV (debe ser mayor que potencialReposo)
     * @param potencialReposo potencial de reposo en mV
     * @param constanteDecaimiento constante de decaimiento tau en ms (debe ser > 0)
     * @param duracionRefractario duración del período refractario en timesteps (debe ser >= 0)
     * @param amplitudLTP amplitud de LTP (debe ser > 0)
     * @param amplitudLTD amplitud de LTD (debe ser > 0)
     * @param tauLTP constante de tiempo para LTP en ms (debe ser > 0)
     * @param tauLTD constante de tiempo para LTD en ms (debe ser > 0)
     * @param frecuenciaMaxima frecuencia máxima en Hz (debe ser > 0)
     * @param modoCodificacion modo de codificación (no puede ser null)
     * @param ventanaDecodificacion tamaño de ventana en timesteps (debe ser > 0)
     * @param tipoInicializacion tipo de inicialización de pesos (no puede ser null)
     * @param pesoMin peso mínimo (debe ser < pesoMax)
     * @param pesoMax peso máximo (debe ser > pesoMin)
     * @param tipoNormalizacion tipo de normalización (no puede ser null)
     * @param valorObjetivoNormalizacion valor objetivo para normalización (debe ser > 0)
     * @param retardoMin retardo mínimo en timesteps (debe ser >= 0)
     * @param retardoMax retardo máximo en timesteps (debe ser >= retardoMin)
     * @param homeostasisActiva indica si homeostasis está activa
     * @param tasaDisparoObjetivo tasa objetivo en Hz (debe ser > 0 si homeostasis activa)
     * @param tasaAjusteHomeostasis tasa de ajuste (debe ser > 0 si homeostasis activa)
     * @param inhibicionLateralActiva indica si inhibición lateral está activa
     * @param radioInhibicion radio de inhibición (debe ser > 0 si inhibición activa)
     * @param fuerzaInhibicion fuerza de inhibición (debe ser > 0 si inhibición activa)
     * @param duracionTimestep duración del timestep en ms (debe ser > 0)
     * @param wtaActivo indica si WTA está activo
     * @param wtaCapaSalida indica si WTA se aplica en la capa de salida
     * @param wtaCapasOcultas indica si WTA se aplica en capas ocultas
     * @param radioWTA radio de competición WTA (0 = global, >0 = local)
     * @param fuerzaWTA fuerza de supresión WTA (debe ser > 0 si WTA activo)
     * @param umbralActivacionWTA umbral mínimo de activación para competir
     * 
     * @throws IllegalArgumentException si algún parámetro es inválido
     */
    public ConfiguracionRed(
            int[] topologia,
            double umbralDisparo,
            double potencialReposo,
            double constanteDecaimiento,
            int duracionRefractario,
            double amplitudLTP,
            double amplitudLTD,
            double tauLTP,
            double tauLTD,
            double frecuenciaMaxima,
            ModoCodificacion modoCodificacion,
            int ventanaDecodificacion,
            TipoInicializacion tipoInicializacion,
            double pesoMin,
            double pesoMax,
            TipoNormalizacion tipoNormalizacion,
            double valorObjetivoNormalizacion,
            int retardoMin,
            int retardoMax,
            boolean homeostasisActiva,
            double tasaDisparoObjetivo,
            double tasaAjusteHomeostasis,
            boolean inhibicionLateralActiva,
            int radioInhibicion,
            double fuerzaInhibicion,
            double duracionTimestep,
            boolean wtaActivo,
            boolean wtaCapaSalida,
            boolean wtaCapasOcultas,
            int radioWTA,
            double fuerzaWTA,
            double umbralActivacionWTA) {
        
        // Validar topología
        if (topologia == null || topologia.length == 0) {
            throw new IllegalArgumentException("La topología debe tener al menos una capa");
        }
        for (int i = 0; i < topologia.length; i++) {
            if (topologia[i] <= 0) {
                throw new IllegalArgumentException(
                    "Capa " + i + " debe tener al menos una neurona, recibido: " + topologia[i]
                );
            }
        }
        this.topologia = Arrays.copyOf(topologia, topologia.length);
        
        // Validar parámetros neuronales
        if (umbralDisparo <= potencialReposo) {
            throw new IllegalArgumentException(
                "Umbral de disparo (" + umbralDisparo + ") debe ser mayor que potencial de reposo (" + potencialReposo + ")"
            );
        }
        this.umbralDisparo = umbralDisparo;
        this.potencialReposo = potencialReposo;
        
        if (constanteDecaimiento <= 0) {
            throw new IllegalArgumentException(
                "Constante de decaimiento debe ser positiva, recibido: " + constanteDecaimiento
            );
        }
        this.constanteDecaimiento = constanteDecaimiento;
        
        if (duracionRefractario < 0) {
            throw new IllegalArgumentException(
                "Duración de período refractario no puede ser negativa: " + duracionRefractario
            );
        }
        this.duracionRefractario = duracionRefractario;
        
        // Validar parámetros STDP
        if (amplitudLTP <= 0) {
            throw new IllegalArgumentException(
                "Amplitud LTP debe ser positiva, recibido: " + amplitudLTP
            );
        }
        this.amplitudLTP = amplitudLTP;
        
        if (amplitudLTD <= 0) {
            throw new IllegalArgumentException(
                "Amplitud LTD debe ser positiva, recibido: " + amplitudLTD
            );
        }
        this.amplitudLTD = amplitudLTD;
        
        if (tauLTP <= 0) {
            throw new IllegalArgumentException(
                "Tau LTP debe ser positivo, recibido: " + tauLTP
            );
        }
        this.tauLTP = tauLTP;
        
        if (tauLTD <= 0) {
            throw new IllegalArgumentException(
                "Tau LTD debe ser positivo, recibido: " + tauLTD
            );
        }
        this.tauLTD = tauLTD;
        
        // Validar parámetros de codificación/decodificación
        if (frecuenciaMaxima <= 0) {
            throw new IllegalArgumentException(
                "Frecuencia máxima debe ser positiva, recibido: " + frecuenciaMaxima
            );
        }
        this.frecuenciaMaxima = frecuenciaMaxima;
        
        if (modoCodificacion == null) {
            throw new IllegalArgumentException("Modo de codificación no puede ser null");
        }
        this.modoCodificacion = modoCodificacion;
        
        if (ventanaDecodificacion <= 0) {
            throw new IllegalArgumentException(
                "Ventana de decodificación debe ser positiva, recibido: " + ventanaDecodificacion
            );
        }
        this.ventanaDecodificacion = ventanaDecodificacion;
        
        // Validar inicialización de pesos
        if (tipoInicializacion == null) {
            throw new IllegalArgumentException("Tipo de inicialización no puede ser null");
        }
        this.tipoInicializacion = tipoInicializacion;
        
        if (pesoMin >= pesoMax) {
            throw new IllegalArgumentException(
                "peso_min (" + pesoMin + ") debe ser menor que peso_max (" + pesoMax + ")"
            );
        }
        this.pesoMin = pesoMin;
        this.pesoMax = pesoMax;
        
        // Validar normalización
        if (tipoNormalizacion == null) {
            throw new IllegalArgumentException("Tipo de normalización no puede ser null");
        }
        this.tipoNormalizacion = tipoNormalizacion;
        
        if (valorObjetivoNormalizacion <= 0) {
            throw new IllegalArgumentException(
                "Valor objetivo de normalización debe ser positivo, recibido: " + valorObjetivoNormalizacion
            );
        }
        this.valorObjetivoNormalizacion = valorObjetivoNormalizacion;
        
        // Validar retardos
        if (retardoMin < 0) {
            throw new IllegalArgumentException(
                "Retardo mínimo no puede ser negativo: " + retardoMin
            );
        }
        this.retardoMin = retardoMin;
        
        if (retardoMax < retardoMin) {
            throw new IllegalArgumentException(
                "Retardo máximo (" + retardoMax + ") debe ser mayor o igual que retardo mínimo (" + retardoMin + ")"
            );
        }
        this.retardoMax = retardoMax;
        
        // Validar homeostasis
        this.homeostasisActiva = homeostasisActiva;
        if (homeostasisActiva) {
            if (tasaDisparoObjetivo <= 0) {
                throw new IllegalArgumentException(
                    "Tasa de disparo objetivo debe ser positiva cuando homeostasis está activa, recibido: " + tasaDisparoObjetivo
                );
            }
            if (tasaAjusteHomeostasis <= 0) {
                throw new IllegalArgumentException(
                    "Tasa de ajuste homeostasis debe ser positiva cuando homeostasis está activa, recibido: " + tasaAjusteHomeostasis
                );
            }
        }
        this.tasaDisparoObjetivo = tasaDisparoObjetivo;
        this.tasaAjusteHomeostasis = tasaAjusteHomeostasis;
        
        // Validar inhibición lateral
        this.inhibicionLateralActiva = inhibicionLateralActiva;
        if (inhibicionLateralActiva) {
            if (radioInhibicion <= 0) {
                throw new IllegalArgumentException(
                    "Radio de inhibición debe ser positivo cuando inhibición lateral está activa, recibido: " + radioInhibicion
                );
            }
            if (fuerzaInhibicion <= 0) {
                throw new IllegalArgumentException(
                    "Fuerza de inhibición debe ser positiva cuando inhibición lateral está activa, recibido: " + fuerzaInhibicion
                );
            }
        }
        this.radioInhibicion = radioInhibicion;
        this.fuerzaInhibicion = fuerzaInhibicion;
        
        // Validar simulación
        if (duracionTimestep <= 0) {
            throw new IllegalArgumentException(
                "Duración del timestep debe ser positiva, recibido: " + duracionTimestep
            );
        }
        this.duracionTimestep = duracionTimestep;

        // Validar WTA
        this.wtaActivo = wtaActivo;
        this.wtaCapaSalida = wtaCapaSalida;
        this.wtaCapasOcultas = wtaCapasOcultas;
        if (wtaActivo) {
            if (radioWTA < 0) {
                throw new IllegalArgumentException(
                    "Radio WTA no puede ser negativo: " + radioWTA);
            }
            if (fuerzaWTA <= 0) {
                throw new IllegalArgumentException(
                    "Fuerza WTA debe ser positiva cuando WTA está activo, recibido: " + fuerzaWTA);
            }
        }
        this.radioWTA = radioWTA;
        this.fuerzaWTA = fuerzaWTA;
        this.umbralActivacionWTA = umbralActivacionWTA;
    }
    
    /**
     * Retorna una copia de la topología de la red.
     * 
     * @return array con los tamaños de cada capa
     */
    public int[] getTopologia() {
        return Arrays.copyOf(topologia, topologia.length);
    }
    
    /**
     * Retorna el número de capas en la red.
     * 
     * @return número de capas
     */
    public int getNumeroCapas() {
        return topologia.length;
    }
    
    /**
     * Retorna el número total de neuronas en la red.
     * 
     * @return suma de neuronas en todas las capas
     */
    public int getNumeroTotalNeuronas() {
        int total = 0;
        for (int tamano : topologia) {
            total += tamano;
        }
        return total;
    }
    
    @Override
    public String toString() {
        return "ConfiguracionRed{" +
                "topologia=" + Arrays.toString(topologia) +
                ", umbralDisparo=" + umbralDisparo +
                ", potencialReposo=" + potencialReposo +
                ", constanteDecaimiento=" + constanteDecaimiento +
                ", duracionRefractario=" + duracionRefractario +
                ", amplitudLTP=" + amplitudLTP +
                ", amplitudLTD=" + amplitudLTD +
                ", tauLTP=" + tauLTP +
                ", tauLTD=" + tauLTD +
                ", frecuenciaMaxima=" + frecuenciaMaxima +
                ", modoCodificacion=" + modoCodificacion +
                ", ventanaDecodificacion=" + ventanaDecodificacion +
                ", tipoInicializacion=" + tipoInicializacion +
                ", pesoMin=" + pesoMin +
                ", pesoMax=" + pesoMax +
                ", tipoNormalizacion=" + tipoNormalizacion +
                ", valorObjetivoNormalizacion=" + valorObjetivoNormalizacion +
                ", retardoMin=" + retardoMin +
                ", retardoMax=" + retardoMax +
                ", homeostasisActiva=" + homeostasisActiva +
                ", tasaDisparoObjetivo=" + tasaDisparoObjetivo +
                ", tasaAjusteHomeostasis=" + tasaAjusteHomeostasis +
                ", inhibicionLateralActiva=" + inhibicionLateralActiva +
                ", radioInhibicion=" + radioInhibicion +
                ", fuerzaInhibicion=" + fuerzaInhibicion +
                ", duracionTimestep=" + duracionTimestep +
                ", wtaActivo=" + wtaActivo +
                ", wtaCapaSalida=" + wtaCapaSalida +
                ", wtaCapasOcultas=" + wtaCapasOcultas +
                ", radioWTA=" + radioWTA +
                ", fuerzaWTA=" + fuerzaWTA +
                ", umbralActivacionWTA=" + umbralActivacionWTA +
                '}';
    }
}
