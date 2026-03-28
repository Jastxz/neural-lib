package es.jastxz.nn.spiking;

import java.util.Random;

/**
 * Gestor de codificación que convierte valores continuos en trenes de spikes.
 * 
 * <p>Implementa diferentes esquemas de codificación temporal (rate coding) para
 * transformar valores de entrada en el rango [0, 1] a secuencias de spikes con
 * frecuencias proporcionales. Soporta codificación Poisson (probabilística),
 * regular (determinística) y burst (ráfagas).</p>
 * 
 * <p>La codificación por tasa de disparo (rate coding) es un esquema fundamental
 * en redes neuronales de spikes donde la información se representa mediante la
 * frecuencia de disparo de las neuronas.</p>
 * 
 * <h3>Modos de Codificación:</h3>
 * <ul>
 *   <li><b>POISSON</b>: Generación probabilística biológicamente realista</li>
 *   <li><b>REGULAR</b>: Espaciado uniforme determinístico</li>
 *   <li><b>BURST</b>: Ráfagas de spikes consecutivos</li>
 * </ul>
 * 
 * <h3>Ejemplo de Uso:</h3>
 * <pre>{@code
 * GestorCodificacion codificador = new GestorCodificacion(100.0, ModoCodificacion.POISSON);
 * double[] valores = {0.8, 0.3, 0.5};
 * List<EventoSpike> spikes = codificador.codificar(valores, 100);
 * }</pre>
 * 
 * @see ModoCodificacion
 * @see EventoSpike
 * @since 1.0
 */
public class GestorCodificacion implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Frecuencia máxima de disparo en Hz.
     * Representa la frecuencia de spikes cuando el valor de entrada es 1.0.
     */
    private final double frecuenciaMaxima;
    
    /**
     * Modo de codificación activo (POISSON, REGULAR o BURST).
     */
    private final ModoCodificacion modo;
    
    /**
     * Generador de números aleatorios para codificación probabilística.
     */
    private final Random random;
    
    /**
     * Construye un nuevo gestor de codificación con los parámetros especificados.
     * 
     * @param frecuenciaMaxima frecuencia máxima de disparo en Hz (debe ser > 0)
     * @param modo modo de codificación a utilizar
     * @throws IllegalArgumentException si frecuenciaMaxima <= 0 o modo es null
     */
    public GestorCodificacion(double frecuenciaMaxima, ModoCodificacion modo) {
        if (frecuenciaMaxima <= 0) {
            throw new IllegalArgumentException(
                "La frecuencia máxima debe ser positiva, recibido: " + frecuenciaMaxima
            );
        }
        if (modo == null) {
            throw new IllegalArgumentException("El modo de codificación no puede ser null");
        }
        
        this.frecuenciaMaxima = frecuenciaMaxima;
        this.modo = modo;
        this.random = new Random();
    }
    
    /**
     * Construye un nuevo gestor de codificación con semilla específica para el generador aleatorio.
     * 
     * <p>Este constructor permite reproducibilidad en tests al fijar la semilla del generador
     * de números aleatorios.</p>
     * 
     * @param frecuenciaMaxima frecuencia máxima de disparo en Hz (debe ser > 0)
     * @param modo modo de codificación a utilizar
     * @param semilla semilla para el generador de números aleatorios
     * @throws IllegalArgumentException si frecuenciaMaxima <= 0 o modo es null
     */
    public GestorCodificacion(double frecuenciaMaxima, ModoCodificacion modo, long semilla) {
        if (frecuenciaMaxima <= 0) {
            throw new IllegalArgumentException(
                "La frecuencia máxima debe ser positiva, recibido: " + frecuenciaMaxima
            );
        }
        if (modo == null) {
            throw new IllegalArgumentException("El modo de codificación no puede ser null");
        }
        
        this.frecuenciaMaxima = frecuenciaMaxima;
        this.modo = modo;
        this.random = new Random(semilla);
    }
    
    /**
     * Obtiene la frecuencia máxima configurada.
     * 
     * @return frecuencia máxima en Hz
     */
    public double getFrecuenciaMaxima() {
        return frecuenciaMaxima;
    }
    
    /**
     * Obtiene el modo de codificación configurado.
     * 
     * @return modo de codificación activo
     */
    public ModoCodificacion getModo() {
        return modo;
    }


    /**
     * Codifica un array de valores continuos en trenes de spikes.
     *
     * <p>Convierte cada valor del array en una secuencia temporal de spikes distribuidos
     * a lo largo de la duración especificada. El método de codificación utilizado depende
     * del modo configurado (POISSON, REGULAR o BURST).</p>
     *
     * <p>Los valores de entrada deben estar en el rango [0, 1], donde:
     * <ul>
     *   <li>0 = sin actividad (cero spikes)</li>
     *   <li>1 = actividad máxima (frecuencia máxima configurada)</li>
     * </ul>
     *
     * @param valores array de valores continuos en [0, 1] a codificar
     * @param duracionTimesteps duración de la presentación en timesteps
     * @return lista de eventos de spike ordenada por timestamp
     * @throws IllegalArgumentException si valores es null o vacío, o si duracionTimesteps <= 0
     */
    public java.util.List<EventoSpike> codificar(double[] valores, int duracionTimesteps) {
        if (valores == null || valores.length == 0) {
            throw new IllegalArgumentException("El array de valores no puede ser null o vacío");
        }
        if (duracionTimesteps <= 0) {
            throw new IllegalArgumentException(
                "La duración debe ser positiva, recibido: " + duracionTimesteps
            );
        }

        java.util.List<EventoSpike> todosLosSpikes = new java.util.ArrayList<>();

        // Codificar cada valor con su índice como neuronaId
        for (int i = 0; i < valores.length; i++) {
            java.util.List<EventoSpike> spikesNeurona;

            switch (modo) {
                case POISSON:
                    spikesNeurona = codificarPoisson(valores[i], i, duracionTimesteps);
                    break;
                case REGULAR:
                    spikesNeurona = codificarRegular(valores[i], i, duracionTimesteps);
                    break;
                case BURST:
                    spikesNeurona = codificarBurst(valores[i], i, duracionTimesteps);
                    break;
                default:
                    throw new IllegalStateException("Modo de codificación no soportado: " + modo);
            }

            todosLosSpikes.addAll(spikesNeurona);
        }

        // Ordenar por timestamp
        todosLosSpikes.sort(null); // Usa compareTo de EventoSpike

        return todosLosSpikes;
    }

    /**
     * Codifica un valor usando el esquema Poisson (probabilístico).
     *
     * <p>Genera spikes de manera probabilística en cada timestep. La probabilidad
     * de generar un spike en cada timestep es proporcional al valor de entrada
     * y a la frecuencia máxima configurada.</p>
     *
     * <p>Fórmula: P(spike) = valor * frecuenciaMaxima * dt</p>
     *
     * <p>Este método es biológicamente realista ya que las neuronas reales exhiben
     * variabilidad en sus patrones de disparo.</p>
     *
     * @param valor valor continuo en [0, 1] a codificar
     * @param neuronaId identificador de la neurona
     * @param duracion duración en timesteps
     * @return lista de eventos de spike generados
     */
    public java.util.List<EventoSpike> codificarPoisson(double valor, int neuronaId, int duracion) {
        java.util.List<EventoSpike> spikes = new java.util.ArrayList<>();

        // Valor 0 no genera spikes
        if (valor <= 0.0) {
            return spikes;
        }

        // Clamp valor a [0, 1]
        valor = Math.max(0.0, Math.min(1.0, valor));

        // dt = 1ms por timestep (asumiendo timesteps de 1ms)
        double dt = 0.001; // 1ms en segundos

        // Probabilidad de spike por timestep
        double probabilidad = valor * frecuenciaMaxima * dt;

        // Generar spikes probabilísticamente
        for (int t = 0; t < duracion; t++) {
            if (random.nextDouble() < probabilidad) {
                // Crear evento de spike
                // Usamos neuronaId como ID, capa 0 (entrada), índice = neuronaId
                EventoSpike spike = new EventoSpike(
                    neuronaId,
                    0, // capa de entrada
                    neuronaId,
                    t,
                    0.0 // potencial no relevante para spikes de entrada
                );
                spikes.add(spike);
            }
        }

        return spikes;
    }

    /**
     * Codifica un valor usando el esquema regular (determinístico).
     *
     * <p>Genera spikes espaciados uniformemente en el tiempo. La frecuencia de spikes
     * es proporcional al valor de entrada, pero los spikes están distribuidos de manera
     * regular en lugar de aleatoria.</p>
     *
     * <p>Este método es determinístico y produce patrones de spikes predecibles, útil
     * para debugging y cuando se requiere reproducibilidad exacta.</p>
     *
     * @param valor valor continuo en [0, 1] a codificar
     * @param neuronaId identificador de la neurona
     * @param duracion duración en timesteps
     * @return lista de eventos de spike generados
     */
    public java.util.List<EventoSpike> codificarRegular(double valor, int neuronaId, int duracion) {
        java.util.List<EventoSpike> spikes = new java.util.ArrayList<>();

        // Valor 0 no genera spikes
        if (valor <= 0.0) {
            return spikes;
        }

        // Clamp valor a [0, 1]
        valor = Math.max(0.0, Math.min(1.0, valor));

        // Calcular frecuencia objetivo
        double frecuencia = valor * frecuenciaMaxima;

        // Calcular intervalo entre spikes en timesteps
        // frecuencia está en Hz, necesitamos intervalo en timesteps (asumiendo 1ms por timestep)
        double intervaloMs = 1000.0 / frecuencia; // intervalo en ms
        double intervaloTimesteps = intervaloMs; // 1 timestep = 1ms

        // Si el intervalo es mayor que la duración, no generamos spikes
        if (intervaloTimesteps > duracion) {
            return spikes;
        }

        // Generar spikes espaciados uniformemente
        double t = 0.0;
        while (t < duracion) {
            EventoSpike spike = new EventoSpike(
                neuronaId,
                0, // capa de entrada
                neuronaId,
                (long) Math.round(t),
                0.0
            );
            spikes.add(spike);
            t += intervaloTimesteps;
        }

        return spikes;
    }

    /**
     * Codifica un valor usando el esquema burst (ráfagas).
     *
     * <p>Genera ráfagas de spikes consecutivos distribuidas a lo largo de la duración.
     * Cada ráfaga contiene múltiples spikes en timesteps consecutivos, seguidos de
     * períodos de silencio.</p>
     *
     * <p>Este patrón es común en neuronas biológicas que exhiben comportamiento de
     * "bursting" donde disparan múltiples spikes en rápida sucesión.</p>
     *
     * @param valor valor continuo en [0, 1] a codificar
     * @param neuronaId identificador de la neurona
     * @param duracion duración en timesteps
     * @return lista de eventos de spike generados
     */
    public java.util.List<EventoSpike> codificarBurst(double valor, int neuronaId, int duracion) {
        java.util.List<EventoSpike> spikes = new java.util.ArrayList<>();

        // Valor 0 no genera spikes
        if (valor <= 0.0) {
            return spikes;
        }

        // Clamp valor a [0, 1]
        valor = Math.max(0.0, Math.min(1.0, valor));

        // Parámetros de burst
        int burstSize = 3; // spikes por ráfaga

        // Calcular frecuencia objetivo y número de bursts
        double frecuencia = valor * frecuenciaMaxima;

        // Número de bursts necesarios para alcanzar la frecuencia objetivo
        // frecuencia en Hz, duración en timesteps (1ms cada uno)
        double duracionSegundos = duracion * 0.001;
        int numSpikesObjetivo = (int) Math.round(frecuencia * duracionSegundos);
        int numBursts = Math.max(1, numSpikesObjetivo / burstSize);

        // Si no hay suficiente duración para bursts, no generamos spikes
        if (numBursts * burstSize > duracion) {
            numBursts = duracion / burstSize;
        }

        if (numBursts == 0) {
            return spikes;
        }

        // Calcular intervalo entre bursts
        double intervaloBursts = (double) duracion / numBursts;

        // Generar bursts distribuidos uniformemente
        for (int b = 0; b < numBursts; b++) {
            long timestampInicio = (long) Math.round(b * intervaloBursts);

            // Generar spikes consecutivos en el burst
            for (int s = 0; s < burstSize && (timestampInicio + s) < duracion; s++) {
                EventoSpike spike = new EventoSpike(
                    neuronaId,
                    0, // capa de entrada
                    neuronaId,
                    timestampInicio + s,
                    0.0
                );
                spikes.add(spike);
            }
        }

        return spikes;
    }

}
