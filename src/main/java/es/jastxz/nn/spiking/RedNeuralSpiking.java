package es.jastxz.nn.spiking;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Red Neuronal de Spikes (Spiking Neural Network - SNN) que implementa el modelo
 * Leaky Integrate-and-Fire (LIF) con plasticidad STDP, codificación rate coding,
 * y período refractario.
 * 
 * <p>Esta clase orquesta la simulación temporal de una red neuronal de spikes,
 * coordinando la propagación de spikes, aplicación de STDP durante aprendizaje,
 * y gestión de eventos con retardos sinápticos.</p>
 * 
 * <p>Características principales:</p>
 * <ul>
 *   <li>Arquitectura multicapa con neuronas organizadas en capas</li>
 *   <li>Simulación temporal discreta con timesteps configurables</li>
 *   <li>Codificación/decodificación rate coding</li>
 *   <li>Aprendizaje mediante STDP (Spike-Timing-Dependent Plasticity)</li>
 *   <li>Retardos sinápticos configurables</li>
 *   <li>Registro de actividad neuronal para análisis</li>
 *   <li>Métricas de rendimiento y costo energético</li>
 *   <li>Persistencia mediante serialización</li>
 * </ul>
 * 
 * @see NeuronaSpiking
 * @see SinapsisSpiking
 * @see ConfiguracionRed
 */
public class RedNeuralSpiking implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Frecuencia de aplicación de homeostasis (cada N timesteps). */
    private static final long FRECUENCIA_HOMEOSTASIS = 100;

    // ========== Arquitectura ==========

    /**
     * Capas de la red, cada capa contiene una lista de neuronas.
     * Las capas están organizadas desde la capa de entrada (índice 0)
     * hasta la capa de salida (último índice).
     */
    private final List<List<NeuronaSpiking>> capas;

    /**
     * Lista de todas las sinapsis (conexiones) de la red.
     * Cada sinapsis conecta una neurona presinaptica con una postsinaptica
     * y tiene un peso y retardo asociados.
     */
    private final List<SinapsisSpiking> sinapsis;

    // ========== Simulación Temporal ==========

    /**
     * Timestep actual de la simulación.
     * Se incrementa con cada llamada a avanzarTimestep().
     */
    private long timestepActual;

    /**
     * Duración de cada timestep en milisegundos.
     * Este valor determina la resolución temporal de la simulación.
     */
    private final double duracionTimestep;

    /**
     * Cola de eventos de spikes pendientes, ordenada por timestamp.
     * Gestiona los retardos sinápticos encolando eventos que se procesarán
     * en timesteps futuros.
     */
    private final ColaEventos colaEventos;

    // ========== Gestores ==========

    /**
     * Gestor de codificación que convierte valores continuos a trenes de spikes.
     * Implementa codificación rate coding (Poisson, regular, burst).
     */
    private final GestorCodificacion codificador;

    /**
     * Gestor de decodificación que convierte trenes de spikes a valores continuos.
     * Cuenta spikes en ventanas temporales y normaliza a rango [0,1].
     */
    private final GestorDecodificacion decodificador;

    /**
     * Gestor de STDP (Spike-Timing-Dependent Plasticity) para aprendizaje.
     * Aplica Long-Term Potentiation (LTP) y Long-Term Depression (LTD)
     * basándose en diferencias temporales entre spikes pre y post-sinápticos.
     */
    private final GestorSTDP gestorSTDP;

    /**
     * Gestor de métricas que calcula y mantiene estadísticas de rendimiento.
     * Registra número de spikes, tasas de disparo, costo energético, etc.
     */
    private final GestorMetricas gestorMetricas;

    /**
     * Registro de actividad neuronal para análisis y visualización.
     * Mantiene historial completo de spikes con timestamps y metadatos.
     */
    private final RegistroActividad registro;

    // ========== Configuración ==========

    /**
     * Configuración inmutable de la red que define todos los parámetros:
     * arquitectura, parámetros neuronales LIF, parámetros STDP,
     * configuración de codificación/decodificación, etc.
     */
    private final ConfiguracionRed configuracion;

    /**
     * Indica si la red está en modo entrenamiento.
     * Cuando está activo, se aplica STDP después de cada spike.
     */
    private boolean modoEntrenamiento;

    // ========== Constructor ==========

    /**
     * Construye una nueva red neuronal de spikes con la configuración especificada.
     *
     * <p>El constructor realiza las siguientes operaciones:</p>
     * <ul>
     *   <li>Valida la arquitectura de la red (al menos una capa, cada capa con al menos una neurona)</li>
     *   <li>Valida que los índices de capa sean consecutivos desde 0</li>
     *   <li>Crea todas las capas y neuronas según la topología</li>
     *   <li>Inicializa todos los gestores (codificador, decodificador, STDP, métricas, registro)</li>
     *   <li>Inicializa el estado temporal (timestepActual = 0, cola de eventos vacía)</li>
     *   <li>Configura homeostasis en neuronas si está activa en la configuración</li>
     * </ul>
     *
     * <p>Después de la construcción, la red está lista para:</p>
     * <ul>
     *   <li>Crear conexiones sinápticas mediante métodos de conexión</li>
     *   <li>Procesar patrones de entrada</li>
     *   <li>Ejecutar simulaciones temporales</li>
     * </ul>
     *
     * <h3>Validaciones realizadas:</h3>
     * <ul>
     *   <li><b>Requisito 15.1:</b> Al menos una capa en la topología</li>
     *   <li><b>Requisito 15.2:</b> Cada capa debe tener al menos una neurona</li>
     *   <li><b>Requisito 15.6:</b> Índices de capa consecutivos comenzando en 0</li>
     * </ul>
     *
     * <h3>Ejemplo de uso:</h3>
     * <pre>{@code
     * ConfiguracionRed config = new ConfiguracionRedBuilder()
     *     .topologia(10, 20, 10)
     *     .parametrosLIF(-55.0, -70.0, 20.0, 2)
     *     .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
     *     .build();
     *
     * RedNeuralSpiking red = new RedNeuralSpiking(config);
     * }</pre>
     *
     * @param configuracion configuración completa de la red (no puede ser null)
     * @throws IllegalArgumentException si la configuración es null
     * @throws IllegalArgumentException si la topología está vacía (Requisito 15.1)
     * @throws IllegalArgumentException si alguna capa tiene cero neuronas (Requisito 15.2)
     * @throws IllegalArgumentException si los índices de capa no son consecutivos (Requisito 15.6)
     *
     * @see ConfiguracionRed
     * @see ConfiguracionRedBuilder
     */
    public RedNeuralSpiking(ConfiguracionRed configuracion) {
        if (configuracion == null) {
            throw new IllegalArgumentException("La configuración no puede ser null");
        }

        // Guardar configuración
        this.configuracion = configuracion;

        // Validar arquitectura (Requisitos 15.1, 15.2)
        int[] topologia = configuracion.getTopologia();

        if (topologia.length == 0) {
            throw new IllegalArgumentException("La topología debe tener al menos una capa");
        }

        for (int i = 0; i < topologia.length; i++) {
            if (topologia[i] <= 0) {
                throw new IllegalArgumentException(
                    "Capa " + i + " debe tener al menos una neurona, recibido: " + topologia[i]
                );
            }
        }

        // Crear capas según topología (Requisito 6.1)
        this.capas = new java.util.ArrayList<>();
        long neuronaIdCounter = 0;

        for (int indiceCapa = 0; indiceCapa < topologia.length; indiceCapa++) {
            List<NeuronaSpiking> capa = new java.util.ArrayList<>();
            int numNeuronas = topologia[indiceCapa];

            for (int indiceNeurona = 0; indiceNeurona < numNeuronas; indiceNeurona++) {
                // Crear neurona con parámetros de la configuración
                NeuronaSpiking neurona = new NeuronaSpiking(
                    neuronaIdCounter++,
                    indiceCapa,
                    indiceNeurona,
                    configuracion.umbralDisparo,
                    configuracion.potencialReposo,
                    configuracion.constanteDecaimiento,
                    configuracion.duracionRefractario
                );

                // Configurar homeostasis si está activa (Requisito 18.6)
                if (configuracion.homeostasisActiva) {
                    neurona.setTasaDisparoObjetivo(configuracion.tasaDisparoObjetivo);
                    neurona.setTasaAjusteHomeostasis(configuracion.tasaAjusteHomeostasis);
                }

                capa.add(neurona);
            }

            capas.add(capa);
        }

        // Validar índices de capa consecutivos desde 0 (Requisito 15.6)
        for (int i = 0; i < capas.size(); i++) {
            // Verificar que el índice de capa de la primera neurona coincide con i
            if (!capas.get(i).isEmpty()) {
                int capaActual = capas.get(i).get(0).getCapa();
                if (capaActual != i) {
                    throw new IllegalArgumentException(
                        "Índices de capa deben ser consecutivos comenzando en 0. " +
                        "Esperado: " + i + ", recibido: " + capaActual
                    );
                }
            }
        }

        // Inicializar lista de sinapsis vacía
        this.sinapsis = new java.util.ArrayList<>();

        // Inicializar estado temporal (Requisito 7.3)
        this.timestepActual = 0;
        this.duracionTimestep = configuracion.duracionTimestep;
        this.colaEventos = new ColaEventos();

        // Inicializar gestores
        this.codificador = new GestorCodificacion(
            configuracion.frecuenciaMaxima,
            configuracion.modoCodificacion
        );

        this.decodificador = new GestorDecodificacion(
            configuracion.ventanaDecodificacion,
            configuracion.frecuenciaMaxima
        );

        this.gestorSTDP = new GestorSTDP(
            configuracion.amplitudLTP,
            configuracion.amplitudLTD,
            configuracion.tauLTP,
            configuracion.tauLTD
        );

        this.gestorMetricas = new GestorMetricas();
        this.registro = new RegistroActividad();

        // Modo entrenamiento desactivado por defecto
        this.modoEntrenamiento = false;
    }

    // ========== Gestión de Conexiones ==========

    /**
     * Crea una nueva conexión sináptica entre dos neuronas de la red.
     *
     * <p>Este método crea una sinapsis que conecta una neurona presinaptica con una
     * neurona postsinaptica, con un peso y retardo especificados. La sinapsis se
     * agrega a la lista de conexiones de la red.</p>
     *
     * <h3>Validaciones realizadas:</h3>
     * <ul>
     *   <li><b>Requisito 15.3:</b> Valida que ambas neuronas existan en la red</li>
     *   <li><b>Requisito 15.4:</b> Valida que no exista una conexión duplicada</li>
     *   <li><b>Requisito 6.3:</b> Permite crear conexiones entre neuronas de diferentes capas</li>
     * </ul>
     *
     * <h3>Ejemplo de uso:</h3>
     * <pre>{@code
     * RedNeuralSpiking red = new RedNeuralSpiking(config);
     *
     * // Obtener neuronas de diferentes capas
     * NeuronaSpiking neurona1 = red.getNeurona(0, 0); // Capa 0, índice 0
     * NeuronaSpiking neurona2 = red.getNeurona(1, 0); // Capa 1, índice 0
     *
     * // Crear conexión con peso 0.5 y retardo 2 timesteps
     * red.crearConexion(neurona1, neurona2, 0.5, 2);
     * }</pre>
     *
     * @param pre neurona presinaptica (origen del spike)
     * @param post neurona postsinaptica (destino del spike)
     * @param peso peso sináptico inicial (debe estar en [pesoMin, pesoMax] de la configuración)
     * @param retardo retardo de transmisión en timesteps (debe ser >= 0)
     * @throws IllegalArgumentException si pre o post son null
     * @throws IllegalArgumentException si pre no existe en la red (Requisito 15.3)
     * @throws IllegalArgumentException si post no existe en la red (Requisito 15.3)
     * @throws IllegalArgumentException si ya existe una conexión entre pre y post (Requisito 15.4)
     * @throws IllegalArgumentException si retardo < 0
     * @throws IllegalArgumentException si peso no está en los límites configurados
     *
     * @see SinapsisSpiking
     * @see #existeNeurona(long)
     * @see #existeConexion(NeuronaSpiking, NeuronaSpiking)
     */
    public void crearConexion(NeuronaSpiking pre, NeuronaSpiking post, double peso, int retardo) {
        // Validar que las neuronas no sean null
        if (pre == null) {
            throw new IllegalArgumentException("La neurona presinaptica no puede ser null");
        }
        if (post == null) {
            throw new IllegalArgumentException("La neurona postsinaptica no puede ser null");
        }

        // Validar que las neuronas existan en la red (Requisito 15.3)
        if (!existeNeurona(pre.getId())) {
            throw new IllegalArgumentException(
                "Neurona presinaptica con ID " + pre.getId() + " no existe en la red"
            );
        }
        if (!existeNeurona(post.getId())) {
            throw new IllegalArgumentException(
                "Neurona postsinaptica con ID " + post.getId() + " no existe en la red"
            );
        }

        // Validar que no exista conexión duplicada (Requisito 15.4)
        if (existeConexion(pre, post)) {
            throw new IllegalArgumentException(
                "Ya existe una conexión entre neurona " + pre.getId() +
                " (capa " + pre.getCapa() + ", índice " + pre.getIndice() + ") y neurona " +
                post.getId() + " (capa " + post.getCapa() + ", índice " + post.getIndice() + ")"
            );
        }

        // Crear la sinapsis con los límites de peso de la configuración
        SinapsisSpiking sinapsis = new SinapsisSpiking(
            pre,
            post,
            peso,
            retardo,
            configuracion.pesoMin,
            configuracion.pesoMax
        );

        // Agregar sinapsis a la lista (Requisito 6.3)
        this.sinapsis.add(sinapsis);
    }

    /**
     * Inicializa los pesos de todas las sinapsis de la red según el tipo especificado.
     *
     * <p>Este método permite configurar los pesos sinápticos iniciales usando diferentes
     * estrategias de inicialización. Los pesos se establecen dentro del rango [min, max]
     * especificado y se validan después de la inicialización.</p>
     *
     * <h3>Tipos de inicialización soportados:</h3>
     * <ul>
     *   <li><b>UNIFORME:</b> Distribución uniforme aleatoria en [min, max]</li>
     *   <li><b>NORMAL:</b> Distribución normal con media = (min+max)/2 y desviación estándar = (max-min)/4,
     *       truncada a [min, max]</li>
     *   <li><b>CONSTANTE:</b> Todos los pesos se establecen al valor medio (min+max)/2</li>
     * </ul>
     *
     * <h3>Validaciones realizadas:</h3>
     * <ul>
     *   <li><b>Requisito 11.4:</b> Valida que min < max</li>
     *   <li><b>Requisito 11.6:</b> Valida que todos los pesos estén en [min, max] después de inicialización</li>
     * </ul>
     *
     * <h3>Ejemplo de uso:</h3>
     * <pre>{@code
     * RedNeuralSpiking red = new RedNeuralSpiking(config);
     * // Crear conexiones...
     * red.crearConexion(n1, n2, 0.0, 1);
     * red.crearConexion(n2, n3, 0.0, 1);
     *
     * // Inicializar pesos con distribución uniforme en [0.3, 0.7]
     * red.inicializarPesos(TipoInicializacion.UNIFORME, 0.3, 0.7);
     * }</pre>
     *
     * @param tipo tipo de inicialización a aplicar (UNIFORME, NORMAL, o CONSTANTE)
     * @param min valor mínimo permitido para los pesos
     * @param max valor máximo permitido para los pesos
     * @throws IllegalArgumentException si tipo es null
     * @throws IllegalArgumentException si min >= max (Requisito 11.4)
     * @throws IllegalArgumentException si tipo es DESDE_ARRAY (no soportado por este método)
     * @throws IllegalStateException si algún peso queda fuera de [min, max] después de inicialización (Requisito 11.6)
     *
     * @see TipoInicializacion
     * @see SinapsisSpiking#ajustarPeso(double)
     */
    public void inicializarPesos(TipoInicializacion tipo, double min, double max) {
        // Validar parámetros
        if (tipo == null) {
            throw new IllegalArgumentException("El tipo de inicialización no puede ser null");
        }

        // Validar rango (Requisito 11.4)
        if (min >= max) {
            throw new IllegalArgumentException(
                "min (" + min + ") debe ser menor que max (" + max + ")"
            );
        }

        // DESDE_ARRAY no es soportado por este método
        if (tipo == TipoInicializacion.DESDE_ARRAY) {
            throw new IllegalArgumentException(
                "TipoInicializacion.DESDE_ARRAY no es soportado por este método. " +
                "Use inicializarPesos(double[] pesos) en su lugar."
            );
        }

        // Crear generador de números aleatorios
        java.util.Random random = new java.util.Random();

        // Aplicar inicialización según el tipo
        for (SinapsisSpiking sinapsis : this.sinapsis) {
            double nuevoPeso;

            switch (tipo) {
                case UNIFORME:
                    // Requisito 11.1: Distribución uniforme en [min, max]
                    nuevoPeso = min + (max - min) * random.nextDouble();
                    break;

                case NORMAL:
                    // Requisito 11.2: Distribución normal truncada a [min, max]
                    // Media en el centro del rango, desviación estándar = (max-min)/4
                    // para que ~95% de valores caigan en [min, max]
                    double media = (min + max) / 2.0;
                    double desviacion = (max - min) / 4.0;

                    // Generar valor con distribución normal
                    nuevoPeso = media + desviacion * random.nextGaussian();

                    // Truncar a [min, max]
                    nuevoPeso = Math.max(min, Math.min(max, nuevoPeso));
                    break;

                case CONSTANTE:
                    // Requisito 11.3: Valor constante (usamos el punto medio del rango)
                    nuevoPeso = (min + max) / 2.0;
                    break;

                default:
                    throw new IllegalArgumentException("Tipo de inicialización no soportado: " + tipo);
            }

            // Establecer el nuevo peso
            sinapsis.setPeso(nuevoPeso);
        }

        // Validar que todos los pesos estén en [min, max] (Requisito 11.6)
        for (SinapsisSpiking sinapsis : this.sinapsis) {
            double peso = sinapsis.getPeso();
            if (peso < min || peso > max) {
                throw new IllegalStateException(
                    "Peso inicializado fuera de límites: " + peso +
                    " no está en [" + min + ", " + max + "]"
                );
            }
        }
    }


    /**
     * Verifica si existe una neurona con el ID dado en la red.
     *
     * @param id identificador único de la neurona
     * @return true si la neurona existe, false en caso contrario
     */
    public boolean existeNeurona(long id) {
        for (List<NeuronaSpiking> capa : capas) {
            for (NeuronaSpiking neurona : capa) {
                if (neurona.getId() == id) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Verifica si ya existe una conexión entre dos neuronas.
     *
     * <p>Este método busca en la lista de sinapsis para determinar si ya existe
     * una conexión desde la neurona presinaptica hacia la neurona postsinaptica.</p>
     *
     * @param pre neurona presinaptica
     * @param post neurona postsinaptica
     * @return true si existe una conexión entre pre y post, false en caso contrario
     */
    public boolean existeConexion(NeuronaSpiking pre, NeuronaSpiking post) {
        for (SinapsisSpiking s : sinapsis) {
            if (s.getPresinaptica().getId() == pre.getId() &&
                s.getPostsinaptica().getId() == post.getId()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Obtiene una neurona específica de la red por su capa e índice.
     *
     * <p>Este método es útil para obtener referencias a neuronas específicas
     * al crear conexiones o consultar el estado de la red.</p>
     *
     * @param capa índice de la capa (0-based)
     * @param indice índice de la neurona dentro de la capa (0-based)
     * @return la neurona en la posición especificada
     * @throws IllegalArgumentException si el índice de capa es inválido
     * @throws IllegalArgumentException si el índice de neurona es inválido
     */
    public NeuronaSpiking getNeurona(int capa, int indice) {
        if (capa < 0 || capa >= capas.size()) {
            throw new IllegalArgumentException(
                "Índice de capa inválido: " + capa + ". La red tiene " + capas.size() + " capas."
            );
        }

        List<NeuronaSpiking> capaNeuronas = capas.get(capa);
        if (indice < 0 || indice >= capaNeuronas.size()) {
            throw new IllegalArgumentException(
                "Índice de neurona inválido: " + indice + ". La capa " + capa +
                " tiene " + capaNeuronas.size() + " neuronas."
            );
        }

        return capaNeuronas.get(indice);
    }


    // ========== Simulación Temporal ==========

    /**
     * Avanza la simulación en un timestep discreto.
     *
     * <p>Este es el método central de la simulación temporal que ejecuta las siguientes
     * operaciones en orden:</p>
     * <ol>
     *   <li><b>Aplicar decaimiento:</b> Todas las neuronas aplican decaimiento exponencial
     *       a su potencial de membrana (Requisito 7.5)</li>
     *   <li><b>Procesar eventos pendientes:</b> Se obtienen todos los eventos de spike
     *       encolados para el timestep actual y se entregan a las neuronas postsinápticas
     *       (Requisito 7.7)</li>
     *   <li><b>Evaluar activación:</b> Cada neurona evalúa si debe generar un spike
     *       basándose en su potencial de membrana y estado refractario, procesando
     *       las capas en orden (Requisito 7.2)</li>
     *   <li><b>Propagar spikes:</b> Los spikes generados se propagan a través de las
     *       sinapsis, creando nuevos eventos con retardo que se encolan para timesteps
     *       futuros (Requisito 7.6)</li>
     *   <li><b>Registrar actividad:</b> Los spikes se registran en métricas y en el
     *       registro de actividad para análisis posterior</li>
     *   <li><b>Aplicar STDP:</b> Si el modo entrenamiento está activo, se aplica
     *       plasticidad sináptica basada en el timing de spikes</li>
     *   <li><b>Incrementar timestep:</b> Se avanza el contador de timestep actual
     *       (Requisito 7.1)</li>
     * </ol>
     *
     * <h3>Orden de operaciones (crítico para corrección):</h3>
     * <p>El orden de las operaciones es importante para garantizar la corrección
     * de la simulación:</p>
     * <ul>
     *   <li>El decaimiento se aplica <b>antes</b> de procesar señales entrantes,
     *       de modo que el potencial decae primero y luego se incrementa por las
     *       señales (Requisito 7.5, Property 18)</li>
     *   <li>Las señales se procesan <b>antes</b> de evaluar activación, para que
     *       las señales del timestep actual puedan contribuir a generar spikes</li>
     *   <li>La evaluación se hace en orden de capas (entrada → ocultas → salida)
     *       para procesamiento feed-forward correcto</li>
     * </ul>
     *
     * <h3>Ejemplo de uso:</h3>
     * <pre>{@code
     * RedNeuralSpiking red = new RedNeuralSpiking(config);
     * // Configurar red y encolar eventos iniciales...
     *
     * // Ejecutar simulación por 100 timesteps
     * for (int i = 0; i < 100; i++) {
     *     red.avanzarTimestep();
     * }
     *
     * // Consultar resultados
     * Map<String, Object> metricas = red.obtenerMetricas();
     * }</pre>
     *
     * <h3>Validaciones y requisitos implementados:</h3>
     * <ul>
     *   <li><b>Requisito 7.1:</b> Avanzar simulación en timesteps discretos</li>
     *   <li><b>Requisito 7.2:</b> Actualizar neuronas en orden de capas</li>
     *   <li><b>Requisito 7.5:</b> Aplicar decaimiento antes de procesar señales</li>
     *   <li><b>Requisito 7.6:</b> Propagar spikes a neuronas conectadas</li>
     *   <li><b>Requisito 7.7:</b> Procesar todas las señales pendientes</li>
     *   <li><b>Property 17:</b> Propagación de spikes entre capas</li>
     *   <li><b>Property 18:</b> Orden correcto de operaciones en timestep</li>
     *   <li><b>Property 19:</b> Completitud de procesamiento de señales</li>
     * </ul>
     *
     * @see #resetearEstadoTemporal()
     * @see NeuronaSpiking#aplicarDecaimiento(double)
     * @see NeuronaSpiking#recibirSeñal(double)
     * @see NeuronaSpiking#evaluarActivacion(long)
     * @see SinapsisSpiking#propagarSpike(long)
     */
    public void avanzarTimestep() {
        // Requisitos: 7.1, 7.2, 7.5, 7.6, 7.7

        // 1. Aplicar decaimiento a todas las neuronas (Requisito 7.5, Property 18)
        // El decaimiento se aplica ANTES de procesar señales entrantes
        for (List<NeuronaSpiking> capa : capas) {
            for (NeuronaSpiking neurona : capa) {
                neurona.aplicarDecaimiento(duracionTimestep);
            }
        }

        // 2. Obtener eventos pendientes de la cola para el timestep actual (Requisito 7.7, Property 19)
        List<EventoSpike> eventosPendientes = colaEventos.obtenerEventos(timestepActual);

        // 3. Procesar cada evento: entregar señal a la neurona postsináptica
        for (EventoSpike evento : eventosPendientes) {
            // Buscar la neurona destino por su ID
            NeuronaSpiking neuronaDestino = buscarNeuronaPorId(evento.getNeuronaId());

            if (neuronaDestino != null) {
                // La señal es el peso sináptico almacenado en el campo potencialMembrana del evento
                double señal = evento.getPotencialMembrana();
                neuronaDestino.recibirSeñal(señal);
            }
        }

        // 4. Evaluar activación de cada neurona en orden de capas (Requisito 7.2)
        // Procesar capas desde entrada hacia salida para propagación feed-forward
        for (List<NeuronaSpiking> capa : capas) {
            for (NeuronaSpiking neurona : capa) {
                // Evaluar si la neurona debe disparar
                boolean disparo = neurona.evaluarActivacion(timestepActual);

                if (disparo) {
                    // 5. Para neuronas que disparan: propagar spike a través de sinapsis (Requisito 7.6, Property 17)
                    propagarSpikeDeNeurona(neurona, timestepActual);

                    // 6. Registrar spike en métricas y registro de actividad
                    gestorMetricas.registrarSpike(neurona.getId(), timestepActual);

                    EventoSpike eventoSpike = new EventoSpike(
                        neurona.getId(),
                        neurona.getCapa(),
                        neurona.getIndice(),
                        timestepActual,
                        neurona.getPotencialMembrana()
                    );
                    registro.registrar(eventoSpike);

                    // 7. Aplicar STDP si está en modo entrenamiento
                    if (modoEntrenamiento) {
                        aplicarSTDPParaNeurona(neurona, timestepActual);
                    }

                    // 8. Aplicar inhibición lateral si está activa (Requisito 19.1-19.5)
                    if (configuracion.inhibicionLateralActiva) {
                        aplicarInhibicionLateral(neurona);
                    }
                }
            }
        }

        // 9. Aplicar homeostasis periódicamente si está activa (Requisito 18.1)
        if (configuracion.homeostasisActiva && timestepActual > 0 && timestepActual % FRECUENCIA_HOMEOSTASIS == 0) {
            for (List<NeuronaSpiking> capa : capas) {
                for (NeuronaSpiking neurona : capa) {
                    neurona.aplicarHomeostasis(
                        timestepActual,
                        FRECUENCIA_HOMEOSTASIS,
                        configuracion.potencialReposo + 1.0,  // umbralMin
                        configuracion.umbralDisparo + 20.0     // umbralMax
                    );
                }
            }
        }

        // 10. Incrementar timestep actual (Requisito 7.1)
        timestepActual++;
    }

    /**
     * Propaga un spike generado por una neurona a través de todas sus sinapsis salientes.
     *
     * <p>Este método busca todas las sinapsis donde la neurona es presinaptica y
     * propaga el spike a través de cada una, creando eventos que se encolan para
     * entrega futura según el retardo de cada sinapsis.</p>
     *
     * @param neurona la neurona que generó el spike
     * @param timestamp el timestamp en el que se generó el spike
     */
    private void propagarSpikeDeNeurona(NeuronaSpiking neurona, long timestamp) {
        // Buscar todas las sinapsis donde esta neurona es presinaptica
        for (SinapsisSpiking sinapsis : this.sinapsis) {
            if (sinapsis.getPresinaptica().getId() == neurona.getId()) {
                // Propagar spike a través de la sinapsis
                EventoSpike eventoFuturo = sinapsis.propagarSpike(timestamp);

                // Encolar evento para entrega futura (timestamp + retardo)
                colaEventos.encolar(eventoFuturo);
            }
        }
    }

    /**
     * Aplica STDP a todas las sinapsis relacionadas con una neurona que acaba de disparar.
     *
     * <p>Cuando una neurona dispara, se aplica STDP a:</p>
     * <ul>
     *   <li>Sinapsis donde es postsináptica (actualizar basándose en spikes presinapticos previos)</li>
     *   <li>Sinapsis donde es presinaptica (actualizar timestamp para futuros cálculos STDP)</li>
     * </ul>
     *
     * @param neurona la neurona que generó el spike
     * @param timestamp el timestamp en el que se generó el spike
     */
    private void aplicarSTDPParaNeurona(NeuronaSpiking neurona, long timestamp) {
        // Aplicar STDP a sinapsis donde esta neurona es postsináptica
        for (SinapsisSpiking sinapsis : this.sinapsis) {
            if (sinapsis.getPostsinaptica().getId() == neurona.getId()) {
                // Actualizar timestamp postsináptico
                sinapsis.setTimestampUltimoSpikePostsinaptico(timestamp);

                // Aplicar STDP (el gestor calcula el cambio de peso basado en dt)
                gestorSTDP.aplicarSTDP(sinapsis, timestamp);
            }
        }
    }

    /**
     * Busca una neurona en la red por su ID único.
     *
     * @param id el identificador único de la neurona
     * @return la neurona con el ID especificado, o null si no se encuentra
     */
    private NeuronaSpiking buscarNeuronaPorId(long id) {
        for (List<NeuronaSpiking> capa : capas) {
            for (NeuronaSpiking neurona : capa) {
                if (neurona.getId() == id) {
                    return neurona;
                }
            }
        }
        return null;
    }

    // ========== Métodos de Consulta ==========

    /**
     * Obtiene el timestep actual de la simulación.
     *
     * @return el timestep actual
     */
    public long getTimestepActual() {
        return timestepActual;
    }

    /**
     * Obtiene la topología de la red (número de neuronas por capa).
     *
     * @return array con el número de neuronas en cada capa
     */
    public int[] getTopologia() {
        return configuracion.getTopologia();
    }

    /**
     * Obtiene la lista de todas las sinapsis de la red.
     *
     * @return lista de sinapsis
     */
    public List<SinapsisSpiking> getSinapsis() {
        return new java.util.ArrayList<>(sinapsis);
    }

    /**
     * Obtiene las métricas de rendimiento de la red.
     *
     * @return mapa con métricas (total de spikes, tasas de disparo, etc.)
     */
    public Map<String, Object> obtenerMetricas() {
        return gestorMetricas.exportarMetricas();
    }

    /**
     * Establece el modo de entrenamiento de la red.
     *
     * <p>Cuando el modo entrenamiento está activo, se aplica STDP después de cada spike
     * para ajustar los pesos sinápticos basándose en el timing de spikes pre y post.</p>
     *
     * @param activar true para activar modo entrenamiento, false para desactivar
     */
    public void setModoEntrenamiento(boolean activar) {
        this.modoEntrenamiento = activar;
    }

    /**
     * Obtiene el estado actual del modo entrenamiento.
     *
     * @return true si el modo entrenamiento está activo, false en caso contrario
     */
    public boolean isModoEntrenamiento() {
        return this.modoEntrenamiento;
    }

    /**
     * Obtiene los spikes generados por una neurona específica.
     *
     * @param capa índice de la capa
     * @param indice índice de la neurona dentro de la capa
     * @return lista de timestamps de spikes, o lista vacía si la neurona no existe
     */
    public List<Long> obtenerSpikes(int capa, int indice) {
        NeuronaSpiking neurona = getNeurona(capa, indice);
        if (neurona == null) return new java.util.ArrayList<>();
        return neurona.getHistorialSpikes();
    }

    /**
     * Obtiene la frecuencia de disparo de una neurona específica.
     *
     * @param capa índice de la capa
     * @param indice índice de la neurona dentro de la capa
     * @param ventanaTemporal ventana temporal en timesteps para calcular la frecuencia
     * @return frecuencia de disparo, o 0.0 si la neurona no existe
     */
    public double obtenerFrecuenciaDisparo(int capa, int indice, long ventanaTemporal) {
        NeuronaSpiking neurona = getNeurona(capa, indice);
        if (neurona == null) return 0.0;
        return neurona.calcularFrecuencia(ventanaTemporal);
    }

    /**
     * Exporta el registro de actividad en formato CSV.
     *
     * @return cadena CSV con formato: timestamp,capa,indice,potencial
     */
    public String exportarRegistroCSV() {
        return registro.exportarCSV();
    }

    /**
     * Calcula el peso promedio global de todas las sinapsis.
     *
     * @return peso promedio, o 0.0 si no hay sinapsis
     */
    public double getPesoPromedioGlobal() {
        if (sinapsis.isEmpty()) return 0.0;
        double suma = 0.0;
        for (SinapsisSpiking s : sinapsis) {
            suma += s.getPeso();
        }
        return suma / sinapsis.size();
    }

    /**
     * Valida la integridad interna de la red.
     *
     * <p>Verifica que todas las sinapsis referencien neuronas que existen en la red
     * y que la topología sea consistente.</p>
     *
     * @return true si la red es íntegra, false si hay inconsistencias
     */
    public boolean validarIntegridad() {
        // Verificar que todas las sinapsis referencien neuronas existentes
        for (SinapsisSpiking s : sinapsis) {
            if (!existeNeurona(s.getPresinaptica().getId())) return false;
            if (!existeNeurona(s.getPostsinaptica().getId())) return false;
        }
        // Verificar que cada capa tenga al menos una neurona
        for (List<NeuronaSpiking> capa : capas) {
            if (capa.isEmpty()) return false;
        }
        return !capas.isEmpty();
    }

    /**
     * Resetea todo el estado temporal de la red a su estado inicial.
     *
     * <p>Este método limpia todo el estado temporal de la simulación, permitiendo
     * que la red sea reutilizada para procesar nuevos patrones sin crear una nueva
     * instancia. Es especialmente útil para procesamiento por lotes donde los patrones
     * deben estar aislados entre sí.</p>
     *
     * <h3>Operaciones realizadas:</h3>
     * <ul>
     *   <li>Resetea el potencial de membrana de todas las neuronas al potencial de reposo</li>
     *   <li>Limpia los historiales de spikes de todas las neuronas</li>
     *   <li>Resetea los timestamps de último spike de todas las neuronas</li>
     *   <li>Limpia la cola de eventos pendientes</li>
     *   <li>Limpia el registro de actividad neuronal</li>
     *   <li>Resetea el timestep actual a 0</li>
     * </ul>
     *
     * <p><b>Nota importante:</b> Este método NO modifica:</p>
     * <ul>
     *   <li>La arquitectura de la red (capas, neuronas, conexiones)</li>
     *   <li>Los pesos sinápticos (se mantienen para preservar el aprendizaje)</li>
     *   <li>Los parámetros de configuración</li>
     *   <li>El modo de entrenamiento</li>
     * </ul>
     *
     * <h3>Ejemplo de uso:</h3>
     * <pre>{@code
     * RedNeuralSpiking red = new RedNeuralSpiking(config);
     *
     * // Procesar primer patrón
     * double[] salida1 = red.procesar(patron1, 100);
     *
     * // Resetear estado temporal antes de procesar siguiente patrón
     * red.resetearEstadoTemporal();
     *
     * // Procesar segundo patrón con estado limpio
     * double[] salida2 = red.procesar(patron2, 100);
     * }</pre>
     *
     * @see #avanzarTimestep()
     */
    public void resetearEstadoTemporal() {
        // Resetear timestep actual a 0 (Requisito 14.2)
        this.timestepActual = 0;

        // Limpiar cola de eventos (Requisito 14.2)
        this.colaEventos.limpiar();

        // Limpiar registro de actividad (Requisito 14.2)
        this.registro.limpiar();

        // Resetear estado temporal de todas las neuronas (Requisito 14.2)
        for (List<NeuronaSpiking> capa : capas) {
            for (NeuronaSpiking neurona : capa) {
                neurona.resetearEstadoTemporal();
            }
        }
    }

    /**
     * Procesa un patrón de entrada a través de la red neuronal de spikes.
     *
     * <p>Este es el método principal para procesar información a través de la red.
     * Realiza las siguientes operaciones en secuencia:</p>
     * <ol>
     *   <li><b>Codificación:</b> Convierte el array de valores continuos de entrada
     *       a eventos de spike usando el codificador configurado (Requisito 3.1)</li>
     *   <li><b>Encolado:</b> Encola todos los eventos de spike codificados en la
     *       cola de eventos para su procesamiento temporal</li>
     *   <li><b>Simulación:</b> Ejecuta la simulación temporal por el número de
     *       timesteps especificado, avanzando la red paso a paso (Requisito 7.1)</li>
     *   <li><b>Recolección:</b> Obtiene todos los spikes generados por las neuronas
     *       de la capa de salida durante la simulación</li>
     *   <li><b>Decodificación:</b> Convierte los trenes de spikes de salida a
     *       valores continuos usando el decodificador configurado (Requisito 4.2)</li>
     * </ol>
     *
     * <h3>Flujo de datos:</h3>
     * <pre>
     * inputs (double[])
     *   → codificador → eventos de spike
     *   → cola de eventos → simulación temporal
     *   → spikes de salida → decodificador
     *   → outputs (double[])
     * </pre>
     *
     * <h3>Ejemplo de uso:</h3>
     * <pre>{@code
     * // Crear y configurar red
     * ConfiguracionRed config = new ConfiguracionRedBuilder()
     *     .topologia(10, 20, 5)  // 10 entradas, 20 ocultas, 5 salidas
     *     .build();
     * RedNeuralSpiking red = new RedNeuralSpiking(config);
     * red.inicializarPesos(TipoInicializacion.UNIFORME, 0.3, 0.7);
     *
     * // Procesar patrón de entrada
     * double[] inputs = {0.8, 0.3, 0.5, 0.9, 0.1, 0.7, 0.4, 0.6, 0.2, 0.0};
     * double[] outputs = red.procesar(inputs, 100);  // Simular por 100 timesteps
     *
     * // outputs contiene 5 valores en [0, 1] correspondientes a las 5 neuronas de salida
     * }</pre>
     *
     * <h3>Validaciones realizadas:</h3>
     * <ul>
     *   <li>El array de inputs no puede ser null o vacío</li>
     *   <li>La duración debe ser positiva (> 0)</li>
     *   <li>El tamaño de inputs debe coincidir con el número de neuronas en la capa de entrada</li>
     * </ul>
     *
     * <h3>Requisitos implementados:</h3>
     * <ul>
     *   <li><b>Requisito 3.1:</b> Codificación de valores continuos a frecuencias de disparo</li>
     *   <li><b>Requisito 4.2:</b> Decodificación de trenes de spikes a valores continuos</li>
     *   <li><b>Requisito 7.1:</b> Simulación temporal discreta por timesteps</li>
     *   <li><b>Requisito 16.2:</b> Codificación temporal de entrada como secuencias</li>
     * </ul>
     *
     * <h3>Notas importantes:</h3>
     * <ul>
     *   <li>Este método NO resetea el estado temporal antes de procesar. Si desea
     *       procesar múltiples patrones de forma aislada, llame a
     *       {@link #resetearEstadoTemporal()} entre patrones.</li>
     *   <li>Los valores de entrada deben estar en el rango [0, 1] para una
     *       codificación óptima. Valores fuera de este rango serán clampeados
     *       por el codificador.</li>
     *   <li>La duración de la simulación afecta la precisión de la decodificación:
     *       duraciones más largas permiten acumular más spikes y obtener valores
     *       de salida más estables.</li>
     * </ul>
     *
     * @param inputs array de valores de entrada en el rango [0, 1], uno por neurona de entrada
     * @param duracionTimesteps número de timesteps a simular (debe ser > 0)
     * @return array de valores de salida en el rango [0, 1], uno por neurona de salida
     * @throws IllegalArgumentException si inputs es null o vacío
     * @throws IllegalArgumentException si duracionTimesteps <= 0
     * @throws IllegalArgumentException si el tamaño de inputs no coincide con el número de neuronas de entrada
     *
     * @see #resetearEstadoTemporal()
     * @see #avanzarTimestep()
     * @see GestorCodificacion#codificar(double[], int)
     * @see GestorDecodificacion#decodificarCapa(List)
     */
    public double[] procesar(double[] inputs, int duracionTimesteps) {
        // Validar parámetros
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("El array de inputs no puede ser null o vacío");
        }
        if (duracionTimesteps <= 0) {
            throw new IllegalArgumentException(
                "La duración debe ser positiva, recibido: " + duracionTimesteps
            );
        }

        // Validar que el tamaño de inputs coincida con la capa de entrada
        int numNeuronasEntrada = capas.get(0).size();
        if (inputs.length != numNeuronasEntrada) {
            throw new IllegalArgumentException(
                "El tamaño de inputs (" + inputs.length +
                ") no coincide con el número de neuronas de entrada (" + numNeuronasEntrada + ")"
            );
        }

        // 1. Codificar inputs a eventos de spike (Requisito 3.1, 16.2)
        java.util.List<EventoSpike> eventosEntrada = codificador.codificar(inputs, duracionTimesteps);

        // 2. Encolar eventos en la cola de eventos
        for (EventoSpike evento : eventosEntrada) {
            // Los eventos de entrada deben ser entregados a las neuronas de la capa de entrada
            // Necesitamos crear eventos que apunten a las neuronas correctas de la capa 0
            NeuronaSpiking neuronaEntrada = capas.get(0).get(evento.getIndice());

            // Crear un evento que será procesado como señal de entrada
            // Usamos un peso fuerte para asegurar que las neuronas de entrada disparen
            // El peso debe ser suficiente para llevar el potencial por encima del umbral
            double pesoSeñal = configuracion.umbralDisparo - configuracion.potencialReposo + 5.0;
            
            EventoSpike eventoParaNeurona = new EventoSpike(
                neuronaEntrada.getId(),
                0,
                evento.getIndice(),
                evento.getTimestamp(),
                pesoSeñal  // Peso de la señal de entrada
            );

            colaEventos.encolar(eventoParaNeurona);
        }

        // 3. Ejecutar simulación por duracionTimesteps (Requisito 7.1)
        for (int i = 0; i < duracionTimesteps; i++) {
            avanzarTimestep();
        }

        // 4. Recolectar spikes de la capa de salida
        // La capa de salida es la última capa
        int indiceCapaSalida = capas.size() - 1;
        List<NeuronaSpiking> capaSalida = capas.get(indiceCapaSalida);

        // Crear lista de listas de spikes, una por neurona de salida
        java.util.List<java.util.List<EventoSpike>> spikesPorNeuronaSalida = new java.util.ArrayList<>();

        for (NeuronaSpiking neurona : capaSalida) {
            // Obtener spikes de esta neurona desde el registro
            java.util.List<EventoSpike> spikesNeurona = registro.obtenerSpikes(neurona.getId());
            spikesPorNeuronaSalida.add(spikesNeurona);
        }

        // 5. Decodificar spikes de capa de salida a valores continuos (Requisito 4.2)
        double[] outputs = decodificador.decodificarCapa(spikesPorNeuronaSalida);

        return outputs;
    }

    /**
     * Procesa un lote de patrones de entrada secuencialmente.
     * 
     * @param batchInputs Array de patrones de entrada, cada uno es un array de valores [0,1]
     * @param duracionPorPatron Duración en timesteps para procesar cada patrón
     * @param resetEntrePatrones Si true, resetea el estado temporal entre patrones (aislamiento);
     *                           Si false, mantiene el estado temporal entre patrones (procesamiento secuencial)
     * @return Array de salidas decodificadas, una por cada patrón de entrada
     * @throws IllegalArgumentException si los parámetros son inválidos
     * 
     * Requisitos: 14.1, 14.2, 14.3, 14.4, 14.5
     */
    public double[][] procesarLote(double[][] batchInputs, int duracionPorPatron, boolean resetEntrePatrones) {
        // Validar parámetros
        if (batchInputs == null || batchInputs.length == 0) {
            throw new IllegalArgumentException("El lote de inputs no puede ser null o vacío");
        }
        if (duracionPorPatron <= 0) {
            throw new IllegalArgumentException(
                "La duración por patrón debe ser positiva, recibido: " + duracionPorPatron
            );
        }

        // Preparar array de salidas
        double[][] batchOutputs = new double[batchInputs.length][];

        // Procesar cada patrón secuencialmente (Requisito 14.1)
        for (int i = 0; i < batchInputs.length; i++) {
            // Si resetEntrePatrones está activo y no es el primer patrón, resetear estado (Requisito 14.2)
            if (resetEntrePatrones && i > 0) {
                resetearEstadoTemporal();
            }

            // Procesar el patrón actual
            double[] outputs = procesar(batchInputs[i], duracionPorPatron);
            batchOutputs[i] = outputs;

            // Las métricas se acumulan automáticamente a través del lote (Requisito 14.3)
            // ya que gestorMetricas y registro mantienen su estado entre llamadas a procesar()
        }

        // Retornar las salidas decodificadas para cada patrón (Requisito 14.4)
        return batchOutputs;
    }

    /**
     * Entrena la red con un lote de patrones usando STDP supervisado.
     *
     * <p>Este método implementa entrenamiento supervisado adaptado para redes spiking.
     * Para cada patrón de entrada y su target correspondiente:</p>
     * <ol>
     *   <li>Activa el modo entrenamiento (STDP se aplica durante simulación)</li>
     *   <li>Procesa el patrón de entrada</li>
     *   <li>Compara la salida con el target</li>
     *   <li>Aplica corrección supervisada: inyecta spikes en neuronas de salida
     *       que deberían haber disparado (target alto) pero no lo hicieron,
     *       y aplica LTD a conexiones de neuronas que dispararon pero no deberían</li>
     * </ol>
     *
     * <p>Requisitos: 5.1, 5.2, 5.3</p>
     *
     * @param batchInputs array de patrones de entrada
     * @param batchTargets array de targets correspondientes (valores en [0,1])
     * @param duracionPorPatron timesteps de simulación por patrón
     * @return error promedio del lote (MSE)
     * @throws IllegalArgumentException si los parámetros son inválidos
     */
    public double entrenar(double[][] batchInputs, double[][] batchTargets, int duracionPorPatron) {
        // Validar parámetros
        if (batchInputs == null || batchInputs.length == 0) {
            throw new IllegalArgumentException("El lote de inputs no puede ser null o vacío");
        }
        if (batchTargets == null || batchTargets.length == 0) {
            throw new IllegalArgumentException("El lote de targets no puede ser null o vacío");
        }
        if (batchInputs.length != batchTargets.length) {
            throw new IllegalArgumentException(
                "El número de inputs (" + batchInputs.length +
                ") no coincide con el número de targets (" + batchTargets.length + ")"
            );
        }
        if (duracionPorPatron <= 0) {
            throw new IllegalArgumentException(
                "La duración por patrón debe ser positiva, recibido: " + duracionPorPatron
            );
        }

        // Validar tamaño de targets vs capa de salida
        int numSalidas = capas.get(capas.size() - 1).size();
        for (int i = 0; i < batchTargets.length; i++) {
            if (batchTargets[i].length != numSalidas) {
                throw new IllegalArgumentException(
                    "El tamaño del target " + i + " (" + batchTargets[i].length +
                    ") no coincide con el número de neuronas de salida (" + numSalidas + ")"
                );
            }
        }

        // Activar modo entrenamiento (Requisito 5.1, 5.2)
        boolean modoAnterior = this.modoEntrenamiento;
        this.modoEntrenamiento = true;

        double errorTotal = 0.0;

        for (int p = 0; p < batchInputs.length; p++) {
            // Resetear estado temporal entre patrones
            resetearEstadoTemporal();

            // Procesar patrón (STDP se aplica automáticamente durante simulación)
            double[] salida = procesar(batchInputs[p], duracionPorPatron);

            // Calcular error MSE para este patrón
            double errorPatron = 0.0;
            for (int j = 0; j < numSalidas; j++) {
                double diff = batchTargets[p][j] - salida[j];
                errorPatron += diff * diff;
            }
            errorPatron /= numSalidas;
            errorTotal += errorPatron;

            // Corrección supervisada basada en diferencia salida-target
            List<NeuronaSpiking> capaSalida = capas.get(capas.size() - 1);
            for (int j = 0; j < numSalidas; j++) {
                double target = batchTargets[p][j];
                double actual = salida[j];
                double error = target - actual;

                if (Math.abs(error) > 0.05) {
                    // Buscar sinapsis que conectan a esta neurona de salida
                    NeuronaSpiking neuronaSalida = capaSalida.get(j);
                    for (SinapsisSpiking sin : this.sinapsis) {
                        if (sin.getPostsinaptica().getId() == neuronaSalida.getId()) {
                            // Ajustar peso proporcionalmente al error
                            double ajuste = error * configuracion.amplitudLTP;
                            sin.ajustarPeso(ajuste);
                        }
                    }
                }
            }
        }

        // Restaurar modo entrenamiento anterior
        this.modoEntrenamiento = modoAnterior;

        // Retornar error promedio del lote
        return errorTotal / batchInputs.length;
    }

    /**
     * Normaliza los pesos de las sinapsis entrantes a neuronas en las capas objetivo.
     *
     * <p>Para cada neurona en las capas objetivo, normaliza los pesos de todas sus
     * sinapsis entrantes según el tipo de normalización especificado, preservando
     * el signo de los pesos.</p>
     *
     * <p>Tipos soportados:</p>
     * <ul>
     *   <li>L1: suma de valores absolutos de pesos entrantes = 1.0</li>
     *   <li>L2: suma de cuadrados de pesos entrantes = 1.0</li>
     * </ul>
     *
     * <p>Requisitos: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6</p>
     *
     * @param tipo tipo de normalización (L1 o L2)
     * @param capasObjetivo índices de las capas a normalizar
     * @throws IllegalArgumentException si tipo o capasObjetivo son null
     */
    public void normalizarPesos(TipoNormalizacion tipo, int[] capasObjetivo) {
        if (tipo == null) {
            throw new IllegalArgumentException("El tipo de normalización no puede ser null");
        }
        if (capasObjetivo == null || capasObjetivo.length == 0) {
            throw new IllegalArgumentException("Las capas objetivo no pueden ser null o vacías");
        }

        for (int indiceCapa : capasObjetivo) {
            if (indiceCapa < 0 || indiceCapa >= capas.size()) {
                throw new IllegalArgumentException("Índice de capa inválido: " + indiceCapa);
            }

            List<NeuronaSpiking> capa = capas.get(indiceCapa);
            for (NeuronaSpiking neurona : capa) {
                // Obtener sinapsis entrantes a esta neurona
                java.util.List<SinapsisSpiking> entrantes = new java.util.ArrayList<>();
                for (SinapsisSpiking s : this.sinapsis) {
                    if (s.getPostsinaptica().getId() == neurona.getId()) {
                        entrantes.add(s);
                    }
                }

                if (entrantes.isEmpty()) continue;

                // Calcular norma según tipo
                double norma = 0.0;
                if (tipo == TipoNormalizacion.L1) {
                    for (SinapsisSpiking s : entrantes) {
                        norma += Math.abs(s.getPeso());
                    }
                } else { // L2
                    for (SinapsisSpiking s : entrantes) {
                        norma += s.getPeso() * s.getPeso();
                    }
                    norma = Math.sqrt(norma);
                }

                // Normalizar preservando signo (Requisito 17.5)
                if (norma > 0) {
                    for (SinapsisSpiking s : entrantes) {
                        double pesoNormalizado = s.getPeso() / norma;
                        // Clamp al rango permitido
                        pesoNormalizado = Math.max(s.getPesoMin(), Math.min(s.getPesoMax(), pesoNormalizado));
                        s.setPeso(pesoNormalizado);
                    }
                }
            }
        }
    }

    // ========== Inhibición Lateral ==========

    /**
     * Calcula las neuronas vecinas dentro de un radio dado en la misma capa.
     *
     * <p>Las neuronas se organizan en un grid 1D dentro de cada capa (por índice).
     * La distancia se calcula como la diferencia absoluta entre índices.
     * Solo se retornan neuronas distintas a la neurona de referencia.</p>
     *
     * @param neurona la neurona de referencia
     * @param radio radio máximo de vecindad (en número de neuronas)
     * @return lista de neuronas vecinas dentro del radio
     * @throws IllegalArgumentException si neurona es null o radio <= 0
     */
    public List<NeuronaSpiking> calcularVecinas(NeuronaSpiking neurona, int radio) {
        if (neurona == null) {
            throw new IllegalArgumentException("La neurona no puede ser null");
        }
        if (radio <= 0) {
            throw new IllegalArgumentException("El radio debe ser positivo, recibido: " + radio);
        }

        int capa = neurona.getCapa();
        if (capa < 0 || capa >= capas.size()) {
            return new java.util.ArrayList<>();
        }

        List<NeuronaSpiking> capaNeuronas = capas.get(capa);
        List<NeuronaSpiking> vecinas = new java.util.ArrayList<>();
        int indiceRef = neurona.getIndice();

        for (NeuronaSpiking candidata : capaNeuronas) {
            if (candidata.getId() == neurona.getId()) continue;
            double distancia = Math.abs(candidata.getIndice() - indiceRef);
            if (distancia <= radio) {
                vecinas.add(candidata);
            }
        }

        return vecinas;
    }

    /**
     * Aplica inhibición lateral desde una neurona que acaba de disparar.
     *
     * <p>Envía señales inhibitorias a neuronas vecinas en la misma capa.
     * La fuerza de la señal decrece linealmente con la distancia:
     * señal = -fuerzaInhibicion * (1 - distancia/radio)</p>
     *
     * @param neurona la neurona que disparó
     */
    private void aplicarInhibicionLateral(NeuronaSpiking neurona) {
        int radio = configuracion.radioInhibicion;
        double fuerza = configuracion.fuerzaInhibicion;

        List<NeuronaSpiking> vecinas = calcularVecinas(neurona, radio);
        int indiceRef = neurona.getIndice();

        for (NeuronaSpiking vecina : vecinas) {
            double distancia = Math.abs(vecina.getIndice() - indiceRef);
            double señalInhibitoria = -fuerza * (1.0 - distancia / radio);
            vecina.recibirSeñal(señalInhibitoria);
        }
    }

    // ========== Persistencia ==========

    /**
     * Guarda la red neuronal en un archivo binario usando serialización Java.
     *
     * @param filename ruta del archivo donde guardar la red
     * @throws IllegalArgumentException si filename es null o vacío
     * @throws java.io.IOException si ocurre un error de escritura
     */
    public void guardar(String filename) throws java.io.IOException {
        if (filename == null || filename.trim().isEmpty()) {
            throw new IllegalArgumentException("El nombre de archivo no puede ser null o vacío");
        }
        try (java.io.ObjectOutputStream oos = new java.io.ObjectOutputStream(
                new java.io.FileOutputStream(filename))) {
            oos.writeObject(this);
        }
    }

    /**
     * Carga una red neuronal desde un archivo binario.
     * Resetea el estado temporal después de cargar.
     *
     * @param filename ruta del archivo a cargar
     * @return la red neuronal cargada con estado temporal reseteado
     * @throws IllegalArgumentException si filename es null o vacío
     * @throws java.io.IOException si ocurre un error de lectura
     * @throws ClassNotFoundException si la clase no se encuentra durante deserialización
     */
    public static RedNeuralSpiking cargar(String filename) throws java.io.IOException, ClassNotFoundException {
        if (filename == null || filename.trim().isEmpty()) {
            throw new IllegalArgumentException("El nombre de archivo no puede ser null o vacío");
        }
        RedNeuralSpiking red;
        try (java.io.ObjectInputStream ois = new java.io.ObjectInputStream(
                new java.io.FileInputStream(filename))) {
            red = (RedNeuralSpiking) ois.readObject();
        }
        // Resetear estado temporal después de cargar (Requisito 12.7)
        red.resetearEstadoTemporal();
        return red;
    }

    // ========== Serialización JSON ==========

    /**
     * Serializa la configuración de la red a formato JSON.
     *
     * @return cadena JSON con la configuración completa de la red
     */
    public String toJSON() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"topologia\": ").append(arrayToJSON(configuracion.topologia)).append(",\n");
        sb.append("  \"umbralDisparo\": ").append(configuracion.umbralDisparo).append(",\n");
        sb.append("  \"potencialReposo\": ").append(configuracion.potencialReposo).append(",\n");
        sb.append("  \"constanteDecaimiento\": ").append(configuracion.constanteDecaimiento).append(",\n");
        sb.append("  \"duracionRefractario\": ").append(configuracion.duracionRefractario).append(",\n");
        sb.append("  \"amplitudLTP\": ").append(configuracion.amplitudLTP).append(",\n");
        sb.append("  \"amplitudLTD\": ").append(configuracion.amplitudLTD).append(",\n");
        sb.append("  \"tauLTP\": ").append(configuracion.tauLTP).append(",\n");
        sb.append("  \"tauLTD\": ").append(configuracion.tauLTD).append(",\n");
        sb.append("  \"frecuenciaMaxima\": ").append(configuracion.frecuenciaMaxima).append(",\n");
        sb.append("  \"modoCodificacion\": \"").append(configuracion.modoCodificacion.name()).append("\",\n");
        sb.append("  \"ventanaDecodificacion\": ").append(configuracion.ventanaDecodificacion).append(",\n");
        sb.append("  \"tipoInicializacion\": \"").append(configuracion.tipoInicializacion.name()).append("\",\n");
        sb.append("  \"pesoMin\": ").append(configuracion.pesoMin).append(",\n");
        sb.append("  \"pesoMax\": ").append(configuracion.pesoMax).append(",\n");
        sb.append("  \"tipoNormalizacion\": \"").append(configuracion.tipoNormalizacion.name()).append("\",\n");
        sb.append("  \"valorObjetivoNormalizacion\": ").append(configuracion.valorObjetivoNormalizacion).append(",\n");
        sb.append("  \"retardoMin\": ").append(configuracion.retardoMin).append(",\n");
        sb.append("  \"retardoMax\": ").append(configuracion.retardoMax).append(",\n");
        sb.append("  \"homeostasisActiva\": ").append(configuracion.homeostasisActiva).append(",\n");
        sb.append("  \"tasaDisparoObjetivo\": ").append(configuracion.tasaDisparoObjetivo).append(",\n");
        sb.append("  \"tasaAjusteHomeostasis\": ").append(configuracion.tasaAjusteHomeostasis).append(",\n");
        sb.append("  \"inhibicionLateralActiva\": ").append(configuracion.inhibicionLateralActiva).append(",\n");
        sb.append("  \"radioInhibicion\": ").append(configuracion.radioInhibicion).append(",\n");
        sb.append("  \"fuerzaInhibicion\": ").append(configuracion.fuerzaInhibicion).append(",\n");
        sb.append("  \"duracionTimestep\": ").append(configuracion.duracionTimestep).append("\n");
        sb.append("}");
        return sb.toString();
    }

    private static String arrayToJSON(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Construye una red neuronal desde una cadena JSON de configuración.
     *
     * @param jsonConfig cadena JSON con la configuración
     * @return nueva red neuronal construida con la configuración parseada
     * @throws IllegalArgumentException si el JSON es null, vacío o tiene formato inválido
     */
    public static RedNeuralSpiking desdeJSON(String jsonConfig) {
        if (jsonConfig == null || jsonConfig.trim().isEmpty()) {
            throw new IllegalArgumentException("La configuración JSON no puede ser null o vacía");
        }

        try {
            java.util.Map<String, String> campos = parsearJSON(jsonConfig);

            ConfiguracionRedBuilder builder = new ConfiguracionRedBuilder();

            // Topología (requerido)
            if (!campos.containsKey("topologia")) {
                throw new IllegalArgumentException("Campo requerido 'topologia' no encontrado en JSON");
            }
            builder.topologia(parsearArrayInt(campos.get("topologia")));

            // Parámetros LIF
            builder.parametrosLIF(
                parseDouble(campos, "umbralDisparo", -55.0),
                parseDouble(campos, "potencialReposo", -70.0),
                parseDouble(campos, "constanteDecaimiento", 20.0),
                parseInt(campos, "duracionRefractario", 2)
            );

            // Parámetros STDP
            builder.parametrosSTDP(
                parseDouble(campos, "amplitudLTP", 0.01),
                parseDouble(campos, "amplitudLTD", 0.012),
                parseDouble(campos, "tauLTP", 20.0),
                parseDouble(campos, "tauLTD", 20.0)
            );

            // Codificación
            builder.frecuenciaMaxima(parseDouble(campos, "frecuenciaMaxima", 100.0));
            if (campos.containsKey("modoCodificacion")) {
                builder.modoCodificacion(ModoCodificacion.valueOf(campos.get("modoCodificacion")));
            }
            builder.ventanaDecodificacion(parseInt(campos, "ventanaDecodificacion", 50));

            // Pesos
            if (campos.containsKey("tipoInicializacion")) {
                builder.tipoInicializacion(TipoInicializacion.valueOf(campos.get("tipoInicializacion")));
            }
            builder.rangoPesos(
                parseDouble(campos, "pesoMin", 0.0),
                parseDouble(campos, "pesoMax", 1.0)
            );

            // Normalización
            if (campos.containsKey("tipoNormalizacion")) {
                builder.tipoNormalizacion(TipoNormalizacion.valueOf(campos.get("tipoNormalizacion")));
            }
            builder.valorObjetivoNormalizacion(parseDouble(campos, "valorObjetivoNormalizacion", 1.0));

            // Retardos
            builder.retardos(
                parseInt(campos, "retardoMin", 1),
                parseInt(campos, "retardoMax", 5)
            );

            // Homeostasis
            boolean homeostasis = parseBoolean(campos, "homeostasisActiva", false);
            builder.homeostasis(homeostasis,
                parseDouble(campos, "tasaDisparoObjetivo", 10.0),
                parseDouble(campos, "tasaAjusteHomeostasis", 0.01)
            );

            // Inhibición lateral
            boolean inhibicion = parseBoolean(campos, "inhibicionLateralActiva", false);
            builder.inhibicionLateral(inhibicion,
                parseInt(campos, "radioInhibicion", 2),
                parseDouble(campos, "fuerzaInhibicion", 0.5)
            );

            // Timestep
            builder.duracionTimestep(parseDouble(campos, "duracionTimestep", 1.0));

            return new RedNeuralSpiking(builder.build());
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("Error parseando JSON: " + e.getMessage(), e);
        }
    }

    private static java.util.Map<String, String> parsearJSON(String json) {
        java.util.Map<String, String> result = new java.util.LinkedHashMap<>();
        // Eliminar llaves externas y espacios
        json = json.trim();
        if (json.startsWith("{")) json = json.substring(1);
        if (json.endsWith("}")) json = json.substring(0, json.length() - 1);

        // Parsear pares clave:valor
        int i = 0;
        while (i < json.length()) {
            // Buscar inicio de clave (comilla)
            int keyStart = json.indexOf('"', i);
            if (keyStart < 0) break;
            int keyEnd = json.indexOf('"', keyStart + 1);
            if (keyEnd < 0) break;
            String key = json.substring(keyStart + 1, keyEnd);

            // Buscar ':'
            int colon = json.indexOf(':', keyEnd + 1);
            if (colon < 0) break;

            // Extraer valor
            int valStart = colon + 1;
            while (valStart < json.length() && json.charAt(valStart) == ' ') valStart++;

            String value;
            if (valStart < json.length() && json.charAt(valStart) == '"') {
                // Valor string
                int valEnd = json.indexOf('"', valStart + 1);
                value = json.substring(valStart + 1, valEnd);
                i = valEnd + 1;
            } else if (valStart < json.length() && json.charAt(valStart) == '[') {
                // Valor array
                int valEnd = json.indexOf(']', valStart);
                value = json.substring(valStart, valEnd + 1);
                i = valEnd + 1;
            } else {
                // Valor numérico o booleano
                int valEnd = valStart;
                while (valEnd < json.length() && json.charAt(valEnd) != ',' && json.charAt(valEnd) != '\n' && json.charAt(valEnd) != '}') {
                    valEnd++;
                }
                value = json.substring(valStart, valEnd).trim();
                i = valEnd;
            }

            result.put(key, value);
            // Avanzar pasado la coma
            int nextComma = json.indexOf(',', i);
            i = (nextComma >= 0) ? nextComma + 1 : json.length();
        }
        return result;
    }

    private static int[] parsearArrayInt(String arrayStr) {
        arrayStr = arrayStr.trim();
        if (arrayStr.startsWith("[")) arrayStr = arrayStr.substring(1);
        if (arrayStr.endsWith("]")) arrayStr = arrayStr.substring(0, arrayStr.length() - 1);
        String[] parts = arrayStr.split(",");
        int[] result = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }
        return result;
    }

    private static double parseDouble(java.util.Map<String, String> campos, String key, double defaultVal) {
        return campos.containsKey(key) ? Double.parseDouble(campos.get(key)) : defaultVal;
    }

    private static int parseInt(java.util.Map<String, String> campos, String key, int defaultVal) {
        return campos.containsKey(key) ? Integer.parseInt(campos.get(key)) : defaultVal;
    }

    private static boolean parseBoolean(java.util.Map<String, String> campos, String key, boolean defaultVal) {
        return campos.containsKey(key) ? Boolean.parseBoolean(campos.get(key)) : defaultVal;
    }

}
