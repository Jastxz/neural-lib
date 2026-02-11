package es.jastxz.nn;

import es.jastxz.nn.enums.EstadoRed;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoConexion;
import es.jastxz.nn.enums.TipoNeurona;

import java.util.*;

/**
 * Red Neuronal Experimental basada en principios biológicos
 * 
 * Implementa:
 * - Arquitectura bidireccional (feed-forward + feedback)
 * - Plasticidad hebiana (STDP)
 * - Predicción y error de predicción
 * - Competición por recursos
 * - Formación y consolidación de engramas
 * 
 * IMPORTANTE - Tres Niveles de Estado:
 * 
 * 1. MEMORIA DE CORTO PLAZO (working memory):
 *    - Estado de activación de neuronas (activa=1, reposo=0)
 *    - Potenciales de membrana actuales
 *    - Funciona como bits de información (pg. 58, 83 Campillo)
 *    - Se mantiene entre procesamientos de la misma tarea
 *    - Se resetea solo al cambiar de tarea (llamar resetear())
 * 
 * 2. MEMORIA DE MEDIO PLAZO (engramas no consolidados):
 *    - Patrones de activación recientes
 *    - Engramas en formación
 *    - Se consolidan durante "sueño" (Fase 7)
 * 
 * 3. MEMORIA DE LARGO PLAZO (conocimiento consolidado):
 *    - Valores almacenados en neuronas (valorAlmacenado)
 *    - Pesos sinápticos de conexiones
 *    - Engramas consolidados
 *    - Persiste indefinidamente (hasta poda por desuso)
 * 
 * Referencias:
 * - "La información se codifica en bits de reposo o activación" (pg. 83 Campillo)
 * - "Cada neurona funciona como un bit de información" (pg. 58 Campillo)
 * - "Engramas: grupos de neuronas que funcionan como bits" (pg. 84-85 Campillo)
 * 
 * Basado en: Apuntes_Neurologicos.md
 */
public class RedNeuralExperimental {
    
    // Arquitectura de capas
    private List<Neurona> capaSensorial;
    private List<List<Neurona>> capasInterneuronas;  // Múltiples subcapas
    private List<Neurona> capaMotora;
    
    // Conectividad
    private List<Conexion> conexiones;
    
    // Sistema de memoria
    private Map<String, Engrama> engramas;
    
    // Control temporal
    private long timestampGlobal;
    private EstadoRed estado;
    
    // Configuración
    private int[] topologia;  // [sensorial, inter1, inter2, ..., motora]
    private double densidadConexiones;  // 0.0 a 1.0
    private Random random;
    
    // Contador para IDs únicos
    private long contadorNeuronas;
    
    /**
     * Constructor con topología configurable
     * 
     * @param topologia Array con número de neuronas por capa [input, hidden1, hidden2, ..., output]
     * @param densidadConexiones Densidad de conexiones entre capas (0.0 a 1.0)
     */
    public RedNeuralExperimental(int[] topologia, double densidadConexiones) {
        if (topologia == null || topologia.length < 2) {
            throw new IllegalArgumentException("La topología debe tener al menos 2 capas (input y output)");
        }
        
        for (int tamano : topologia) {
            if (tamano <= 0) {
                throw new IllegalArgumentException("Cada capa debe tener al menos 1 neurona");
            }
        }
        
        if (densidadConexiones < 0.0 || densidadConexiones > 1.0) {
            throw new IllegalArgumentException("La densidad debe estar entre 0.0 y 1.0");
        }
        
        this.topologia = topologia.clone();
        this.densidadConexiones = densidadConexiones;
        this.random = new Random();
        this.contadorNeuronas = 0L;
        
        this.capaSensorial = new ArrayList<>();
        this.capasInterneuronas = new ArrayList<>();
        this.capaMotora = new ArrayList<>();
        this.conexiones = new ArrayList<>();
        this.engramas = new HashMap<>();
        
        this.timestampGlobal = 0L;
        this.estado = EstadoRed.ACTIVO;
        
        inicializarCapas();
        generarConexiones();
    }
    
    /**
     * Inicializa las capas de neuronas según la topología
     */
    private void inicializarCapas() {
        // Capa sensorial (primera capa)
        for (int i = 0; i < topologia[0]; i++) {
            Neurona neurona = new Neurona(
                contadorNeuronas++,
                TipoNeurona.SENSORIAL,
                randomValue(),
                PotencialMemoria.REPOSO  // Potencial de reposo
            );
            capaSensorial.add(neurona);
        }
        
        // Capas de interneuronas (capas intermedias)
        for (int capa = 1; capa < topologia.length - 1; capa++) {
            List<Neurona> capaInter = new ArrayList<>();
            for (int i = 0; i < topologia[capa]; i++) {
                Neurona neurona = new Neurona(
                    contadorNeuronas++,
                    TipoNeurona.INTER,
                    randomValue(),
                    PotencialMemoria.REPOSO
                );
                capaInter.add(neurona);
            }
            capasInterneuronas.add(capaInter);
        }
        
        // Capa motora (última capa)
        for (int i = 0; i < topologia[topologia.length - 1]; i++) {
            Neurona neurona = new Neurona(
                contadorNeuronas++,
                TipoNeurona.MOTORA,
                randomValue(),
                PotencialMemoria.REPOSO
            );
            capaMotora.add(neurona);
        }
    }
    
    /**
     * Genera conexiones entre capas según densidad configurada
     * Incluye conexiones feed-forward y feedback
     */
    private void generarConexiones() {
        // Conexiones feed-forward
        if (!capasInterneuronas.isEmpty()) {
            // sensorial → primera inter
            generarConexionesEntrCapas(capaSensorial, capasInterneuronas.get(0), true);
            
            // Conexiones entre capas de interneuronas
            for (int i = 0; i < capasInterneuronas.size() - 1; i++) {
                generarConexionesEntrCapas(capasInterneuronas.get(i), capasInterneuronas.get(i + 1), true);
            }
            
            // última inter → motora
            generarConexionesEntrCapas(
                capasInterneuronas.get(capasInterneuronas.size() - 1),
                capaMotora,
                true
            );
        } else {
            // Si no hay capas intermedias, conectar directamente sensorial → motora
            generarConexionesEntrCapas(capaSensorial, capaMotora, true);
        }
        
        // Conexiones feedback (bidireccionales)
        generarConexionesFeedback();
    }
    
    /**
     * Genera conexiones entre dos capas
     * 
     * @param capaOrigen Capa presináptica
     * @param capaDestino Capa postsináptica
     * @param permitirDiadicas Si se permiten sinapsis diádicas/triádicas
     */
    private void generarConexionesEntrCapas(List<Neurona> capaOrigen, List<Neurona> capaDestino, boolean permitirDiadicas) {
        for (Neurona pre : capaOrigen) {
            for (Neurona post : capaDestino) {
                // Decidir si crear conexión según densidad
                if (random.nextDouble() < densidadConexiones) {
                    double peso = randomValue();  // Peso inicial aleatorio [-1, 1]
                    
                    // 66% de probabilidad de sinapsis diádica (inspirado en C. elegans)
                    if (permitirDiadicas && random.nextDouble() < 0.66 && capaDestino.size() > 1) {
                        // Sinapsis diádica: 1 pre → 2 post
                        Neurona post2 = capaDestino.get(random.nextInt(capaDestino.size()));
                        if (post2 != post) {
                            List<Neurona> postsinápticas = Arrays.asList(post, post2);
                            Conexion conexion = new Conexion(pre, postsinápticas, peso, TipoConexion.QUIMICA);
                            conexiones.add(conexion);
                            continue;  // Ya creamos la conexión diádica
                        }
                    }
                    
                    // Conexión simple 1→1
                    Conexion conexion = new Conexion(pre, post, peso, TipoConexion.QUIMICA);
                    conexiones.add(conexion);
                }
            }
        }
    }
    
    /**
     * Genera conexiones feedback (de capas posteriores a anteriores)
     * Implementa el principio de retroalimentación (pg. 61 Eagleman)
     */
    private void generarConexionesFeedback() {
        // Feedback: motora → última inter
        if (!capasInterneuronas.isEmpty()) {
            generarConexionesEntrCapas(
                capaMotora,
                capasInterneuronas.get(capasInterneuronas.size() - 1),
                false  // Feedback sin sinapsis múltiples para simplificar
            );
        }
        
        // Feedback entre capas de interneuronas (de posterior a anterior)
        for (int i = capasInterneuronas.size() - 1; i > 0; i--) {
            generarConexionesEntrCapas(
                capasInterneuronas.get(i),
                capasInterneuronas.get(i - 1),
                false
            );
        }
    }
    
    /**
     * Genera valor aleatorio entre -1 y 1
     */
    private double randomValue() {
        return (random.nextDouble() * 2.0) - 1.0;
    }
    
    /**
     * Procesa inputs y genera outputs
     * Combina propagación feed-forward y feedback
     * 
     * IMPORTANTE: NO resetea el estado de activación (memoria de corto plazo)
     * Las neuronas mantienen su estado como bits de memoria (pg. 83, 58 Campillo)
     * Para resetear memoria de corto plazo, llamar explícitamente a resetear()
     * 
     * @param inputs Array de valores de entrada (debe coincidir con tamaño de capa sensorial)
     * @return Array de valores de salida (tamaño de capa motora)
     */
    public double[] procesar(double[] inputs) {
        if (inputs == null || inputs.length != capaSensorial.size()) {
            throw new IllegalArgumentException(
                "Los inputs deben tener tamaño " + capaSensorial.size() + 
                ", recibido: " + (inputs == null ? "null" : inputs.length)
            );
        }
        
        // NO resetear estado - las neuronas mantienen su activación como memoria
        // (pg. 83 Campillo: "bits de reposo o activación")
        
        // Establecer inputs en capa sensorial
        establecerInputs(inputs);
        
        // Propagación feed-forward
        propagarHaciaAdelante();
        
        // Propagación feedback (refinamiento)
        propagarHaciaAtras();
        
        // Avanzar tiempo
        avanzarTiempo(1L);
        
        // Extraer outputs
        return getOutputs();
    }
    
    /**
     * Resetea solo el estado transitorio (potenciales, activaciones)
     * Preserva el conocimiento aprendido (valores almacenados, pesos sinápticos)
     */
    private void resetearEstadoTransitorio() {
        for (Neurona n : capaSensorial) n.resetear();
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona n : capa) n.resetear();
        }
        for (Neurona n : capaMotora) n.resetear();
    }
    
    /**
     * Establece los valores de entrada en la capa sensorial
     */
    private void establecerInputs(double[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            Neurona neurona = capaSensorial.get(i);
            
            // Establecer valor almacenado
            neurona.setValorAlmacenado(inputs[i]);
            
            // Si el input es significativo, activar la neurona
            if (Math.abs(inputs[i]) > 0.1) {
                neurona.setPotencial(PotencialMemoria.PICO);
                // Evaluar para marcar como activa
                neurona.evaluar(timestampGlobal);
            }
        }
    }
    
    /**
     * Propagación feed-forward: de sensorial hacia motora
     * Cada capa evalúa sus neuronas basándose en inputs de la capa anterior
     */
    private void propagarHaciaAdelante() {
        // Evaluar capas de interneuronas
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona neurona : capa) {
                neurona.evaluar(timestampGlobal);
            }
        }
        
        // Evaluar capa motora
        for (Neurona neurona : capaMotora) {
            neurona.evaluar(timestampGlobal);
        }
    }
    
    /**
     * Propagación feedback: de motora hacia sensorial
     * Implementa retroalimentación para refinamiento (pg. 61 Eagleman)
     * 
     * En esta versión simplificada, el feedback modula la activación
     * de capas anteriores basándose en la actividad de capas posteriores
     */
    private void propagarHaciaAtras() {
        // Feedback desde capa motora
        if (!capasInterneuronas.isEmpty()) {
            aplicarFeedbackACapa(capaMotora, capasInterneuronas.get(capasInterneuronas.size() - 1));
        }
        
        // Feedback entre capas de interneuronas (de posterior a anterior)
        for (int i = capasInterneuronas.size() - 1; i > 0; i--) {
            aplicarFeedbackACapa(capasInterneuronas.get(i), capasInterneuronas.get(i - 1));
        }
    }
    
    /**
     * Aplica feedback de una capa posterior a una anterior
     * Modula la activación basándose en conexiones feedback
     */
    private void aplicarFeedbackACapa(List<Neurona> capaOrigen, List<Neurona> capaDestino) {
        for (Neurona destino : capaDestino) {
            double feedbackTotal = 0.0;
            int contadorFeedback = 0;
            
            // Buscar conexiones feedback desde capaOrigen
            for (Conexion axon : destino.getAxones()) {
                // Si alguna postsináptica está en capaOrigen y activa
                for (Neurona post : axon.getPostsinápticas()) {
                    if (capaOrigen.contains(post) && post.estaActiva()) {
                        feedbackTotal += axon.getPeso() * post.getPotencial();
                        contadorFeedback++;
                    }
                }
            }
            
            // Aplicar modulación por feedback
            if (contadorFeedback > 0) {
                double modulacion = feedbackTotal / contadorFeedback;
                // Ajustar valor almacenado basándose en feedback
                double nuevoValor = destino.getValorAlmacenado() + (modulacion * 0.1);
                destino.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
            }
        }
    }
    
    /**
     * Extrae los valores de salida de la capa motora
     */
    private double[] getOutputs() {
        double[] outputs = new double[capaMotora.size()];
        
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neurona = capaMotora.get(i);
            
            // Combinar potencial y valor almacenado
            // Si está activa, usar potencial normalizado
            if (neurona.estaActiva()) {
                outputs[i] = neurona.getPotencial() / PotencialMemoria.PICO.getValor();
            } else {
                // Si no está activa, usar valor almacenado
                outputs[i] = neurona.getValorAlmacenado();
            }
        }
        
        return outputs;
    }
    
    /**
     * Resetea todas las neuronas a estado de reposo
     * 
     * IMPORTANTE: Esto borra la memoria de corto plazo (working memory)
     * Las neuronas funcionan como bits: activa=1, reposo=0 (pg. 58, 83 Campillo)
     * 
     * Usar cuando:
     * - Se inicia una nueva tarea completamente diferente
     * - Se quiere limpiar el contexto de procesamiento
     * 
     * NO usar entre procesamientos de la misma tarea secuencial
     * (la red necesita mantener su estado para contexto)
     */
    public void resetear() {
        resetearEstadoTransitorio();
    }
    
    /**
     * Avanza el reloj interno de la red
     */
    public void avanzarTiempo(long delta) {
        this.timestampGlobal += delta;
    }
    
    // Getters
    public List<Neurona> getCapaSensorial() {
        return new ArrayList<>(capaSensorial);
    }
    
    public List<List<Neurona>> getCapasInterneuronas() {
        List<List<Neurona>> copia = new ArrayList<>();
        for (List<Neurona> capa : capasInterneuronas) {
            copia.add(new ArrayList<>(capa));
        }
        return copia;
    }
    
    public List<Neurona> getCapaMotora() {
        return new ArrayList<>(capaMotora);
    }
    
    public List<Conexion> getConexiones() {
        return new ArrayList<>(conexiones);
    }
    
    public Map<String, Engrama> getEngramas() {
        return new HashMap<>(engramas);
    }
    
    public long getTimestampGlobal() {
        return timestampGlobal;
    }
    
    public EstadoRed getEstado() {
        return estado;
    }
    
    public int[] getTopologia() {
        return topologia.clone();
    }
    
    public int getTotalNeuronas() {
        int total = capaSensorial.size() + capaMotora.size();
        for (List<Neurona> capa : capasInterneuronas) {
            total += capa.size();
        }
        return total;
    }
    
    public int getTotalConexiones() {
        return conexiones.size();
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("RedNeuralExperimental {\n");
        sb.append("  Topología: ").append(Arrays.toString(topologia)).append("\n");
        sb.append("  Total neuronas: ").append(getTotalNeuronas()).append("\n");
        sb.append("  Total conexiones: ").append(getTotalConexiones()).append("\n");
        sb.append("  Densidad: ").append(String.format("%.2f", densidadConexiones)).append("\n");
        sb.append("  Estado: ").append(estado).append("\n");
        sb.append("  Timestamp: ").append(timestampGlobal).append("\n");
        sb.append("  Engramas: ").append(engramas.size()).append("\n");
        sb.append("}");
        return sb.toString();
    }
}
