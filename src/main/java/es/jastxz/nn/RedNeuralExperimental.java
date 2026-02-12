package es.jastxz.nn;

import es.jastxz.nn.enums.EstadoRed;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoConexion;
import es.jastxz.nn.enums.TipoNeurona;
import es.jastxz.nn.experimental.EntrenadorHebiano;
import es.jastxz.nn.experimental.PropagadorSeñal;
import es.jastxz.nn.experimental.GestorEngramas;
import es.jastxz.nn.experimental.GestorPredicciones;
import es.jastxz.nn.experimental.GestorCompeticion;

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
    // (gestionado por gestorEngramas)
    
    // Control temporal
    private long timestampGlobal;
    private EstadoRed estado;
    
    // Configuración
    private int[] topologia;  // [sensorial, inter1, inter2, ..., motora]
    private double densidadConexiones;  // 0.0 a 1.0
    private Random random;
    
    // Contador para IDs únicos
    private long contadorNeuronas;
    
    // Componentes auxiliares
    private final PropagadorSeñal propagador;
    private final EntrenadorHebiano entrenador;
    private final GestorEngramas gestorEngramas;
    private final GestorPredicciones gestorPredicciones;
    private final GestorCompeticion gestorCompeticion;
    
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
        
        this.timestampGlobal = 0L;
        this.estado = EstadoRed.ACTIVO;
        
        this.propagador = new PropagadorSeñal();
        this.entrenador = new EntrenadorHebiano();
        this.gestorEngramas = new GestorEngramas();
        this.gestorPredicciones = new GestorPredicciones(topologia[topologia.length - 1]);
        this.gestorCompeticion = new GestorCompeticion();
        
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
                    // Peso inicial positivo pequeño para feed-forward (permite aprendizaje hebiano)
                    double peso = random.nextDouble() * 0.3 + 0.1;  // [0.1, 0.4]
                    
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
        // Validar que no estamos en consolidación
        if (estado == EstadoRed.CONSOLIDANDO) {
            throw new IllegalStateException("No se puede procesar durante consolidación");
        }
        
        if (inputs == null || inputs.length != capaSensorial.size()) {
            throw new IllegalArgumentException(
                "Los inputs deben tener tamaño " + capaSensorial.size() + 
                ", recibido: " + (inputs == null ? "null" : inputs.length)
            );
        }
        
        // NO resetear estado - las neuronas mantienen su activación como memoria
        // (pg. 83 Campillo: "bits de reposo o activación")
        
        // Si modo predictivo está activo, calcular predicciones antes de procesar
        if (gestorPredicciones.estaActivo()) {
            gestorPredicciones.calcularPredicciones(capasInterneuronas, capaMotora);
        }
        
        // Establecer inputs en capa sensorial
        propagador.establecerInputs(capaSensorial, inputs, timestampGlobal);
        
        // Propagación feed-forward
        propagador.propagarHaciaAdelante(capasInterneuronas, capaMotora, timestampGlobal);
        
        // Propagación feedback (refinamiento)
        propagador.propagarHaciaAtras(capaMotora, capasInterneuronas);
        
        // Si detección de engramas está activa, detectar patrones
        if (gestorEngramas.esDeteccionActiva()) {
            gestorEngramas.detectarYFormarEngramas(capasInterneuronas, timestampGlobal);
        }
        
        // Si modo predictivo, calcular errores y ajustar modelo
        if (gestorPredicciones.estaActivo()) {
            gestorPredicciones.calcularErroresPrediccion(capasInterneuronas, capaMotora);
            gestorPredicciones.ajustarModeloPredictivo(capasInterneuronas, capaMotora);
        }
        
        // Avanzar tiempo
        avanzarTiempo(1L);
        
        // Extraer outputs
        return propagador.getOutputs(capaMotora);
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
        
        // Si modo predictivo está activo, reinicializar predicciones
        if (gestorPredicciones.estaActivo()) {
            gestorPredicciones.inicializarPredicciones(capasInterneuronas, capaMotora);
        }
    }
    
    /**
     * Avanza el reloj interno de la red
     */
    public void avanzarTiempo(long delta) {
        this.timestampGlobal += delta;
    }
    
    /**
     * Entrena la red mediante plasticidad hebiana
     * Implementa el principio: "neuronas que se activan juntas, se conectan"
     * (pg. 46-47 Eagleman)
     * 
     * @param inputs Array de valores de entrada
     * @param targets Array de valores objetivo (supervisión débil)
     * @param iteraciones Número de veces que se presenta el patrón
     */
    public void entrenar(double[] inputs, double[] targets, int iteraciones) {
        if (targets == null || targets.length != capaMotora.size()) {
            throw new IllegalArgumentException(
                "Los targets deben tener tamaño " + capaMotora.size() + 
                ", recibido: " + (targets == null ? "null" : targets.length)
            );
        }
        
        for (int i = 0; i < iteraciones; i++) {
            // Procesar inputs
            double[] outputs = procesar(inputs);
            
            // Calcular error
            double[] errores = new double[targets.length];
            for (int j = 0; j < targets.length; j++) {
                errores[j] = targets[j] - outputs[j];
            }
            
            // Aplicar plasticidad hebiana a todas las conexiones
            entrenador.aplicarPlasticidadHebianaGlobal(conexiones, timestampGlobal);
            
            // Modular aprendizaje basándose en error (supervisión débil)
            entrenador.modularAprendizajePorError(capaMotora, capasInterneuronas, errores);
            
            // Avanzar tiempo para ventana temporal
            avanzarTiempo(10L);
        }
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
        return gestorEngramas.getEngramas();
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
    
    /**
     * Activa o desactiva el modo predictivo
     * En modo predictivo, cada capa predice la activación de la siguiente
     * y solo se propagan los errores de predicción (pg. 64 Eagleman)
     * 
     * @param activar true para activar, false para desactivar
     */
    public void activarModoPredictivo(boolean activar) {
        gestorPredicciones.activar(activar, capasInterneuronas, capaMotora);
    }
    
    /**
     * Verifica si el modo predictivo está activo
     */
    public boolean esModoPredictivo() {
        return gestorPredicciones.estaActivo();
    }
    
    /**
     * Obtiene los errores de predicción de la última propagación
     */
    public double[] getErroresPrediccion() {
        return gestorPredicciones.getErroresPrediccion();
    }
    
    /**
     * Obtiene el número de capas que tienen predicciones
     */
    public int getNumeroCapasConPredicciones() {
        return gestorPredicciones.getNumeroCapasConPredicciones();
    }
    

    
    /**
     * Activa o desactiva la competición por recursos
     * Implementa el principio de "supervivencia del más útil"
     * (pg. 18-19, 229-230 Eagleman)
     * 
     * @param activar true para activar, false para desactivar
     */
    public void activarCompeticionRecursos(boolean activar) {
        gestorCompeticion.activar(activar, timestampGlobal);
    }
    
    /**
     * Verifica si la competición por recursos está activa
     */
    public boolean esCompeticionActiva() {
        return gestorCompeticion.estaActiva();
    }
    
    /**
     * Ejecuta competición por recursos entre neuronas y conexiones
     * Elementos usados ganan recursos, elementos no usados los pierden
     * (pg. 229-230 Eagleman: "cualquier red que se active con frecuencia gana más territorio")
     */
    public void competirPorRecursos() {
        gestorCompeticion.competir(capaSensorial, capasInterneuronas, capaMotora, 
                                   conexiones, timestampGlobal);
    }
    
    /**
     * Poda elementos sin recursos suficientes
     * Elimina neuronas y conexiones que han perdido la competición
     * (pg. 18-19 Eagleman: "solo sobreviven aquellas que sean importantes")
     * 
     * @return Número de elementos podados
     */
    public int podarElementos() {
        return gestorCompeticion.podarElementos(conexiones);
    }
    
    /**
     * Activa o desactiva la detección automática de engramas
     * Cuando está activa, la red detecta patrones de activación repetidos
     * y forma engramas automáticamente
     * (pg. 85 Campillo: "el aprendizaje culmina en un entramado de engramas")
     * 
     * @param activar true para activar, false para desactivar
     */
    public void activarDeteccionEngramas(boolean activar) {
        gestorEngramas.activarDeteccion(activar);
    }
    
    /**
     * Verifica si la detección automática de engramas está activa
     */
    public boolean esDeteccionEngramasActiva() {
        return gestorEngramas.esDeteccionActiva();
    }
    
    /**
     * Forma un engrama manualmente con las neuronas especificadas
     * Un engrama es un conjunto de neuronas que codifican un recuerdo
     * (pg. 84-85 Campillo: "conjunto de neuronas que funcionan como bits para guardar un recuerdo")
     * 
     * @param id Identificador único del engrama
     * @param participantes Lista de neuronas que forman el engrama
     */
    public void formarEngrama(String id, List<Neurona> participantes) {
        gestorEngramas.formarEngrama(id, participantes, timestampGlobal);
    }
    
    /**
     * Activa un engrama existente
     * La activación facilita las neuronas del engrama, reduciendo su umbral
     * (pg. 93 Campillo: "el cerebro revisa otros recuerdos para completar aquellos que no lo estén")
     * 
     * @param id Identificador del engrama a activar
     */
    public void activarEngrama(String id) {
        gestorEngramas.activarEngrama(id, timestampGlobal);
    }
    
    /**
     * Inicia el proceso de consolidación ("sueño")
     * Durante la consolidación, la red no puede procesar inputs
     * (pg. 87 Campillo: "Durante el sueño el cerebro revisa los engramas")
     */
    public void iniciarConsolidacion() {
        if (estado == EstadoRed.CONSOLIDANDO) {
            throw new IllegalStateException("La red ya está en proceso de consolidación");
        }
        this.estado = EstadoRed.CONSOLIDANDO;
    }
    
    /**
     * Ejecuta el ciclo de consolidación
     * Revisa engramas formados, fortalece los importantes, debilita los irrelevantes
     * (pg. 87 Campillo: "consolida los más importantes")
     * (pg. 89 Campillo: "Olvido gradual pero parcial")
     */
    public void consolidar() {
        if (estado != EstadoRed.CONSOLIDANDO) {
            throw new IllegalStateException("Debe iniciar consolidación primero");
        }
        
        Map<String, Engrama> engramas = gestorEngramas.getEngramasInterno();
        List<String> engramasAEliminar = new ArrayList<>();
        
        for (Map.Entry<String, Engrama> entry : engramas.entrySet()) {
            String id = entry.getKey();
            Engrama engrama = entry.getValue();
            
            // Calcular tiempo desde última activación
            long tiempoSinUso = timestampGlobal - engrama.getTimestampUltimaActivacion();
            
            // Si el engrama ha sido usado recientemente, fortalecerlo
            if (tiempoSinUso < 30L) {
                // Engrama relevante: aumentar su relevancia
                double nuevaRelevancia = Math.min(2.0, engrama.getRelevancia() * 1.1);
                engrama.setRelevancia(nuevaRelevancia);
            } else {
                // Engrama no usado: degradar
                double factorDegradacion = 0.80;  // Pierde 20% de relevancia
                double nuevaRelevancia = engrama.getRelevancia() * factorDegradacion;
                engrama.setRelevancia(nuevaRelevancia);
                
                // Si la relevancia es muy baja, marcar para eliminación
                if (nuevaRelevancia < 0.15) {
                    engramasAEliminar.add(id);
                }
            }
        }
        
        // Eliminar engramas irrelevantes
        for (String id : engramasAEliminar) {
            gestorEngramas.eliminarEngrama(id);
        }
        
        // Ajuste fino de pesos sinápticos (consolidación de conocimiento)
        consolidarPesosSinapticos();
    }
    
    /**
     * Finaliza el proceso de consolidación
     * Vuelve la red al estado ACTIVO
     */
    public void finalizarConsolidacion() {
        if (estado != EstadoRed.CONSOLIDANDO) {
            throw new IllegalStateException("La red no está en proceso de consolidación");
        }
        this.estado = EstadoRed.ACTIVO;
    }
    
    /**
     * Consolida pesos sinápticos
     * Ajusta ligeramente los pesos basándose en uso reciente
     */
    private void consolidarPesosSinapticos() {
        long ventanaTemporal = 100L;
        
        for (Conexion conexion : conexiones) {
            long tiempoSinUso = timestampGlobal - conexion.getTimestampUltimaActivacion();
            
            if (tiempoSinUso < ventanaTemporal) {
                // Conexión usada recientemente: reforzar ligeramente
                double ajuste = 0.02;  // 2% de refuerzo
                double nuevoPeso = conexion.getPeso() * (1.0 + ajuste);
                conexion.setPeso(Math.min(1.0, nuevoPeso));  // Clamp a 1.0
            } else {
                // Conexión no usada: debilitar ligeramente
                double ajuste = 0.01;  // 1% de debilitamiento
                double nuevoPeso = conexion.getPeso() * (1.0 - ajuste);
                conexion.setPeso(Math.max(0.0, nuevoPeso));  // Clamp a 0.0
            }
        }
    }
    
    /**
     * Obtiene estadísticas generales de la red
     * Útil para monitorización y debugging
     * 
     * @return Map con estadísticas clave
     */
    public Map<String, Object> getEstadisticas() {
        Map<String, Object> stats = new HashMap<>();
        
        // Información básica
        stats.put("totalNeuronas", getTotalNeuronas());
        stats.put("totalConexiones", getTotalConexiones());
        stats.put("totalEngramas", gestorEngramas.getEngramas().size());
        stats.put("timestampGlobal", timestampGlobal);
        stats.put("estado", estado.toString());
        stats.put("topologia", topologia);
        stats.put("densidadConexiones", densidadConexiones);
        
        // Información de sistemas activos
        stats.put("modoPredictivo", gestorPredicciones.estaActivo());
        stats.put("competicionActiva", gestorCompeticion.estaActiva());
        stats.put("deteccionEngramas", gestorEngramas.esDeteccionActiva());
        
        // Estadísticas de activación
        long neuronasActivas = contarNeuronasActivas();
        stats.put("neuronasActivas", neuronasActivas);
        stats.put("porcentajeActivacion", (double)neuronasActivas / getTotalNeuronas() * 100.0);
        
        return stats;
    }
    
    /**
     * Visualiza el estado de activación de todas las neuronas
     * Útil para debugging y comprensión del procesamiento
     * 
     * @return String con representación visual de activaciones
     */
    public String visualizarActivaciones() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Activaciones de Red Neural ===\n\n");
        
        // Capa sensorial
        sb.append("Capa Sensorial (").append(capaSensorial.size()).append(" neuronas):\n");
        for (int i = 0; i < capaSensorial.size(); i++) {
            Neurona n = capaSensorial.get(i);
            sb.append(String.format("  [%d] %s (pot: %.2f)\n", 
                i, n.estaActiva() ? "●" : "○", n.getPotencial()));
        }
        sb.append("\n");
        
        // Capas intermedias
        for (int capa = 0; capa < capasInterneuronas.size(); capa++) {
            List<Neurona> capaInter = capasInterneuronas.get(capa);
            sb.append("Capa Inter ").append(capa + 1).append(" (")
              .append(capaInter.size()).append(" neuronas):\n");
            for (int i = 0; i < capaInter.size(); i++) {
                Neurona n = capaInter.get(i);
                sb.append(String.format("  [%d] %s (pot: %.2f)\n", 
                    i, n.estaActiva() ? "●" : "○", n.getPotencial()));
            }
            sb.append("\n");
        }
        
        // Capa motora
        sb.append("Capa Motora (").append(capaMotora.size()).append(" neuronas):\n");
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona n = capaMotora.get(i);
            sb.append(String.format("  [%d] %s (pot: %.2f, val: %.2f)\n", 
                i, n.estaActiva() ? "●" : "○", n.getPotencial(), n.getValorAlmacenado()));
        }
        
        return sb.toString();
    }
    
    /**
     * Analiza los engramas formados en la red
     * Proporciona información detallada sobre memoria
     * 
     * @return String con análisis de engramas
     */
    public String analizarEngramas() {
        StringBuilder sb = new StringBuilder();
        Map<String, Engrama> engramas = gestorEngramas.getEngramas();
        
        sb.append("=== Análisis de Engramas ===\n\n");
        sb.append("Total de engramas: ").append(engramas.size()).append("\n\n");
        
        if (engramas.isEmpty()) {
            sb.append("No hay engramas formados.\n");
            return sb.toString();
        }
        
        for (Map.Entry<String, Engrama> entry : engramas.entrySet()) {
            String id = entry.getKey();
            Engrama engrama = entry.getValue();
            
            sb.append("Engrama: ").append(id).append("\n");
            sb.append("  Neuronas: ").append(engrama.getNeuronas().size()).append("\n");
            sb.append("  Relevancia: ").append(String.format("%.2f", engrama.getRelevancia())).append("\n");
            sb.append("  Fuerza: ").append(String.format("%.2f", engrama.getFuerza())).append("\n");
            sb.append("  Activaciones: ").append(engrama.getContadorActivaciones()).append("\n");
            sb.append("  Última activación: ").append(engrama.getTimestampUltimaActivacion()).append("\n");
            sb.append("  Tiempo sin uso: ").append(timestampGlobal - engrama.getTimestampUltimaActivacion()).append("\n");
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    /**
     * Genera reporte de asignación de recursos
     * Muestra cómo se distribuyen los recursos entre neuronas y conexiones
     * 
     * @return String con reporte de recursos
     */
    public String reporteRecursos() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Reporte de Recursos ===\n\n");
        
        // Recursos de neuronas
        double sumaRecursosNeuronas = 0.0;
        int contadorNeuronas = 0;
        
        for (Neurona n : capaSensorial) {
            sumaRecursosNeuronas += n.getRecursosAsignados();
            contadorNeuronas++;
        }
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona n : capa) {
                sumaRecursosNeuronas += n.getRecursosAsignados();
                contadorNeuronas++;
            }
        }
        for (Neurona n : capaMotora) {
            sumaRecursosNeuronas += n.getRecursosAsignados();
            contadorNeuronas++;
        }
        
        double promedioNeuronas = contadorNeuronas > 0 ? 
            sumaRecursosNeuronas / contadorNeuronas : 0.0;
        
        sb.append("Neuronas:\n");
        sb.append("  Total: ").append(contadorNeuronas).append("\n");
        sb.append("  Recursos promedio: ").append(String.format("%.3f", promedioNeuronas)).append("\n");
        sb.append("\n");
        
        // Recursos de conexiones
        double sumaRecursosConexiones = 0.0;
        for (Conexion c : conexiones) {
            sumaRecursosConexiones += c.getRecursosAsignados();
        }
        
        double promedioConexiones = conexiones.size() > 0 ? 
            sumaRecursosConexiones / conexiones.size() : 0.0;
        
        sb.append("Conexiones:\n");
        sb.append("  Total: ").append(conexiones.size()).append("\n");
        sb.append("  Recursos promedio: ").append(String.format("%.3f", promedioConexiones)).append("\n");
        
        return sb.toString();
    }
    
    /**
     * Cuenta el número de neuronas activas en toda la red
     */
    private long contarNeuronasActivas() {
        long contador = 0;
        
        for (Neurona n : capaSensorial) {
            if (n.estaActiva()) contador++;
        }
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona n : capa) {
                if (n.estaActiva()) contador++;
            }
        }
        for (Neurona n : capaMotora) {
            if (n.estaActiva()) contador++;
        }
        
        return contador;
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
        sb.append("  Engramas: ").append(gestorEngramas.getEngramas().size()).append("\n");
        sb.append("}");
        return sb.toString();
    }
}
