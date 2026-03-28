package es.jastxz.nn.spiking;

import es.jastxz.nn.benchmark.*;
import es.jastxz.nn.genetico.*;

import java.util.*;

/**
 * Fachada de alto nivel que optimiza automáticamente todos los hiperparámetros
 * de una SNN mediante un Algoritmo Genético en dos fases.
 *
 * <h3>Estrategia de optimización en dos fases:</h3>
 * <ol>
 *   <li><b>Fase 1 — Hiperparámetros:</b> Topología fija al máximo permitido.
 *       El AG evoluciona solo LIF, STDP, codificación y regulación.
 *       Esto evita que el AG se distraiga optimizando el tamaño de la red
 *       cuando lo que necesita ajustar son los parámetros de la dinámica neuronal.</li>
 *   <li><b>Fase 2 — Topología:</b> Con los mejores hiperparámetros encontrados,
 *       el AG evoluciona solo la topología (capas y neuronas) para encontrar
 *       la red más compacta que mantenga la precisión.</li>
 * </ol>
 *
 * <h3>Uso mínimo:</h3>
 * <pre>{@code
 * double[][] inputs  = {{0,0},{0,1},{1,0},{1,1}};
 * double[][] targets = {{0},{1},{1},{0}};
 *
 * ResultadoOptimizacion resultado = OptimizadorSNN.optimizar(inputs, targets);
 *
 * RedNeuralSpiking red = resultado.red();
 * double[] salida = red.procesar(new double[]{1, 0}, 100);
 * }</pre>
 *
 * <h3>Uso con configuración personalizada:</h3>
 * <pre>{@code
 * ResultadoOptimizacion resultado = new OptimizadorSNN.Builder()
 *     .maxGeneraciones(200)
 *     .tamañoPoblacion(80)
 *     .duracionTimesteps(150)
 *     .semilla(42L)
 *     .optimizar(inputs, targets);
 * }</pre>
 *
 * @since 1.2
 * @see ResultadoOptimizacion
 */
public class OptimizadorSNN {

    private OptimizadorSNN() {
        // Utility class
    }

    /**
     * Optimiza una SNN con configuración por defecto.
     *
     * @param inputs  datos de entrada (cada fila es un patrón)
     * @param targets datos objetivo (cada fila es la salida esperada)
     * @return resultado con la mejor red entrenada y métricas de la evolución
     * @throws IllegalArgumentException si los datos son inválidos
     */
    public static ResultadoOptimizacion optimizar(double[][] inputs, double[][] targets) {
        return new Builder().optimizar(inputs, targets);
    }

    /**
     * Builder para personalizar los parámetros del proceso de optimización.
     */
    public static class Builder {

        // --- Parámetros del AG ---
        private int tamañoPoblacion = 50;
        private int maxGeneraciones = 100;
        private int generacionesEstancamiento = 15;
        private double probabilidadCruce = 0.8;
        private double probabilidadMutacion = 0.15;
        private int limiteTopologico = 128;

        // --- Parámetros de evaluación ---
        private int duracionTimesteps = 100;
        private int epocasEntrenamiento = 10;
        private int numIntentos = 3;

        // --- Pesos de fitness ---
        private double pesoPrecision = 0.8;
        private double pesoEnergia = 0.1;
        private double pesoTamanio = 0.1;

        // --- Reproducibilidad ---
        private Long semilla = null;

        public Builder tamañoPoblacion(int v) { this.tamañoPoblacion = v; return this; }
        public Builder maxGeneraciones(int v) { this.maxGeneraciones = v; return this; }
        public Builder generacionesEstancamiento(int v) { this.generacionesEstancamiento = v; return this; }
        public Builder probabilidadCruce(double v) { this.probabilidadCruce = v; return this; }
        public Builder probabilidadMutacion(double v) { this.probabilidadMutacion = v; return this; }
        public Builder limiteTopologico(int v) { this.limiteTopologico = v; return this; }
        public Builder duracionTimesteps(int v) { this.duracionTimesteps = v; return this; }
        public Builder epocasEntrenamiento(int v) { this.epocasEntrenamiento = v; return this; }
        public Builder numIntentos(int v) { this.numIntentos = v; return this; }
        public Builder pesoPrecision(double v) { this.pesoPrecision = v; return this; }
        public Builder pesoEnergia(double v) { this.pesoEnergia = v; return this; }
        public Builder pesoTamanio(double v) { this.pesoTamanio = v; return this; }
        public Builder semilla(long v) { this.semilla = v; return this; }

        /**
         * Ejecuta la optimización en dos fases.
         *
         * <p>Fase 1: topología fija al máximo, evoluciona hiperparámetros.
         * Fase 2: hiperparámetros fijos (los mejores de fase 1), evoluciona topología.</p>
         *
         * @param inputs  datos de entrada
         * @param targets datos objetivo
         * @return resultado con la mejor red y métricas
         * @throws IllegalArgumentException si los datos son inválidos
         */
        public ResultadoOptimizacion optimizar(double[][] inputs, double[][] targets) {
            validarDatos(inputs, targets);

            int dimEntrada = inputs[0].length;
            int dimSalida = targets[0].length;

            ResultadoOptimizacion mejorGlobal = null;

            for (int intento = 0; intento < numIntentos; intento++) {
                long semillaActual = (semilla != null)
                        ? semilla + intento
                        : System.nanoTime();

                // === FASE 1: Optimizar hiperparámetros con topología máxima ===
                int[] topologiaMaxima = calcularTopologiaMaxima(
                        dimEntrada, dimSalida, limiteTopologico);

                ResultadoFase fase1 = ejecutarFase(
                        inputs, targets, dimEntrada, dimSalida, semillaActual,
                        Set.of(BloqueFuncional.TOPOLOGIA), topologiaMaxima,
                        maxGeneraciones);

                // === FASE 2: Optimizar topología con mejores hiperparámetros ===
                Set<BloqueFuncional> bloquesHiperparams = Set.of(
                        BloqueFuncional.LIF, BloqueFuncional.STDP,
                        BloqueFuncional.CODIFICACION, BloqueFuncional.REGULACION,
                        BloqueFuncional.COMPETICION);

                // Usar la mitad de generaciones para fase 2 (topología converge más rápido)
                int gensFase2 = Math.max(5, maxGeneraciones / 2);

                ResultadoFase fase2 = ejecutarFaseConCromosoma(
                        inputs, targets, dimEntrada, dimSalida,
                        semillaActual + 1000,
                        bloquesHiperparams, fase1.mejorCromosoma(),
                        gensFase2);

                // Elegir el mejor resultado entre ambas fases
                ResultadoFase mejorFase = fase2.precision() >= fase1.precision()
                        ? fase2 : fase1;

                RedNeuralSpiking red = reconstruirRed(
                        mejorFase.mejorIndividuo(), inputs, targets,
                        duracionTimesteps, epocasEntrenamiento, semillaActual);

                ResultadoOptimizacion resultado = new ResultadoOptimizacion(
                        red, mejorFase.mejorIndividuo().configuracionRed(),
                        mejorFase.precision(), mejorFase.fitness(),
                        mejorFase.informe());

                if (mejorGlobal == null
                        || resultado.precision() > mejorGlobal.precision()) {
                    mejorGlobal = resultado;
                }

                if (resultado.precision() >= 1.0) break;
            }

            return mejorGlobal;
        }

        // --- Ejecución de fases ---

        /**
         * Ejecuta una fase del AG con bloques congelados y topología fija opcional.
         */
        private ResultadoFase ejecutarFase(
                double[][] inputs, double[][] targets,
                int dimEntrada, int dimSalida, long semillaActual,
                Set<BloqueFuncional> bloquesCongelados,
                int[] topologiaFija, int generaciones) {

            ConfiguracionAG configAG = crearConfigAG(
                    generaciones, semillaActual);

            EvaluadorConDatos evaluador = new EvaluadorConDatos(
                    inputs, targets, dimEntrada, dimSalida,
                    configAG, duracionTimesteps);

            Random random = new Random(semillaActual);
            FabricaIndividuos fabrica = new FabricaIndividuos(
                    dimEntrada, dimSalida, limiteTopologico, random);

            // Generar población con topología fija si se especifica
            MotorEvolutivo motor = crearMotor(
                    configAG, fabrica, evaluador, random,
                    bloquesCongelados);

            // Si hay topología fija, inyectar población inicial
            InformeEvolucion informe;
            if (topologiaFija != null) {
                informe = evolucionarConTopologiaFija(
                        motor, fabrica, evaluador, configAG,
                        topologiaFija, bloquesCongelados, random);
            } else {
                informe = motor.evolucionar();
            }

            return extraerResultadoFase(informe);
        }

        /**
         * Ejecuta una fase del AG inyectando los bloques no congelados
         * de un cromosoma base (resultado de una fase anterior).
         */
        private ResultadoFase ejecutarFaseConCromosoma(
                double[][] inputs, double[][] targets,
                int dimEntrada, int dimSalida, long semillaActual,
                Set<BloqueFuncional> bloquesCongelados,
                Cromosoma cromosomaBase, int generaciones) {

            ConfiguracionAG configAG = crearConfigAG(
                    generaciones, semillaActual);

            EvaluadorConDatos evaluador = new EvaluadorConDatos(
                    inputs, targets, dimEntrada, dimSalida,
                    configAG, duracionTimesteps);

            Random random = new Random(semillaActual);
            FabricaIndividuos fabrica = new FabricaIndividuos(
                    dimEntrada, dimSalida, limiteTopologico, random);

            MotorEvolutivo motor = crearMotor(
                    configAG, fabrica, evaluador, random,
                    bloquesCongelados);

            InformeEvolucion informe = evolucionarConCromosomaBase(
                    motor, fabrica, evaluador, configAG,
                    cromosomaBase, bloquesCongelados, random);

            return extraerResultadoFase(informe);
        }

        // --- Helpers de construcción ---

        private ConfiguracionAG crearConfigAG(int generaciones, long semillaActual) {
            double sumaPesos = pesoPrecision + pesoEnergia + pesoTamanio;
            double pPrec = pesoPrecision / sumaPesos;
            double pEner = pesoEnergia / sumaPesos;
            double pTam  = pesoTamanio / sumaPesos;

            return new ConfiguracionAG(
                    tamañoPoblacion, generaciones, generacionesEstancamiento,
                    probabilidadCruce, 2, probabilidadMutacion,
                    Math.min(3, tamañoPoblacion), 2, 0.15,
                    pPrec, pEner, pTam,
                    limiteTopologico, epocasEntrenamiento, 1,
                    semillaActual);
        }

        private MotorEvolutivo crearMotor(
                ConfiguracionAG configAG, FabricaIndividuos fabrica,
                EvaluadorConDatos evaluador, Random random,
                Set<BloqueFuncional> bloquesCongelados) {

            SelectorTorneo selector = new SelectorTorneo(
                    configAG.tamañoTorneo(), configAG.numElites(),
                    configAG.porcentajeNoElite(), random);
            OperadorCruce cruce = new OperadorCruce(
                    configAG.probabilidadCruce(), configAG.puntosCorte(),
                    limiteTopologico, fabrica, random, bloquesCongelados);
            OperadorMutacion mutacion = new OperadorMutacion(
                    configAG.probabilidadMutacion(), limiteTopologico,
                    fabrica, random, bloquesCongelados);

            return new MotorEvolutivo(
                    configAG, fabrica, evaluador, selector, cruce, mutacion);
        }

        /**
         * Evoluciona con topología fija: genera población inicial donde
         * todos los individuos tienen la topología especificada.
         */
        private InformeEvolucion evolucionarConTopologiaFija(
                MotorEvolutivo motor, FabricaIndividuos fabrica,
                EvaluadorConDatos evaluador, ConfiguracionAG configAG,
                int[] topologiaFija,
                Set<BloqueFuncional> bloquesCongelados, Random random) {
            List<Individuo> poblacion = new ArrayList<>();
            for (int i = 0; i < configAG.tamañoPoblacion(); i++) {
                poblacion.add(fabrica.generarConTopologiaFija(topologiaFija));
            }
            return motor.evolucionar(poblacion);
        }

        /**
         * Evoluciona inyectando los bloques congelados de un cromosoma base
         * en toda la población inicial.
         */
        private InformeEvolucion evolucionarConCromosomaBase(
                MotorEvolutivo motor, FabricaIndividuos fabrica,
                EvaluadorConDatos evaluador, ConfiguracionAG configAG,
                Cromosoma cromosomaBase,
                Set<BloqueFuncional> bloquesCongelados, Random random) {
            List<Individuo> poblacion = new ArrayList<>();
            for (int i = 0; i < configAG.tamañoPoblacion(); i++) {
                Individuo aleatorio = fabrica.generarAleatorio();
                // Reemplazar los bloques congelados con los del cromosoma base
                Cromosoma cromosoma = aleatorio.cromosoma();
                for (BloqueFuncional bloque : bloquesCongelados) {
                    cromosoma = cromosoma.conBloque(bloque,
                            cromosomaBase.genesDeBloque(bloque));
                }
                ConfiguracionRed config = fabrica.construirConfiguracion(cromosoma);
                poblacion.add(Individuo.sinEvaluar(cromosoma, config));
            }
            return motor.evolucionar(poblacion);
        }

        private ResultadoFase extraerResultadoFase(InformeEvolucion informe) {
            Individuo mejor = informe.mejorIndividuo();
            double precision = 0.0;
            if (mejor.resultadoBenchmark() != null) {
                precision = mejor.resultadoBenchmark().precisionFinal();
            }
            return new ResultadoFase(
                    mejor, mejor.cromosoma(), precision,
                    mejor.fitness(), informe);
        }
    }

    // ==================== Resultado intermedio de fase ====================

    record ResultadoFase(
            Individuo mejorIndividuo,
            Cromosoma mejorCromosoma,
            double precision,
            double fitness,
            InformeEvolucion informe) {}

    // ==================== Evaluador interno ====================

    /**
     * Evaluador que entrena y evalúa la SNN usando la ConfiguracionRed
     * real del individuo, delegando a {@link RecolectorMetricas} con el
     * overload que acepta {@link ConfiguracionRed}.
     *
     * <p>Esto es esencial para que el AG pueda optimizar hiperparámetros:
     * cada individuo tiene su propia configuración LIF, STDP, codificación, etc.
     * y la evaluación debe usar exactamente esos parámetros.</p>
     */
    static class EvaluadorConDatos extends EvaluadorFitness {

        private final double[][] inputs;
        private final double[][] targets;
        private final ConfiguracionAG config;
        private final int duracionTimesteps;
        private final RecolectorMetricas recolectorInterno;

        EvaluadorConDatos(double[][] inputs, double[][] targets,
                          int dimEntrada, int dimSalida,
                          ConfiguracionAG config, int duracionTimesteps) {
            super(nivelDesde(dimEntrada, dimSalida),
                  new RecolectorMetricas(),
                  config.pesoPrecision(), config.pesoEnergia(), config.pesoTamanio(),
                  config.limiteTopologico(), config.epocasBenchmark(),
                  1, config.semilla());
            this.inputs = inputs;
            this.targets = targets;
            this.config = config;
            this.duracionTimesteps = duracionTimesteps;
            this.recolectorInterno = new RecolectorMetricas();
        }

        @Override
        public Individuo evaluar(Individuo individuo) {
            try {
                ConfiguracionRed configRed = individuo.configuracionRed();

                ResultadoBenchmark resultado = recolectorInterno.ejecutarYRecolectar(
                        configRed, config.epocasBenchmark(), duracionTimesteps,
                        config.semilla(), inputs, targets);

                double fitness = calcularFitness(resultado);
                return individuo.conEvaluacion(fitness, resultado);
            } catch (Exception e) {
                return individuo.conEvaluacion(0.0, null);
            }
        }
    }

    // ==================== Helpers ====================

    /**
     * Calcula una topología con el máximo de neuronas ocultas permitido.
     * Usa una sola capa oculta con el máximo de neuronas que cabe en el límite.
     */
    static int[] calcularTopologiaMaxima(int dimEntrada, int dimSalida, int limiteTopologico) {
        int neuronasOcultas = Math.max(1, limiteTopologico - dimEntrada - dimSalida);
        return new int[]{dimEntrada, neuronasOcultas, dimSalida};
    }

    static NivelComplejidad nivelDesde(int dimEntrada, int dimSalida) {
        for (NivelComplejidad nivel : NivelComplejidad.values()) {
            if (nivel.getDimensionEntrada() == dimEntrada
                    && nivel.getDimensionSalida() == dimSalida) {
                return nivel;
            }
        }
        return NivelComplejidad.TRIVIAL;
    }

    /**
     * Reconstruye y entrena la red con la mejor configuración encontrada.
     */
    static RedNeuralSpiking reconstruirRed(
            Individuo mejor, double[][] inputs, double[][] targets,
            int duracionTimesteps, int epocas, long semilla) {

        ConfiguracionRed configRed = mejor.configuracionRed();
        RedNeuralSpiking red = new RedNeuralSpiking(configRed);

        int[] topologia = configRed.getTopologia();
        Random rng = new Random(semilla);
        for (int capa = 0; capa < topologia.length - 1; capa++) {
            for (int i = 0; i < topologia[capa]; i++) {
                for (int j = 0; j < topologia[capa + 1]; j++) {
                    double peso = configRed.pesoMin
                            + (configRed.pesoMax - configRed.pesoMin) * rng.nextDouble();
                    red.crearConexion(
                            red.getNeurona(capa, i),
                            red.getNeurona(capa + 1, j),
                            peso, 1);
                }
            }
        }

        for (int e = 0; e < epocas; e++) {
            red.entrenar(inputs, targets, duracionTimesteps);
        }

        return red;
    }

    static void validarDatos(double[][] inputs, double[][] targets) {
        if (inputs == null || targets == null) {
            throw new IllegalArgumentException("inputs y targets no pueden ser null");
        }
        if (inputs.length == 0) {
            throw new IllegalArgumentException("inputs no puede estar vacío");
        }
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException(
                    "inputs y targets deben tener el mismo número de patrones: "
                            + inputs.length + " vs " + targets.length);
        }
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] == null || targets[i] == null) {
                throw new IllegalArgumentException(
                        "Patrón " + i + " contiene null");
            }
        }
    }
}
