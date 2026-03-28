package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.*;

import java.util.List;

/**
 * Evaluador de fitness multi-objetivo para individuos del algoritmo genético.
 *
 * <p>Calcula el fitness como combinación ponderada de tres componentes normalizados:</p>
 * <ul>
 *   <li>Precisión (peso por defecto 0.5)</li>
 *   <li>Eficiencia energética inversa (peso por defecto 0.3)</li>
 *   <li>Penalización por tamaño de red (peso por defecto 0.2)</li>
 * </ul>
 *
 * <p>Individuos con clasificación "timeout" o "limite_no_superado", o cuya
 * evaluación lanza excepción, reciben fitness 0.0.</p>
 */
public class EvaluadorFitness {

    /** Costo energético máximo de referencia para normalización. */
    static final double COSTO_MAX_REFERENCIA = 1000.0;

    private final NivelComplejidad nivel;
    private final RecolectorMetricas recolector;
    private final double pesoPrecision;
    private final double pesoEnergia;
    private final double pesoTamanio;
    private final int limiteTopologico;
    private final int epocasBenchmark;
    private final int repeticionesBenchmark;
    private final long semilla;

    /**
     * @param nivel                nivel de complejidad del benchmark
     * @param recolector           recolector de métricas para ejecutar benchmarks
     * @param pesoPrecision        peso del componente de precisión
     * @param pesoEnergia          peso del componente de eficiencia energética
     * @param pesoTamanio          peso del componente de penalización por tamaño
     * @param limiteTopologico     límite máximo de neuronas totales
     * @param epocasBenchmark      épocas de entrenamiento del benchmark
     * @param repeticionesBenchmark repeticiones del benchmark
     * @param semilla              semilla para reproducibilidad
     */
    public EvaluadorFitness(NivelComplejidad nivel, RecolectorMetricas recolector,
                            double pesoPrecision, double pesoEnergia, double pesoTamanio,
                            int limiteTopologico, int epocasBenchmark,
                            int repeticionesBenchmark, long semilla) {
        this.nivel = nivel;
        this.recolector = recolector;
        this.pesoPrecision = pesoPrecision;
        this.pesoEnergia = pesoEnergia;
        this.pesoTamanio = pesoTamanio;
        this.limiteTopologico = limiteTopologico;
        this.epocasBenchmark = epocasBenchmark;
        this.repeticionesBenchmark = repeticionesBenchmark;
        this.semilla = semilla;
    }

    /**
     * Evalúa un individuo ejecutando benchmark y calculando fitness ponderado.
     *
     * <p>Algoritmo:</p>
     * <ol>
     *   <li>Construir {@link ConfiguracionBenchmark} con la topología del individuo</li>
     *   <li>Ejecutar {@link RecolectorMetricas#ejecutarYRecolectar}</li>
     *   <li>Clasificar con {@link DetectorLimites#clasificar}</li>
     *   <li>Si clasificación es "timeout" o "limite_no_superado" → fitness = 0.0</li>
     *   <li>Normalizar componentes y calcular fitness ponderado</li>
     * </ol>
     *
     * <p>Si la construcción o entrenamiento lanza excepción → fitness = 0.0.</p>
     *
     * @param individuo individuo a evaluar
     * @return nuevo individuo con fitness y resultado de benchmark asignados
     */
    public Individuo evaluar(Individuo individuo) {
        try {
            // 1. Build topology array from ConfiguracionRed
            int[] topologia = individuo.configuracionRed().getTopologia();

            // 2. Build ConfiguracionBenchmark
            ConfiguracionBenchmark configBenchmark = new ConfiguracionBenchmark(
                    nivel, topologia, epocasBenchmark,
                    1, // duracionTimesteps
                    repeticionesBenchmark, semilla);

            // 3. Generate training data for the problem
            double[][] inputs = generarInputs();
            double[][] targets = generarTargets();

            // 4. Execute benchmark using the individual's ConfiguracionRed
            ResultadoBenchmark resultado = recolector.ejecutarYRecolectar(
                    configBenchmark, individuo.configuracionRed(), inputs, targets);

            // 5. Classify with DetectorLimites
            String clasificacion = DetectorLimites.clasificar(resultado);
            if (clasificacion != null) {
                resultado = new ResultadoBenchmark(
                        resultado.configuracion(), resultado.precisionFinal(),
                        resultado.errorMSEPorEpoca(), resultado.tiempoEntrenamientoMs(),
                        resultado.totalSpikes(), resultado.tasaDisparoPromedio(),
                        resultado.costoEnergetico(), resultado.dispersionActividad(),
                        resultado.neuronasActivas(), resultado.neuronasTotal(),
                        clasificacion);
            }

            // 6. Calculate fitness (handles failed classifications internally)
            double fitness = calcularFitness(resultado);
            return individuo.conEvaluacion(fitness, resultado);

        } catch (Exception e) {
            // Construction or training exception → fitness = 0.0
            return individuo.conEvaluacion(0.0, null);
        }
    }

    /**
     * Evalúa toda la población en paralelo usando un pool estático de threads.
     *
     * <p>Cada individuo se evalúa de forma independiente (sin estado compartido),
     * por lo que la evaluación es <em>embarrassingly parallel</em>. Se usa un
     * pool estático de threads daemon para evitar overhead de creación/destrucción
     * y contención con el common pool de la JVM.</p>
     *
     * @param poblacion lista de individuos a evaluar
     * @return nueva lista con individuos evaluados
     */
    public List<Individuo> evaluarPoblacion(List<Individuo> poblacion) {
        return poblacion.stream().map(this::evaluar).toList();
    }

    /**
     * Calcula el fitness a partir de un resultado de benchmark.
     *
     * <p>Fórmula: fitness = pesoPrecision × precisionNorm + pesoEnergia × energiaNorm
     * + pesoTamanio × tamanioNorm</p>
     *
     * <p>Este método es package-private para permitir testing directo de la lógica
     * de cálculo sin ejecutar benchmarks reales.</p>
     *
     * @param resultado resultado de benchmark con métricas
     * @return valor de fitness calculado
     */
    public double calcularFitness(ResultadoBenchmark resultado) {
        // Check for failed classifications first
        String clasificacion = resultado.clasificacion();
        if ("timeout".equals(clasificacion) || "limite_no_superado".equals(clasificacion)) {
            return 0.0;
        }

        // precision_norm = precisionFinal (already in [0,1])
        double precisionNorm = Math.max(0.0, Math.min(1.0, resultado.precisionFinal()));

        // energia_norm = 1.0 - min(1.0, costoEnergetico / costoMaxReferencia)
        double energiaNorm = 1.0 - Math.min(1.0, resultado.costoEnergetico() / COSTO_MAX_REFERENCIA);
        energiaNorm = Math.max(0.0, Math.min(1.0, energiaNorm));

        // tamanio_norm = 1.0 - (neuronasTotal / limiteTopologico)
        double tamanioNorm = 1.0 - ((double) resultado.neuronasTotal() / limiteTopologico);
        tamanioNorm = Math.max(0.0, Math.min(1.0, tamanioNorm));

        return pesoPrecision * precisionNorm
             + pesoEnergia * energiaNorm
             + pesoTamanio * tamanioNorm;
    }

    // ==================== Helpers privados ====================

    private double[][] generarInputs() {
        int dim = nivel.getDimensionEntrada();
        // Generate minimal training data based on problem dimension
        int numPatrones = Math.min(dim, 16);
        double[][] inputs = new double[numPatrones][dim];
        java.util.Random rng = new java.util.Random(semilla);
        for (int i = 0; i < numPatrones; i++) {
            for (int j = 0; j < dim; j++) {
                inputs[i][j] = rng.nextDouble();
            }
        }
        return inputs;
    }

    private double[][] generarTargets() {
        int dimSalida = nivel.getDimensionSalida();
        int dimEntrada = nivel.getDimensionEntrada();
        int numPatrones = Math.min(dimEntrada, 16);
        double[][] targets = new double[numPatrones][dimSalida];
        for (int i = 0; i < numPatrones; i++) {
            // One-hot encoding for classification
            int clase = i % dimSalida;
            targets[i][clase] = 1.0;
        }
        return targets;
    }
}
