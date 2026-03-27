package es.jastxz.nn.benchmark;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ConfiguracionRedBuilder;
import es.jastxz.nn.spiking.RedNeuralSpiking;
import es.jastxz.nn.spiking.TipoInicializacion;

import java.util.Map;
import java.util.Random;

/**
 * Recolector de métricas que ejecuta un benchmark individual sobre la SNN
 * y devuelve un {@link ResultadoBenchmark} con todas las métricas capturadas.
 *
 * <p>Ofrece dos modos de ejecución:</p>
 * <ul>
 *   <li>{@link #ejecutarYRecolectar(ConfiguracionBenchmark, double[][], double[][])}:
 *       usa parámetros de referencia internos (para benchmarks standalone).</li>
 *   <li>{@link #ejecutarYRecolectar(ConfiguracionBenchmark, ConfiguracionRed, double[][], double[][])}:
 *       usa la {@link ConfiguracionRed} proporcionada (para el AG y optimizador).</li>
 * </ul>
 *
 * @since 1.1
 */
public class RecolectorMetricas {

    /**
     * Ejecuta un benchmark usando parámetros de referencia internos.
     *
     * <p>Crea la red con hiperparámetros fijos optimizados para benchmarks
     * generales. Útil para comparaciones estandarizadas donde se quiere
     * aislar el efecto de la topología.</p>
     *
     * @param config  configuración del benchmark (topología, épocas, semilla, etc.)
     * @param inputs  datos de entrada para entrenamiento y evaluación
     * @param targets datos objetivo para entrenamiento y evaluación
     * @return resultado con todas las métricas capturadas
     */
    public ResultadoBenchmark ejecutarYRecolectar(
            ConfiguracionBenchmark config,
            double[][] inputs,
            double[][] targets) {

        ConfiguracionRed configRed = crearConfigReferencia(config.topologia());
        return ejecutarConConfig(config, configRed, inputs, targets);
    }

    /**
     * Ejecuta un benchmark usando la {@link ConfiguracionRed} proporcionada.
     *
     * <p>Permite que el AG y el optimizador evalúen individuos con sus propios
     * hiperparámetros evolucionados (LIF, STDP, codificación, regulación),
     * en lugar de usar los parámetros de referencia internos.</p>
     *
     * @param config    configuración del benchmark (topología, épocas, semilla, etc.)
     * @param configRed configuración de la red con hiperparámetros específicos
     * @param inputs    datos de entrada para entrenamiento y evaluación
     * @param targets   datos objetivo para entrenamiento y evaluación
     * @return resultado con todas las métricas capturadas
     */
    public ResultadoBenchmark ejecutarYRecolectar(
            ConfiguracionBenchmark config,
            ConfiguracionRed configRed,
            double[][] inputs,
            double[][] targets) {

        return ejecutarConConfig(config, configRed, inputs, targets);
    }

    // ==================== Implementación compartida ====================

    /**
     * Implementación compartida que crea la red, la entrena y recolecta métricas.
     */
    private ResultadoBenchmark ejecutarConConfig(
            ConfiguracionBenchmark config,
            ConfiguracionRed configRed,
            double[][] inputs,
            double[][] targets) {

        return ejecutarInterno(config, configRed, config.epocas(),
                config.duracionTimesteps(), config.semilla(), inputs, targets);
    }

    /**
     * Ejecuta un benchmark usando la {@link ConfiguracionRed} proporcionada,
     * sin necesidad de un {@link ConfiguracionBenchmark}.
     *
     * <p>Útil cuando las dimensiones del problema no coinciden con ningún
     * {@link NivelComplejidad} predefinido (por ejemplo, problemas con 1 o 3
     * entradas que no encajan en los niveles del enum).</p>
     *
     * @param configRed configuración de la red con hiperparámetros específicos
     * @param epocas    número de épocas de entrenamiento
     * @param duracion  timesteps por patrón
     * @param semilla   semilla para reproducibilidad
     * @param inputs    datos de entrada
     * @param targets   datos objetivo
     * @return resultado con todas las métricas capturadas
     */
    public ResultadoBenchmark ejecutarYRecolectar(
            ConfiguracionRed configRed,
            int epocas, int duracion, long semilla,
            double[][] inputs, double[][] targets) {

        return ejecutarInterno(null, configRed, epocas, duracion, semilla, inputs, targets);
    }

    /**
     * Implementación interna compartida por todos los overloads.
     */
    private ResultadoBenchmark ejecutarInterno(
            ConfiguracionBenchmark configBench,
            ConfiguracionRed configRed,
            int epocas, int duracion, long semilla,
            double[][] inputs, double[][] targets) {

        long inicio = System.currentTimeMillis();

        // 1. Crear red
        RedNeuralSpiking red = new RedNeuralSpiking(configRed);

        // 2. Crear conexiones fully-connected entre capas adyacentes
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

        // 3. Entrenar
        double[] msePorEpoca = new double[epocas];
        for (int e = 0; e < epocas; e++) {
            double mse = red.entrenar(inputs, targets, duracion);
            msePorEpoca[e] = Double.isNaN(mse) || Double.isInfinite(mse) ? 0.0 : mse;
        }

        long tiempoMs = System.currentTimeMillis() - inicio;

        // 4. Evaluar precisión
        double[][] salidas = red.procesarLote(inputs, duracion, true);
        double precision = calcularPrecision(salidas, targets);

        // 5. Extraer métricas de la red
        Map<String, Object> metricas = red.obtenerMetricas();
        long totalSpikes = (Long) metricas.get("totalSpikes");
        double tasaPromedio = (Double) metricas.get("tasaPromedioGlobal");
        double costoEnergetico = (Double) metricas.get("costoEnergetico");
        double dispersion = (Double) metricas.get("dispersionActividad");
        int neuronasActivas = (Integer) metricas.get("neuronasActivas");
        int neuronasTotal = configRed.getNumeroTotalNeuronas();

        // Si no hay ConfiguracionBenchmark, crear una sintética para el record
        if (configBench == null) {
            configBench = crearConfigBenchmarkSintetica(topologia, epocas, duracion, semilla);
        }

        return new ResultadoBenchmark(
                configBench, precision, msePorEpoca, tiempoMs,
                totalSpikes, tasaPromedio, costoEnergetico, dispersion,
                neuronasActivas, neuronasTotal, null);
    }

    // ==================== Config de referencia ====================

    /**
     * Crea un {@link ConfiguracionBenchmark} sintético para dimensiones
     * que no coinciden con ningún {@link NivelComplejidad} predefinido.
     *
     * <p>Busca un nivel compatible; si no existe, usa {@code TRIVIAL} con
     * la topología ajustada a sus dimensiones (2 entradas, 1 salida).</p>
     */
    private ConfiguracionBenchmark crearConfigBenchmarkSintetica(
            int[] topologia, int epocas, int duracion, long semilla) {

        int dimEntrada = topologia[0];
        int dimSalida = topologia[topologia.length - 1];

        // Buscar nivel compatible
        for (NivelComplejidad nivel : NivelComplejidad.values()) {
            if (nivel.getDimensionEntrada() == dimEntrada
                    && nivel.getDimensionSalida() == dimSalida) {
                return new ConfiguracionBenchmark(nivel, topologia, epocas, duracion, 1, semilla);
            }
        }

        // Fallback: usar TRIVIAL con topología adaptada (solo para el record)
        int[] topologiaTrivial = new int[topologia.length];
        topologiaTrivial[0] = NivelComplejidad.TRIVIAL.getDimensionEntrada();
        System.arraycopy(topologia, 1, topologiaTrivial, 1, topologia.length - 2);
        topologiaTrivial[topologiaTrivial.length - 1] = NivelComplejidad.TRIVIAL.getDimensionSalida();
        return new ConfiguracionBenchmark(
                NivelComplejidad.TRIVIAL, topologiaTrivial, epocas, duracion, 1, semilla);
    }

    /**
     * Crea una {@link ConfiguracionRed} con hiperparámetros de referencia fijos.
     *
     * <p>Estos valores están optimizados para benchmarks generales y proporcionan
     * un baseline estable para comparaciones estandarizadas.</p>
     *
     * @param topologia topología de la red [entrada, ocultas..., salida]
     * @return configuración de referencia con parámetros fijos
     */
    private ConfiguracionRed crearConfigReferencia(int[] topologia) {
        return new ConfiguracionRedBuilder()
                .topologia(topologia)
                .parametrosLIF(-55.0, -70.0, 20.0, 2)
                .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
                .inicializacionPesos(TipoInicializacion.UNIFORME, 0.0, 1.0)
                .build();
    }

    // ==================== Helpers estáticos ====================

    /**
     * Calcula la precisión (porcentaje de aciertos) comparando salidas con targets.
     *
     * <p>Para cada patrón, compara el argmax de la salida con el argmax del target.
     * Si ambos coinciden, se cuenta como acierto.</p>
     *
     * @param salidas salidas de la red (una fila por patrón)
     * @param targets targets esperados (una fila por patrón)
     * @return precisión en [0.0, 1.0]
     */
    public static double calcularPrecision(double[][] salidas, double[][] targets) {
        if (salidas == null || targets == null || salidas.length == 0) {
            return 0.0;
        }
        int aciertos = 0;
        for (int i = 0; i < salidas.length; i++) {
            if (argmax(salidas[i]) == argmax(targets[i])) {
                aciertos++;
            }
        }
        return (double) aciertos / salidas.length;
    }

    /**
     * Devuelve el índice del valor máximo en un array.
     *
     * @param array array de valores
     * @return índice del valor máximo
     */
    static int argmax(double[] array) {
        if (array == null || array.length == 0) return 0;
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    /**
     * Calcula el número total de neuronas en una topología.
     *
     * @param topologia array con tamaños de cada capa
     * @return suma de todas las neuronas
     */
    public static int calcularNeuronasTotal(int[] topologia) {
        int total = 0;
        for (int n : topologia) {
            total += n;
        }
        return total;
    }
}
