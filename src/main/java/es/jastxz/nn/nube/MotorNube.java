package es.jastxz.nn.nube;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import es.jastxz.math.Matrix;
import es.jastxz.nn.NeuralNetwork;

/**
 * Orquestador principal del Método de la Nube Aleatoria.
 *
 * <p>Genera una nube de redes neuronales con pesos aleatorios, evalúa cada una
 * contra un umbral de acierto, reduce progresivamente sus neuronas para encontrar
 * la estructura mínima viable, y refina la mejor red encontrada mediante
 * backpropagation clásico.</p>
 *
 * @see ConfiguracionNube
 * @see PoliticaEliminacion
 */
public class MotorNube {

    private final ConfiguracionNube config;
    private final double[][] entradas;
    private final double[][] objetivos;
    private final PoliticaEliminacion politica;
    private final Random random;

    /**
     * Crea un motor con la política de eliminación secuencial por defecto.
     *
     * @param config    configuración del método
     * @param entradas  datos de entrada (una fila por muestra)
     * @param objetivos valores objetivo (una fila por muestra)
     * @throws IllegalArgumentException si los datos son inválidos o incoherentes con la topología
     */
    public MotorNube(ConfiguracionNube config, double[][] entradas, double[][] objetivos) {
        this(config, entradas, objetivos, new PoliticaEliminacionSecuencial());
    }

    /**
     * Crea un motor con una política de eliminación personalizada.
     *
     * @param config    configuración del método
     * @param entradas  datos de entrada (una fila por muestra)
     * @param objetivos valores objetivo (una fila por muestra)
     * @param politica  política de eliminación a aplicar durante la reducción
     * @throws IllegalArgumentException si los datos son inválidos o incoherentes con la topología
     */
    public MotorNube(ConfiguracionNube config, double[][] entradas, double[][] objetivos,
                     PoliticaEliminacion politica) {
        if (entradas == null || entradas.length == 0) {
            throw new IllegalArgumentException("Los datos de entrada no pueden ser nulos ni vacíos");
        }
        if (objetivos == null || objetivos.length == 0) {
            throw new IllegalArgumentException("Los datos objetivo no pueden ser nulos ni vacíos");
        }
        if (entradas.length != objetivos.length) {
            throw new IllegalArgumentException(
                    "entradas y objetivos deben tener el mismo número de muestras (entradas: "
                            + entradas.length + ", objetivos: " + objetivos.length + ")");
        }

        int[] topologia = config.topologiaInicial();
        if (entradas[0].length != topologia[0]) {
            throw new IllegalArgumentException(
                    "La dimensión de entrada (" + entradas[0].length
                            + ") no coincide con la capa de entrada de la topología (" + topologia[0] + ")");
        }
        if (objetivos[0].length != topologia[topologia.length - 1]) {
            throw new IllegalArgumentException(
                    "La dimensión de objetivo (" + objetivos[0].length
                            + ") no coincide con la capa de salida de la topología ("
                            + topologia[topologia.length - 1] + ")");
        }

        this.config = config;
        this.entradas = entradas;
        this.objetivos = objetivos;
        this.politica = politica;
        this.random = new Random(config.semilla());
    }

    /**
     * Genera la nube de redes neuronales con pesos aleatorios reproducibles.
     *
     * <p>Cada red se crea con la topología inicial de la configuración y pesos
     * generados a partir del {@link Random} central (derivando sub-semillas),
     * garantizando reproducibilidad para la misma semilla.</p>
     *
     * @return lista de redes neuronales generadas
     */
    List<NeuralNetwork> generarNube() {
        int n = config.tamañoNube();
        int[] topologia = config.topologiaInicial();
        List<NeuralNetwork> nube = new ArrayList<>(n);

        for (int r = 0; r < n; r++) {
            long subSemilla = random.nextLong();
            Random subRandom = new Random(subSemilla);

            List<Matrix> weights = new ArrayList<>(topologia.length - 1);
            List<Matrix> biases = new ArrayList<>(topologia.length - 1);

            for (int i = 0; i < topologia.length - 1; i++) {
                int filas = topologia[i + 1];
                int cols = topologia[i];

                double[][] wData = new double[filas][cols];
                for (int f = 0; f < filas; f++) {
                    for (int c = 0; c < cols; c++) {
                        wData[f][c] = subRandom.nextDouble() * 2.0 - 1.0;
                    }
                }
                weights.add(new Matrix(wData));

                double[][] bData = new double[filas][1];
                for (int f = 0; f < filas; f++) {
                    bData[f][0] = subRandom.nextDouble() * 2.0 - 1.0;
                }
                biases.add(new Matrix(bData));
            }

            nube.add(new NeuralNetwork(topologia, weights, biases));
        }

        return nube;
    }

    /**
     * Evalúa la precisión de una red sobre todos los datos de entrenamiento.
     *
     * <p>Para cada muestra, ejecuta feedforward y compara el argmax de la salida
     * con el argmax del objetivo. La precisión es el ratio de predicciones correctas
     * sobre el total de muestras.</p>
     *
     * @param red la red neuronal a evaluar
     * @return precisión en el rango [0.0, 1.0]
     */
    double evaluar(NeuralNetwork red) {
        int correctas = 0;
        for (int i = 0; i < entradas.length; i++) {
            double[] salida = red.feedForward(entradas[i]);
            if (argmax(salida) == argmax(objetivos[i])) {
                correctas++;
            }
        }
        return (double) correctas / entradas.length;
    }

    /**
     * Retorna el índice del valor máximo en el array.
     *
     * @param arr array de valores
     * @return índice del valor máximo
     */
    private static int argmax(double[] arr) {
        int idx = 0;
        double max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }

    /**
     * Reconstruye una red neuronal con una topología reducida, preservando los pesos
     * y biases de las neuronas no eliminadas (submatriz superior-izquierda).
     *
     * <p>Para cada transición de capa {@code i}:
     * <ul>
     *   <li>La matriz de pesos se recorta a {@code nuevaTopologia[i+1]} filas
     *       y {@code nuevaTopologia[i]} columnas (submatriz superior-izquierda).</li>
     *   <li>La matriz de biases se recorta a {@code nuevaTopologia[i+1]} filas.</li>
     * </ul>
     *
     * @param original       la red original cuyos pesos se preservan
     * @param nuevaTopologia la topología reducida deseada
     * @return nueva red con la topología reducida y pesos preservados
     */
    NeuralNetwork reconstruirRed(NeuralNetwork original, int[] nuevaTopologia) {
        // Filtrar capas ocultas con 0 neuronas para obtener una topología válida
        // Mantener siempre la capa de entrada (primera) y salida (última)
        List<Integer> topoFiltrada = new ArrayList<>();
        topoFiltrada.add(nuevaTopologia[0]); // capa de entrada
        for (int i = 1; i < nuevaTopologia.length - 1; i++) {
            if (nuevaTopologia[i] > 0) {
                topoFiltrada.add(nuevaTopologia[i]);
            }
        }
        topoFiltrada.add(nuevaTopologia[nuevaTopologia.length - 1]); // capa de salida

        // Si solo quedan entrada y salida (sin ocultas), no es una red válida
        if (topoFiltrada.size() < 3) {
            return null;
        }

        int[] topoLimpia = topoFiltrada.stream().mapToInt(Integer::intValue).toArray();

        // Reconstruir pesos mapeando desde la topología original
        List<Matrix> pesosOriginales = original.getWeights();
        List<Matrix> biasesOriginales = original.getBiases();

        List<Matrix> nuevosPesos = new ArrayList<>(topoLimpia.length - 1);
        List<Matrix> nuevosBiases = new ArrayList<>(topoLimpia.length - 1);

        // Mapear índices de la topología filtrada a la topología con ceros
        // topoLimpia[j] corresponde a nuevaTopologia[indicesOriginales[j]]
        List<Integer> indicesOriginales = new ArrayList<>();
        indicesOriginales.add(0); // entrada
        for (int i = 1; i < nuevaTopologia.length - 1; i++) {
            if (nuevaTopologia[i] > 0) {
                indicesOriginales.add(i);
            }
        }
        indicesOriginales.add(nuevaTopologia.length - 1); // salida

        for (int j = 0; j < topoLimpia.length - 1; j++) {
            int idxOrigen = indicesOriginales.get(j);
            int idxDestino = indicesOriginales.get(j + 1);

            int filasNuevas = topoLimpia[j + 1];
            int colsNuevas = topoLimpia[j];

            // Si las capas son adyacentes en la topología original, copiar submatriz
            if (idxDestino == idxOrigen + 1) {
                double[][] wOriginal = pesosOriginales.get(idxOrigen).getData();
                double[][] wNuevo = new double[filasNuevas][colsNuevas];
                for (int f = 0; f < filasNuevas; f++) {
                    System.arraycopy(wOriginal[f], 0, wNuevo[f], 0, colsNuevas);
                }
                nuevosPesos.add(new Matrix(wNuevo));

                double[][] bOriginal = biasesOriginales.get(idxOrigen).getData();
                double[][] bNuevo = new double[filasNuevas][1];
                for (int f = 0; f < filasNuevas; f++) {
                    bNuevo[f][0] = bOriginal[f][0];
                }
                nuevosBiases.add(new Matrix(bNuevo));
            } else {
                // Capas no adyacentes (se eliminaron capas intermedias):
                // Multiplicar las matrices de pesos intermedias para colapsar la conexión
                Matrix producto = pesosOriginales.get(idxOrigen);
                for (int k = idxOrigen + 1; k < idxDestino; k++) {
                    producto = Matrix.multiply(pesosOriginales.get(k), producto);
                }
                // Recortar al tamaño necesario
                double[][] pData = producto.getData();
                double[][] wNuevo = new double[filasNuevas][colsNuevas];
                for (int f = 0; f < filasNuevas; f++) {
                    System.arraycopy(pData[f], 0, wNuevo[f], 0, colsNuevas);
                }
                nuevosPesos.add(new Matrix(wNuevo));

                // Usar biases de la capa destino
                double[][] bOriginal = biasesOriginales.get(idxDestino - 1).getData();
                double[][] bNuevo = new double[filasNuevas][1];
                for (int f = 0; f < filasNuevas; f++) {
                    bNuevo[f][0] = bOriginal[f][0];
                }
                nuevosBiases.add(new Matrix(bNuevo));
            }
        }

        return new NeuralNetwork(topoLimpia, nuevosPesos, nuevosBiases);
    }
    /**
     * Refina una red neuronal mediante backpropagation clásico.
     *
     * <p>Entrena la red durante el número de épocas y con la tasa de aprendizaje
     * especificados en la configuración, iterando sobre todos los datos de
     * entrenamiento en cada época.</p>
     *
     * @param red la red a refinar (se modifica in-place)
     */
    private void refinar(NeuralNetwork red) {
        red.setLearningRate(config.tasaAprendizaje());
        for (int epoca = 0; epoca < config.epocasRefinamiento(); epoca++) {
            for (int i = 0; i < entradas.length; i++) {
                red.train(entradas[i], objetivos[i]);
            }
        }
    }

    /**
     * Ejecuta el ciclo completo del Método de la Nube Aleatoria.
     *
     * <p>Orquesta: generación de la nube → reducción completa → refinamiento
     * de la mejor red → construcción del informe de resultados.</p>
     *
     * @return informe con los resultados del proceso
     */
    public InformeNube ejecutar() {
        long inicio = System.nanoTime();

        List<NeuralNetwork> nube = generarNube();
        ResultadoReduccion resultado = ejecutarReduccionCompleta(nube);

        long tiempoMs = (System.nanoTime() - inicio) / 1_000_000;

        if (resultado.mejorRed() != null) {
            refinar(resultado.mejorRed());
            double precisionFinal = evaluar(resultado.mejorRed());
            tiempoMs = (System.nanoTime() - inicio) / 1_000_000;
            return new InformeNube(
                    resultado.mejorRed(),
                    precisionFinal,
                    resultado.mejorRed().getTopology(),
                    nube.size(),
                    resultado.totalReducciones(),
                    tiempoMs,
                    true
            );
        } else {
            return new InformeNube(
                    null,
                    0.0,
                    null,
                    nube.size(),
                    resultado.totalReducciones(),
                    tiempoMs,
                    false
            );
        }
    }


    /**
     * Resultado del proceso de reducción sobre toda la nube.
     *
     * @param mejorRed         la red con mayor precisión que superó el umbral, o null
     * @param mejorPrecision   la precisión de la mejor red (0.0 si ninguna superó el umbral)
     * @param totalReducciones número total de reducciones aplicadas en todo el proceso
     */
    private record ResultadoReduccion(NeuralNetwork mejorRed, double mejorPrecision, int totalReducciones) {}

    /**
     * Ejecuta el ciclo completo de reducción sobre todas las redes de la nube.
     *
     * <p>Para cada red candidata, aplica iterativamente la política de eliminación:
     * <ol>
     *   <li>Evalúa la red actual.</li>
     *   <li>Si la precisión supera el umbral y es mayor que la mejor registrada,
     *       actualiza la mejor configuración.</li>
     *   <li>Aplica la política de eliminación para obtener la siguiente topología reducida.</li>
     *   <li>Si la política retorna {@code null} (todas las capas ocultas a 0), detiene
     *       la reducción de esta red y pasa a la siguiente.</li>
     *   <li>Reconstruye la red con la topología reducida preservando pesos.</li>
     *   <li>Repite desde el paso 1.</li>
     * </ol>
     *
     * @param nube lista de redes candidatas generadas
     * @return resultado con la mejor red encontrada, su precisión y el total de reducciones
     */
    ResultadoReduccion ejecutarReduccionCompleta(List<NeuralNetwork> nube) {
        NeuralNetwork mejorRed = null;
        double mejorPrecision = 0.0;
        int totalReducciones = 0;

        for (NeuralNetwork red : nube) {
            NeuralNetwork redActual = red;
            int[] topologiaActual = red.getTopology();

            while (true) {
                double precision = evaluar(redActual);

                if (precision > config.umbralAcierto() && precision > mejorPrecision) {
                    mejorRed = redActual;
                    mejorPrecision = precision;
                }

                int[] nuevaTopologia = politica.siguienteReduccion(topologiaActual, config.neuronasEliminar());
                if (nuevaTopologia == null) {
                    break;
                }

                totalReducciones++;
                NeuralNetwork redReducida = reconstruirRed(redActual, nuevaTopologia);
                if (redReducida == null) {
                    break; // No quedan capas ocultas válidas
                }
                redActual = redReducida;
                topologiaActual = redReducida.getTopology();
            }
        }

        return new ResultadoReduccion(mejorRed, mejorPrecision, totalReducciones);
    }

}
