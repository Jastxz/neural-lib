package es.jastxz.nn.genetico;

/**
 * Configuración inmutable del Algoritmo Genético.
 *
 * <p>Agrupa todos los parámetros del AG: tamaño de población, criterios de parada,
 * probabilidades de operadores genéticos, parámetros de selección, pesos de fitness
 * y límite topológico.</p>
 *
 * <p>Validaciones del constructor compacto:</p>
 * <ul>
 *   <li>Los pesos de fitness deben sumar 1.0 (tolerancia 1e-6)</li>
 *   <li>El tamaño de torneo no puede exceder el tamaño de población</li>
 *   <li>Los puntos de corte deben estar en [1, 4]</li>
 *   <li>El número de élites debe ser menor que el tamaño de población</li>
 *   <li>El porcentaje de no-élite debe estar en [0.0, 1.0]</li>
 *   <li>El límite topológico debe ser positivo</li>
 * </ul>
 *
 * @param tamañoPoblacion           tamaño de la población (default 30)
 * @param maxGeneraciones           número máximo de generaciones (default 50)
 * @param generacionesEstancamiento generaciones sin mejora para parar (default 10)
 * @param probabilidadCruce         probabilidad de aplicar cruce (default 0.8)
 * @param puntosCorte               puntos de corte del cruce (default 2)
 * @param probabilidadMutacion      probabilidad de mutación por gen (default 0.1)
 * @param tamañoTorneo              individuos por torneo (default 3)
 * @param numElites                 élites preservadas por generación (default 2)
 * @param porcentajeNoElite         fracción mínima de padres no-élite (default 0.15)
 * @param pesoPrecision             peso del componente de precisión (default 0.5)
 * @param pesoEnergia               peso del componente de eficiencia energética (default 0.3)
 * @param pesoTamanio               peso del componente de penalización por tamaño (default 0.2)
 * @param limiteTopologico          máximo de neuronas totales (default 512)
 * @param epocasBenchmark           épocas de entrenamiento del benchmark (default 10)
 * @param repeticionesBenchmark     repeticiones del benchmark (default 1)
 * @param semilla                   semilla para reproducibilidad (default System.nanoTime())
 */
public record ConfiguracionAG(
        int tamañoPoblacion,
        int maxGeneraciones,
        int generacionesEstancamiento,
        double probabilidadCruce,
        int puntosCorte,
        double probabilidadMutacion,
        int tamañoTorneo,
        int numElites,
        double porcentajeNoElite,
        double pesoPrecision,
        double pesoEnergia,
        double pesoTamanio,
        int limiteTopologico,
        int epocasBenchmark,
        int repeticionesBenchmark,
        long semilla
) {

    /** Tolerancia para la validación de suma de pesos. */
    private static final double TOLERANCIA_PESOS = 1e-6;

    /**
     * Constructor compacto con validaciones.
     */
    public ConfiguracionAG {
        double sumaPesos = pesoPrecision + pesoEnergia + pesoTamanio;
        if (Math.abs(sumaPesos - 1.0) > TOLERANCIA_PESOS) {
            throw new IllegalArgumentException(
                    "Los pesos de fitness deben sumar 1.0 (suma actual: " + sumaPesos + ")");
        }
        if (tamañoTorneo > tamañoPoblacion) {
            throw new IllegalArgumentException(
                    "tamañoTorneo (" + tamañoTorneo + ") no puede ser mayor que tamañoPoblacion (" + tamañoPoblacion + ")");
        }
        if (puntosCorte < 1 || puntosCorte > 4) {
            throw new IllegalArgumentException(
                    "puntosCorte debe estar en [1, 4] (valor: " + puntosCorte + ")");
        }
        if (numElites >= tamañoPoblacion) {
            throw new IllegalArgumentException(
                    "numElites (" + numElites + ") debe ser menor que tamañoPoblacion (" + tamañoPoblacion + ")");
        }
        if (porcentajeNoElite < 0.0 || porcentajeNoElite > 1.0) {
            throw new IllegalArgumentException(
                    "porcentajeNoElite debe estar en [0.0, 1.0] (valor: " + porcentajeNoElite + ")");
        }
        if (limiteTopologico <= 0) {
            throw new IllegalArgumentException(
                    "limiteTopologico debe ser positivo (valor: " + limiteTopologico + ")");
        }
    }
}
