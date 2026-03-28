package es.jastxz.nn.spiking;

/**
 * Ejemplos de uso de la Red Neuronal de Spikes (SNN).
 *
 * <p>Esta clase contiene ejemplos prácticos que demuestran las principales
 * funcionalidades de la red neuronal spiking.</p>
 *
 * @since 1.0
 */
public class SpikingNetworkExample {

    /**
     * Ejemplo 1: Red simple para clasificación XOR.
     * Crea una red 2-4-1, entrena con patrones XOR y evalúa.
     */
    public static void ejemploXOR() {
        System.out.println("=== Ejemplo 1: Clasificación XOR ===");

        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 4, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones completas entre capas
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 4; i++)
            red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, 0), 0.5, 1);

        // Datos XOR
        double[][] inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        double[][] targets = {{0.0}, {1.0}, {1.0}, {0.0}};

        // Entrenar
        for (int epoca = 0; epoca < 5; epoca++) {
            double error = red.entrenar(inputs, targets, 100);
            System.out.println("Época " + epoca + " - Error: " + String.format("%.4f", error));
        }

        // Evaluar
        for (double[] input : inputs) {
            double[] salida = red.procesar(input, 100);
            System.out.printf("Input: [%.0f, %.0f] -> Salida: %.3f%n",
                input[0], input[1], salida[0]);
        }
    }

    /**
     * Ejemplo 2: Red con STDP para aprendizaje temporal.
     * Demuestra cómo STDP ajusta pesos basándose en timing de spikes.
     */
    public static void ejemploSTDP() {
        System.out.println("\n=== Ejemplo 2: Aprendizaje STDP ===");

        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(3, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .homeostasis(true, 10.0, 0.01)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);

        // Crear conexiones
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, j), 0.5, 1);

        System.out.println("Peso promedio antes: " + String.format("%.4f", red.getPesoPromedioGlobal()));

        // Entrenar con STDP
        red.setModoEntrenamiento(true);
        red.procesar(new double[]{0.8, 0.2, 0.5}, 200);
        red.setModoEntrenamiento(false);

        System.out.println("Peso promedio después: " + String.format("%.4f", red.getPesoPromedioGlobal()));
    }

    /**
     * Ejemplo 3: Guardar y cargar modelo.
     * Demuestra persistencia binaria de la red.
     */
    public static void ejemploPersistencia() throws Exception {
        System.out.println("\n=== Ejemplo 3: Persistencia ===");

        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 1);
        for (int i = 0; i < 3; i++)
            red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, 0), 0.5, 1);

        // Guardar
        String archivo = "/tmp/ejemplo_snn.bin";
        red.guardar(archivo);
        System.out.println("Red guardada en: " + archivo);

        // Cargar
        RedNeuralSpiking cargada = RedNeuralSpiking.cargar(archivo);
        System.out.println("Red cargada. Topología: " + java.util.Arrays.toString(cargada.getTopologia()));

        double[] salida = cargada.procesar(new double[]{0.7, 0.3}, 100);
        System.out.println("Salida tras cargar: " + salida[0]);
    }

    /**
     * Ejemplo 4: Configuración desde JSON.
     * Demuestra serialización/deserialización JSON de configuración.
     */
    public static void ejemploJSON() {
        System.out.println("\n=== Ejemplo 4: Configuración JSON ===");

        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 4, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .homeostasis(true, 10.0, 0.01)
            .inhibicionLateral(true, 2, 0.5)
            .build();

        RedNeuralSpiking red = new RedNeuralSpiking(config);
        String json = red.toJSON();
        System.out.println("JSON generado:");
        System.out.println(json);

        // Reconstruir desde JSON
        RedNeuralSpiking reconstruida = RedNeuralSpiking.desdeJSON(json);
        System.out.println("\nRed reconstruida. Topología: " +
            java.util.Arrays.toString(reconstruida.getTopologia()));
        System.out.println("Integridad: " + reconstruida.validarIntegridad());
    }

    public static void main(String[] args) throws Exception {
        ejemploXOR();
        ejemploSTDP();
        ejemploPersistencia();
        ejemploJSON();
    }
}
