package es.jastxz.nn.genetico;

/**
 * Builder con valores por defecto para {@link ConfiguracionAG}.
 *
 * <p>Proporciona métodos fluidos para configurar cada parámetro del AG.
 * Los valores por defecto son razonables para la mayoría de problemas.</p>
 *
 * <p>Ejemplo de uso:</p>
 * <pre>{@code
 * ConfiguracionAG config = new ConfiguracionAGBuilder()
 *     .tamañoPoblacion(50)
 *     .maxGeneraciones(100)
 *     .pesosFitness(0.6, 0.2, 0.2)
 *     .build();
 * }</pre>
 */
public class ConfiguracionAGBuilder {

    private int tamañoPoblacion = 30;
    private int maxGeneraciones = 50;
    private int generacionesEstancamiento = 10;
    private double probabilidadCruce = 0.8;
    private int puntosCorte = 2;
    private double probabilidadMutacion = 0.1;
    private int tamañoTorneo = 3;
    private int numElites = 2;
    private double porcentajeNoElite = 0.15;
    private double pesoPrecision = 0.5;
    private double pesoEnergia = 0.3;
    private double pesoTamanio = 0.2;
    private int limiteTopologico = 512;
    private int epocasBenchmark = 10;
    private int repeticionesBenchmark = 1;
    private long semilla = System.nanoTime();

    public ConfiguracionAGBuilder tamañoPoblacion(int v) { this.tamañoPoblacion = v; return this; }
    public ConfiguracionAGBuilder maxGeneraciones(int v) { this.maxGeneraciones = v; return this; }
    public ConfiguracionAGBuilder generacionesEstancamiento(int v) { this.generacionesEstancamiento = v; return this; }
    public ConfiguracionAGBuilder probabilidadCruce(double v) { this.probabilidadCruce = v; return this; }
    public ConfiguracionAGBuilder puntosCorte(int v) { this.puntosCorte = v; return this; }
    public ConfiguracionAGBuilder probabilidadMutacion(double v) { this.probabilidadMutacion = v; return this; }
    public ConfiguracionAGBuilder tamañoTorneo(int v) { this.tamañoTorneo = v; return this; }
    public ConfiguracionAGBuilder numElites(int v) { this.numElites = v; return this; }
    public ConfiguracionAGBuilder porcentajeNoElite(double v) { this.porcentajeNoElite = v; return this; }

    /**
     * Establece los tres pesos de fitness simultáneamente.
     *
     * @param precision peso del componente de precisión
     * @param energia   peso del componente de eficiencia energética
     * @param tamanio   peso del componente de penalización por tamaño
     * @return este builder
     */
    public ConfiguracionAGBuilder pesosFitness(double precision, double energia, double tamanio) {
        this.pesoPrecision = precision;
        this.pesoEnergia = energia;
        this.pesoTamanio = tamanio;
        return this;
    }

    public ConfiguracionAGBuilder limiteTopologico(int v) { this.limiteTopologico = v; return this; }
    public ConfiguracionAGBuilder epocasBenchmark(int v) { this.epocasBenchmark = v; return this; }
    public ConfiguracionAGBuilder repeticionesBenchmark(int v) { this.repeticionesBenchmark = v; return this; }
    public ConfiguracionAGBuilder semilla(long v) { this.semilla = v; return this; }

    /**
     * Construye y retorna una {@link ConfiguracionAG} validada.
     *
     * @return configuración del AG con los parámetros establecidos
     * @throws IllegalArgumentException si alguna validación falla
     */
    public ConfiguracionAG build() {
        return new ConfiguracionAG(
                tamañoPoblacion, maxGeneraciones, generacionesEstancamiento,
                probabilidadCruce, puntosCorte, probabilidadMutacion,
                tamañoTorneo, numElites, porcentajeNoElite,
                pesoPrecision, pesoEnergia, pesoTamanio,
                limiteTopologico, epocasBenchmark, repeticionesBenchmark,
                semilla);
    }
}
