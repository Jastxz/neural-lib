package es.jastxz.nn.nube;

/**
 * Builder con valores por defecto para {@link ConfiguracionNube}.
 *
 * <p>Proporciona métodos fluidos para configurar cada parámetro del Método de la Nube Aleatoria.
 * Los valores por defecto son razonables para la mayoría de problemas.</p>
 *
 * <p>Ejemplo de uso:</p>
 * <pre>{@code
 * ConfiguracionNube config = new ConfiguracionNubeBuilder()
 *     .tamañoNube(20)
 *     .topologiaInicial(2, 8, 4, 1)
 *     .umbralAcierto(0.7)
 *     .build();
 * }</pre>
 */
public class ConfiguracionNubeBuilder {

    private int tamañoNube = 10;
    private int[] topologiaInicial = {2, 4, 1};
    private double umbralAcierto = 0.5;
    private int neuronasEliminar = 1;
    private int epocasRefinamiento = 1000;
    private double tasaAprendizaje = 0.1;
    private long semilla = System.nanoTime();

    public ConfiguracionNubeBuilder tamañoNube(int v) { this.tamañoNube = v; return this; }
    public ConfiguracionNubeBuilder topologiaInicial(int... v) { this.topologiaInicial = v; return this; }
    public ConfiguracionNubeBuilder umbralAcierto(double v) { this.umbralAcierto = v; return this; }
    public ConfiguracionNubeBuilder neuronasEliminar(int v) { this.neuronasEliminar = v; return this; }
    public ConfiguracionNubeBuilder epocasRefinamiento(int v) { this.epocasRefinamiento = v; return this; }
    public ConfiguracionNubeBuilder tasaAprendizaje(double v) { this.tasaAprendizaje = v; return this; }
    public ConfiguracionNubeBuilder semilla(long v) { this.semilla = v; return this; }

    /**
     * Construye y retorna una {@link ConfiguracionNube} validada.
     *
     * @return configuración de la nube con los parámetros establecidos
     * @throws IllegalArgumentException si alguna validación falla
     */
    public ConfiguracionNube build() {
        return new ConfiguracionNube(
                tamañoNube, topologiaInicial, umbralAcierto,
                neuronasEliminar, epocasRefinamiento, tasaAprendizaje,
                semilla);
    }
}
