package es.jastxz.nn.benchmark;

/**
 * Niveles de complejidad de problemas para benchmarks de la SNN,
 * clasificados por el tamaño del espacio de estados.
 *
 * <p>Cada nivel define un problema representativo con sus dimensiones
 * de entrada/salida y el tamaño aproximado de su espacio de estados.</p>
 *
 * @since 1.1
 */
public enum NivelComplejidad {

    /** Problema trivial: Puertas Lógicas (AND, OR, XOR). Espacio de estados ~4. */
    TRIVIAL("Puertas Lógicas", 4, 2, 1),

    /** Problema de baja complejidad: 3 en Raya. Espacio de estados ~5.478. */
    BAJO("3 en Raya", 5_478, 10, 9),

    /** Problema de complejidad media: Gatos. Espacio de estados ~100.000. */
    MEDIO("Gatos", 100_000, 25, 25),

    /**
     * Problema de alta complejidad: Damas. Espacio de estados ~5×10^20.
     * Se usa {@code Long.MAX_VALUE} como aproximación ya que el valor real
     * (500.000.000.000.000.000.000) excede el rango de {@code long}.
     */
    ALTO("Damas", Long.MAX_VALUE, 32, 32);

    private final String nombreProblema;
    private final long espacioEstados;
    private final int dimensionEntrada;
    private final int dimensionSalida;

    NivelComplejidad(String nombreProblema, long espacioEstados,
                     int dimensionEntrada, int dimensionSalida) {
        this.nombreProblema = nombreProblema;
        this.espacioEstados = espacioEstados;
        this.dimensionEntrada = dimensionEntrada;
        this.dimensionSalida = dimensionSalida;
    }

    public String getNombreProblema() {
        return nombreProblema;
    }

    public long getEspacioEstados() {
        return espacioEstados;
    }

    public int getDimensionEntrada() {
        return dimensionEntrada;
    }

    public int getDimensionSalida() {
        return dimensionSalida;
    }
}
