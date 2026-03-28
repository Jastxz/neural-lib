package es.jastxz.nn.genetico;

/**
 * Unidad mínima del cromosoma que representa un hiperparámetro individual
 * con su tipo, valor actual y rango válido.
 *
 * <p>Interfaz sellada con cuatro implementaciones: {@link GenEntero},
 * {@link GenReal}, {@link GenBooleano} y {@link GenEnum}.</p>
 *
 * @param <T> tipo del valor del gen
 */
public sealed interface Gen<T> {

    /** Nombre identificador del gen. */
    String nombre();

    /** Valor actual del gen. */
    T valor();

    /** Valor mínimo permitido. */
    T minimo();

    /** Valor máximo permitido. */
    T maximo();

    /**
     * Crea una copia del gen con un nuevo valor, aplicando clamping al rango válido.
     *
     * @param nuevoValor valor deseado
     * @return nuevo gen con el valor ajustado al rango
     */
    Gen<T> conValor(T nuevoValor);

    /**
     * Gen de tipo entero con rango {@code [minimo, maximo]}.
     * El constructor compacto valida que el valor esté dentro del rango.
     * Se aceptan tanto primitivos como boxed gracias a autoboxing.
     */
    record GenEntero(String nombre, Integer valor, Integer minimo, Integer maximo)
            implements Gen<Integer> {

        public GenEntero {
            if (minimo > maximo)
                throw new IllegalArgumentException(
                        nombre + ": minimo (" + minimo + ") > maximo (" + maximo + ")");
            if (valor < minimo || valor > maximo)
                throw new IllegalArgumentException(
                        nombre + ": " + valor + " fuera de [" + minimo + ", " + maximo + "]");
        }

        @Override
        public Gen<Integer> conValor(Integer v) {
            int clamped = Math.clamp(v, (int) minimo, (int) maximo);
            return new GenEntero(nombre, clamped, minimo, maximo);
        }
    }

    /**
     * Gen de tipo real (double) con rango {@code [minimo, maximo]}.
     * El constructor compacto valida que el valor esté dentro del rango.
     */
    record GenReal(String nombre, Double valor, Double minimo, Double maximo)
            implements Gen<Double> {

        public GenReal {
            if (minimo > maximo)
                throw new IllegalArgumentException(
                        nombre + ": minimo (" + minimo + ") > maximo (" + maximo + ")");
            if (valor < minimo || valor > maximo)
                throw new IllegalArgumentException(
                        nombre + ": " + valor + " fuera de [" + minimo + ", " + maximo + "]");
        }

        @Override
        public Gen<Double> conValor(Double v) {
            double clamped = Math.clamp(v, (double) minimo, (double) maximo);
            return new GenReal(nombre, clamped, minimo, maximo);
        }
    }

    /**
     * Gen booleano. No tiene rango configurable: mínimo es {@code false}, máximo es {@code true}.
     */
    record GenBooleano(String nombre, Boolean valor)
            implements Gen<Boolean> {

        @Override public Boolean minimo() { return false; }
        @Override public Boolean maximo() { return true; }

        @Override
        public Gen<Boolean> conValor(Boolean v) {
            return new GenBooleano(nombre, v);
        }
    }

    /**
     * Gen enumerado. El rango se define por las constantes del enum.
     *
     * @param <E> tipo del enum
     */
    record GenEnum<E extends Enum<E>>(String nombre, E valor, Class<E> tipoEnum)
            implements Gen<E> {

        @Override
        public E minimo() {
            return tipoEnum.getEnumConstants()[0];
        }

        @Override
        public E maximo() {
            E[] vals = tipoEnum.getEnumConstants();
            return vals[vals.length - 1];
        }

        @Override
        public Gen<E> conValor(E v) {
            return new GenEnum<>(nombre, v, tipoEnum);
        }
    }
}
