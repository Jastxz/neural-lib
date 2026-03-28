package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.ResultadoBenchmark;
import es.jastxz.nn.spiking.ConfiguracionRed;

/**
 * Unidad de la población del algoritmo genético.
 *
 * <p>Un individuo es inmutable y asocia un {@link Cromosoma} con su
 * {@link ConfiguracionRed} correspondiente, el fitness obtenido tras
 * evaluación y el {@link ResultadoBenchmark} que lo produjo.</p>
 *
 * <p>Los individuos se ordenan de mayor a menor fitness para facilitar
 * la selección por torneo y la preservación de élites.</p>
 *
 * @param cromosoma        codificación genética del individuo
 * @param fitness          valor de aptitud (-1 si no evaluado)
 * @param configuracionRed configuración de red asociada al cromosoma
 * @param resultadoBenchmark resultado de benchmark (null si no evaluado)
 */
public record Individuo(
    Cromosoma cromosoma,
    double fitness,
    ConfiguracionRed configuracionRed,
    ResultadoBenchmark resultadoBenchmark
) implements Comparable<Individuo> {

    /**
     * Crea un individuo sin evaluar (fitness = -1, sin resultado de benchmark).
     *
     * @param cromosoma codificación genética
     * @param config    configuración de red construida a partir del cromosoma
     * @return individuo pendiente de evaluación
     */
    public static Individuo sinEvaluar(Cromosoma cromosoma, ConfiguracionRed config) {
        return new Individuo(cromosoma, -1.0, config, null);
    }

    /**
     * Crea una copia de este individuo con el fitness y resultado de benchmark asignados.
     *
     * @param fitness   valor de aptitud calculado
     * @param resultado resultado de benchmark obtenido
     * @return nuevo individuo evaluado
     */
    public Individuo conEvaluacion(double fitness, ResultadoBenchmark resultado) {
        return new Individuo(this.cromosoma, fitness, this.configuracionRed, resultado);
    }

    /**
     * Compara individuos por fitness en orden descendente (mayor fitness primero).
     *
     * @param otro individuo a comparar
     * @return valor negativo si este individuo tiene mayor fitness,
     *         positivo si tiene menor, cero si son iguales
     */
    @Override
    public int compareTo(Individuo otro) {
        return Double.compare(otro.fitness, this.fitness);
    }
}
