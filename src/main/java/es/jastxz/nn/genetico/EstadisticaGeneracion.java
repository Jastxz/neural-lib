package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;

/**
 * Estadísticas de una generación del algoritmo genético.
 *
 * @param numero             número de la generación
 * @param mejorFitness       mejor fitness de la generación
 * @param fitnessPromedio    fitness promedio de la generación
 * @param peorFitness        peor fitness de la generación
 * @param mejorConfiguracion configuración de red del mejor individuo
 */
public record EstadisticaGeneracion(
    int numero,
    double mejorFitness,
    double fitnessPromedio,
    double peorFitness,
    ConfiguracionRed mejorConfiguracion
) {}
