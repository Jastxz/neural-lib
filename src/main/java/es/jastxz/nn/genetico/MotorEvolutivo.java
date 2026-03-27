package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.NivelComplejidad;
import es.jastxz.nn.benchmark.RecolectorMetricas;

import java.util.*;

/**
 * Orquestador del ciclo evolutivo del algoritmo genético.
 *
 * <p>Ejecuta el ciclo completo: generación inicial → evaluación → selección →
 * cruce → mutación → nueva población, repitiendo hasta cumplir un criterio
 * de parada (máximo de generaciones o estancamiento).</p>
 *
 * <p>Preserva los mejores N individuos (élites) de cada generación sin
 * modificación y rastrea el mejor individuo global a lo largo de toda
 * la evolución.</p>
 */
public class MotorEvolutivo {

    private final ConfiguracionAG config;
    private final NivelComplejidad nivel;
    private final EvaluadorFitness evaluador;
    private final SelectorTorneo selector;
    private final OperadorCruce cruce;
    private final OperadorMutacion mutacion;
    private final FabricaIndividuos fabrica;

    /**
     * Constructor público que instancia todos los componentes internos.
     *
     * @param config configuración del algoritmo genético
     * @param nivel  nivel de complejidad del benchmark
     */
    public MotorEvolutivo(ConfiguracionAG config, NivelComplejidad nivel) {
        this.config = config;
        this.nivel = nivel;

        Random random = new Random(config.semilla());

        this.fabrica = new FabricaIndividuos(
                nivel.getDimensionEntrada(), nivel.getDimensionSalida(),
                config.limiteTopologico(), random);

        this.evaluador = new EvaluadorFitness(
                nivel, new RecolectorMetricas(),
                config.pesoPrecision(), config.pesoEnergia(), config.pesoTamanio(),
                config.limiteTopologico(), config.epocasBenchmark(),
                config.repeticionesBenchmark(), config.semilla());

        this.selector = new SelectorTorneo(
                config.tamañoTorneo(), config.numElites(),
                config.porcentajeNoElite(), random);

        this.cruce = new OperadorCruce(
                config.probabilidadCruce(), config.puntosCorte(),
                config.limiteTopologico(), fabrica, random);

        this.mutacion = new OperadorMutacion(
                config.probabilidadMutacion(), config.limiteTopologico(),
                fabrica, random);
    }

    /**
     * Constructor que acepta todos los componentes directamente.
     *
     * @param config    configuración del AG
     * @param fabrica   fábrica de individuos
     * @param evaluador evaluador de fitness
     * @param selector  selector de padres por torneo
     * @param cruce     operador de cruce
     * @param mutacion  operador de mutación
     */
    public MotorEvolutivo(ConfiguracionAG config, FabricaIndividuos fabrica,
                   EvaluadorFitness evaluador, SelectorTorneo selector,
                   OperadorCruce cruce, OperadorMutacion mutacion) {
        this.config = config;
        this.nivel = null; // no necesario cuando se inyectan componentes
        this.evaluador = evaluador;
        this.selector = selector;
        this.cruce = cruce;
        this.mutacion = mutacion;
        this.fabrica = fabrica;
    }

    /**
     * Ejecuta el ciclo evolutivo completo.
     *
     * @return informe de evolución con resultados y estadísticas
     */
    public InformeEvolucion evolucionar() {
        List<Individuo> poblacion = new ArrayList<>();
        for (int i = 0; i < config.tamañoPoblacion(); i++) {
            poblacion.add(fabrica.generarAleatorio());
        }
        return evolucionar(poblacion);
    }

    /**
     * Ejecuta el ciclo evolutivo con una población inicial proporcionada.
     *
     * <p>Permite inyectar individuos con características específicas
     * (por ejemplo, topología fija o hiperparámetros heredados de una
     * fase anterior de optimización).</p>
     *
     * @param poblacionInicial población inicial a evaluar y evolucionar
     * @return informe de evolución con resultados y estadísticas
     */
    public InformeEvolucion evolucionar(List<Individuo> poblacionInicial) {
        // 1. Evaluar población inicial
        List<Individuo> poblacion = new ArrayList<>(
                evaluador.evaluarPoblacion(poblacionInicial));

        // 3. Rastrear mejor global
        Individuo mejorGlobal = Collections.min(poblacion); // min con Comparable = mayor fitness
        int generacionMejor = 0;
        int generacionesSinMejora = 0;
        List<EstadisticaGeneracion> historial = new ArrayList<>();
        String motivoParada = "max_generaciones";

        // 4. Ciclo evolutivo
        int totalGeneraciones = 0;
        for (int gen = 0; gen < config.maxGeneraciones(); gen++) {
            totalGeneraciones = gen + 1;

            // a. Ordenar población por fitness descendente
            poblacion.sort(Comparator.naturalOrder());

            // b. Preservar élites (primeros numElites)
            int numElites = Math.min(config.numElites(), poblacion.size());
            List<Individuo> elites = new ArrayList<>(poblacion.subList(0, numElites));

            // c. Seleccionar parejas de padres
            List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

            // d-e. Cruzar y mutar para generar descendientes
            List<Individuo> descendientes = new ArrayList<>();
            for (SelectorTorneo.Pareja pareja : parejas) {
                Individuo hijo = cruce.cruzar(pareja.padre1(), pareja.padre2());
                hijo = mutacion.mutar(hijo);
                descendientes.add(hijo);
            }

            // f. Formar nueva población = élites + descendientes
            List<Individuo> nuevaPoblacion = new ArrayList<>(elites);
            nuevaPoblacion.addAll(descendientes);

            // Ajustar tamaño si es necesario (rellenar con más descendientes o truncar)
            while (nuevaPoblacion.size() < config.tamañoPoblacion() && !descendientes.isEmpty()) {
                // Si faltan individuos, generar más
                Individuo extra = fabrica.generarAleatorio();
                nuevaPoblacion.add(extra);
            }
            if (nuevaPoblacion.size() > config.tamañoPoblacion()) {
                nuevaPoblacion = new ArrayList<>(nuevaPoblacion.subList(0, config.tamañoPoblacion()));
            }

            // g. Evaluar nueva población
            nuevaPoblacion = new ArrayList<>(evaluador.evaluarPoblacion(nuevaPoblacion));

            // h. Verificar mejora del mejor global
            Individuo mejorGeneracion = Collections.min(nuevaPoblacion);
            if (mejorGeneracion.fitness() > mejorGlobal.fitness() * 1.01) {
                mejorGlobal = mejorGeneracion;
                generacionMejor = gen;
                generacionesSinMejora = 0;
            } else {
                // Actualizar mejor global si hay mejora absoluta (aunque no >1%)
                if (mejorGeneracion.fitness() > mejorGlobal.fitness()) {
                    mejorGlobal = mejorGeneracion;
                    generacionMejor = gen;
                }
                generacionesSinMejora++;
            }

            // i. Registrar estadísticas de generación
            double mejorFitness = mejorGeneracion.fitness();
            double peorFitness = Collections.max(nuevaPoblacion).fitness(); // max con Comparable = menor fitness
            double sumaFitness = 0;
            for (Individuo ind : nuevaPoblacion) {
                sumaFitness += ind.fitness();
            }
            double promedioFitness = sumaFitness / nuevaPoblacion.size();

            historial.add(new EstadisticaGeneracion(
                    gen, mejorFitness, promedioFitness, peorFitness,
                    mejorGeneracion.configuracionRed()));

            // j. Verificar estancamiento
            if (generacionesSinMejora >= config.generacionesEstancamiento()) {
                motivoParada = "estancamiento";
                break;
            }

            // k. Actualizar población
            poblacion = nuevaPoblacion;
        }

        // 6. Retornar informe
        return new InformeEvolucion(
                mejorGlobal, generacionMejor, totalGeneraciones,
                motivoParada, historial, config);
    }
}
