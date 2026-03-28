package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ConfiguracionRedBuilder;
import es.jastxz.nn.spiking.ModoCodificacion;
import net.jqwik.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link SelectorTorneo}.
 *
 * <p>Genera poblaciones de individuos evaluados con fitness conocidos
 * y verifica propiedades universales de la selección por torneo.</p>
 */
class SelectorTorneoPropertyTest {

    // ==================== Property 8: Ganador del torneo es el mejor de los k seleccionados ====================

    // Feature: genetic-algorithm-hyperparameters, Property 8: Ganador del torneo es el mejor de los k seleccionados
    /**
     * Para cualquier ejecución de torneo con k individuos seleccionados aleatoriamente
     * de la población, el individuo retornado como ganador debe tener un fitness mayor
     * o igual al de todos los demás individuos del torneo.
     *
     * <p>We test this by setting tournament size = population size, which makes the
     * tournament deterministic: the winner must always be the individual with the
     * highest fitness.</p>
     *
     * <p><b>Validates: Requirements 4.1, 4.2</b></p>
     */
    @Property(tries = 100)
    void ganadorTorneoEsMejor(@ForAll("semilla") long semilla,
                              @ForAll("tamañoPoblacion") int tamPoblacion) {
        List<Individuo> poblacion = crearPoblacionEvaluada(tamPoblacion, semilla);

        // Use tournament size = population size → must return the best
        SelectorTorneo selector = new SelectorTorneo(tamPoblacion, 2, 0.15, new Random(semilla));
        List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

        // Find the best individual in the population
        double mejorFitness = poblacion.stream()
                .mapToDouble(Individuo::fitness)
                .max()
                .orElse(Double.NEGATIVE_INFINITY);

        // When k = N, every tournament winner must be the best individual
        // Non-elite pairs are selected from non-elite subset, so we check tournament pairs
        int numParejas = (tamPoblacion - 2) / 2;
        int parejasNoElite = (int) Math.ceil(numParejas * 0.15);

        for (int i = parejasNoElite; i < parejas.size(); i++) {
            SelectorTorneo.Pareja pareja = parejas.get(i);
            assertTrue(pareja.padre1().fitness() == mejorFitness
                            || pareja.padre2().fitness() == mejorFitness,
                    "Con torneo de tamaño N, al menos un padre debe ser el mejor. "
                            + "Mejor fitness=" + mejorFitness
                            + ", padre1=" + pareja.padre1().fitness()
                            + ", padre2=" + pareja.padre2().fitness());
        }
    }

    // ==================== Property 9: Padres siempre distintos ====================

    // Feature: genetic-algorithm-hyperparameters, Property 9: Padres siempre distintos
    /**
     * Para cualquier pareja de padres seleccionada para cruce, padre1 y padre2
     * deben ser individuos distintos (diferente referencia/cromosoma).
     *
     * <p><b>Validates: Requirement 4.3</b></p>
     */
    @Property(tries = 100)
    void padresDistintos(@ForAll("semilla") long semilla,
                         @ForAll("tamañoPoblacion") int tamPoblacion,
                         @ForAll("tamañoTorneo") int tamTorneo) {
        int k = Math.min(tamTorneo, tamPoblacion);
        List<Individuo> poblacion = crearPoblacionEvaluada(tamPoblacion, semilla);

        SelectorTorneo selector = new SelectorTorneo(k, 2, 0.15, new Random(semilla + 1));
        List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

        for (SelectorTorneo.Pareja pareja : parejas) {
            assertNotSame(pareja.padre1(), pareja.padre2(),
                    "padre1 y padre2 deben ser individuos distintos");
        }
    }

    // ==================== Property 11: Diversidad mínima de padres no-élite ====================

    // Feature: genetic-algorithm-hyperparameters, Property 11: Diversidad mínima de padres no-élite
    /**
     * Para cualquier generación, al menos el 15% de los padres seleccionados deben
     * provenir de individuos que no están entre los N mejores (no-élite).
     *
     * <p><b>Validates: Requirement 4.5</b></p>
     */
    @Property(tries = 100)
    void diversidadNoElite(@ForAll("semilla") long semilla,
                           @ForAll("tamañoPoblacionGrande") int tamPoblacion) {
        int numElites = 2;
        double porcentajeNoElite = 0.15;
        List<Individuo> poblacion = crearPoblacionEvaluada(tamPoblacion, semilla);

        SelectorTorneo selector = new SelectorTorneo(3, numElites, porcentajeNoElite, new Random(semilla + 1));
        List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

        if (parejas.isEmpty()) {
            return;
        }

        // Identify elites (top numElites by fitness)
        List<Individuo> ordenada = new ArrayList<>(poblacion);
        ordenada.sort(Comparator.naturalOrder());
        Set<Individuo> elites = new HashSet<>(ordenada.subList(0, Math.min(numElites, ordenada.size())));

        // Count parents that are non-elite
        int totalPadres = parejas.size() * 2;
        int padresNoElite = 0;
        for (SelectorTorneo.Pareja pareja : parejas) {
            if (!elites.contains(pareja.padre1())) padresNoElite++;
            if (!elites.contains(pareja.padre2())) padresNoElite++;
        }

        double fraccionNoElite = (double) padresNoElite / totalPadres;
        assertTrue(fraccionNoElite >= porcentajeNoElite - 0.01,
                "Fracción de padres no-élite=" + fraccionNoElite
                        + " es menor que el mínimo requerido=" + porcentajeNoElite
                        + " (padresNoElite=" + padresNoElite + ", totalPadres=" + totalPadres + ")");
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Long> semilla() {
        return Arbitraries.longs();
    }

    @Provide
    Arbitrary<Integer> tamañoPoblacion() {
        return Arbitraries.integers().between(10, 30);
    }

    @Provide
    Arbitrary<Integer> tamañoPoblacionGrande() {
        return Arbitraries.integers().between(20, 40);
    }

    @Provide
    Arbitrary<Integer> tamañoTorneo() {
        return Arbitraries.integers().between(2, 10);
    }

    // ==================== Helpers ====================

    private static Cromosoma cromosomaSimple() {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                new Gen.GenEntero("capasOcultas", 2, 1, 10),
                new Gen.GenEntero("neuronasCapa_0", 32, 1, 512),
                new Gen.GenEntero("neuronasCapa_1", 16, 1, 512)));
        bloques.put(BloqueFuncional.LIF, List.of(
                new Gen.GenReal("umbralDisparo", -55.0, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", -70.0, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", 20.0, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", 2, 1, 10)));
        bloques.put(BloqueFuncional.STDP, List.of(
                new Gen.GenReal("amplitudLTP", 0.01, 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", 0.012, 0.001, 0.1),
                new Gen.GenReal("tauLTP", 20.0, 5.0, 50.0),
                new Gen.GenReal("tauLTD", 20.0, 5.0, 50.0)));
        bloques.put(BloqueFuncional.CODIFICACION, List.of(
                new Gen.GenReal("frecuenciaMaxima", 100.0, 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion", ModoCodificacion.POISSON, ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", 50, 10, 200)));
        bloques.put(BloqueFuncional.REGULACION, List.of(
                new Gen.GenBooleano("homeostasisActiva", true),
                new Gen.GenReal("tasaDisparoObjetivo", 10.0, 1.0, 50.0),
                new Gen.GenReal("tasaAjusteHomeostasis", 0.01, 0.001, 0.1),
                new Gen.GenBooleano("inhibicionLateralActiva", false),
                new Gen.GenEntero("radioInhibicion", 2, 1, 5),
                new Gen.GenReal("fuerzaInhibicion", 0.5, 0.1, 2.0)));
        return new Cromosoma(bloques);
    }

    private static ConfiguracionRed configSimple() {
        return new ConfiguracionRedBuilder()
                .topologia(10, 32, 16, 5)
                .parametrosLIF(-55.0, -70.0, 20.0, 2)
                .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
                .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
                .homeostasis(true, 10.0, 0.01)
                .inhibicionLateral(false, 2, 0.5)
                .build();
    }

    /**
     * Creates a population of evaluated individuals with distinct fitness values.
     */
    private List<Individuo> crearPoblacionEvaluada(int tamaño, long semilla) {
        Random rng = new Random(semilla);
        Cromosoma cromosoma = cromosomaSimple();
        ConfiguracionRed config = configSimple();
        List<Individuo> poblacion = new ArrayList<>();
        for (int i = 0; i < tamaño; i++) {
            double fitness = 0.1 + (i * 0.05) + rng.nextDouble() * 0.01;
            Individuo ind = Individuo.sinEvaluar(cromosoma, config)
                    .conEvaluacion(fitness, null);
            poblacion.add(ind);
        }
        return poblacion;
    }
}
