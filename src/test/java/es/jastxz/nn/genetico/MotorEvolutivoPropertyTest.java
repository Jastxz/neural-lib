package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.NivelComplejidad;
import es.jastxz.nn.benchmark.RecolectorMetricas;
import net.jqwik.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link MotorEvolutivo}.
 *
 * <p>Usa stubs de {@link EvaluadorFitness} que asignan fitness determinista
 * sin ejecutar benchmarks reales, permitiendo tests rápidos y reproducibles.</p>
 */
class MotorEvolutivoPropertyTest {

    private static final int DIMENSION_ENTRADA = 4;
    private static final int DIMENSION_SALIDA = 2;
    private static final int LIMITE_TOPOLOGICO = 512;

    // ==================== Stubs para testing ====================

    /**
     * Stub que asigna fitness basado en el hashCode del cromosoma.
     * Garantiza que el mismo cromosoma siempre recibe el mismo fitness,
     * lo que permite verificar preservación de élites correctamente.
     */
    private static class EvaluadorFitnessDeterminista extends EvaluadorFitness {

        EvaluadorFitnessDeterminista() {
            super(NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                    0.5, 0.3, 0.2, LIMITE_TOPOLOGICO, 1, 1, 42L);
        }

        @Override
        public Individuo evaluar(Individuo individuo) {
            // Fitness determinista basado en el cromosoma: normalizado a [0.1, 0.9]
            double fitness = 0.1 + 0.8 * ((individuo.cromosoma().hashCode() & 0x7FFFFFFF) % 10000) / 10000.0;
            return individuo.conEvaluacion(fitness, null);
        }

        @Override
        public List<Individuo> evaluarPoblacion(List<Individuo> poblacion) {
            return poblacion.stream().map(this::evaluar).toList();
        }
    }

    /**
     * Stub que siempre retorna el mismo fitness (para test de estancamiento).
     */
    private static class EvaluadorFitnessConstante extends EvaluadorFitness {
        private final double fitnessConstante;

        EvaluadorFitnessConstante(double fitnessConstante) {
            super(NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                    0.5, 0.3, 0.2, LIMITE_TOPOLOGICO, 1, 1, 42L);
            this.fitnessConstante = fitnessConstante;
        }

        @Override
        public Individuo evaluar(Individuo individuo) {
            return individuo.conEvaluacion(fitnessConstante, null);
        }

        @Override
        public List<Individuo> evaluarPoblacion(List<Individuo> poblacion) {
            return poblacion.stream().map(this::evaluar).toList();
        }
    }

    /**
     * Stub que asigna fitness creciente con un contador global.
     */
    private static class EvaluadorFitnessCreciente extends EvaluadorFitness {
        private int contador = 0;

        EvaluadorFitnessCreciente() {
            super(NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                    0.5, 0.3, 0.2, LIMITE_TOPOLOGICO, 1, 1, 42L);
        }

        @Override
        public Individuo evaluar(Individuo individuo) {
            return individuo.conEvaluacion(0.1 + (contador++ * 0.001), null);
        }

        @Override
        public List<Individuo> evaluarPoblacion(List<Individuo> poblacion) {
            return poblacion.stream().map(this::evaluar).toList();
        }
    }

    // ==================== Helpers ====================

    private MotorEvolutivo crearMotor(ConfiguracionAG config, EvaluadorFitness evaluador, long semilla) {
        Random random = new Random(semilla);
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, random);
        SelectorTorneo selector = new SelectorTorneo(
                config.tamañoTorneo(), config.numElites(),
                config.porcentajeNoElite(), random);
        OperadorCruce cruce = new OperadorCruce(
                config.probabilidadCruce(), config.puntosCorte(),
                LIMITE_TOPOLOGICO, fabrica, random);
        OperadorMutacion mutacion = new OperadorMutacion(
                config.probabilidadMutacion(), LIMITE_TOPOLOGICO, fabrica, random);

        return new MotorEvolutivo(config, fabrica, evaluador, selector, cruce, mutacion);
    }

    private ConfiguracionAG crearConfigPequeña(long semilla, int generaciones, int estancamiento) {
        return new ConfiguracionAGBuilder()
                .tamañoPoblacion(6)
                .maxGeneraciones(generaciones)
                .generacionesEstancamiento(estancamiento)
                .numElites(2)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(LIMITE_TOPOLOGICO)
                .semilla(semilla)
                .build();
    }

    // ==================== Property 10: Preservación de élites ====================

    // Feature: genetic-algorithm-hyperparameters, Property 10: Preservación de élites
    /**
     * Verifica que los N mejores individuos de una generación están presentes
     * sin modificación en la siguiente generación.
     *
     * <p>Con un evaluador determinista (mismo cromosoma → mismo fitness), si las
     * élites se preservan correctamente, el mejor fitness del historial nunca
     * debe decrecer entre generaciones consecutivas, ya que los mejores individuos
     * se mantienen en la población.</p>
     *
     * <p><b>Validates: Requisito 4.4</b></p>
     */
    @Property(tries = 100)
    void elitesPreservadas(@ForAll("semilla") long semilla) {
        ConfiguracionAG config = crearConfigPequeña(semilla, 5, 10);
        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor = crearMotor(config, evaluador, semilla);

        InformeEvolucion informe = motor.evolucionar();

        // Con élites preservadas y evaluación determinista, el mejor fitness
        // de cada generación no debe decrecer respecto a la anterior
        List<EstadisticaGeneracion> historial = informe.historial();
        for (int i = 1; i < historial.size(); i++) {
            double fitnessPrev = historial.get(i - 1).mejorFitness();
            double fitnessCurr = historial.get(i).mejorFitness();
            assertTrue(fitnessCurr >= fitnessPrev - 1e-9,
                    "El mejor fitness no debe decrecer entre generaciones " +
                    (i - 1) + " (" + fitnessPrev + ") y " + i + " (" + fitnessCurr + "). " +
                    "Esto indica que las élites no se preservaron correctamente.");
        }
    }

    // ==================== Property 17: Criterios de parada respetados ====================

    // Feature: genetic-algorithm-hyperparameters, Property 17: Criterios de parada respetados
    /**
     * Verifica que:
     * - totalGeneraciones <= maxGeneraciones
     * - Si motivoParada es "estancamiento", el mejor fitness no mejoró >1% en las
     *   últimas N generaciones consecutivas
     *
     * <p><b>Validates: Requisitos 7.2, 7.3</b></p>
     */
    @Property(tries = 100)
    void criteriosParadaRespetados(@ForAll("semilla") long semilla) {
        int maxGen = 5;
        int estancamiento = 2;
        ConfiguracionAG config = crearConfigPequeña(semilla, maxGen, estancamiento);

        // Usar evaluador determinista para resultados reproducibles
        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor = crearMotor(config, evaluador, semilla);

        InformeEvolucion informe = motor.evolucionar();

        // 1. totalGeneraciones <= maxGeneraciones
        assertTrue(informe.totalGeneraciones() <= maxGen,
                "totalGeneraciones=" + informe.totalGeneraciones() +
                " excede maxGeneraciones=" + maxGen);

        // 2. Si motivoParada es "estancamiento", verificar que no hubo mejora >1%
        if ("estancamiento".equals(informe.motivoParada())) {
            List<EstadisticaGeneracion> historial = informe.historial();
            int n = historial.size();
            if (n >= estancamiento) {
                // Verificar que en las últimas 'estancamiento' generaciones,
                // el mejor fitness no mejoró más de 1%
                double mejorAnterior = historial.get(n - estancamiento - 1 >= 0 ?
                        n - estancamiento - 1 : 0).mejorFitness();
                for (int i = Math.max(0, n - estancamiento); i < n; i++) {
                    double fitnessCurr = historial.get(i).mejorFitness();
                    assertTrue(fitnessCurr <= mejorAnterior * 1.01 + 1e-9,
                            "En estancamiento, generación " + i + " tiene fitness " +
                            fitnessCurr + " que mejora >1% respecto a " + mejorAnterior);
                }
            }
        }

        // 3. motivoParada debe ser uno de los dos valores válidos
        assertTrue("max_generaciones".equals(informe.motivoParada()) ||
                   "estancamiento".equals(informe.motivoParada()),
                "motivoParada debe ser 'max_generaciones' o 'estancamiento', " +
                "pero fue: " + informe.motivoParada());
    }

    // ==================== Property 17 con estancamiento forzado ====================

    // Feature: genetic-algorithm-hyperparameters, Property 17: Criterios de parada respetados (estancamiento)
    /**
     * Verifica que con fitness constante, el motor se detiene por estancamiento.
     *
     * <p><b>Validates: Requisitos 7.2, 7.3</b></p>
     */
    @Property(tries = 100)
    void estancamientoConFitnessConstante(@ForAll("semilla") long semilla) {
        int maxGen = 10;
        int estancamiento = 2;
        ConfiguracionAG config = crearConfigPequeña(semilla, maxGen, estancamiento);

        EvaluadorFitnessConstante evaluador = new EvaluadorFitnessConstante(0.5);
        MotorEvolutivo motor = crearMotor(config, evaluador, semilla);

        InformeEvolucion informe = motor.evolucionar();

        // Con fitness constante, debe parar por estancamiento
        assertEquals("estancamiento", informe.motivoParada(),
                "Con fitness constante, el motor debe parar por estancamiento");
        assertTrue(informe.totalGeneraciones() <= estancamiento + 1,
                "Con estancamiento=" + estancamiento + ", debe parar en ≤" +
                (estancamiento + 1) + " generaciones, pero ejecutó " +
                informe.totalGeneraciones());
    }

    // ==================== Property 18: Mejor global es el mejor de toda la evolución ====================

    // Feature: genetic-algorithm-hyperparameters, Property 18: Mejor global es el mejor de toda la evolución
    /**
     * Verifica que mejorIndividuo.fitness() >= mejorFitness de cada generación en el historial.
     *
     * <p><b>Validates: Requisito 7.5</b></p>
     */
    @Property(tries = 100)
    void mejorGlobalEsMejorHistorial(@ForAll("semilla") long semilla) {
        ConfiguracionAG config = crearConfigPequeña(semilla, 5, 10);
        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor = crearMotor(config, evaluador, semilla);

        InformeEvolucion informe = motor.evolucionar();

        double mejorGlobalFitness = informe.mejorIndividuo().fitness();

        // El mejor global debe tener fitness >= mejor fitness de cada generación
        for (EstadisticaGeneracion estadistica : informe.historial()) {
            assertTrue(mejorGlobalFitness >= estadistica.mejorFitness() - 1e-9,
                    "El mejor global (fitness=" + mejorGlobalFitness +
                    ") debe ser >= mejor de generación " + estadistica.numero() +
                    " (fitness=" + estadistica.mejorFitness() + ")");
        }
    }

    // ==================== Property 19: Reproducibilidad con semilla ====================

    // Feature: genetic-algorithm-hyperparameters, Property 19: Reproducibilidad con semilla
    /**
     * Verifica que dos ejecuciones con misma semilla y configuración producen
     * el mismo resultado (mismo mejor fitness y cromosoma).
     *
     * <p><b>Validates: Requisito 7.7</b></p>
     */
    @Property(tries = 100)
    void reproducibilidadConSemilla(@ForAll("semilla") long semilla) {
        ConfiguracionAG config = crearConfigPequeña(semilla, 3, 10);

        // Primera ejecución
        EvaluadorFitnessDeterminista evaluador1 = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor1 = crearMotor(config, evaluador1, semilla);
        InformeEvolucion informe1 = motor1.evolucionar();

        // Segunda ejecución con misma semilla y configuración
        EvaluadorFitnessDeterminista evaluador2 = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor2 = crearMotor(config, evaluador2, semilla);
        InformeEvolucion informe2 = motor2.evolucionar();

        // Mismo mejor fitness
        assertEquals(informe1.mejorIndividuo().fitness(),
                informe2.mejorIndividuo().fitness(), 1e-9,
                "Dos ejecuciones con misma semilla deben producir el mismo mejor fitness");

        // Mismo cromosoma (mismos genes)
        assertEquals(informe1.mejorIndividuo().cromosoma().genesOrdenados().toString(),
                informe2.mejorIndividuo().cromosoma().genesOrdenados().toString(),
                "Dos ejecuciones con misma semilla deben producir el mismo cromosoma");

        // Mismo número de generaciones
        assertEquals(informe1.totalGeneraciones(), informe2.totalGeneraciones(),
                "Dos ejecuciones con misma semilla deben ejecutar el mismo número de generaciones");

        // Mismo motivo de parada
        assertEquals(informe1.motivoParada(), informe2.motivoParada(),
                "Dos ejecuciones con misma semilla deben tener el mismo motivo de parada");
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Long> semilla() {
        return Arbitraries.longs();
    }
}
