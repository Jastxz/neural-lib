package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.NivelComplejidad;
import es.jastxz.nn.benchmark.RecolectorMetricas;
import net.jqwik.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para {@link InformeEvolucion}.
 *
 * <p>Usa stubs de {@link EvaluadorFitness} para verificar que el informe
 * generado por {@link MotorEvolutivo} contiene todos los campos requeridos.</p>
 */
class InformeEvolucionPropertyTest {

    private static final int DIMENSION_ENTRADA = 4;
    private static final int DIMENSION_SALIDA = 2;
    private static final int LIMITE_TOPOLOGICO = 512;

    // ==================== Stub para testing ====================

    /**
     * Stub que asigna fitness determinista basado en el hashCode del cromosoma.
     */
    private static class EvaluadorFitnessDeterminista extends EvaluadorFitness {

        EvaluadorFitnessDeterminista() {
            super(NivelComplejidad.TRIVIAL, new RecolectorMetricas(),
                    0.5, 0.3, 0.2, LIMITE_TOPOLOGICO, 1, 1, 42L);
        }

        @Override
        public Individuo evaluar(Individuo individuo) {
            double fitness = 0.1 + 0.8 * ((individuo.cromosoma().hashCode() & 0x7FFFFFFF) % 10000) / 10000.0;
            return individuo.conEvaluacion(fitness, null);
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

    // ==================== Property 21: Informe completo ====================

    // Feature: genetic-algorithm-hyperparameters, Property 21: Informe completo
    /**
     * Verifica que el informe contiene todos los campos requeridos:
     * <ul>
     *   <li>mejorIndividuo con fitness &gt; -1</li>
     *   <li>generacionMejor &gt;= 0</li>
     *   <li>totalGeneraciones &gt; 0</li>
     *   <li>motivoParada es "max_generaciones" o "estancamiento"</li>
     *   <li>historial tiene una entrada por generación ejecutada</li>
     *   <li>configuracionAG no es null</li>
     * </ul>
     *
     * <p><b>Validates: Requisitos 10.1, 10.2, 10.3, 10.5</b></p>
     */
    @Property(tries = 100)
    void informeCompletoTieneCampos(@ForAll("semilla") long semilla) {
        ConfiguracionAG config = new ConfiguracionAGBuilder()
                .tamañoPoblacion(6)
                .maxGeneraciones(5)
                .generacionesEstancamiento(10)
                .numElites(2)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(LIMITE_TOPOLOGICO)
                .semilla(semilla)
                .build();

        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor = crearMotor(config, evaluador, semilla);

        InformeEvolucion informe = motor.evolucionar();

        // mejorIndividuo con fitness > -1
        assertNotNull(informe.mejorIndividuo(),
                "mejorIndividuo no debe ser null");
        assertTrue(informe.mejorIndividuo().fitness() > -1,
                "mejorIndividuo debe tener fitness > -1, pero fue: " +
                informe.mejorIndividuo().fitness());

        // generacionMejor >= 0
        assertTrue(informe.generacionMejor() >= 0,
                "generacionMejor debe ser >= 0, pero fue: " + informe.generacionMejor());

        // totalGeneraciones > 0
        assertTrue(informe.totalGeneraciones() > 0,
                "totalGeneraciones debe ser > 0, pero fue: " + informe.totalGeneraciones());

        // motivoParada es uno de los dos valores válidos
        assertTrue("max_generaciones".equals(informe.motivoParada()) ||
                   "estancamiento".equals(informe.motivoParada()),
                "motivoParada debe ser 'max_generaciones' o 'estancamiento', " +
                "pero fue: " + informe.motivoParada());

        // historial tiene una entrada por generación ejecutada
        assertNotNull(informe.historial(),
                "historial no debe ser null");
        assertEquals(informe.totalGeneraciones(), informe.historial().size(),
                "historial debe tener una entrada por generación ejecutada. " +
                "totalGeneraciones=" + informe.totalGeneraciones() +
                ", historial.size()=" + informe.historial().size());

        // configuracionAG no es null
        assertNotNull(informe.configuracionAG(),
                "configuracionAG no debe ser null");
    }

    // ==================== Providers ====================

    @Provide
    Arbitrary<Long> semilla() {
        return Arbitraries.longs();
    }
}
