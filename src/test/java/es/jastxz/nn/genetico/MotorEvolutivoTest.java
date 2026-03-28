package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.NivelComplejidad;
import es.jastxz.nn.benchmark.RecolectorMetricas;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link MotorEvolutivo}.
 *
 * <p>Usa stubs de {@link EvaluadorFitness} para evitar ejecutar benchmarks reales.</p>
 */
class MotorEvolutivoTest {

    private static final int DIMENSION_ENTRADA = 4;
    private static final int DIMENSION_SALIDA = 2;
    private static final int LIMITE_TOPOLOGICO = 512;
    private static final long SEMILLA = 42L;

    // ==================== Stubs ====================

    /**
     * Stub que asigna fitness determinista basado en hashCode del cromosoma.
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

    // ==================== Helpers ====================

    private MotorEvolutivo crearMotor(ConfiguracionAG config, EvaluadorFitness evaluador) {
        Random random = new Random(SEMILLA);
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

    // ==================== Tests ====================

    /**
     * Verifica evolución con 1 generación: el informe debe tener exactamente
     * 1 generación en el historial y motivo de parada "max_generaciones".
     */
    @Test
    void evolucionConUnaGeneracion() {
        ConfiguracionAG config = new ConfiguracionAGBuilder()
                .tamañoPoblacion(6)
                .maxGeneraciones(1)
                .generacionesEstancamiento(10)
                .numElites(2)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(LIMITE_TOPOLOGICO)
                .semilla(SEMILLA)
                .build();

        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor = crearMotor(config, evaluador);

        InformeEvolucion informe = motor.evolucionar();

        assertEquals(1, informe.totalGeneraciones(),
                "Debe haber exactamente 1 generación");
        assertEquals(1, informe.historial().size(),
                "El historial debe tener 1 entrada");
        assertEquals("max_generaciones", informe.motivoParada(),
                "Con 1 generación, el motivo debe ser max_generaciones");
        assertTrue(informe.mejorIndividuo().fitness() > -1,
                "El mejor individuo debe haber sido evaluado");
        assertNotNull(informe.configuracionAG(),
                "El informe debe incluir la configuración del AG");
    }

    /**
     * Verifica parada por estancamiento: con fitness constante y
     * generacionesEstancamiento=1, el motor debe parar rápidamente.
     */
    @Test
    void paradaPorEstancamiento() {
        ConfiguracionAG config = new ConfiguracionAGBuilder()
                .tamañoPoblacion(6)
                .maxGeneraciones(10)
                .generacionesEstancamiento(1)
                .numElites(2)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(LIMITE_TOPOLOGICO)
                .semilla(SEMILLA)
                .build();

        EvaluadorFitnessConstante evaluador = new EvaluadorFitnessConstante(0.5);
        MotorEvolutivo motor = crearMotor(config, evaluador);

        InformeEvolucion informe = motor.evolucionar();

        assertEquals("estancamiento", informe.motivoParada(),
                "Con fitness constante, debe parar por estancamiento");
        assertTrue(informe.totalGeneraciones() <= 2,
                "Con generacionesEstancamiento=1, debe parar en ≤2 generaciones, " +
                "pero ejecutó " + informe.totalGeneraciones());
        assertEquals(0.5, informe.mejorIndividuo().fitness(), 1e-9,
                "El mejor fitness debe ser 0.5 (constante)");
    }

    /**
     * Verifica que el informe tiene el motivo de parada correcto.
     */
    @Test
    void informeTieneMotivoParadaCorrecto() {
        // Test con max_generaciones
        ConfiguracionAG config = new ConfiguracionAGBuilder()
                .tamañoPoblacion(6)
                .maxGeneraciones(3)
                .generacionesEstancamiento(100) // alto para evitar estancamiento
                .numElites(2)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(LIMITE_TOPOLOGICO)
                .semilla(SEMILLA)
                .build();

        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        MotorEvolutivo motor = crearMotor(config, evaluador);

        InformeEvolucion informe = motor.evolucionar();

        // Con estancamiento alto, debería llegar a max_generaciones
        assertTrue("max_generaciones".equals(informe.motivoParada()) ||
                   "estancamiento".equals(informe.motivoParada()),
                "motivoParada debe ser válido: " + informe.motivoParada());
        assertTrue(informe.totalGeneraciones() <= 3,
                "No debe exceder maxGeneraciones=3");
    }
}
