package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.NivelComplejidad;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de integración para el ciclo evolutivo completo del AG.
 *
 * <p>Estos tests usan el sistema de benchmark real con configuraciones
 * mínimas (población pequeña, pocas generaciones) y
 * {@link NivelComplejidad#TRIVIAL} para ejecución rápida.</p>
 *
 * <p>Valida: Requisitos 7.1, 9.1, 9.2, 9.3, 10.1</p>
 */
class IntegracionAGTest {

    /**
     * Crea una configuración mínima para tests de integración.
     */
    private ConfiguracionAG crearConfigMinima() {
        return new ConfiguracionAGBuilder()
                .tamañoPoblacion(4)
                .maxGeneraciones(2)
                .generacionesEstancamiento(5)
                .numElites(1)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(512)
                .semilla(42L)
                .build();
    }

    /**
     * Test 1: Ejecución completa del ciclo evolutivo con benchmark real.
     *
     * <p>Verifica que el motor evolutivo puede ejecutar el ciclo completo
     * usando el constructor público con ConfiguracionAG y NivelComplejidad,
     * y que el informe resultante tiene una estructura válida.</p>
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void cicloEvolutivoCompletoConBenchmarkReal() {
        ConfiguracionAG config = crearConfigMinima();

        MotorEvolutivo motor = new MotorEvolutivo(config, NivelComplejidad.TRIVIAL);
        InformeEvolucion informe = motor.evolucionar();

        // Verificar estructura básica del informe
        assertNotNull(informe);
        assertNotNull(informe.mejorIndividuo());
        assertTrue(informe.mejorIndividuo().fitness() >= 0.0);
        assertTrue(informe.totalGeneraciones() > 0);
        assertTrue(informe.totalGeneraciones() <= 2);
        assertNotNull(informe.historial());
        assertEquals(informe.totalGeneraciones(), informe.historial().size());
        assertNotNull(informe.configuracionAG());
    }

    /**
     * Test 2: Consistencia del informe con la ejecución.
     *
     * <p>Verifica que los datos del informe son internamente consistentes:
     * motivo de parada válido, generación del mejor dentro de rango,
     * y estadísticas por generación coherentes.</p>
     */
    @Test
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void informeConsistenteConEjecucion() {
        ConfiguracionAG config = crearConfigMinima();

        MotorEvolutivo motor = new MotorEvolutivo(config, NivelComplejidad.TRIVIAL);
        InformeEvolucion informe = motor.evolucionar();

        // motivoParada debe ser válido
        assertTrue("max_generaciones".equals(informe.motivoParada())
                        || "estancamiento".equals(informe.motivoParada()),
                "motivoParada inesperado: " + informe.motivoParada());

        // generacionMejor debe estar dentro del rango
        assertTrue(informe.generacionMejor() >= 0);
        assertTrue(informe.generacionMejor() < informe.totalGeneraciones());

        // Las entradas del historial deben tener datos válidos
        for (EstadisticaGeneracion est : informe.historial()) {
            assertTrue(est.mejorFitness() >= est.peorFitness(),
                    "mejorFitness (" + est.mejorFitness() + ") < peorFitness (" + est.peorFitness() + ")");
            assertTrue(est.fitnessPromedio() >= est.peorFitness(),
                    "fitnessPromedio (" + est.fitnessPromedio() + ") < peorFitness (" + est.peorFitness() + ")");
            assertTrue(est.fitnessPromedio() <= est.mejorFitness(),
                    "fitnessPromedio (" + est.fitnessPromedio() + ") > mejorFitness (" + est.mejorFitness() + ")");
        }
    }
}
