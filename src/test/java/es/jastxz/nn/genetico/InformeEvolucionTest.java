package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.NivelComplejidad;
import es.jastxz.nn.benchmark.RecolectorMetricas;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link InformeEvolucion}.
 *
 * <p>Verifica que {@code imprimir()} produce salida no vacía y que
 * contiene información clave del informe.</p>
 */
class InformeEvolucionTest {

    private static final int DIMENSION_ENTRADA = 4;
    private static final int DIMENSION_SALIDA = 2;
    private static final int LIMITE_TOPOLOGICO = 512;

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

    private InformeEvolucion ejecutarEvolucion() {
        long semilla = 12345L;
        ConfiguracionAG config = new ConfiguracionAGBuilder()
                .tamañoPoblacion(6)
                .maxGeneraciones(3)
                .generacionesEstancamiento(10)
                .numElites(2)
                .tamañoTorneo(2)
                .probabilidadCruce(0.8)
                .probabilidadMutacion(0.1)
                .puntosCorte(1)
                .limiteTopologico(LIMITE_TOPOLOGICO)
                .semilla(semilla)
                .build();

        Random random = new Random(semilla);
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIMENSION_ENTRADA, DIMENSION_SALIDA, LIMITE_TOPOLOGICO, random);
        EvaluadorFitnessDeterminista evaluador = new EvaluadorFitnessDeterminista();
        SelectorTorneo selector = new SelectorTorneo(
                config.tamañoTorneo(), config.numElites(),
                config.porcentajeNoElite(), random);
        OperadorCruce cruce = new OperadorCruce(
                config.probabilidadCruce(), config.puntosCorte(),
                LIMITE_TOPOLOGICO, fabrica, random);
        OperadorMutacion mutacion = new OperadorMutacion(
                config.probabilidadMutacion(), LIMITE_TOPOLOGICO, fabrica, random);

        MotorEvolutivo motor = new MotorEvolutivo(config, fabrica, evaluador, selector, cruce, mutacion);
        return motor.evolucionar();
    }

    // Requisito 10.4: imprimir() produce salida no vacía
    @Test
    void imprimirProduceSalidaNoVacia() {
        InformeEvolucion informe = ejecutarEvolucion();

        // Capturar System.out
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream original = System.out;
        try {
            System.setOut(new PrintStream(baos));
            informe.imprimir();
        } finally {
            System.setOut(original);
        }

        String salida = baos.toString();
        assertFalse(salida.isEmpty(), "imprimir() debe producir salida no vacía");
    }

    // Requisito 10.4: la salida contiene información clave
    @Test
    void imprimirContieneDatosClave() {
        InformeEvolucion informe = ejecutarEvolucion();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        PrintStream original = System.out;
        try {
            System.setOut(new PrintStream(baos));
            informe.imprimir();
        } finally {
            System.setOut(original);
        }

        String salida = baos.toString();

        // Contiene fitness del mejor individuo
        assertTrue(salida.contains(String.format("%.6f", informe.mejorIndividuo().fitness())),
                "La salida debe contener el fitness del mejor individuo");

        // Contiene generación donde se encontró
        assertTrue(salida.contains(String.valueOf(informe.generacionMejor())),
                "La salida debe contener la generación del mejor individuo");

        // Contiene motivo de parada
        assertTrue(salida.contains(informe.motivoParada()),
                "La salida debe contener el motivo de parada");

        // Contiene total de generaciones
        assertTrue(salida.contains(String.valueOf(informe.totalGeneraciones())),
                "La salida debe contener el total de generaciones");

        // Contiene límite topológico
        assertTrue(salida.contains(String.valueOf(informe.configuracionAG().limiteTopologico())),
                "La salida debe contener el límite topológico");
    }
}
