package es.jastxz.nn.genetico;

import es.jastxz.nn.benchmark.ResultadoBenchmark;
import es.jastxz.nn.genetico.Gen.GenBooleano;
import es.jastxz.nn.genetico.Gen.GenEntero;
import es.jastxz.nn.genetico.Gen.GenEnum;
import es.jastxz.nn.genetico.Gen.GenReal;
import es.jastxz.nn.spiking.ConfiguracionRedBuilder;
import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link Individuo}.
 * Valida: Requisitos 1.7, 3.1
 */
class IndividuoTest {

    // ── Helpers ─────────────────────────────────────────────────────────

    private static Cromosoma cromosomaSimple() {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                new GenEntero("capasOcultas", 2, 1, 10),
                new GenEntero("neuronasCapa_0", 32, 1, 512),
                new GenEntero("neuronasCapa_1", 16, 1, 512)));
        bloques.put(BloqueFuncional.LIF, List.of(
                new GenReal("umbralDisparo", -55.0, -60.0, -40.0),
                new GenReal("potencialReposo", -70.0, -80.0, -60.0),
                new GenReal("constanteDecaimiento", 20.0, 5.0, 50.0),
                new GenEntero("duracionRefractario", 2, 1, 10)));
        bloques.put(BloqueFuncional.STDP, List.of(
                new GenReal("amplitudLTP", 0.01, 0.001, 0.1),
                new GenReal("amplitudLTD", 0.012, 0.001, 0.1),
                new GenReal("tauLTP", 20.0, 5.0, 50.0),
                new GenReal("tauLTD", 20.0, 5.0, 50.0)));
        bloques.put(BloqueFuncional.CODIFICACION, List.of(
                new GenReal("frecuenciaMaxima", 100.0, 10.0, 500.0),
                new GenEnum<>("modoCodificacion", ModoCodificacion.POISSON, ModoCodificacion.class),
                new GenEntero("ventanaDecodificacion", 50, 10, 200)));
        bloques.put(BloqueFuncional.REGULACION, List.of(
                new GenBooleano("homeostasisActiva", true),
                new GenReal("tasaDisparoObjetivo", 10.0, 1.0, 50.0),
                new GenReal("tasaAjusteHomeostasis", 0.01, 0.001, 0.1),
                new GenBooleano("inhibicionLateralActiva", false),
                new GenEntero("radioInhibicion", 2, 1, 5),
                new GenReal("fuerzaInhibicion", 0.5, 0.1, 2.0)));
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

    // ── sinEvaluar ──────────────────────────────────────────────────────

    @Nested
    class SinEvaluar {

        @Test
        void creaIndividuoConFitnessMenosUno() {
            var cromosoma = cromosomaSimple();
            var config = configSimple();

            var individuo = Individuo.sinEvaluar(cromosoma, config);

            assertEquals(-1.0, individuo.fitness());
        }

        @Test
        void creaIndividuoSinResultadoBenchmark() {
            var cromosoma = cromosomaSimple();
            var config = configSimple();

            var individuo = Individuo.sinEvaluar(cromosoma, config);

            assertNull(individuo.resultadoBenchmark());
        }

        @Test
        void preservaCromosomaYConfiguracion() {
            var cromosoma = cromosomaSimple();
            var config = configSimple();

            var individuo = Individuo.sinEvaluar(cromosoma, config);

            assertSame(cromosoma, individuo.cromosoma());
            assertSame(config, individuo.configuracionRed());
        }
    }

    // ── conEvaluacion ───────────────────────────────────────────────────

    @Nested
    class ConEvaluacion {

        @Test
        void asignaFitnessYResultado() {
            var individuo = Individuo.sinEvaluar(cromosomaSimple(), configSimple());
            var resultado = new ResultadoBenchmark(
                    null, 0.85, new double[]{0.5, 0.3}, 1000L,
                    500L, 5.0, 10.0, 1.5, 40, 63, null);

            var evaluado = individuo.conEvaluacion(0.75, resultado);

            assertEquals(0.75, evaluado.fitness());
            assertSame(resultado, evaluado.resultadoBenchmark());
        }

        @Test
        void preservaCromosomaYConfiguracion() {
            var cromosoma = cromosomaSimple();
            var config = configSimple();
            var individuo = Individuo.sinEvaluar(cromosoma, config);
            var resultado = new ResultadoBenchmark(
                    null, 0.9, new double[]{0.2}, 500L,
                    200L, 3.0, 5.0, 0.8, 30, 63, null);

            var evaluado = individuo.conEvaluacion(0.9, resultado);

            assertSame(cromosoma, evaluado.cromosoma());
            assertSame(config, evaluado.configuracionRed());
        }

        @Test
        void noModificaIndividuoOriginal() {
            var individuo = Individuo.sinEvaluar(cromosomaSimple(), configSimple());
            var resultado = new ResultadoBenchmark(
                    null, 0.8, new double[]{0.4}, 800L,
                    400L, 4.0, 8.0, 1.2, 35, 63, null);

            individuo.conEvaluacion(0.8, resultado);

            assertEquals(-1.0, individuo.fitness());
            assertNull(individuo.resultadoBenchmark());
        }
    }

    // ── compareTo ───────────────────────────────────────────────────────

    @Nested
    class CompareTo {

        @Test
        void mayorFitnessPrimero() {
            var alto = Individuo.sinEvaluar(cromosomaSimple(), configSimple())
                    .conEvaluacion(0.9, null);
            var bajo = Individuo.sinEvaluar(cromosomaSimple(), configSimple())
                    .conEvaluacion(0.3, null);

            assertTrue(alto.compareTo(bajo) < 0, "Mayor fitness debe ir primero (compareTo negativo)");
            assertTrue(bajo.compareTo(alto) > 0, "Menor fitness debe ir después (compareTo positivo)");
        }

        @Test
        void mismoFitnessRetornaCero() {
            var a = Individuo.sinEvaluar(cromosomaSimple(), configSimple())
                    .conEvaluacion(0.5, null);
            var b = Individuo.sinEvaluar(cromosomaSimple(), configSimple())
                    .conEvaluacion(0.5, null);

            assertEquals(0, a.compareTo(b));
        }

        @Test
        void ordenNaturalEsDescendentePorFitness() {
            var individuos = List.of(
                    Individuo.sinEvaluar(cromosomaSimple(), configSimple()).conEvaluacion(0.3, null),
                    Individuo.sinEvaluar(cromosomaSimple(), configSimple()).conEvaluacion(0.9, null),
                    Individuo.sinEvaluar(cromosomaSimple(), configSimple()).conEvaluacion(0.6, null));

            var ordenados = individuos.stream().sorted().toList();

            assertEquals(0.9, ordenados.get(0).fitness());
            assertEquals(0.6, ordenados.get(1).fitness());
            assertEquals(0.3, ordenados.get(2).fitness());
        }
    }
}
