package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ConfiguracionRedBuilder;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link SelectorTorneo}.
 */
class SelectorTorneoTest {

    @Test
    void torneoConK1_retornaAleatorio() {
        // k=1 means the tournament picks a single random individual
        List<Individuo> poblacion = crearPoblacionConFitness(20);

        // Run multiple times to verify it doesn't always return the best
        Set<Double> fitnessSeleccionados = new HashSet<>();
        for (int i = 0; i < 50; i++) {
            SelectorTorneo selector = new SelectorTorneo(1, 2, 0.15, new Random(i));
            List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);
            for (SelectorTorneo.Pareja p : parejas) {
                fitnessSeleccionados.add(p.padre1().fitness());
                fitnessSeleccionados.add(p.padre2().fitness());
            }
        }

        // With k=1, we should see variety in selected parents (not always the best)
        assertTrue(fitnessSeleccionados.size() > 1,
                "Con k=1, debería haber variedad en los padres seleccionados, "
                        + "pero solo se encontraron " + fitnessSeleccionados.size() + " fitness distintos");
    }

    @Test
    void torneoConKIgualN_retornaElMejor() {
        // k=N means deterministic selection of the best individual
        List<Individuo> poblacion = crearPoblacionConFitness(20);
        double mejorFitness = poblacion.stream()
                .mapToDouble(Individuo::fitness)
                .max()
                .orElse(0);

        SelectorTorneo selector = new SelectorTorneo(20, 2, 0.15, new Random(42));
        List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

        int numParejas = (20 - 2) / 2; // 9
        int parejasNoElite = (int) Math.ceil(numParejas * 0.15); // 2

        // The tournament-selected pairs (after the non-elite ones) should have the best
        for (int i = parejasNoElite; i < parejas.size(); i++) {
            SelectorTorneo.Pareja p = parejas.get(i);
            assertTrue(p.padre1().fitness() == mejorFitness || p.padre2().fitness() == mejorFitness,
                    "Con k=N, al menos un padre de la pareja de torneo debe ser el mejor. "
                            + "Pareja " + i + ": padre1=" + p.padre1().fitness()
                            + ", padre2=" + p.padre2().fitness()
                            + ", mejorFitness=" + mejorFitness);
        }
    }

    @Test
    void seleccionarParejas_padresSiempreDistintos() {
        List<Individuo> poblacion = crearPoblacionConFitness(20);

        SelectorTorneo selector = new SelectorTorneo(3, 2, 0.15, new Random(42));
        List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

        for (SelectorTorneo.Pareja p : parejas) {
            assertNotSame(p.padre1(), p.padre2(),
                    "padre1 y padre2 deben ser individuos distintos");
        }
    }

    @Test
    void pareja_mejorPadre_retornaPadreConMayorFitness() {
        Individuo ind1 = crearIndividuoConFitness(0.8);
        Individuo ind2 = crearIndividuoConFitness(0.5);

        SelectorTorneo.Pareja pareja = new SelectorTorneo.Pareja(ind1, ind2);
        assertSame(ind1, pareja.mejorPadre());
        assertSame(ind2, pareja.peorPadre());

        // Reverse order
        SelectorTorneo.Pareja parejaInversa = new SelectorTorneo.Pareja(ind2, ind1);
        assertSame(ind1, parejaInversa.mejorPadre());
        assertSame(ind2, parejaInversa.peorPadre());
    }

    @Test
    void seleccionarParejas_numeroCorrecto() {
        List<Individuo> poblacion = crearPoblacionConFitness(20);
        int numElites = 2;
        int numParejas = (20 - numElites) / 2; // 9

        SelectorTorneo selector = new SelectorTorneo(3, numElites, 0.15, new Random(42));
        List<SelectorTorneo.Pareja> parejas = selector.seleccionarParejas(poblacion);

        assertEquals(numParejas, parejas.size(),
                "Debe haber (tamañoPoblacion - numElites) / 2 parejas");
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

    private static Individuo crearIndividuoConFitness(double fitness) {
        return Individuo.sinEvaluar(cromosomaSimple(), configSimple())
                .conEvaluacion(fitness, null);
    }

    private static List<Individuo> crearPoblacionConFitness(int tamaño) {
        List<Individuo> poblacion = new ArrayList<>();
        for (int i = 0; i < tamaño; i++) {
            double fitness = 0.1 + (i * 0.05);
            poblacion.add(crearIndividuoConFitness(fitness));
        }
        return poblacion;
    }
}
