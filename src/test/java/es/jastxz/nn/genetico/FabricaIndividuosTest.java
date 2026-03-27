package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link FabricaIndividuos}.
 */
class FabricaIndividuosTest {

    private FabricaIndividuos fabrica;

    @BeforeEach
    void setUp() {
        fabrica = new FabricaIndividuos(10, 5, 512, new Random(42));
    }

    @Test
    void generarAleatorio_produceIndividuoValido() {
        Individuo individuo = fabrica.generarAleatorio();

        assertNotNull(individuo);
        assertNotNull(individuo.cromosoma());
        assertNotNull(individuo.configuracionRed());
        assertEquals(-1.0, individuo.fitness());
        assertNull(individuo.resultadoBenchmark());
    }

    @Test
    void generarAleatorio_respetaLimiteTopologico() {
        // Usar un límite bajo para forzar reparación
        var fabricaPequena = new FabricaIndividuos(10, 5, 50, new Random(42));
        for (int i = 0; i < 20; i++) {
            Individuo individuo = fabricaPequena.generarAleatorio();
            int total = individuo.cromosoma().neuronasTotal(10, 5);
            assertTrue(total <= 50,
                    "neuronasTotal=" + total + " excede límite 50");
        }
    }

    @Test
    void generarAleatorio_umbralMayorQueReposo() {
        for (int i = 0; i < 50; i++) {
            Individuo individuo = fabrica.generarAleatorio();
            List<Gen<?>> lif = individuo.cromosoma().genesDeBloque(BloqueFuncional.LIF);
            double umbral = ((Number) lif.get(0).valor()).doubleValue();
            double reposo = ((Number) lif.get(1).valor()).doubleValue();
            assertTrue(umbral > reposo,
                    "umbralDisparo=" + umbral + " no es mayor que potencialReposo=" + reposo);
        }
    }

    @Test
    void construirConfiguracion_produceConfiguracionValida() {
        Individuo individuo = fabrica.generarAleatorio();
        // Should not throw
        ConfiguracionRed config = fabrica.construirConfiguracion(individuo.cromosoma());
        assertNotNull(config);
    }

    @Test
    void repararTopologia_casoConocido() {
        // Crear cromosoma que excede el límite: 3 capas con 200 neuronas cada una = 600 ocultas
        // Con entrada=10, salida=5, total=615 > 512
        // maxOcultas = 512 - 10 - 5 = 497
        // factor = 497/600 ≈ 0.828
        // Cada capa: round(200 * 0.828) = round(165.67) = 166
        // Suma = 498 > 497 → ajustar última: 166 - 1 = 165
        List<Gen<?>> topologia = List.of(
                new Gen.GenEntero("capasOcultas", 3, 1, 10),
                new Gen.GenEntero("neuronasCapa_0", 200, 1, 512),
                new Gen.GenEntero("neuronasCapa_1", 200, 1, 512),
                new Gen.GenEntero("neuronasCapa_2", 200, 1, 512)
        );

        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        bloques.put(BloqueFuncional.TOPOLOGIA, topologia);
        bloques.put(BloqueFuncional.LIF, crearGenesLIF());
        bloques.put(BloqueFuncional.STDP, crearGenesSTDP());
        bloques.put(BloqueFuncional.CODIFICACION, crearGenesCodificacion());
        bloques.put(BloqueFuncional.REGULACION, crearGenesRegulacion());
        Cromosoma cromosoma = new Cromosoma(bloques);

        Cromosoma reparado = fabrica.repararTopologia(cromosoma);
        int total = reparado.neuronasTotal(10, 5);
        assertTrue(total <= 512, "neuronasTotal=" + total + " excede 512");

        // Cada capa debe tener al menos 1 neurona
        List<Gen<?>> topReparada = reparado.genesDeBloque(BloqueFuncional.TOPOLOGIA);
        for (int i = 1; i < topReparada.size(); i++) {
            int neuronas = ((Number) topReparada.get(i).valor()).intValue();
            assertTrue(neuronas >= 1, "Capa " + (i - 1) + " tiene " + neuronas + " neuronas");
        }
    }

    @Test
    void repararTopologia_noModificaSiDentroDelLimite() {
        List<Gen<?>> topologia = List.of(
                new Gen.GenEntero("capasOcultas", 2, 1, 10),
                new Gen.GenEntero("neuronasCapa_0", 10, 1, 512),
                new Gen.GenEntero("neuronasCapa_1", 10, 1, 512)
        );

        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        bloques.put(BloqueFuncional.TOPOLOGIA, topologia);
        bloques.put(BloqueFuncional.LIF, crearGenesLIF());
        bloques.put(BloqueFuncional.STDP, crearGenesSTDP());
        bloques.put(BloqueFuncional.CODIFICACION, crearGenesCodificacion());
        bloques.put(BloqueFuncional.REGULACION, crearGenesRegulacion());
        Cromosoma cromosoma = new Cromosoma(bloques);

        Cromosoma reparado = fabrica.repararTopologia(cromosoma);
        // Neuronas should be unchanged
        List<Gen<?>> topReparada = reparado.genesDeBloque(BloqueFuncional.TOPOLOGIA);
        assertEquals(10, ((Number) topReparada.get(1).valor()).intValue());
        assertEquals(10, ((Number) topReparada.get(2).valor()).intValue());
    }

    // ==================== Helpers ====================

    private List<Gen<?>> crearGenesLIF() {
        return List.of(
                new Gen.GenReal("umbralDisparo", -50.0, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", -70.0, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", 20.0, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", 2, 1, 10)
        );
    }

    private List<Gen<?>> crearGenesSTDP() {
        return List.of(
                new Gen.GenReal("amplitudLTP", 0.01, 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", 0.012, 0.001, 0.1),
                new Gen.GenReal("tauLTP", 20.0, 5.0, 50.0),
                new Gen.GenReal("tauLTD", 20.0, 5.0, 50.0)
        );
    }

    private List<Gen<?>> crearGenesCodificacion() {
        return List.of(
                new Gen.GenReal("frecuenciaMaxima", 100.0, 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion",
                        es.jastxz.nn.spiking.ModoCodificacion.POISSON,
                        es.jastxz.nn.spiking.ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", 50, 10, 200)
        );
    }

    private List<Gen<?>> crearGenesRegulacion() {
        return List.of(
                new Gen.GenBooleano("homeostasisActiva", true),
                new Gen.GenReal("tasaDisparoObjetivo", 10.0, 1.0, 50.0),
                new Gen.GenReal("tasaAjusteHomeostasis", 0.01, 0.001, 0.1),
                new Gen.GenBooleano("inhibicionLateralActiva", false),
                new Gen.GenEntero("radioInhibicion", 2, 1, 5),
                new Gen.GenReal("fuerzaInhibicion", 0.5, 0.1, 2.0)
        );
    }
}
