package es.jastxz.nn.genetico;

import es.jastxz.nn.genetico.Gen.GenBooleano;
import es.jastxz.nn.genetico.Gen.GenEntero;
import es.jastxz.nn.genetico.Gen.GenEnum;
import es.jastxz.nn.genetico.Gen.GenReal;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link Cromosoma}.
 * Valida: Requisitos 1.1, 2.1
 */
class CromosomaTest {

    // ── Helper ──────────────────────────────────────────────────────────

    /**
     * Construye un Cromosoma simple con valores conocidos para testing.
     * TOPOLOGIA: capasOcultas=2, neuronasCapa_0=32, neuronasCapa_1=16
     */
    private static Cromosoma cromosomaSimple() {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);

        bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                new GenEntero("capasOcultas", 2, 1, 10),
                new GenEntero("neuronasCapa_0", 32, 1, 512),
                new GenEntero("neuronasCapa_1", 16, 1, 512)
        ));
        bloques.put(BloqueFuncional.LIF, List.of(
                new GenReal("umbralDisparo", -55.0, -60.0, -40.0),
                new GenReal("potencialReposo", -70.0, -80.0, -60.0),
                new GenReal("constanteDecaimiento", 20.0, 5.0, 50.0),
                new GenEntero("duracionRefractario", 2, 1, 10)
        ));
        bloques.put(BloqueFuncional.STDP, List.of(
                new GenReal("amplitudLTP", 0.01, 0.001, 0.1),
                new GenReal("amplitudLTD", 0.012, 0.001, 0.1),
                new GenReal("tauLTP", 20.0, 5.0, 50.0),
                new GenReal("tauLTD", 20.0, 5.0, 50.0)
        ));
        bloques.put(BloqueFuncional.CODIFICACION, List.of(
                new GenReal("frecuenciaMaxima", 100.0, 10.0, 500.0),
                new GenEnum<>("modoCodificacion", ModoCodificacion.POISSON, ModoCodificacion.class),
                new GenEntero("ventanaDecodificacion", 50, 10, 200)
        ));
        bloques.put(BloqueFuncional.REGULACION, List.of(
                new GenBooleano("homeostasisActiva", true),
                new GenReal("tasaDisparoObjetivo", 10.0, 1.0, 50.0),
                new GenReal("tasaAjusteHomeostasis", 0.01, 0.001, 0.1),
                new GenBooleano("inhibicionLateralActiva", false),
                new GenEntero("radioInhibicion", 2, 1, 5),
                new GenReal("fuerzaInhibicion", 0.5, 0.1, 2.0)
        ));

        return new Cromosoma(bloques);
    }

    // ── genesDeBloque ───────────────────────────────────────────────────

    @Nested
    class GenesDeBloque {

        @Test
        void retornaGenesDelBloqueTopologia() {
            Cromosoma c = cromosomaSimple();
            List<Gen<?>> topologia = c.genesDeBloque(BloqueFuncional.TOPOLOGIA);

            assertEquals(3, topologia.size());
            assertEquals("capasOcultas", topologia.get(0).nombre());
            assertEquals(2, topologia.get(0).valor());
            assertEquals("neuronasCapa_0", topologia.get(1).nombre());
            assertEquals(32, topologia.get(1).valor());
            assertEquals("neuronasCapa_1", topologia.get(2).nombre());
            assertEquals(16, topologia.get(2).valor());
        }

        @Test
        void retornaGenesDelBloqueLIF() {
            Cromosoma c = cromosomaSimple();
            List<Gen<?>> lif = c.genesDeBloque(BloqueFuncional.LIF);

            assertEquals(4, lif.size());
            assertEquals("umbralDisparo", lif.get(0).nombre());
            assertEquals("duracionRefractario", lif.get(3).nombre());
        }

        @Test
        void retornaGenesDelBloqueSTDP() {
            Cromosoma c = cromosomaSimple();
            List<Gen<?>> stdp = c.genesDeBloque(BloqueFuncional.STDP);

            assertEquals(4, stdp.size());
            assertEquals("amplitudLTP", stdp.get(0).nombre());
        }

        @Test
        void retornaGenesDelBloqueCodificacion() {
            Cromosoma c = cromosomaSimple();
            List<Gen<?>> cod = c.genesDeBloque(BloqueFuncional.CODIFICACION);

            assertEquals(3, cod.size());
            assertEquals("modoCodificacion", cod.get(1).nombre());
            assertEquals(ModoCodificacion.POISSON, cod.get(1).valor());
        }

        @Test
        void retornaGenesDelBloqueRegulacion() {
            Cromosoma c = cromosomaSimple();
            List<Gen<?>> reg = c.genesDeBloque(BloqueFuncional.REGULACION);

            assertEquals(6, reg.size());
            assertEquals("homeostasisActiva", reg.get(0).nombre());
            assertEquals(true, reg.get(0).valor());
        }

        @Test
        void bloqueInexistenteRetornaListaVacia() {
            // Cromosoma con solo TOPOLOGIA
            var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
            bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                    new GenEntero("capasOcultas", 1, 1, 10)
            ));
            Cromosoma c = new Cromosoma(bloques);

            assertTrue(c.genesDeBloque(BloqueFuncional.LIF).isEmpty());
        }
    }

    // ── genesOrdenados ──────────────────────────────────────────────────

    @Nested
    class GenesOrdenados {

        @Test
        void retornaTodosLosGenesEnOrdenDeBloques() {
            Cromosoma c = cromosomaSimple();
            List<Gen<?>> genes = c.genesOrdenados();

            // 3 TOPOLOGIA + 4 LIF + 4 STDP + 3 CODIFICACION + 6 REGULACION = 20
            assertEquals(20, genes.size());

            // Primer gen: TOPOLOGIA
            assertEquals("capasOcultas", genes.get(0).nombre());
            // Primer gen de LIF (posición 3)
            assertEquals("umbralDisparo", genes.get(3).nombre());
            // Primer gen de STDP (posición 7)
            assertEquals("amplitudLTP", genes.get(7).nombre());
            // Primer gen de CODIFICACION (posición 11)
            assertEquals("frecuenciaMaxima", genes.get(11).nombre());
            // Primer gen de REGULACION (posición 14)
            assertEquals("homeostasisActiva", genes.get(14).nombre());
        }
    }

    // ── neuronasTotal ───────────────────────────────────────────────────

    @Nested
    class NeuronasTotal {

        @Test
        void calculaCorrectamenteConCapasOcultas() {
            Cromosoma c = cromosomaSimple();
            // entrada=10, ocultas=32+16=48, salida=3 → total=61
            assertEquals(61, c.neuronasTotal(10, 3));
        }

        @Test
        void calculaConUnaCapaOculta() {
            var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
            bloques.put(BloqueFuncional.TOPOLOGIA, List.of(
                    new GenEntero("capasOcultas", 1, 1, 10),
                    new GenEntero("neuronasCapa_0", 100, 1, 512)
            ));
            Cromosoma c = new Cromosoma(bloques);

            // entrada=5, ocultas=100, salida=2 → total=107
            assertEquals(107, c.neuronasTotal(5, 2));
        }

        @Test
        void entradaYSalidaSeSumanCorrectamente() {
            Cromosoma c = cromosomaSimple();
            // ocultas=32+16=48
            // Con entrada=0, salida=0 → solo ocultas
            assertEquals(48, c.neuronasTotal(0, 0));
            // Con entrada=100, salida=50 → 100+48+50=198
            assertEquals(198, c.neuronasTotal(100, 50));
        }
    }

    // ── conBloque ───────────────────────────────────────────────────────

    @Nested
    class ConBloque {

        @Test
        void reemplazaBloqueYRetornaNuevoCromosoma() {
            Cromosoma original = cromosomaSimple();
            List<Gen<?>> nuevaTopologia = List.of(
                    new GenEntero("capasOcultas", 3, 1, 10),
                    new GenEntero("neuronasCapa_0", 64, 1, 512),
                    new GenEntero("neuronasCapa_1", 32, 1, 512),
                    new GenEntero("neuronasCapa_2", 16, 1, 512)
            );

            Cromosoma modificado = original.conBloque(BloqueFuncional.TOPOLOGIA, nuevaTopologia);

            // Nuevo cromosoma tiene la topología reemplazada
            assertEquals(4, modificado.genesDeBloque(BloqueFuncional.TOPOLOGIA).size());
            assertEquals(3, modificado.genesDeBloque(BloqueFuncional.TOPOLOGIA).get(0).valor());
            assertEquals(64, modificado.genesDeBloque(BloqueFuncional.TOPOLOGIA).get(1).valor());
        }

        @Test
        void noModificaElCromosomaOriginal() {
            Cromosoma original = cromosomaSimple();
            List<Gen<?>> nuevaTopologia = List.of(
                    new GenEntero("capasOcultas", 1, 1, 10),
                    new GenEntero("neuronasCapa_0", 256, 1, 512)
            );

            original.conBloque(BloqueFuncional.TOPOLOGIA, nuevaTopologia);

            // Original no cambia (inmutabilidad)
            assertEquals(3, original.genesDeBloque(BloqueFuncional.TOPOLOGIA).size());
            assertEquals(2, original.genesDeBloque(BloqueFuncional.TOPOLOGIA).get(0).valor());
        }

        @Test
        void otrosBloquesNoSeModifican() {
            Cromosoma original = cromosomaSimple();
            List<Gen<?>> nuevaTopologia = List.of(
                    new GenEntero("capasOcultas", 1, 1, 10),
                    new GenEntero("neuronasCapa_0", 128, 1, 512)
            );

            Cromosoma modificado = original.conBloque(BloqueFuncional.TOPOLOGIA, nuevaTopologia);

            // LIF no cambia
            assertEquals(
                    original.genesDeBloque(BloqueFuncional.LIF),
                    modificado.genesDeBloque(BloqueFuncional.LIF)
            );
            // STDP no cambia
            assertEquals(
                    original.genesDeBloque(BloqueFuncional.STDP),
                    modificado.genesDeBloque(BloqueFuncional.STDP)
            );
        }

        @Test
        void neuronasActualizadasTrasCambioDeTopologia() {
            Cromosoma original = cromosomaSimple();
            // Original: 32+16=48 ocultas

            List<Gen<?>> nuevaTopologia = List.of(
                    new GenEntero("capasOcultas", 1, 1, 10),
                    new GenEntero("neuronasCapa_0", 256, 1, 512)
            );
            Cromosoma modificado = original.conBloque(BloqueFuncional.TOPOLOGIA, nuevaTopologia);

            // entrada=10, ocultas=256, salida=3 → 269
            assertEquals(269, modificado.neuronasTotal(10, 3));
        }
    }
}
