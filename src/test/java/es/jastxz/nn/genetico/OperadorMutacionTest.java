package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.ConfiguracionRed;
import es.jastxz.nn.spiking.ModoCodificacion;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link OperadorMutacion}.
 */
class OperadorMutacionTest {

    private static final int DIM_ENTRADA = 10;
    private static final int DIM_SALIDA = 5;
    private static final int LIMITE_TOPOLOGICO = 512;

    @Test
    void mutacionConProbabilidad0_noModifica() {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(42));
        Individuo original = fabrica.generarAleatorio();

        OperadorMutacion mutacion = new OperadorMutacion(
                0.0, LIMITE_TOPOLOGICO, fabrica, new Random(42));
        Individuo mutado = mutacion.mutar(original);

        // All genes should be identical
        for (BloqueFuncional bloque : BloqueFuncional.values()) {
            List<Gen<?>> genesOrig = original.cromosoma().genesDeBloque(bloque);
            List<Gen<?>> genesMut = mutado.cromosoma().genesDeBloque(bloque);
            assertEquals(genesOrig.size(), genesMut.size(),
                    "Bloque " + bloque + " debe tener mismo tamaño");
            for (int i = 0; i < genesOrig.size(); i++) {
                assertEquals(genesOrig.get(i).valor(), genesMut.get(i).valor(),
                        "Gen " + genesOrig.get(i).nombre() + " no debe cambiar");
            }
        }
    }

    @Test
    void mutacionDeCapas_añadirCapas() {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(42));

        // Create a chromosome with 2 layers
        Cromosoma cromosoma = crearCromosomaConCapas(2);
        ConfiguracionRed config = fabrica.construirConfiguracion(cromosoma);
        Individuo original = Individuo.sinEvaluar(cromosoma, config).conEvaluacion(0.5, null);

        // Use a seed that will increase capasOcultas (we force mutation with prob 1.0)
        // We need to find a seed where capasOcultas increases
        boolean encontrado = false;
        for (long seed = 0; seed < 1000; seed++) {
            Random rng = new Random(seed);
            OperadorMutacion mutacion = new OperadorMutacion(
                    1.0, LIMITE_TOPOLOGICO, fabrica, rng);
            Individuo mutado = mutacion.mutar(original);

            List<Gen<?>> topMutado = mutado.cromosoma().genesDeBloque(BloqueFuncional.TOPOLOGIA);
            int capasMutadas = ((Number) topMutado.get(0).valor()).intValue();

            if (capasMutadas > 2) {
                // Verify new layers have neurons in [1, 512]
                assertEquals(capasMutadas + 1, topMutado.size(),
                        "Debe haber capasOcultas + 1 genes de topología");
                for (int i = 1; i < topMutado.size(); i++) {
                    int neuronas = ((Number) topMutado.get(i).valor()).intValue();
                    assertTrue(neuronas >= 1 && neuronas <= 512,
                            "Capa " + (i - 1) + " debe tener neuronas en [1, 512], fue: " + neuronas);
                }
                encontrado = true;
                break;
            }
        }
        assertTrue(encontrado, "Debe encontrar al menos una semilla que aumente capas");
    }

    @Test
    void mutacionDeCapas_eliminarCapas() {
        FabricaIndividuos fabrica = new FabricaIndividuos(
                DIM_ENTRADA, DIM_SALIDA, LIMITE_TOPOLOGICO, new Random(42));

        // Create a chromosome with 5 layers
        Cromosoma cromosoma = crearCromosomaConCapas(5);
        ConfiguracionRed config = fabrica.construirConfiguracion(cromosoma);
        Individuo original = Individuo.sinEvaluar(cromosoma, config).conEvaluacion(0.5, null);

        // Find a seed where capasOcultas decreases
        boolean encontrado = false;
        for (long seed = 0; seed < 1000; seed++) {
            Random rng = new Random(seed);
            OperadorMutacion mutacion = new OperadorMutacion(
                    1.0, LIMITE_TOPOLOGICO, fabrica, rng);
            Individuo mutado = mutacion.mutar(original);

            List<Gen<?>> topMutado = mutado.cromosoma().genesDeBloque(BloqueFuncional.TOPOLOGIA);
            int capasMutadas = ((Number) topMutado.get(0).valor()).intValue();

            if (capasMutadas < 5) {
                // Verify the number of layer genes matches
                assertEquals(capasMutadas + 1, topMutado.size(),
                        "Debe haber capasOcultas + 1 genes de topología");
                // Verify remaining layers are valid
                for (int i = 1; i < topMutado.size(); i++) {
                    int neuronas = ((Number) topMutado.get(i).valor()).intValue();
                    assertTrue(neuronas >= 1 && neuronas <= 512,
                            "Capa " + (i - 1) + " debe tener neuronas en [1, 512], fue: " + neuronas);
                }
                encontrado = true;
                break;
            }
        }
        assertTrue(encontrado, "Debe encontrar al menos una semilla que reduzca capas");
    }

    // ==================== Helpers ====================

    private static Cromosoma crearCromosomaConCapas(int numCapas) {
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);

        List<Gen<?>> topologia = new ArrayList<>();
        topologia.add(new Gen.GenEntero("capasOcultas", numCapas, 1, 10));
        for (int i = 0; i < numCapas; i++) {
            topologia.add(new Gen.GenEntero("neuronasCapa_" + i, 20 + i * 5, 1, 512));
        }
        bloques.put(BloqueFuncional.TOPOLOGIA, topologia);

        bloques.put(BloqueFuncional.LIF, List.of(
                new Gen.GenReal("umbralDisparo", -50.0, -60.0, -40.0),
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

        bloques.put(BloqueFuncional.COMPETICION, List.of(
                new Gen.GenBooleano("wtaActivo", false),
                new Gen.GenBooleano("wtaCapaSalida", false),
                new Gen.GenBooleano("wtaCapasOcultas", false),
                new Gen.GenEntero("radioWTA", 0, 0, 10),
                new Gen.GenReal("fuerzaWTA", 2.0, 0.5, 5.0),
                new Gen.GenReal("umbralActivacionWTA", 0.1, 0.0, 0.5)));

        return new Cromosoma(bloques);
    }
}
