package es.jastxz.nn.benchmark;

import net.jqwik.api.*;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de propiedades jqwik para la completitud del generador de topologías
 * en {@link GeneradorTopologias}.
 *
 * <p><b>Validates: Requisitos 2.1, 2.2, 2.4</b></p>
 *
 * @since 1.1
 */
class GeneradorTopologiasPropertyTest {

    private static final int[] CAPAS_OCULTAS_ESPERADAS = {1, 2, 3, 4};
    private static final int[] FACTORES_ESPERADOS = {1, 2};

    /**
     * Propiedad: para cualquier par de dimensiones de entrada y salida válidas,
     * el generador produce exactamente 8 topologías, cada una comienza con entrada,
     * termina con salida, las capas ocultas tienen el tamaño correcto según el factor,
     * y todas las combinaciones de capas y factores están representadas.
     */
    @Property(tries = 100)
    void completitudDelGeneradorDeTopologias(
            @ForAll("dimensionPositiva") int entrada,
            @ForAll("dimensionPositiva") int salida) {

        List<int[]> topologias = GeneradorTopologias.generar(entrada, salida);

        // 1. Exactamente 8 topologías (4 capas × 2 factores)
        assertEquals(8, topologias.size(),
            "Debe generar exactamente 8 topologías (4 capas × 2 factores)");

        // 2. Cada topología comienza con entrada y termina con salida
        for (int[] t : topologias) {
            assertEquals(entrada, t[0],
                "Toda topología debe comenzar con la dimensión de entrada");
            assertEquals(salida, t[t.length - 1],
                "Toda topología debe terminar con la dimensión de salida");
        }

        // 3. Capas ocultas tienen el tamaño correcto (entrada * factor)
        for (int[] t : topologias) {
            int numCapasOcultas = t.length - 2;
            assertTrue(numCapasOcultas >= 1 && numCapasOcultas <= 4,
                "Número de capas ocultas debe estar entre 1 y 4, fue: " + numCapasOcultas);
            int valorOculta = t[1];
            assertTrue(
                valorOculta == entrada * 1 || valorOculta == entrada * 2,
                "Capa oculta debe ser entrada*factor, fue: " + valorOculta +
                " para entrada=" + entrada);
            for (int i = 1; i <= numCapasOcultas; i++) {
                assertEquals(valorOculta, t[i],
                    "Todas las capas ocultas deben tener el mismo tamaño");
            }
        }

        // 4. Todas las 4 cantidades de capas ocultas están representadas
        Set<Integer> capasPresentes = topologias.stream()
            .map(t -> t.length - 2)
            .collect(Collectors.toSet());
        for (int capas : CAPAS_OCULTAS_ESPERADAS) {
            assertTrue(capasPresentes.contains(capas),
                "Debe haber topologías con " + capas + " capa(s) oculta(s)");
        }

        // 5. Todos los 2 factores están representados
        Set<Integer> factoresPresentes = topologias.stream()
            .map(t -> t[1])
            .collect(Collectors.toSet());
        for (int factor : FACTORES_ESPERADOS) {
            assertTrue(factoresPresentes.contains(entrada * factor),
                "Debe haber topologías con factor " + factor +
                "x (neuronas ocultas=" + (entrada * factor) + ")");
        }
    }

    @Provide
    Arbitrary<Integer> dimensionPositiva() {
        return Arbitraries.integers().between(1, 100);
    }
}
