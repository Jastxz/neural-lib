package es.jastxz.nn.benchmark;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link GeneradorTopologias}.
 */
class GeneradorTopologiasTest {

    @Test
    void generaExactamente8Topologias() {
        List<int[]> topologias = GeneradorTopologias.generar(10, 9);
        assertEquals(8, topologias.size());
    }

    @Test
    void todasLasTopologiasComienzanConEntrada() {
        List<int[]> topologias = GeneradorTopologias.generar(10, 9);
        for (int[] t : topologias) {
            assertEquals(10, t[0], "Primera capa debe ser la entrada");
        }
    }

    @Test
    void todasLasTopologiasTerminanConSalida() {
        List<int[]> topologias = GeneradorTopologias.generar(10, 9);
        for (int[] t : topologias) {
            assertEquals(9, t[t.length - 1], "Última capa debe ser la salida");
        }
    }

    @Test
    void topologiaCon1CapaOcultaFactor1x() {
        List<int[]> topologias = GeneradorTopologias.generar(10, 9);
        assertArrayEquals(new int[]{10, 10, 9}, topologias.get(0));
    }

    @Test
    void topologiaCon1CapaOcultaFactor2x() {
        List<int[]> topologias = GeneradorTopologias.generar(10, 9);
        assertArrayEquals(new int[]{10, 20, 9}, topologias.get(1));
    }

    @Test
    void topologiaCon4CapasOcultasFactor2x() {
        List<int[]> topologias = GeneradorTopologias.generar(10, 9);
        // 4 capas ocultas (índice base 6), factor 2x (offset 1) → [10, 20, 20, 20, 20, 9]
        assertArrayEquals(new int[]{10, 20, 20, 20, 20, 9}, topologias.get(7));
    }

    @Test
    void tamanosCrecientesPorCapasOcultas() {
        List<int[]> topologias = GeneradorTopologias.generar(5, 3);
        // 1 capa oculta → tamaño 3
        assertEquals(3, topologias.get(0).length);
        // 2 capas ocultas → tamaño 4
        assertEquals(4, topologias.get(2).length);
        // 3 capas ocultas → tamaño 5
        assertEquals(5, topologias.get(4).length);
        // 4 capas ocultas → tamaño 6
        assertEquals(6, topologias.get(6).length);
    }

    @Test
    void capasOcultasConFactorCorrecto() {
        List<int[]> topologias = GeneradorTopologias.generar(5, 3);
        // 2 capas ocultas, factor 2x → [5, 10, 10, 3]
        int[] t = topologias.get(3);
        assertEquals(5, t[0]);
        assertEquals(10, t[1]);
        assertEquals(10, t[2]);
        assertEquals(3, t[3]);
    }

    @Test
    void funcionaConDimensionesTriviales() {
        List<int[]> topologias = GeneradorTopologias.generar(2, 1);
        assertEquals(8, topologias.size());
        assertArrayEquals(new int[]{2, 2, 1}, topologias.get(0));
    }
}
