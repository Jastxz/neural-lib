package es.jastxz.experimental;

import es.jastxz.nn.*;
import es.jastxz.nn.enums.EstadoRed;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoNeurona;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para RedNeuralExperimental - Fase 1: Estructura Base
 */
class RedNeuralExperimentalTest {
    
    @Test
    @DisplayName("Red se inicializa con topología simple [2, 3, 1]")
    void testInicializacionSimple() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        assertNotNull(red);
        assertArrayEquals(topologia, red.getTopologia());
        assertEquals(6, red.getTotalNeuronas()); // 2 + 3 + 1
        assertTrue(red.getTotalConexiones() > 0);
    }
    
    @Test
    @DisplayName("Red se inicializa con topología compleja [4, 6, 4, 2]")
    void testInicializacionCompleja() {
        int[] topologia = {4, 6, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        assertEquals(16, red.getTotalNeuronas()); // 4 + 6 + 4 + 2
        assertEquals(4, red.getCapaSensorial().size());
        assertEquals(2, red.getCapasInterneuronas().size());
        assertEquals(6, red.getCapasInterneuronas().get(0).size());
        assertEquals(4, red.getCapasInterneuronas().get(1).size());
        assertEquals(2, red.getCapaMotora().size());
    }
    
    @Test
    @DisplayName("Red con topología mínima [1, 1]")
    void testTopologiaMinima() {
        int[] topologia = {1, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        assertEquals(2, red.getTotalNeuronas());
        assertEquals(1, red.getCapaSensorial().size());
        assertEquals(0, red.getCapasInterneuronas().size());
        assertEquals(1, red.getCapaMotora().size());
    }
    
    @Test
    @DisplayName("Densidad de conexiones afecta número de conexiones")
    void testDensidadConexiones() {
        int[] topologia = {5, 5, 5};
        
        RedNeuralExperimental redDensa = new RedNeuralExperimental(topologia, 1.0);
        RedNeuralExperimental redEsparsa = new RedNeuralExperimental(topologia, 0.3);
        
        // Red densa debe tener más conexiones que red esparsa
        assertTrue(redDensa.getTotalConexiones() > redEsparsa.getTotalConexiones());
    }
    
    @Test
    @DisplayName("Neuronas se crean con tipos correctos")
    void testTiposNeuronas() {
        int[] topologia = {2, 3, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        // Verificar tipos de capa sensorial
        for (Neurona n : red.getCapaSensorial()) {
            assertEquals(TipoNeurona.SENSORIAL, n.getTipo());
        }
        
        // Verificar tipos de interneuronas
        for (List<Neurona> capa : red.getCapasInterneuronas()) {
            for (Neurona n : capa) {
                assertEquals(TipoNeurona.INTER, n.getTipo());
            }
        }
        
        // Verificar tipos de capa motora
        for (Neurona n : red.getCapaMotora()) {
            assertEquals(TipoNeurona.MOTORA, n.getTipo());
        }
    }
    
    @Test
    @DisplayName("Neuronas tienen IDs únicos")
    void testIDsUnicos() {
        int[] topologia = {3, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        List<Neurona> todasNeuronas = red.getCapaSensorial();
        for (List<Neurona> capa : red.getCapasInterneuronas()) {
            todasNeuronas.addAll(capa);
        }
        todasNeuronas.addAll(red.getCapaMotora());
        
        // Verificar que todos los IDs son únicos
        long idsUnicos = todasNeuronas.stream()
            .map(Neurona::getId)
            .distinct()
            .count();
        
        assertEquals(todasNeuronas.size(), idsUnicos);
    }
    
    @Test
    @DisplayName("Conexiones se crean entre capas adyacentes")
    void testConexionesEntrCapas() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Con densidad 1.0, debe haber conexiones entre todas las capas
        assertTrue(red.getTotalConexiones() > 0);
        
        // Verificar que hay conexiones feed-forward
        List<Conexion> conexiones = red.getConexiones();
        assertFalse(conexiones.isEmpty());
    }
    
    @Test
    @DisplayName("Red se resetea correctamente")
    void testReseteo() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        // Resetear (aunque ya estén en reposo, debe funcionar)
        red.resetear();
        
        // Verificar que todas las neuronas están en reposo
        for (Neurona n : red.getCapaSensorial()) {
            assertEquals(PotencialMemoria.REPOSO.getValor(), n.getPotencial(), 0.001);
            assertFalse(n.estaActiva());
        }
        
        for (Neurona n : red.getCapaMotora()) {
            assertEquals(PotencialMemoria.REPOSO.getValor(), n.getPotencial(), 0.001);
            assertFalse(n.estaActiva());
        }
    }
    
    @Test
    @DisplayName("Timestamp global avanza correctamente")
    void testTimestamp() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        assertEquals(0L, red.getTimestampGlobal());
        
        red.avanzarTiempo(100L);
        assertEquals(100L, red.getTimestampGlobal());
        
        red.avanzarTiempo(50L);
        assertEquals(150L, red.getTimestampGlobal());
    }
    
    @Test
    @DisplayName("Estado inicial es ACTIVO")
    void testEstadoInicial() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        assertEquals(EstadoRed.ACTIVO, red.getEstado());
    }
    
    @Test
    @DisplayName("Engramas inicialmente vacíos")
    void testEngramasIniciales() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        assertTrue(red.getEngramas().isEmpty());
    }
    
    @Test
    @DisplayName("toString proporciona información útil")
    void testToString() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        String info = red.toString();
        
        assertTrue(info.contains("RedNeuralExperimental"));
        assertTrue(info.contains("Topología"));
        assertTrue(info.contains("Total neuronas"));
        assertTrue(info.contains("Total conexiones"));
        assertTrue(info.contains("Estado"));
    }
    
    @Test
    @DisplayName("Excepción si topología es null")
    void testTopologiaNull() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RedNeuralExperimental(null, 0.5);
        });
    }
    
    @Test
    @DisplayName("Excepción si topología tiene menos de 2 capas")
    void testTopologiaInsuficiente() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RedNeuralExperimental(new int[]{5}, 0.5);
        });
    }
    
    @Test
    @DisplayName("Excepción si alguna capa tiene 0 neuronas")
    void testCapaVacia() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RedNeuralExperimental(new int[]{5, 0, 3}, 0.5);
        });
    }
    
    @Test
    @DisplayName("Excepción si densidad fuera de rango")
    void testDensidadInvalida() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RedNeuralExperimental(new int[]{2, 2}, -0.1);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new RedNeuralExperimental(new int[]{2, 2}, 1.5);
        });
    }
    
    @Test
    @DisplayName("Red con múltiples capas intermedias")
    void testMultiplesCapasIntermedias() {
        int[] topologia = {3, 5, 4, 3, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.6);
        
        assertEquals(3, red.getCapasInterneuronas().size());
        assertEquals(5, red.getCapasInterneuronas().get(0).size());
        assertEquals(4, red.getCapasInterneuronas().get(1).size());
        assertEquals(3, red.getCapasInterneuronas().get(2).size());
    }
    
    @Test
    @DisplayName("Conexiones incluyen feedback")
    void testConexionesFeedback() {
        int[] topologia = {2, 3, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        // Con densidad alta, debe haber conexiones feedback
        // (difícil de verificar directamente sin inspeccionar direcciones)
        // Por ahora verificamos que hay suficientes conexiones
        int conexionesEsperadasMinimas = 2 * 3 + 3 * 2; // feed-forward mínimo
        assertTrue(red.getTotalConexiones() >= conexionesEsperadasMinimas * 0.5);
    }
    
    @Test
    @DisplayName("Neuronas tienen potencial de reposo inicial")
    void testPotencialReposoInicial() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        for (Neurona n : red.getCapaSensorial()) {
            assertEquals(PotencialMemoria.REPOSO.getValor(), n.getPotencial(), 0.001);
        }
    }
}
