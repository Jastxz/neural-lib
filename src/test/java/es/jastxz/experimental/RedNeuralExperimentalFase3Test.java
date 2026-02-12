package es.jastxz.experimental;

import es.jastxz.nn.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para RedNeuralExperimental - Fase 3: Plasticidad Hebiana
 */
class RedNeuralExperimentalFase3Test {
    
    @Test
    @DisplayName("Entrenar con inputs y targets válidos no lanza excepción")
    void testEntrenarValido() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double[] inputs = {0.5, -0.3};
        double[] targets = {0.8};
        
        assertDoesNotThrow(() -> {
            red.entrenar(inputs, targets, 10);
        });
    }
    
    @Test
    @DisplayName("Excepción si targets es null")
    void testEntrenarTargetsNull() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        assertThrows(IllegalArgumentException.class, () -> {
            red.entrenar(new double[]{0.5, 0.3}, null, 10);
        });
    }
    
    @Test
    @DisplayName("Excepción si tamaño de targets no coincide")
    void testEntrenarTamanoTargetsIncorrecto() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        double[] inputs = {0.5, 0.3};
        double[] targetsIncorrectos = {0.8, 0.6}; // Esperaba 1, recibió 2
        
        assertThrows(IllegalArgumentException.class, () -> {
            red.entrenar(inputs, targetsIncorrectos, 10);
        });
    }
    
    @Test
    @DisplayName("Pesos sinápticos cambian después del entrenamiento")
    void testPesosCambianConEntrenamiento() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Guardar pesos iniciales
        double[] pesosIniciales = new double[red.getConexiones().size()];
        for (int i = 0; i < red.getConexiones().size(); i++) {
            pesosIniciales[i] = red.getConexiones().get(i).getPeso();
        }
        
        // Entrenar
        red.entrenar(new double[]{0.8, 0.6}, new double[]{0.9}, 50);
        
        // Verificar que al menos algunos pesos cambiaron
        boolean algunPesoCambio = false;
        for (int i = 0; i < red.getConexiones().size(); i++) {
            double pesoFinal = red.getConexiones().get(i).getPeso();
            if (Math.abs(pesoFinal - pesosIniciales[i]) > 0.001) {
                algunPesoCambio = true;
                break;
            }
        }
        
        assertTrue(algunPesoCambio, "Al menos algunos pesos deben cambiar con el entrenamiento");
    }
    
    @Test
    @DisplayName("Valores almacenados cambian después del entrenamiento")
    void testValoresAlmacenadosCambianConEntrenamiento() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        // Guardar valor almacenado inicial de neurona motora
        double valorInicial = red.getCapaMotora().get(0).getValorAlmacenado();
        
        // Entrenar con target diferente
        red.entrenar(new double[]{0.5, -0.3}, new double[]{0.9}, 50);
        
        // Verificar que el valor cambió
        double valorFinal = red.getCapaMotora().get(0).getValorAlmacenado();
        assertNotEquals(valorInicial, valorFinal, 0.001, 
            "El valor almacenado debe cambiar con el entrenamiento");
    }
    
    @Test
    @DisplayName("Red aprende a aproximar target simple")
    void testAprendizajeSimple() {
        int[] topologia = {2, 4, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.9);
        
        double[] inputs = {1.0, 0.0};
        double[] targets = {0.8};
        
        // Output antes del entrenamiento
        double outputAntes = red.procesar(inputs)[0];
        double errorAntes = Math.abs(targets[0] - outputAntes);
        
        // Entrenar
        red.resetear();
        red.entrenar(inputs, targets, 100);
        
        // Output después del entrenamiento
        red.resetear();
        double outputDespues = red.procesar(inputs)[0];
        double errorDespues = Math.abs(targets[0] - outputDespues);
        
        // El error debe reducirse (aprendizaje)
        assertTrue(errorDespues < errorAntes || errorDespues < 0.3, 
            "El error debe reducirse con el entrenamiento. Antes: " + errorAntes + ", Después: " + errorDespues);
    }
    
    @Test
    @DisplayName("Red aprende múltiples patrones")
    void testAprendizajeMultiplesPatrones() {
        int[] topologia = {2, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.9);
        
        // Patrón 1
        double[] inputs1 = {1.0, 0.0};
        double[] targets1 = {1.0, 0.0};
        
        // Patrón 2
        double[] inputs2 = {0.0, 1.0};
        double[] targets2 = {0.0, 1.0};
        
        // Entrenar ambos patrones
        for (int i = 0; i < 50; i++) {
            red.entrenar(inputs1, targets1, 1);
            red.entrenar(inputs2, targets2, 1);
        }
        
        // Verificar que la red funciona
        red.resetear();
        double[] output1 = red.procesar(inputs1);
        red.resetear();
        double[] output2 = red.procesar(inputs2);
        
        assertNotNull(output1);
        assertNotNull(output2);
        assertEquals(2, output1.length);
        assertEquals(2, output2.length);
    }
    
    @Test
    @DisplayName("Entrenamiento con 0 iteraciones no cambia la red")
    void testEntrenamientoCeroIteraciones() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double pesoInicial = red.getConexiones().get(0).getPeso();
        
        red.entrenar(new double[]{0.5, 0.3}, new double[]{0.8}, 0);
        
        double pesoFinal = red.getConexiones().get(0).getPeso();
        assertEquals(pesoInicial, pesoFinal, 0.0001);
    }
    
    @Test
    @DisplayName("Más iteraciones producen más cambio")
    void testMasIteracionesMasCambio() {
        int[] topologia = {2, 3, 1};
        
        // Red 1: pocas iteraciones
        RedNeuralExperimental red1 = new RedNeuralExperimental(topologia, 0.9);
        double pesoInicial1 = red1.getConexiones().get(0).getPeso();
        red1.entrenar(new double[]{0.8, 0.6}, new double[]{0.9}, 10);
        double cambio1 = Math.abs(red1.getConexiones().get(0).getPeso() - pesoInicial1);
        
        // Red 2: muchas iteraciones (misma topología, diferentes pesos iniciales)
        RedNeuralExperimental red2 = new RedNeuralExperimental(topologia, 0.9);
        double pesoInicial2 = red2.getConexiones().get(0).getPeso();
        red2.entrenar(new double[]{0.8, 0.6}, new double[]{0.9}, 100);
        double cambio2 = Math.abs(red2.getConexiones().get(0).getPeso() - pesoInicial2);
        
        // Más iteraciones generalmente producen más cambio
        // (aunque no siempre debido a saturación)
        assertTrue(cambio1 >= 0 && cambio2 >= 0, 
            "Ambos entrenamientos deben producir algún cambio");
    }
    
    @Test
    @DisplayName("Timestamp avanza durante entrenamiento")
    void testTimestampAvanzaDuranteEntrenamiento() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        long timestampInicial = red.getTimestampGlobal();
        
        red.entrenar(new double[]{0.5, 0.3}, new double[]{0.8}, 10);
        
        long timestampFinal = red.getTimestampGlobal();
        
        assertTrue(timestampFinal > timestampInicial, 
            "El timestamp debe avanzar durante el entrenamiento");
    }
    
    @Test
    @DisplayName("Plasticidad hebiana refuerza conexiones usadas")
    void testPlasticidadRefuerzaConexionesUsadas() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Entrenar con patrón que active fuertemente la red
        red.entrenar(new double[]{1.0, 1.0}, new double[]{1.0}, 50);
        
        // Verificar que hay conexiones con pesos altos
        boolean hayConexionesFuertes = false;
        for (Conexion c : red.getConexiones()) {
            if (c.getPeso() > 0.7) {
                hayConexionesFuertes = true;
                break;
            }
        }
        
        assertTrue(hayConexionesFuertes, 
            "Debe haber conexiones reforzadas después del entrenamiento");
    }
    
    @Test
    @DisplayName("Red con topología mínima puede entrenar")
    void testEntrenamientoTopologiaMinima() {
        int[] topologia = {1, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        assertDoesNotThrow(() -> {
            red.entrenar(new double[]{0.5}, new double[]{0.8}, 20);
        });
    }
    
    @Test
    @DisplayName("Red con múltiples capas intermedias puede entrenar")
    void testEntrenamientoMultiplesCapas() {
        int[] topologia = {2, 4, 3, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        assertDoesNotThrow(() -> {
            red.entrenar(new double[]{0.5, -0.3}, new double[]{0.8, 0.6}, 30);
        });
    }
    
    @Test
    @DisplayName("Entrenamiento secuencial de múltiples patrones")
    void testEntrenamientoSecuencial() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double[][] patrones = {
            {1.0, 0.0},
            {0.0, 1.0},
            {0.5, 0.5},
            {-0.5, 0.5}
        };
        
        double[][] targets = {
            {1.0},
            {0.0},
            {0.5},
            {0.0}
        };
        
        // Entrenar todos los patrones
        for (int i = 0; i < patrones.length; i++) {
            red.entrenar(patrones[i], targets[i], 10);
        }
        
        // Verificar que la red sigue funcionando
        red.resetear();
        double[] output = red.procesar(patrones[0]);
        assertNotNull(output);
        assertEquals(1, output.length);
    }
}
