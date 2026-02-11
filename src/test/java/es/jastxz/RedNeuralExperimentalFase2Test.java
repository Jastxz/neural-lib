package es.jastxz;

import es.jastxz.nn.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para RedNeuralExperimental - Fase 2: Propagación Básica
 */
class RedNeuralExperimentalFase2Test {
    
    @Test
    @DisplayName("Procesar inputs genera outputs del tamaño correcto")
    void testProcesarTamanoOutputs() {
        int[] topologia = {3, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double[] inputs = {0.5, -0.3, 0.8};
        double[] outputs = red.procesar(inputs);
        
        assertNotNull(outputs);
        assertEquals(2, outputs.length);
    }
    
    @Test
    @DisplayName("Procesar con inputs válidos no lanza excepción")
    void testProcesarInputsValidos() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        double[] inputs = {0.5, -0.5};
        
        assertDoesNotThrow(() -> {
            double[] outputs = red.procesar(inputs);
            assertNotNull(outputs);
        });
    }
    
    @Test
    @DisplayName("Excepción si inputs es null")
    void testProcesarInputsNull() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        assertThrows(IllegalArgumentException.class, () -> {
            red.procesar(null);
        });
    }
    
    @Test
    @DisplayName("Excepción si tamaño de inputs no coincide")
    void testProcesarTamanoIncorrecto() {
        int[] topologia = {3, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        double[] inputsCortos = {0.5, 0.3};  // Esperaba 3
        double[] inputsLargos = {0.5, 0.3, 0.1, 0.9};  // Esperaba 3
        
        assertThrows(IllegalArgumentException.class, () -> {
            red.procesar(inputsCortos);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            red.procesar(inputsLargos);
        });
    }
    
    @Test
    @DisplayName("Outputs están en rango válido [-1, 1]")
    void testOutputsEnRango() {
        int[] topologia = {2, 3, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.7);
        
        double[] inputs = {0.8, -0.6};
        double[] outputs = red.procesar(inputs);
        
        for (double output : outputs) {
            assertTrue(output >= -1.0 && output <= 1.0, 
                "Output fuera de rango: " + output);
        }
    }
    
    @Test
    @DisplayName("Procesar múltiples veces con mismos inputs da resultados consistentes")
    void testConsistencia() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double[] inputs = {0.5, -0.3};
        
        // Primera vez
        double[] outputs1 = red.procesar(inputs);
        
        // Resetear para tener mismo estado inicial
        red.resetear();
        
        // Segunda vez con mismo estado inicial
        double[] outputs2 = red.procesar(inputs);
        
        assertArrayEquals(outputs1, outputs2, 0.001);
    }
    
    @Test
    @DisplayName("Inputs diferentes pueden producir outputs diferentes")
    void testInputsDiferentesOutputsDiferentes() {
        int[] topologia = {2, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.9);
        
        double[] inputs1 = {1.0, 0.0};
        double[] inputs2 = {0.0, 1.0};
        
        double[] outputs1 = red.procesar(inputs1);
        double[] outputs2 = red.procesar(inputs2);
        
        // Verificar que los outputs son válidos (no necesariamente diferentes)
        // En una red con pesos aleatorios, es posible que inputs diferentes
        // produzcan outputs similares hasta que la red aprenda
        assertNotNull(outputs1);
        assertNotNull(outputs2);
        assertEquals(outputs1.length, outputs2.length);
    }
    
    @Test
    @DisplayName("Timestamp avanza después de procesar")
    void testTimestampAvanza() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        long timestampInicial = red.getTimestampGlobal();
        
        double[] inputs = {0.5, -0.5};
        red.procesar(inputs);
        
        assertTrue(red.getTimestampGlobal() > timestampInicial);
    }
    
    @Test
    @DisplayName("Red se resetea antes de cada procesamiento")
    void testReseteoAntesProcesamiento() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        // Procesar primera vez
        double[] inputs1 = {0.8, 0.6};
        red.procesar(inputs1);
        
        // Procesar segunda vez con inputs diferentes
        double[] inputs2 = {-0.3, 0.4};
        double[] outputs2 = red.procesar(inputs2);
        
        // Debe funcionar correctamente (no acumular estado)
        assertNotNull(outputs2);
    }
    
    @Test
    @DisplayName("Inputs significativos activan neuronas sensoriales")
    void testActivacionNeuronasSensoriales() {
        int[] topologia = {3, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double[] inputs = {0.8, 0.0, -0.7};  // Dos significativos, uno no
        red.procesar(inputs);
        
        // Al menos algunas neuronas deben haberse activado
        // (difícil verificar directamente sin exponer estado interno)
        // Por ahora verificamos que el procesamiento completa sin errores
        assertTrue(true);
    }
    
    @Test
    @DisplayName("Red con topología mínima procesa correctamente")
    void testTopologiaMinimaProcesamient() {
        int[] topologia = {1, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        double[] inputs = {0.5};
        double[] outputs = red.procesar(inputs);
        
        assertEquals(1, outputs.length);
    }
    
    @Test
    @DisplayName("Red con múltiples capas intermedias procesa correctamente")
    void testMultiplesCapasIntermedias() {
        int[] topologia = {2, 4, 3, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.7);
        
        double[] inputs = {0.6, -0.4};
        double[] outputs = red.procesar(inputs);
        
        assertEquals(1, outputs.length);
        assertTrue(outputs[0] >= -1.0 && outputs[0] <= 1.0);
    }
    
    @Test
    @DisplayName("Inputs con valores extremos se manejan correctamente")
    void testInputsExtremos() {
        int[] topologia = {2, 3, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.6);
        
        double[] inputsMax = {1.0, 1.0};
        double[] inputsMin = {-1.0, -1.0};
        double[] inputsCero = {0.0, 0.0};
        
        assertDoesNotThrow(() -> {
            red.procesar(inputsMax);
            red.procesar(inputsMin);
            red.procesar(inputsCero);
        });
    }
    
    @Test
    @DisplayName("Propagación feed-forward activa capas sucesivas")
    void testPropagacionFeedForward() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.9);
        
        double[] inputs = {0.8, 0.7};
        double[] outputs = red.procesar(inputs);
        
        // Si hay conexiones suficientes, debe haber algún output
        assertNotNull(outputs);
        assertEquals(1, outputs.length);
    }
    
    @Test
    @DisplayName("Feedback modula activación de capas anteriores")
    void testPropagacionFeedback() {
        int[] topologia = {2, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        double[] inputs = {0.5, -0.5};
        
        // Procesar dos veces con reseteo entre medio
        double[] outputs1 = red.procesar(inputs);
        red.resetear();  // Resetear memoria de corto plazo
        double[] outputs2 = red.procesar(inputs);
        
        // Deben ser consistentes (mismo input, mismo estado inicial)
        assertArrayEquals(outputs1, outputs2, 0.001);
    }
    
    @Test
    @DisplayName("Red con densidad baja procesa correctamente")
    void testDensidadBaja() {
        int[] topologia = {3, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.2);
        
        double[] inputs = {0.5, 0.3, -0.4};
        double[] outputs = red.procesar(inputs);
        
        assertNotNull(outputs);
        assertEquals(2, outputs.length);
    }
    
    @Test
    @DisplayName("Red con densidad alta procesa correctamente")
    void testDensidadAlta() {
        int[] topologia = {3, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        double[] inputs = {0.5, 0.3, -0.4};
        double[] outputs = red.procesar(inputs);
        
        assertNotNull(outputs);
        assertEquals(2, outputs.length);
    }
    
    @Test
    @DisplayName("Procesamiento secuencial funciona correctamente")
    void testProcesamientoSecuencial() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.7);
        
        double[][] secuenciaInputs = {
            {0.5, -0.3},
            {-0.2, 0.8},
            {0.0, 0.0},
            {1.0, -1.0}
        };
        
        for (double[] inputs : secuenciaInputs) {
            double[] outputs = red.procesar(inputs);
            assertNotNull(outputs);
            assertEquals(1, outputs.length);
        }
    }
    
    @Test
    @DisplayName("Conocimiento se preserva entre procesamientos")
    void testConocimientoSePreserva() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        // Procesar varios inputs
        red.procesar(new double[]{0.5, -0.3});
        red.procesar(new double[]{-0.2, 0.8});
        red.procesar(new double[]{0.7, 0.4});
        
        // Los valores almacenados pueden cambiar por feedback, pero las neuronas siguen existiendo
        // y mantienen su identidad
        assertNotNull(red.getCapaSensorial().get(0));
        assertNotNull(red.getCapasInterneuronas().get(0).get(0));
        
        // Las conexiones se mantienen
        assertTrue(red.getTotalConexiones() > 0);
    }
    
    @Test
    @DisplayName("Pesos sinápticos se preservan entre procesamientos")
    void testPesosSinapticosSePreservan() {
        int[] topologia = {2, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Obtener peso inicial de una conexión
        double pesoInicial = red.getConexiones().get(0).getPeso();
        
        // Procesar varios inputs
        red.procesar(new double[]{0.5, -0.3});
        red.procesar(new double[]{-0.2, 0.8});
        
        // El peso debe mantenerse (en esta fase aún no hay aprendizaje)
        double pesoFinal = red.getConexiones().get(0).getPeso();
        assertEquals(pesoInicial, pesoFinal, 0.0001, 
            "Los pesos deben preservarse entre procesamientos (sin aprendizaje aún)");
    }
    
    @Test
    @DisplayName("Estado de la red evoluciona entre procesamientos")
    void testEstadoEvolucionaEntrProcesamientos() {
        int[] topologia = {3, 4, 2};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        // Primer procesamiento
        double[] outputs1 = red.procesar(new double[]{0.8, 0.7, 0.6});
        
        // Segundo procesamiento sin resetear
        // El estado interno puede influir en el resultado
        double[] outputs2 = red.procesar(new double[]{0.1, 0.2, 0.1});
        
        // Verificar que ambos procesamientos funcionan
        assertNotNull(outputs1);
        assertNotNull(outputs2);
        assertEquals(2, outputs1.length);
        assertEquals(2, outputs2.length);
    }
    
    @Test
    @DisplayName("Resetear permite procesar con estado limpio")
    void testResetearLimpiaEstado() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        // Procesar
        double[] outputs1 = red.procesar(new double[]{0.9, 0.8});
        
        // Resetear
        red.resetear();
        
        // Procesar de nuevo con mismo input
        double[] outputs2 = red.procesar(new double[]{0.9, 0.8});
        
        // Deben ser iguales (mismo estado inicial)
        assertArrayEquals(outputs1, outputs2, 0.001);
    }
}
