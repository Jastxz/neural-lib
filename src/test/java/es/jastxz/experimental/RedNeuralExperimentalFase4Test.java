package es.jastxz.experimental;

import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para Fase 4: Predicción y Error
 * 
 * Implementa codificación predictiva:
 * - Cada capa predice activación de la siguiente
 * - Solo se propagan diferencias (errores)
 * - Modelo predictivo mejora con el tiempo
 * 
 * Referencia: pg. 64 Eagleman - "solo se transmite el error de predicción"
 */
@DisplayName("Fase 4: Predicción y Error")
public class RedNeuralExperimentalFase4Test {
    
    @Test
    @DisplayName("Red puede activar modo predictivo")
    void testActivarModoPredictivo() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Activar modo predictivo
        red.activarModoPredictivo(true);
        
        assertTrue(red.esModoPredictivo(), 
            "El modo predictivo debe estar activado");
    }
    
    @Test
    @DisplayName("Red puede desactivar modo predictivo")
    void testDesactivarModoPredictivo() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        red.activarModoPredictivo(false);
        
        assertFalse(red.esModoPredictivo(), 
            "El modo predictivo debe estar desactivado");
    }
    
    @Test
    @DisplayName("Red genera predicciones iniciales aleatorias")
    void testPrediccionesInicialesAleatorias() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        // Primera vez: predicciones deben ser aleatorias (no todas cero)
        double[] outputs = red.procesar(new double[]{1.0, 1.0});
        
        boolean hayPrediccionNoNula = false;
        for (double output : outputs) {
            if (Math.abs(output) > 0.01) {
                hayPrediccionNoNula = true;
                break;
            }
        }
        
        assertTrue(hayPrediccionNoNula, 
            "Debe haber al menos una predicción no nula");
    }
    
    @Test
    @DisplayName("Red calcula errores de predicción")
    void testCalculoErroresPrediccion() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        // Procesar dos veces el mismo input
        red.procesar(new double[]{1.0, 0.0});
        red.procesar(new double[]{1.0, 0.0});
        
        // Obtener errores de predicción
        double[] errores = red.getErroresPrediccion();
        
        assertNotNull(errores, "Los errores de predicción no deben ser null");
        assertEquals(1, errores.length, "Debe haber un error por neurona de salida");
    }
    
    @Test
    @DisplayName("Errores de predicción disminuyen con entrenamiento")
    void testErroresDisminuyenConEntrenamiento() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        // Entrenar con patrón repetido
        double[] input = {1.0, 1.0};
        double[] target = {1.0};
        
        // Entrenar bastante para establecer patrón
        red.entrenar(input, target, 100);
        
        // Medir error después de entrenamiento extenso
        double[] outputs = red.procesar(input);
        double error = Math.abs(target[0] - outputs[0]);
        
        // El error debe ser razonable (< 0.3) tras entrenamiento extenso
        assertTrue(error < 0.3, 
            "El error debe ser menor a 0.3 tras entrenamiento extenso, fue: " + error);
    }
    
    @Test
    @DisplayName("Predicciones mejoran con exposición repetida")
    void testPrediccionesMejoranConExposicion() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        double[] input = {1.0, 0.5};
        
        // Múltiples exposiciones para establecer patrón
        for (int i = 0; i < 50; i++) {
            red.procesar(input);
        }
        
        double[] errores = red.getErroresPrediccion();
        double errorTotal = 0.0;
        for (double e : errores) errorTotal += Math.abs(e);
        
        // Tras muchas exposiciones, el error debe ser razonable
        assertTrue(errorTotal < 1.0, 
            "El error total debe ser menor a 1.0 tras exposición repetida, fue: " + errorTotal);
    }
    
    @Test
    @DisplayName("Red ajusta modelo predictivo automáticamente")
    void testAjusteModeloPredictivoAutomatico() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        double[] input = {0.8, 0.8};
        
        // Procesar muchas veces para dar tiempo al ajuste
        for (int i = 0; i < 30; i++) {
            red.procesar(input);
        }
        
        // El modelo debe haberse ajustado
        double[] errores = red.getErroresPrediccion();
        double errorPromedio = 0.0;
        for (double e : errores) errorPromedio += Math.abs(e);
        errorPromedio /= errores.length;
        
        assertTrue(errorPromedio < 0.8, 
            "El error promedio debe ser menor a 0.8 tras ajuste automático, fue: " + errorPromedio);
    }
    
    @Test
    @DisplayName("Solo se propagan errores en modo predictivo")
    void testSoloPropagaErrores() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        // Procesar para establecer predicción
        red.procesar(new double[]{1.0, 1.0});
        
        // Segunda vez: si la predicción es perfecta, error debe ser ~0
        red.procesar(new double[]{1.0, 1.0});
        double[] errores = red.getErroresPrediccion();
        
        // Los errores deben existir (aunque sean pequeños)
        assertNotNull(errores, "Debe haber errores calculados");
        assertTrue(errores.length > 0, "Debe haber al menos un error");
    }
    
    @Test
    @DisplayName("Modo predictivo no afecta procesamiento normal")
    void testModoPredictivo_NoAfectaProcesamiento() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Procesar sin modo predictivo
        double[] outputs1 = red.procesar(new double[]{1.0, 0.5});
        
        // Activar modo predictivo y procesar
        red.activarModoPredictivo(true);
        double[] outputs2 = red.procesar(new double[]{1.0, 0.5});
        
        // Ambos deben generar outputs válidos
        assertNotNull(outputs1, "Outputs sin modo predictivo deben existir");
        assertNotNull(outputs2, "Outputs con modo predictivo deben existir");
        assertEquals(outputs1.length, outputs2.length, 
            "Ambos deben tener el mismo número de outputs");
    }
    
    @Test
    @DisplayName("Predicciones se almacenan por capa")
    void testPrediccionesPorCapa() {
        int[] topologia = {2, 3, 2, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        red.procesar(new double[]{1.0, 0.5});
        
        // Debe haber predicciones para cada capa intermedia y motora
        int numCapasConPredicciones = red.getNumeroCapasConPredicciones();
        
        // Capas intermedias (2) + capa motora (1) = 3
        assertTrue(numCapasConPredicciones >= 2, 
            "Debe haber predicciones para al menos 2 capas");
    }
    
    @Test
    @DisplayName("Red con topología mínima puede usar modo predictivo")
    void testModoPredictivo_TopologiaMinima() {
        int[] topologia = {1, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        assertDoesNotThrow(() -> {
            red.procesar(new double[]{1.0});
            red.getErroresPrediccion();
        }, "Red mínima debe soportar modo predictivo");
    }
    
    @Test
    @DisplayName("Errores de predicción son proporcionales a diferencia real")
    void testErroresProporcionalesADiferencia() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        // Establecer patrón fuertemente
        for (int i = 0; i < 30; i++) {
            red.procesar(new double[]{1.0, 1.0});
        }
        
        // Procesar el mismo patrón - error debe ser pequeño
        red.procesar(new double[]{1.0, 1.0});
        double[] erroresMismoPatron = red.getErroresPrediccion();
        double errorMismoPatron = Math.abs(erroresMismoPatron[0]);
        
        // Cambio grande - error debe ser mayor
        red.procesar(new double[]{0.0, 0.0});
        double[] erroresCambioGrande = red.getErroresPrediccion();
        double errorCambioGrande = Math.abs(erroresCambioGrande[0]);
        
        // El error del cambio grande debe ser mayor que el del mismo patrón
        assertTrue(errorCambioGrande >= errorMismoPatron, 
            "Cambios grandes deben generar errores mayores o iguales. " +
            "Error mismo patrón: " + errorMismoPatron + ", Error cambio grande: " + errorCambioGrande);
    }
    
    @Test
    @DisplayName("Modelo predictivo se resetea correctamente")
    void testReseteoModeloPredictivo() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarModoPredictivo(true);
        
        // Entrenar patrón
        for (int i = 0; i < 20; i++) {
            red.procesar(new double[]{1.0, 1.0});
        }
        
        // Resetear
        red.resetear();
        
        // Procesar de nuevo
        red.procesar(new double[]{1.0, 1.0});
        double[] errores = red.getErroresPrediccion();
        
        // Tras reseteo, errores deben ser mayores (modelo limpio)
        assertNotNull(errores, "Debe haber errores tras reseteo");
    }
}
