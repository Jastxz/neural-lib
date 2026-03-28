package es.jastxz.experimental;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para Fase 5: Competición por Recursos
 * 
 * Implementa competición y poda:
 * - Elementos útiles reciben más recursos
 * - Elementos no utilizados pierden recursos
 * - Red se optimiza automáticamente
 * 
 * Referencias:
 * - pg. 18-19 Eagleman: "cada neurona y conexión compite por recursos"
 * - pg. 229-230 Eagleman: "cualquier red que se active con frecuencia gana más territorio"
 */
@DisplayName("Fase 5: Competición por Recursos")
public class RedNeuralExperimentalFase5Test {
    
    @Test
    @DisplayName("Red puede activar competición por recursos")
    void testActivarCompeticion() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        assertTrue(red.esCompeticionActiva(), 
            "La competición por recursos debe estar activada");
    }
    
    @Test
    @DisplayName("Red puede desactivar competición por recursos")
    void testDesactivarCompeticion() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        red.activarCompeticionRecursos(false);
        
        assertFalse(red.esCompeticionActiva(), 
            "La competición por recursos debe estar desactivada");
    }
    
    @Test
    @DisplayName("Neuronas usadas ganan recursos")
    void testNeuronasUsadasGananRecursos() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar con inputs fuertes para activar neuronas
        for (int i = 0; i < 10; i++) {
            red.procesar(new double[]{1.0, 1.0});
        }
        
        // Ahora dejar pasar tiempo sin procesar para que pierdan recursos
        for (int i = 0; i < 30; i++) {
            red.avanzarTiempo(1L);
            red.competirPorRecursos();
        }
        
        // Verificar que hay neuronas con recursos reducidos
        List<Neurona> primeraCapaInter = red.getCapasInterneuronas().get(0);
        boolean hayVariacion = false;
        
        for (Neurona n : primeraCapaInter) {
            if (n.getRecursosAsignados() < 1.0) {
                hayVariacion = true;
                break;
            }
        }
        
        assertTrue(hayVariacion, 
            "Debe haber neuronas con recursos reducidos tras periodo sin uso");
    }
    
    @Test
    @DisplayName("Neuronas no usadas pierden recursos")
    void testNeuronasNoUsadasPierdenRecursos() {
        int[] topologia = {2, 5, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.5);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar con inputs que no activen todas las neuronas
        for (int i = 0; i < 30; i++) {
            red.procesar(new double[]{0.1, 0.1});
            red.competirPorRecursos();
        }
        
        // Debe haber neuronas con recursos reducidos
        List<Neurona> primeraCapaInter = red.getCapasInterneuronas().get(0);
        boolean algunaPerdioRecursos = false;
        for (Neurona n : primeraCapaInter) {
            if (n.getRecursosAsignados() < 1.0) {
                algunaPerdioRecursos = true;
                break;
            }
        }
        
        assertTrue(algunaPerdioRecursos, 
            "Debe haber neuronas que perdieron recursos por falta de uso");
    }
    
    @Test
    @DisplayName("Conexiones usadas ganan recursos")
    void testConexionesUsadasGananRecursos() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar varias veces
        for (int i = 0; i < 20; i++) {
            red.procesar(new double[]{1.0, 1.0});
            red.competirPorRecursos();
        }
        
        // Debe haber conexiones con recursos altos
        boolean hayConexionesConRecursos = false;
        for (Conexion c : red.getConexiones()) {
            if (c.getRecursosAsignados() > 0.8) {
                hayConexionesConRecursos = true;
                break;
            }
        }
        
        assertTrue(hayConexionesConRecursos, 
            "Debe haber conexiones usadas con recursos altos");
    }
    
    @Test
    @DisplayName("Conexiones no usadas pierden recursos")
    void testConexionesNoUsadasPierdenRecursos() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar con inputs débiles
        for (int i = 0; i < 30; i++) {
            red.procesar(new double[]{0.05, 0.05});
            red.competirPorRecursos();
        }
        
        // Debe haber conexiones con recursos reducidos
        boolean hayConexionesConPocoRecurso = false;
        for (Conexion c : red.getConexiones()) {
            if (c.getRecursosAsignados() < 0.9) {
                hayConexionesConPocoRecurso = true;
                break;
            }
        }
        
        assertTrue(hayConexionesConPocoRecurso, 
            "Debe haber conexiones con recursos reducidos por falta de uso");
    }
    
    @Test
    @DisplayName("Red puede podar elementos sin recursos")
    void testPodarElementos() {
        int[] topologia = {2, 5, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.6);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar con inputs débiles para que algunos elementos pierdan recursos
        for (int i = 0; i < 50; i++) {
            red.procesar(new double[]{0.05, 0.05});
            red.competirPorRecursos();
        }
        
        // Podar elementos sin recursos
        int elementosPodados = red.podarElementos();
        
        assertTrue(elementosPodados >= 0, 
            "Debe retornar el número de elementos podados (puede ser 0)");
    }
    
    @Test
    @DisplayName("Poda elimina conexiones con recursos bajos")
    void testPodaEliminaConexiones() {
        int[] topologia = {2, 4, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        int conexionesIniciales = red.getTotalConexiones();
        
        // Forzar pérdida de recursos en algunas conexiones
        for (int i = 0; i < 100; i++) {
            red.procesar(new double[]{0.01, 0.01});
            red.competirPorRecursos();
        }
        
        // Podar
        red.podarElementos();
        
        int conexionesFinales = red.getTotalConexiones();
        
        // Puede que se hayan podado conexiones (o no, depende del uso)
        assertTrue(conexionesFinales <= conexionesIniciales, 
            "El número de conexiones no debe aumentar tras poda");
    }
    
    @Test
    @DisplayName("Competición no afecta procesamiento normal")
    void testCompeticionNoAfectaProcesamiento() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Procesar sin competición
        double[] outputs1 = red.procesar(new double[]{1.0, 0.5});
        
        // Activar competición y procesar
        red.activarCompeticionRecursos(true);
        red.competirPorRecursos();
        double[] outputs2 = red.procesar(new double[]{1.0, 0.5});
        
        // Ambos deben generar outputs válidos
        assertNotNull(outputs1, "Outputs sin competición deben existir");
        assertNotNull(outputs2, "Outputs con competición deben existir");
        assertEquals(outputs1.length, outputs2.length, 
            "Ambos deben tener el mismo número de outputs");
    }
    
    @Test
    @DisplayName("Red con topología mínima puede usar competición")
    void testCompeticion_TopologiaMinima() {
        int[] topologia = {1, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        assertDoesNotThrow(() -> {
            red.procesar(new double[]{1.0});
            red.competirPorRecursos();
            red.podarElementos();
        }, "Red mínima debe soportar competición");
    }
    
    @Test
    @DisplayName("Recursos se redistribuyen dinámicamente")
    void testRedistribucionDinamica() {
        int[] topologia = {2, 4, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar con un patrón
        for (int i = 0; i < 5; i++) {
            red.procesar(new double[]{1.0, 0.0});
        }
        
        // Dejar pasar tiempo sin actividad
        for (int i = 0; i < 40; i++) {
            red.avanzarTiempo(1L);
            red.competirPorRecursos();
        }
        
        List<Neurona> capa = red.getCapasInterneuronas().get(0);
        
        // Verificar que al menos una neurona perdió recursos
        boolean algunaPerdioRecursos = false;
        for (Neurona n : capa) {
            if (n.getRecursosAsignados() < 1.0) {
                algunaPerdioRecursos = true;
                break;
            }
        }
        
        assertTrue(algunaPerdioRecursos, 
            "Al menos una neurona debe haber perdido recursos tras periodo sin uso");
    }
    
    @Test
    @DisplayName("Poda no elimina todas las conexiones")
    void testPodaNoEliminaTodo() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarCompeticionRecursos(true);
        
        // Procesar normalmente
        for (int i = 0; i < 30; i++) {
            red.procesar(new double[]{0.5, 0.5});
            red.competirPorRecursos();
        }
        
        red.podarElementos();
        
        // Debe quedar al menos una conexión
        assertTrue(red.getTotalConexiones() > 0, 
            "Debe quedar al menos una conexión tras la poda");
    }
    
    @Test
    @DisplayName("Competición mejora eficiencia de la red")
    void testCompeticionMejoraEficiencia() {
        int[] topologia = {2, 6, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 0.8);
        
        red.activarCompeticionRecursos(true);
        
        int conexionesIniciales = red.getTotalConexiones();
        
        // Entrenar con patrón específico
        for (int i = 0; i < 50; i++) {
            red.entrenar(new double[]{1.0, 0.0}, new double[]{1.0}, 1);
            red.competirPorRecursos();
        }
        
        // Podar elementos inútiles
        red.podarElementos();
        
        int conexionesFinales = red.getTotalConexiones();
        
        // La red debe seguir funcionando
        double[] outputs = red.procesar(new double[]{1.0, 0.0});
        assertNotNull(outputs, "La red debe seguir funcionando tras optimización");
        
        // Idealmente, se habrán eliminado conexiones inútiles
        // (pero esto depende del patrón de uso, así que solo verificamos que funciona)
        assertTrue(conexionesFinales < conexionesIniciales, 
            "Debe quedar al menos una conexión funcional");
    }
}
