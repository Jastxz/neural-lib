package es.jastxz.experimental;

import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para Fase 8: Utilidades y Debugging
 * 
 * Implementa herramientas de monitorización y análisis:
 * - Estadísticas generales de la red
 * - Visualización de activaciones
 * - Análisis de engramas
 * - Reporte de recursos
 */
public class RedNeuralExperimentalFase8Test {
    
    /**
     * Test 1: Verificar que getEstadisticas devuelve información completa
     */
    @Test
    public void testGetEstadisticas() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{3, 5, 4, 2}, 0.7);
        
        Map<String, Object> stats = red.getEstadisticas();
        
        assertNotNull(stats);
        assertTrue(stats.containsKey("totalNeuronas"));
        assertTrue(stats.containsKey("totalConexiones"));
        assertTrue(stats.containsKey("totalEngramas"));
        assertTrue(stats.containsKey("timestampGlobal"));
        assertTrue(stats.containsKey("estado"));
        assertTrue(stats.containsKey("topologia"));
        
        assertEquals(14, stats.get("totalNeuronas"));  // 3+5+4+2
        assertTrue((int)stats.get("totalConexiones") > 0);
    }
    
    /**
     * Test 2: Verificar que visualizarActivaciones muestra estado de neuronas
     */
    @Test
    public void testVisualizarActivaciones() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        // Procesar algo para activar neuronas
        red.procesar(new double[]{1.0, 0.0});
        
        String visualizacion = red.visualizarActivaciones();
        
        assertNotNull(visualizacion);
        assertTrue(visualizacion.contains("Capa Sensorial"));
        assertTrue(visualizacion.contains("Capa Motora"));
        assertTrue(visualizacion.length() > 50);  // Debe tener contenido significativo
    }
    
    /**
     * Test 3: Verificar que analizarEngramas proporciona información detallada
     */
    @Test
    public void testAnalizarEngramas() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        // Sin engramas
        String analisis = red.analizarEngramas();
        assertNotNull(analisis);
        assertTrue(analisis.contains("0") || analisis.contains("Total"));
        
        // Activar detección y crear engramas
        red.activarDeteccionEngramas(true);
        red.procesar(new double[]{1.0, 0.0});
        red.procesar(new double[]{1.0, 0.0});
        
        analisis = red.analizarEngramas();
        assertTrue(analisis.contains("engrama") || analisis.contains("Engrama") || analisis.contains("Total"));
    }
    
    /**
     * Test 4: Verificar que reporteRecursos muestra asignación de recursos
     */
    @Test
    public void testReporteRecursos() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        String reporte = red.reporteRecursos();
        
        assertNotNull(reporte);
        assertTrue(reporte.contains("Recursos"));
        assertTrue(reporte.contains("Neuronas") || reporte.contains("neuronas"));
        assertTrue(reporte.contains("Conexiones") || reporte.contains("conexiones"));
    }
    
    /**
     * Test 5: Verificar que estadísticas se actualizan tras procesamiento
     */
    @Test
    public void testEstadisticasSeActualizan() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        Map<String, Object> statsAntes = red.getEstadisticas();
        long timestampAntes = (long)statsAntes.get("timestampGlobal");
        
        red.procesar(new double[]{1.0, 0.0});
        
        Map<String, Object> statsDespues = red.getEstadisticas();
        long timestampDespues = (long)statsDespues.get("timestampGlobal");
        
        assertTrue(timestampDespues > timestampAntes);
    }
    
    /**
     * Test 6: Verificar que estadísticas incluyen información de predicción
     */
    @Test
    public void testEstadisticasConPrediccion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        red.activarModoPredictivo(true);
        red.procesar(new double[]{1.0, 0.0});
        
        Map<String, Object> stats = red.getEstadisticas();
        
        assertTrue(stats.containsKey("modoPredictivo"));
        assertEquals(true, stats.get("modoPredictivo"));
    }
    
    /**
     * Test 7: Verificar que estadísticas incluyen información de competición
     */
    @Test
    public void testEstadisticasConCompeticion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        red.activarCompeticionRecursos(true);
        
        Map<String, Object> stats = red.getEstadisticas();
        
        assertTrue(stats.containsKey("competicionActiva"));
        assertEquals(true, stats.get("competicionActiva"));
    }
    
    /**
     * Test 8: Verificar que reporte de recursos muestra promedios
     */
    @Test
    public void testReporteRecursosConPromedios() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        // Activar competición y procesar
        red.activarCompeticionRecursos(true);
        red.procesar(new double[]{1.0, 0.0});
        red.competirPorRecursos();
        
        String reporte = red.reporteRecursos();
        
        assertTrue(reporte.contains("promedio") || reporte.contains("Promedio"));
    }
    
    /**
     * Test 9: Verificar que análisis de engramas muestra detalles
     */
    @Test
    public void testAnalisisEngramasDetallado() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        // Crear engrama manualmente
        red.formarEngrama("test", 
            new java.util.ArrayList<>(red.getCapasInterneuronas().get(0).subList(0, 2)));
        
        String analisis = red.analizarEngramas();
        
        assertTrue(analisis.contains("test") || analisis.contains("1 engrama"));
        assertTrue(analisis.contains("relevancia") || analisis.contains("Relevancia"));
    }
    
    /**
     * Test 10: Verificar que visualización muestra capas intermedias
     */
    @Test
    public void testVisualizacionCapasIntermedias() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 4, 2}, 0.8);
        
        red.procesar(new double[]{1.0, 0.0});
        
        String visualizacion = red.visualizarActivaciones();
        
        assertTrue(visualizacion.contains("Inter") || visualizacion.contains("Capa"));
    }
    
    /**
     * Test 11: Verificar que estadísticas incluyen densidad de conexiones
     */
    @Test
    public void testEstadisticasDensidad() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.7);
        
        Map<String, Object> stats = red.getEstadisticas();
        
        assertTrue(stats.containsKey("densidadConexiones"));
        assertEquals(0.7, (double)stats.get("densidadConexiones"), 0.01);
    }
    
    /**
     * Test 12: Verificar que toString proporciona resumen útil
     */
    @Test
    public void testToString() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        String resumen = red.toString();
        
        assertNotNull(resumen);
        assertTrue(resumen.contains("RedNeuralExperimental"));
        assertTrue(resumen.contains("Topología") || resumen.contains("topología"));
    }
    
    /**
     * Test 13: Verificar que estadísticas funcionan tras consolidación
     */
    @Test
    public void testEstadisticasTrasConsolidacion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        red.iniciarConsolidacion();
        red.consolidar();
        red.finalizarConsolidacion();
        
        Map<String, Object> stats = red.getEstadisticas();
        
        assertNotNull(stats);
        assertEquals("ACTIVO", stats.get("estado").toString());
    }
    
    /**
     * Test 14: Verificar que reporte de recursos funciona sin competición
     */
    @Test
    public void testReporteRecursosSinCompeticion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        String reporte = red.reporteRecursos();
        
        assertNotNull(reporte);
        assertTrue(reporte.length() > 0);
    }
}
