package es.jastxz.experimental;

import es.jastxz.nn.experimental.GestorConsolidacionAdaptativa;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para GestorConsolidacionAdaptativa
 * Valida medición de tiempos, cálculo de intervalo adaptativo y ventana deslizante
 */
class GestorConsolidacionAdaptativaTest {
    
    private GestorConsolidacionAdaptativa gestor;
    
    @BeforeEach
    void setUp() {
        gestor = new GestorConsolidacionAdaptativa();
    }
    
    @Test
    @DisplayName("Gestor se inicializa correctamente")
    void testInicializacion() {
        assertFalse(gestor.estaInicializado(), "No debe estar inicializado al crear");
        assertEquals(0, gestor.getContadorIteraciones(), "Contador debe ser 0");
        assertEquals(50, gestor.getIntervaloActual(), "Intervalo inicial debe ser 50");
        assertEquals(0.0, gestor.getTiempoPromedioMs(), 0.001, "Tiempo promedio debe ser 0");
    }
    
    @Test
    @DisplayName("Ventana deslizante mantiene tamaño máximo de 10")
    void testVentanaDeslizante() {
        // Registrar 15 tiempos
        for (int i = 0; i < 15; i++) {
            gestor.registrarTiempo(10L);
        }
        
        // La ventana debe tener máximo 10 elementos
        assertEquals(10, gestor.getTamañoVentana(), "Ventana debe tener máximo 10 elementos");
    }
    
    @Test
    @DisplayName("Inicialización ocurre después de 10 iteraciones")
    void testInicializacionDespuesDe10Iteraciones() {
        // Registrar 9 tiempos
        for (int i = 0; i < 9; i++) {
            gestor.registrarTiempo(20L);
            assertFalse(gestor.estaInicializado(), "No debe inicializar antes de 10 iteraciones");
        }
        
        // Registrar el décimo tiempo
        gestor.registrarTiempo(20L);
        assertTrue(gestor.estaInicializado(), "Debe inicializar después de 10 iteraciones");
    }
    
    @Test
    @DisplayName("Cálculo de intervalo con fórmula correcta")
    void testCalculoIntervalo() {
        // Registrar 10 tiempos de 40ms cada uno
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(40L);
        }
        
        // Promedio = 40ms, intervalo = max(10, min(100, 40/2)) = 20
        assertEquals(20, gestor.getIntervaloActual(), "Intervalo debe ser 20 con promedio de 40ms");
        assertEquals(40.0, gestor.getTiempoPromedioMs(), 0.001, "Tiempo promedio debe ser 40ms");
    }
    
    @Test
    @DisplayName("Intervalo mínimo es 10")
    void testIntervaloMinimo() {
        // Registrar tiempos muy pequeños (5ms)
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(5L);
        }
        
        // Promedio = 5ms, intervalo = max(10, min(100, 5/2)) = 10
        assertEquals(10, gestor.getIntervaloActual(), "Intervalo mínimo debe ser 10");
    }
    
    @Test
    @DisplayName("Intervalo máximo es 100")
    void testIntervaloMaximo() {
        // Registrar tiempos muy grandes (500ms)
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(500L);
        }
        
        // Promedio = 500ms, intervalo = max(10, min(100, 500/2)) = 100
        assertEquals(100, gestor.getIntervaloActual(), "Intervalo máximo debe ser 100");
    }
    
    @Test
    @DisplayName("Ajuste continuo cada 10 iteraciones")
    void testAjusteContinuo() {
        // Primera fase: 10 iteraciones de 40ms
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(40L);
        }
        assertEquals(20, gestor.getIntervaloActual(), "Intervalo inicial debe ser 20");
        
        // Segunda fase: 10 iteraciones de 80ms (ventana deslizante)
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(80L);
        }
        
        // Ahora el promedio es 80ms, intervalo = 40
        assertEquals(40, gestor.getIntervaloActual(), "Intervalo debe ajustarse a 40");
        assertEquals(80.0, gestor.getTiempoPromedioMs(), 0.001, "Tiempo promedio debe ser 80ms");
    }
    
    @Test
    @DisplayName("debeConsolidar usa intervalo inicial antes de inicializar")
    void testDebeConsolidarAntesDeInicializar() {
        // Antes de inicializar, usa intervalo inicial (50)
        for (int i = 1; i <= 100; i++) {
            gestor.registrarTiempo(10L);
            
            if (i % 50 == 0 && i < 100) {
                assertTrue(gestor.debeConsolidar(), 
                    "Debe consolidar cada 50 iteraciones antes de inicializar");
            }
        }
    }
    
    @Test
    @DisplayName("debeConsolidar usa intervalo adaptativo después de inicializar")
    void testDebeConsolidarDespuesDeInicializar() {
        // Inicializar con tiempos de 40ms -> intervalo = 20
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(40L);
        }
        
        assertTrue(gestor.estaInicializado(), "Debe estar inicializado");
        assertEquals(20, gestor.getIntervaloActual(), "Intervalo debe ser 20");
        
        // Registrar 10 iteraciones más
        for (int i = 0; i < 10; i++) {
            gestor.registrarTiempo(40L);
        }
        
        // En iteración 20 debe consolidar (múltiplo de 20)
        assertTrue(gestor.debeConsolidar(), "Debe consolidar en iteración 20");
    }
    
    @Test
    @DisplayName("Tiempo promedio se calcula correctamente con valores mixtos")
    void testTiempoPromedioMixto() {
        // Registrar tiempos variados
        gestor.registrarTiempo(10L);
        gestor.registrarTiempo(20L);
        gestor.registrarTiempo(30L);
        gestor.registrarTiempo(40L);
        gestor.registrarTiempo(50L);
        
        // Promedio = (10+20+30+40+50)/5 = 30
        assertEquals(30.0, gestor.getTiempoPromedioMs(), 0.001, 
            "Tiempo promedio debe ser 30ms");
    }
    
    @Test
    @DisplayName("Contador de iteraciones incrementa correctamente")
    void testContadorIteraciones() {
        assertEquals(0, gestor.getContadorIteraciones());
        
        for (int i = 1; i <= 25; i++) {
            gestor.registrarTiempo(10L);
            assertEquals(i, gestor.getContadorIteraciones(), 
                "Contador debe ser " + i);
        }
    }
}
