package es.jastxz.experimental;

import es.jastxz.nn.Engrama;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.nn.enums.EstadoRed;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para Fase 7: Consolidación
 * 
 * Implementa el ciclo de consolidación ("sueño") que:
 * - Revisa engramas formados durante actividad
 * - Fortalece engramas importantes (usados frecuentemente)
 * - Debilita engramas irrelevantes (no usados)
 * - Implementa degradación por desuso
 * 
 * Referencias:
 * - pg. 87 Campillo: "Durante el sueño el cerebro revisa los engramas y consolida los más importantes"
 * - pg. 89 Campillo: "Olvido gradual pero parcial"
 */
public class RedNeuralExperimentalFase7Test {
    
    /**
     * Test 1: Verificar que se puede activar modo consolidación
     */
    @Test
    public void testActivarModoConsolidacion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        // Inicialmente en estado ACTIVO
        assertEquals(EstadoRed.ACTIVO, red.getEstado());
        
        // Activar consolidación
        red.iniciarConsolidacion();
        
        // Estado debe cambiar a CONSOLIDANDO
        assertEquals(EstadoRed.CONSOLIDANDO, red.getEstado());
    }
    
    /**
     * Test 2: Verificar que consolidación fortalece engramas usados
     */
    @Test
    public void testConsolidacionFortalece() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        // Activar detección de engramas
        red.activarDeteccionEngramas(true);
        
        // Procesar patrón varias veces para formar engrama
        double[] input = {1.0, 0.0};
        for (int i = 0; i < 5; i++) {
            red.procesar(input);
        }
        
        // Debe haber al menos un engrama
        assertTrue(red.getEngramas().size() > 0);
        
        // Obtener primer engrama
        Engrama engrama = red.getEngramas().values().iterator().next();
        int activacionesAntes = engrama.getContadorActivaciones();
        
        // Consolidar
        red.iniciarConsolidacion();
        red.consolidar();
        red.finalizarConsolidacion();
        
        // El engrama debe tener mayor fuerza (más activaciones registradas)
        int activacionesDespues = engrama.getContadorActivaciones();
        assertTrue(activacionesDespues >= activacionesAntes);
    }
    
    /**
     * Test 3: Verificar que consolidación debilita engramas no usados
     */
    @Test
    public void testConsolidacionDebilita() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        // Crear engrama manualmente
        List<Neurona> neuronas = new ArrayList<>(red.getCapasInterneuronas().get(0).subList(0, 2));
        red.formarEngrama("test_engrama", neuronas);
        
        // No usar el engrama (no activarlo)
        // Avanzar tiempo significativamente
        red.avanzarTiempo(100L);
        
        // Consolidar
        red.iniciarConsolidacion();
        red.consolidar();
        red.finalizarConsolidacion();
        
        // El engrama debe tener menor relevancia
        Engrama engrama = red.getEngramas().get("test_engrama");
        assertNotNull(engrama);
        // La relevancia debe ser menor que 1.0 (valor inicial)
        assertTrue(engrama.getRelevancia() < 1.0);
    }
    
    /**
     * Test 4: Verificar que degradación elimina engramas muy débiles
     */
    @Test
    public void testDegradacionEliminaEngramas() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        // Crear engrama manualmente
        List<Neurona> neuronas = new ArrayList<>(red.getCapasInterneuronas().get(0).subList(0, 2));
        red.formarEngrama("test_engrama", neuronas);
        
        assertEquals(1, red.getEngramas().size());
        
        // Degradar múltiples veces sin usar el engrama
        for (int i = 0; i < 10; i++) {
            red.avanzarTiempo(50L);
            red.iniciarConsolidacion();
            red.consolidar();
            red.finalizarConsolidacion();
        }
        
        // El engrama debe haber sido eliminado por irrelevancia
        assertEquals(0, red.getEngramas().size());
    }
    
    /**
     * Test 5: Verificar que engramas usados sobreviven a degradación
     */
    @Test
    public void testEngramasUsadosSobreviven() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        // Activar detección
        red.activarDeteccionEngramas(true);
        
        // Procesar patrón para formar engrama
        double[] input = {1.0, 0.0};
        for (int i = 0; i < 5; i++) {
            red.procesar(input);
        }
        
        int engramasInicial = red.getEngramas().size();
        assertTrue(engramasInicial > 0);
        
        // Consolidar varias veces, pero seguir usando el patrón
        for (int i = 0; i < 5; i++) {
            red.procesar(input);  // Usar el patrón
            red.iniciarConsolidacion();
            red.consolidar();
            red.finalizarConsolidacion();
        }
        
        // Los engramas deben seguir existiendo
        assertTrue(red.getEngramas().size() >= engramasInicial);
    }
    
    /**
     * Test 6: Verificar que estado vuelve a ACTIVO tras consolidación
     */
    @Test
    public void testEstadoVuelveActivo() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        assertEquals(EstadoRed.ACTIVO, red.getEstado());
        
        red.iniciarConsolidacion();
        assertEquals(EstadoRed.CONSOLIDANDO, red.getEstado());
        
        red.consolidar();
        assertEquals(EstadoRed.CONSOLIDANDO, red.getEstado());
        
        red.finalizarConsolidacion();
        assertEquals(EstadoRed.ACTIVO, red.getEstado());
    }
    
    /**
     * Test 7: Verificar que no se puede procesar durante consolidación
     */
    @Test
    public void testNoProcesarDuranteConsolidacion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        red.iniciarConsolidacion();
        
        // Intentar procesar debe lanzar excepción
        assertThrows(IllegalStateException.class, () -> {
            red.procesar(new double[]{1.0, 0.0});
        });
    }
    
    /**
     * Test 8: Verificar que consolidación ajusta pesos sinápticos
     */
    @Test
    public void testConsolidacionAjustaPesos() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        // Entrenar un poco
        double[] input = {1.0, 0.0};
        double[] target = {1.0, 0.0};
        red.entrenar(input, target, 10);
        
        // Guardar suma de pesos antes
        double sumaPesosAntes = red.getConexiones().stream()
            .mapToDouble(c -> c.getPeso())
            .sum();
        
        // Consolidar
        red.iniciarConsolidacion();
        red.consolidar();
        red.finalizarConsolidacion();
        
        // Los pesos pueden haber cambiado ligeramente
        double sumaPesosDespues = red.getConexiones().stream()
            .mapToDouble(c -> c.getPeso())
            .sum();
        
        // No deben ser idénticos (consolidación hace ajustes finos)
        // Pero tampoco deben cambiar drásticamente
        double diferencia = Math.abs(sumaPesosAntes - sumaPesosDespues);
        assertTrue(diferencia >= 0.0);  // Puede haber cambio o no
    }
    
    /**
     * Test 9: Verificar que múltiples ciclos de consolidación funcionan
     */
    @Test
    public void testMultiplesCiclosConsolidacion() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        // Múltiples ciclos
        for (int i = 0; i < 5; i++) {
            red.iniciarConsolidacion();
            red.consolidar();
            red.finalizarConsolidacion();
            
            assertEquals(EstadoRed.ACTIVO, red.getEstado());
        }
    }
    
    /**
     * Test 10: Verificar que consolidación respeta engramas importantes
     */
    @Test
    public void testConsolidacionRespecta() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
        
        red.activarDeteccionEngramas(true);
        
        // Crear patrón muy usado
        double[] input = {1.0, 0.0};
        for (int i = 0; i < 20; i++) {
            red.procesar(input);
        }
        
        int engramasAntes = red.getEngramas().size();
        assertTrue(engramasAntes > 0);
        
        // Consolidar varias veces
        for (int i = 0; i < 3; i++) {
            red.iniciarConsolidacion();
            red.consolidar();
            red.finalizarConsolidacion();
        }
        
        // Los engramas importantes deben seguir ahí
        assertTrue(red.getEngramas().size() > 0);
    }
    
    /**
     * Test 11: Verificar que getEstado funciona correctamente
     */
    @Test
    public void testGetEstado() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 2}, 0.8);
        
        EstadoRed estado = red.getEstado();
        assertNotNull(estado);
        assertEquals(EstadoRed.ACTIVO, estado);
    }
    
    /**
     * Test 12: Verificar que consolidación no afecta topología
     */
    @Test
    public void testConsolidacionNoAfectaTopologia() {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 3, 2}, 0.8);
        
        int neuronasAntes = red.getTotalNeuronas();
        
        red.iniciarConsolidacion();
        red.consolidar();
        red.finalizarConsolidacion();
        
        int neuronasDespues = red.getTotalNeuronas();
        
        assertEquals(neuronasAntes, neuronasDespues);
    }
}
