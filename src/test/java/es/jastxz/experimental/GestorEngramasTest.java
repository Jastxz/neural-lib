package es.jastxz.experimental;

import es.jastxz.nn.Engrama;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoNeurona;
import es.jastxz.nn.experimental.GestorEngramas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para GestorEngramas con nuevas funcionalidades
 * Valida límite del 22%, división, fusión, clustering y métricas
 */
class GestorEngramasTest {
    
    private GestorEngramas gestor;
    private List<Neurona> neuronas;
    private static final int TOTAL_NEURONAS = 100;
    
    @BeforeEach
    void setUp() {
        gestor = new GestorEngramas(TOTAL_NEURONAS);
        
        // Crear neuronas de prueba
        neuronas = new ArrayList<>();
        for (int i = 0; i < TOTAL_NEURONAS; i++) {
            neuronas.add(new Neurona(i, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO));
        }
    }
    
    @Test
    @DisplayName("Gestor se inicializa correctamente")
    void testInicializacion() {
        assertFalse(gestor.esDeteccionActiva(), "Detección debe estar desactivada");
        assertEquals(0, gestor.getEngramas().size(), "No debe haber engramas inicialmente");
        
        GestorEngramas.EstadisticasEngramas stats = gestor.getEstadisticas();
        assertEquals(0, stats.totalActuales);
        assertEquals(0, stats.totalFormados);
        assertEquals(0, stats.totalFusionados);
        assertEquals(0, stats.totalPodados);
    }
    
    @Test
    @DisplayName("Límite del 22% se aplica correctamente")
    void testLimite22Porciento() {
        // Límite = 100 * 0.22 = 22 neuronas
        int limite = 22;
        
        // Crear lista con 30 neuronas (supera el límite)
        List<Neurona> neuronasGrandes = neuronas.subList(0, 30);
        
        // Formar engrama manualmente (simula lo que haría detectarYFormarEngramas)
        gestor.activarDeteccion(true);
        
        // Simular formación con límite
        List<List<Neurona>> capas = new ArrayList<>();
        capas.add(neuronasGrandes);
        
        // Activar todas las neuronas correctamente
        for (Neurona n : neuronasGrandes) {
            n.activar(1000L);
        }
        
        gestor.detectarYFormarEngramas(capas, 1000L);
        
        // Verificar que se formaron engramas
        assertTrue(gestor.getEngramas().size() > 0, "Debe formar al menos un engrama");
        
        // Verificar que ningún engrama supera el límite del 22%
        for (Engrama e : gestor.getEngramas().values()) {
            assertTrue(e.getNeuronas().size() <= limite, 
                "Engrama no debe superar " + limite + " neuronas, tiene: " + e.getNeuronas().size());
        }
    }
    
    @Test
    @DisplayName("División automática de patrones grandes")
    void testDivisionAutomatica() {
        // Crear lista con 50 neuronas (supera límite de 22)
        List<Neurona> neuronasGrandes = neuronas.subList(0, 50);
        
        gestor.activarDeteccion(true);
        
        List<List<Neurona>> capas = new ArrayList<>();
        capas.add(neuronasGrandes);
        
        // Activar todas correctamente
        for (Neurona n : neuronasGrandes) {
            n.activar(1000L);
        }
        
        gestor.detectarYFormarEngramas(capas, 1000L);
        
        // Debe crear múltiples engramas (chunks)
        int totalEngramas = gestor.getEngramas().size();
        assertTrue(totalEngramas >= 2, 
            "Debe dividir en al menos 2 engramas, creó: " + totalEngramas);
        
        // Verificar que la suma de neuronas es aproximadamente 50
        int totalNeuronasEnEngramas = 0;
        for (Engrama e : gestor.getEngramas().values()) {
            totalNeuronasEnEngramas += e.getNeuronas().size();
        }
        
        // Puede haber duplicados por solapamiento, pero debe estar cerca
        assertTrue(totalNeuronasEnEngramas >= 40, 
            "Total de neuronas en engramas debe ser al menos 40, es: " + totalNeuronasEnEngramas);
    }
    
    @Test
    @DisplayName("Fusión de engramas similares (>90%)")
    void testFusionEngramasSimilares() {
        // Crear dos engramas muy similares manualmente
        List<Neurona> grupo1 = neuronas.subList(0, 10);
        List<Neurona> grupo2 = neuronas.subList(0, 10); // Idéntico
        
        gestor.formarEngrama("engrama1", grupo1, 1000L);
        gestor.formarEngrama("engrama2", grupo2, 1000L);
        
        assertEquals(2, gestor.getEngramas().size(), "Debe haber 2 engramas antes de optimizar");
        
        // Optimizar (debe fusionar)
        gestor.optimizarEngramas();
        
        // Debe quedar solo 1 engrama (fusionados)
        assertEquals(1, gestor.getEngramas().size(), 
            "Debe quedar 1 engrama después de fusionar similares");
        
        GestorEngramas.EstadisticasEngramas stats = gestor.getEstadisticas();
        assertEquals(1, stats.totalFusionados, "Debe registrar 1 fusión");
    }
    
    @Test
    @DisplayName("No fusiona engramas diferentes (<90%)")
    void testNoFusionEngramasDiferentes() {
        // Crear dos engramas diferentes
        List<Neurona> grupo1 = neuronas.subList(0, 10);
        List<Neurona> grupo2 = neuronas.subList(20, 30); // Sin solapamiento
        
        gestor.formarEngrama("engrama1", grupo1, 1000L);
        gestor.formarEngrama("engrama2", grupo2, 1000L);
        
        assertEquals(2, gestor.getEngramas().size());
        
        // Optimizar (no debe fusionar)
        gestor.optimizarEngramas();
        
        // Deben seguir siendo 2
        assertEquals(2, gestor.getEngramas().size(), 
            "No debe fusionar engramas diferentes");
    }
    
    @Test
    @DisplayName("Poda de engramas con baja relevancia (<0.15)")
    void testPodaEngramasBajaRelevancia() {
        // Crear engrama con baja relevancia
        List<Neurona> grupo = neuronas.subList(0, 5);
        gestor.formarEngrama("engrama_debil", grupo, 1000L);
        
        // Obtener el engrama y reducir su relevancia
        Engrama engrama = gestor.getEngramas().get("engrama_debil");
        engrama.setRelevancia(0.10); // Menor que 0.15
        
        assertEquals(1, gestor.getEngramas().size());
        
        // Optimizar (debe podar)
        gestor.optimizarEngramas();
        
        // Debe haber sido eliminado
        assertEquals(0, gestor.getEngramas().size(), 
            "Debe podar engramas con relevancia < 0.15");
        
        GestorEngramas.EstadisticasEngramas stats = gestor.getEstadisticas();
        assertEquals(1, stats.totalPodados, "Debe registrar 1 poda");
    }
    
    @Test
    @DisplayName("Clustering de engramas grandes (>15% de neuronas)")
    void testClusteringEngramasGrandes() {
        // Crear engrama grande (>15% = >15 neuronas)
        // Mezclar neuronas de diferentes "capas" simuladas
        List<Neurona> grupoGrande = new ArrayList<>();
        grupoGrande.addAll(neuronas.subList(0, 10));   // "Capa 0"
        grupoGrande.addAll(neuronas.subList(50, 60));  // "Capa 50"
        
        gestor.formarEngrama("engrama_grande", grupoGrande, 1000L);
        
        assertEquals(1, gestor.getEngramas().size());
        
        // Optimizar (puede aplicar clustering si detecta grupos naturales)
        gestor.optimizarEngramas();
        
        // El clustering puede o no dividir dependiendo de la heurística
        // Solo verificamos que no falla
        assertTrue(gestor.getEngramas().size() >= 1, 
            "Debe mantener al menos el engrama original");
    }
    
    @Test
    @DisplayName("Estadísticas se calculan correctamente")
    void testEstadisticas() {
        // Formar varios engramas
        gestor.formarEngrama("e1", neuronas.subList(0, 5), 1000L);
        gestor.formarEngrama("e2", neuronas.subList(10, 20), 1000L);
        gestor.formarEngrama("e3", neuronas.subList(30, 35), 1000L);
        
        GestorEngramas.EstadisticasEngramas stats = gestor.getEstadisticas();
        
        assertEquals(3, stats.totalActuales, "Debe haber 3 engramas actuales");
        assertEquals(3, stats.totalFormados, "Debe haber formado 3 engramas");
        assertEquals(0, stats.totalFusionados, "No debe haber fusiones");
        assertEquals(0, stats.totalPodados, "No debe haber podas");
        
        // Tamaño promedio = (5 + 10 + 5) / 3 = 6.67
        assertTrue(stats.tamañoPromedio >= 6.0 && stats.tamañoPromedio <= 7.0, 
            "Tamaño promedio debe ser ~6.67, es: " + stats.tamañoPromedio);
        
        assertEquals(5, stats.tamañoMin, "Tamaño mínimo debe ser 5");
        assertEquals(10, stats.tamañoMax, "Tamaño máximo debe ser 10");
        
        // Porcentaje promedio = 6.67 / 100 * 100 = 6.67%
        assertTrue(stats.porcentajePromedioNeuronas >= 6.0 && stats.porcentajePromedioNeuronas <= 7.0,
            "Porcentaje debe ser ~6.67%, es: " + stats.porcentajePromedioNeuronas);
    }
    
    @Test
    @DisplayName("Activar y desactivar detección")
    void testActivarDesactivarDeteccion() {
        assertFalse(gestor.esDeteccionActiva());
        
        gestor.activarDeteccion(true);
        assertTrue(gestor.esDeteccionActiva());
        
        gestor.activarDeteccion(false);
        assertFalse(gestor.esDeteccionActiva());
    }
    
    @Test
    @DisplayName("Formar y eliminar engramas")
    void testFormarYEliminarEngramas() {
        List<Neurona> grupo = neuronas.subList(0, 5);
        
        gestor.formarEngrama("test_id", grupo, 1000L);
        assertEquals(1, gestor.getEngramas().size());
        assertTrue(gestor.getEngramas().containsKey("test_id"));
        
        gestor.eliminarEngrama("test_id");
        assertEquals(0, gestor.getEngramas().size());
        assertFalse(gestor.getEngramas().containsKey("test_id"));
    }
    
    @Test
    @DisplayName("Activar engrama existente")
    void testActivarEngrama() {
        List<Neurona> grupo = neuronas.subList(0, 5);
        gestor.formarEngrama("test_id", grupo, 1000L);
        
        Engrama engrama = gestor.getEngramas().get("test_id");
        int activacionesInicial = engrama.getContadorActivaciones();
        
        gestor.activarEngrama("test_id", 2000L);
        
        assertEquals(activacionesInicial + 1, engrama.getContadorActivaciones(),
            "Contador de activaciones debe incrementar");
        assertEquals(2000L, engrama.getTimestampUltimaActivacion(),
            "Timestamp debe actualizarse");
    }
    
    @Test
    @DisplayName("Fusión respeta límite del 22%")
    void testFusionRespetaLimite() {
        // Crear dos engramas que juntos superarían el límite
        List<Neurona> grupo1 = neuronas.subList(0, 15);
        List<Neurona> grupo2 = neuronas.subList(10, 25); // Solapamiento parcial
        
        gestor.formarEngrama("e1", grupo1, 1000L);
        gestor.formarEngrama("e2", grupo2, 1000L);
        
        // Optimizar
        gestor.optimizarEngramas();
        
        // Verificar que ningún engrama supera el límite
        for (Engrama e : gestor.getEngramas().values()) {
            assertTrue(e.getNeuronas().size() <= 22,
                "Engrama fusionado no debe superar límite de 22 neuronas");
        }
    }
}
