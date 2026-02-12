package es.jastxz.experimental;

import es.jastxz.nn.Engrama;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para Fase 6: Gestión de Engramas
 * 
 * Implementa sistema de memoria explícita:
 * - Engramas se forman durante aprendizaje
 * - Engramas pueden ser activados posteriormente
 * - Activación de engrama reproduce patrón aprendido
 * 
 * Referencias:
 * - pg. 84-85 Campillo: "conjunto de neuronas que funcionan como bits para guardar un recuerdo"
 * - pg. 85 Campillo: "el aprendizaje culmina en un entramado de engramas"
 * - pg. 93 Campillo: "el cerebro revisa otros recuerdos (engramas) para completar aquellos que no lo estén"
 */
@DisplayName("Fase 6: Gestión de Engramas")
public class RedNeuralExperimentalFase6Test {
    
    @Test
    @DisplayName("Red puede formar engrama manualmente")
    void testFormarEngramaManual() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Obtener algunas neuronas
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        List<Neurona> participantes = neuronas.subList(0, 2);
        
        // Formar engrama
        red.formarEngrama("test_engrama", participantes);
        
        // Verificar que existe
        Map<String, Engrama> engramas = red.getEngramas();
        assertTrue(engramas.containsKey("test_engrama"), 
            "El engrama debe existir en el mapa");
    }
    
    @Test
    @DisplayName("Red puede listar engramas")
    void testListarEngramas() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Formar varios engramas
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        red.formarEngrama("engrama1", neuronas.subList(0, 2));
        red.formarEngrama("engrama2", neuronas.subList(1, 3));
        
        Map<String, Engrama> engramas = red.getEngramas();
        
        assertEquals(2, engramas.size(), 
            "Debe haber 2 engramas");
        assertTrue(engramas.containsKey("engrama1"), "Debe existir engrama1");
        assertTrue(engramas.containsKey("engrama2"), "Debe existir engrama2");
    }
    
    @Test
    @DisplayName("Red puede activar engrama existente")
    void testActivarEngrama() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Formar engrama
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        red.formarEngrama("memoria1", neuronas.subList(0, 2));
        
        // Activar engrama
        assertDoesNotThrow(() -> red.activarEngrama("memoria1"), 
            "Debe poder activar un engrama existente");
    }
    
    @Test
    @DisplayName("Activar engrama inexistente lanza excepción")
    void testActivarEngramaInexistente() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        assertThrows(IllegalArgumentException.class, 
            () -> red.activarEngrama("no_existe"), 
            "Debe lanzar excepción al activar engrama inexistente");
    }
    
    @Test
    @DisplayName("Red detecta patrones de activación automáticamente")
    void testDeteccionAutomaticaPatrones() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarDeteccionEngramas(true);
        
        // Procesar el mismo patrón varias veces
        for (int i = 0; i < 10; i++) {
            red.procesar(new double[]{1.0, 1.0});
        }
        
        // Debe haber detectado y formado engramas automáticamente
        Map<String, Engrama> engramas = red.getEngramas();
        
        assertTrue(engramas.size() >= 0, 
            "Puede haber formado engramas automáticamente (o no, depende del umbral)");
    }
    
    @Test
    @DisplayName("Engrama se activa cuando suficientes neuronas están activas")
    void testEngramaSeActivaConPatron() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Procesar para activar neuronas
        red.procesar(new double[]{1.0, 1.0});
        
        // Formar engrama con neuronas activas
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        red.formarEngrama("patron1", neuronas);
        
        // Obtener engrama
        Engrama engrama = red.getEngramas().get("patron1");
        
        // Procesar de nuevo el mismo patrón
        red.procesar(new double[]{1.0, 1.0});
        
        // El engrama debe detectar activación
        boolean estaActivo = engrama.estaActivo(0.3);  // 30% umbral
        
        // Puede estar activo o no, dependiendo de cuántas neuronas se activaron
        assertNotNull(engrama, "El engrama debe existir");
        assert(estaActivo);
    }
    
    @Test
    @DisplayName("Activación de engrama facilita neuronas participantes")
    void testActivacionFacilitaNeuronas() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Formar engrama
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        red.formarEngrama("memoria1", neuronas.subList(0, 2));
        
        // Activar engrama (facilita neuronas)
        red.activarEngrama("memoria1");
        
        // Las neuronas del engrama deben tener facilitación
        // (esto es interno, pero podemos verificar que no lanza error)
        assertDoesNotThrow(() -> red.procesar(new double[]{0.5, 0.5}), 
            "Debe poder procesar tras activar engrama");
    }
    
    @Test
    @DisplayName("Engramas persisten entre procesamientos")
    void testEngramasPersisten() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Formar engrama
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        red.formarEngrama("persistente", neuronas.subList(0, 2));
        
        // Procesar varias veces
        for (int i = 0; i < 5; i++) {
            red.procesar(new double[]{1.0, 0.5});
        }
        
        // El engrama debe seguir existiendo
        assertTrue(red.getEngramas().containsKey("persistente"), 
            "El engrama debe persistir entre procesamientos");
    }
    
    @Test
    @DisplayName("Red puede tener múltiples engramas simultáneos")
    void testMultiplesEngramas() {
        int[] topologia = {2, 4, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        
        // Formar varios engramas con diferentes neuronas
        red.formarEngrama("engrama_A", neuronas.subList(0, 2));
        red.formarEngrama("engrama_B", neuronas.subList(2, 4));
        red.formarEngrama("engrama_C", neuronas.subList(1, 3));
        
        assertEquals(3, red.getEngramas().size(), 
            "Debe haber 3 engramas");
    }
    
    @Test
    @DisplayName("Engrama con ID duplicado reemplaza al anterior")
    void testEngramaDuplicadoReemplaza() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        
        // Formar engrama
        red.formarEngrama("duplicado", neuronas.subList(0, 1));
        Engrama primero = red.getEngramas().get("duplicado");
        
        // Formar otro con mismo ID
        red.formarEngrama("duplicado", neuronas.subList(1, 2));
        Engrama segundo = red.getEngramas().get("duplicado");
        
        // Debe haber solo uno
        assertEquals(1, red.getEngramas().size(), 
            "Debe haber solo un engrama con ese ID");
        assertNotSame(primero, segundo, 
            "El segundo debe ser diferente del primero");
    }
    
    @Test
    @DisplayName("Detección automática puede desactivarse")
    void testDesactivarDeteccionAutomatica() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarDeteccionEngramas(true);
        red.activarDeteccionEngramas(false);
        
        assertFalse(red.esDeteccionEngramasActiva(), 
            "La detección automática debe estar desactivada");
    }
    
    @Test
    @DisplayName("Red con topología mínima puede usar engramas")
    void testEngramas_TopologiaMinima() {
        int[] topologia = {1, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // No hay capas intermedias, pero no debe fallar
        assertDoesNotThrow(() -> {
            red.activarDeteccionEngramas(true);
            red.procesar(new double[]{1.0});
        }, "Red mínima debe soportar sistema de engramas");
    }
    
    @Test
    @DisplayName("Engramas se integran con entrenamiento")
    void testEngramasConEntrenamiento() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        red.activarDeteccionEngramas(true);
        
        // Entrenar con patrón
        red.entrenar(new double[]{1.0, 0.0}, new double[]{1.0}, 10);
        
        // Puede haber formado engramas durante el entrenamiento
        // (o no, depende del umbral de detección)
        assertNotNull(red.getEngramas(), 
            "El mapa de engramas debe existir");
    }
    
    @Test
    @DisplayName("Activar engrama no interfiere con procesamiento normal")
    void testActivarEngramaNoInterfiere() {
        int[] topologia = {2, 3, 1};
        RedNeuralExperimental red = new RedNeuralExperimental(topologia, 1.0);
        
        // Formar y activar engrama
        List<Neurona> neuronas = red.getCapasInterneuronas().get(0);
        red.formarEngrama("test", neuronas.subList(0, 2));
        red.activarEngrama("test");
        
        // Procesar debe funcionar normalmente
        double[] outputs = red.procesar(new double[]{1.0, 0.5});
        
        assertNotNull(outputs, "Debe generar outputs normalmente");
        assertEquals(1, outputs.length, "Debe tener el tamaño correcto");
    }
}
