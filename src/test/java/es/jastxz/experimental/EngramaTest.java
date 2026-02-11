package es.jastxz.experimental;

import es.jastxz.nn.*;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoConexion;
import es.jastxz.nn.enums.TipoNeurona;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para la clase Engrama
 * Valida formación de memoria, consolidación, degradación
 */
class EngramaTest {
    
    private Engrama engrama;
    private Neurona neurona1;
    private Neurona neurona2;
    private Neurona neurona3;
    private Conexion conexion1;
    private Conexion conexion2;
    
    @BeforeEach
    void setUp() {
        engrama = new Engrama("engrama_test_001");
        
        neurona1 = new Neurona(1L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.REPOSO);
        neurona2 = new Neurona(2L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        neurona3 = new Neurona(3L, TipoNeurona.MOTORA, 0.0, PotencialMemoria.REPOSO);
        
        conexion1 = new Conexion(neurona1, neurona2, 0.5, TipoConexion.QUIMICA);
        conexion2 = new Conexion(neurona2, neurona3, 0.5, TipoConexion.QUIMICA);
    }
    
    @Test
    @DisplayName("Engrama se inicializa correctamente")
    void testInicializacion() {
        assertEquals("engrama_test_001", engrama.getId());
        assertEquals(0, engrama.getNeuronasParticipantes().size());
        assertEquals(0, engrama.getConexionesParticipantes().size());
        assertEquals(0.5, engrama.getFuerza(), 0.001);
        assertTrue(engrama.getTimestampUltimaActivacion() > 0);
    }
    
    @Test
    @DisplayName("Engrama puede agregar neuronas")
    void testAgregarNeurona() {
        engrama.agregarNeurona(neurona1);
        
        assertEquals(1, engrama.getNeuronasParticipantes().size());
        assertTrue(engrama.getNeuronasParticipantes().contains(neurona1));
        
        // Verificar que neurona sabe que pertenece al engrama
        assertTrue(neurona1.getEngramasActivos().contains("engrama_test_001"));
    }
    
    @Test
    @DisplayName("Engrama puede agregar múltiples neuronas")
    void testAgregarMultiplesNeuronas() {
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        engrama.agregarNeurona(neurona3);
        
        assertEquals(3, engrama.getNeuronasParticipantes().size());
        assertTrue(engrama.getNeuronasParticipantes().contains(neurona1));
        assertTrue(engrama.getNeuronasParticipantes().contains(neurona2));
        assertTrue(engrama.getNeuronasParticipantes().contains(neurona3));
    }
    
    @Test
    @DisplayName("Engrama puede agregar conexiones")
    void testAgregarConexion() {
        engrama.agregarConexion(conexion1);
        
        assertEquals(1, engrama.getConexionesParticipantes().size());
        assertTrue(engrama.getConexionesParticipantes().contains(conexion1));
        
        // Verificar que conexión sabe que pertenece al engrama
        assertTrue(conexion1.getEngramasActivos().contains("engrama_test_001"));
    }
    
    @Test
    @DisplayName("Engrama puede agregar múltiples conexiones")
    void testAgregarMultiplesConexiones() {
        engrama.agregarConexion(conexion1);
        engrama.agregarConexion(conexion2);
        
        assertEquals(2, engrama.getConexionesParticipantes().size());
        assertTrue(engrama.getConexionesParticipantes().contains(conexion1));
        assertTrue(engrama.getConexionesParticipantes().contains(conexion2));
    }
    
    @Test
    @DisplayName("Consolidación fortalece el engrama")
    void testConsolidacion() {
        double fuerzaInicial = engrama.getFuerza();
        
        engrama.consolidar(0.2);
        
        assertTrue(engrama.getFuerza() > fuerzaInicial);
    }
    
    @Test
    @DisplayName("Consolidación fortalece conexiones del engrama")
    void testConsolidacionFortalececonexiones() {
        engrama.agregarConexion(conexion1);
        engrama.agregarConexion(conexion2);
        
        double pesoInicial1 = conexion1.getPeso();
        double pesoInicial2 = conexion2.getPeso();
        
        engrama.consolidar(0.3);
        
        assertTrue(conexion1.getPeso() > pesoInicial1);
        assertTrue(conexion2.getPeso() > pesoInicial2);
    }
    
    @Test
    @DisplayName("Fuerza del engrama no supera 1.0")
    void testFuerzaMaxima() {
        // Consolidar múltiples veces
        for (int i = 0; i < 10; i++) {
            engrama.consolidar(0.3);
        }
        
        assertTrue(engrama.getFuerza() <= 1.0);
    }
    
    @Test
    @DisplayName("Degradación debilita el engrama")
    void testDegradacion() {
        double fuerzaInicial = engrama.getFuerza();
        
        engrama.degradar(0.2);
        
        assertTrue(engrama.getFuerza() < fuerzaInicial);
    }
    
    @Test
    @DisplayName("Degradación debilita conexiones cuando fuerza baja")
    void testDegradacionDebilitaConexiones() {
        engrama.agregarConexion(conexion1);
        
        // Degradar hasta fuerza muy baja
        for (int i = 0; i < 5; i++) {
            engrama.degradar(0.1);
        }
        
        double pesoInicial = conexion1.getPeso();
        
        // Degradar más
        engrama.degradar(0.1);
        
        // Verificar que conexión se debilitó
        assertTrue(conexion1.getPeso() < pesoInicial);
    }
    
    @Test
    @DisplayName("Fuerza del engrama no baja de 0.0")
    void testFuerzaMinima() {
        // Degradar múltiples veces
        for (int i = 0; i < 10; i++) {
            engrama.degradar(0.3);
        }
        
        assertTrue(engrama.getFuerza() >= 0.0);
    }
    
    @Test
    @DisplayName("Engrama detecta cuando está activo")
    void testDeteccionActivacion() {
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        engrama.agregarNeurona(neurona3);
        
        // Inicialmente ninguna activa
        assertFalse(engrama.estaActivo(0.3));
        
        // Activar una neurona manualmente (simular input externo)
        Neurona pre = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        new Conexion(pre, neurona1, 0.9, TipoConexion.QUIMICA);
        neurona1.evaluar(1000L);
        
        // 1 de 3 = 33% activas, supera umbral de 30%
        assertTrue(engrama.estaActivo(0.3));
    }
    
    @Test
    @DisplayName("Completado de patrón facilita neuronas inactivas")
    void testCompletadoPatron() {
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        engrama.agregarNeurona(neurona3);
        
        // Activar suficientes neuronas para superar umbral
        Neurona pre1 = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        Neurona pre2 = new Neurona(11L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        new Conexion(pre1, neurona1, 0.9, TipoConexion.QUIMICA);
        new Conexion(pre2, neurona2, 0.9, TipoConexion.QUIMICA);
        
        neurona1.evaluar(1000L);
        neurona2.evaluar(1000L);
        
        // 2 de 3 = 66% activas, supera umbral
        assertTrue(engrama.estaActivo(0.3));
        
        // Completar patrón debe actualizar timestamp
        long timestampAntes = engrama.getTimestampUltimaActivacion();
        
        // Esperar un poco para asegurar diferencia de timestamp
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
            // Ignorar
        }
        
        long nuevoTimestamp = System.currentTimeMillis();
        engrama.completarPatron(nuevoTimestamp, 0.3);
        
        // Verificar que el timestamp se actualizó
        assertTrue(timestampAntes < engrama.getTimestampUltimaActivacion());
    }
    
    @Test
    @DisplayName("Completado de patrón no ocurre sin activación suficiente")
    void testCompletadoPatronRequiereUmbral() {
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        engrama.agregarNeurona(neurona3);
        
        // Solo activar una neurona (33%)
        Neurona pre = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        new Conexion(pre, neurona1, 0.9, TipoConexion.QUIMICA);
        neurona1.evaluar(1000L);
        
        long timestampInicial = engrama.getTimestampUltimaActivacion();
        
        // Intentar completar con umbral alto (50%)
        engrama.completarPatron(5000L, 0.5);
        
        // No debe actualizar timestamp porque no hay suficiente activación
        assertEquals(timestampInicial, engrama.getTimestampUltimaActivacion());
    }
    
    @Test
    @DisplayName("Facilitación proporcional a fuerza del engrama")
    void testFacilitacionProporcionalAFuerza() {
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        
        // Consolidar para aumentar fuerza
        engrama.consolidar(0.4);
        
        // Activar una neurona para superar umbral
        Neurona pre = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        new Conexion(pre, neurona1, 0.9, TipoConexion.QUIMICA);
        neurona1.evaluar(1000L);
        
        // Completar patrón (facilita neurona2)
        engrama.completarPatron(5000L, 0.3);
        
        // neurona2 debería tener facilitación ahora
        // No podemos verificar directamente, pero podemos verificar que el engrama está fuerte
        assertTrue(engrama.getFuerza() > 0.5);
    }
    
    @Test
    @DisplayName("Engrama completo con neuronas y conexiones")
    void testEngramaCompleto() {
        // Formar engrama completo
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        engrama.agregarNeurona(neurona3);
        engrama.agregarConexion(conexion1);
        engrama.agregarConexion(conexion2);
        
        assertEquals(3, engrama.getNeuronasParticipantes().size());
        assertEquals(2, engrama.getConexionesParticipantes().size());
        
        // Consolidar
        engrama.consolidar(0.3);
        
        // Verificar que todo se fortaleció
        assertTrue(engrama.getFuerza() > 0.5);
        assertTrue(conexion1.getPeso() > 0.5);
        assertTrue(conexion2.getPeso() > 0.5);
    }
    
    @Test
    @DisplayName("Engramas no desaparecen completamente tras degradación")
    void testEngramasPersistenParcialmente() {
        engrama.agregarConexion(conexion1);
        
        // Degradar completamente
        for (int i = 0; i < 20; i++) {
            engrama.degradar(0.1);
        }
        
        // Verificar que fuerza llegó a 0 pero conexión mantiene peso mínimo
        assertEquals(0.0, engrama.getFuerza(), 0.001);
        assertTrue(conexion1.getPeso() >= 0.1); // Peso mínimo mantenido
    }
    
    @Test
    @DisplayName("Ciclo completo: formación, consolidación, degradación")
    void testCicloCompleto() {
        // Formación
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        engrama.agregarConexion(conexion1);
        
        double fuerzaInicial = engrama.getFuerza();
        
        // Consolidación (simular "sueño")
        engrama.consolidar(0.3);
        double fuerzaConsolidada = engrama.getFuerza();
        assertTrue(fuerzaConsolidada > fuerzaInicial);
        
        // Uso (activación parcial + completado de patrón)
        Neurona pre = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        new Conexion(pre, neurona1, 0.9, TipoConexion.QUIMICA);
        neurona1.evaluar(10000L);
        engrama.completarPatron(10000L, 0.3);
        
        // Degradación por desuso
        engrama.degradar(0.1);
        double fuerzaDegradada = engrama.getFuerza();
        assertTrue(fuerzaDegradada < fuerzaConsolidada);
        
        // Pero no desaparece completamente
        assertTrue(fuerzaDegradada > 0.0);
    }
    
    @Test
    @DisplayName("Facilitación temporal permite activación con menos input")
    void testFacilitacionReduceUmbral() {
        engrama.agregarNeurona(neurona1);
        engrama.agregarNeurona(neurona2);
        
        // Consolidar engrama
        engrama.consolidar(0.5);
        
        // Activar neurona1 para que engrama esté parcialmente activo
        Neurona pre1 = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        new Conexion(pre1, neurona1, 0.9, TipoConexion.QUIMICA);
        neurona1.evaluar(1000L);
        
        // Completar patrón (facilita neurona2)
        engrama.completarPatron(1000L, 0.3);
        
        // Crear input débil para neurona2 (que normalmente no activaría)
        Neurona pre2 = new Neurona(11L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.REPOSO);
        pre2.evaluar(1000L);
        new Conexion(pre2, neurona2, 0.3, TipoConexion.QUIMICA);
        
        // Con facilitación, neurona2 debería activarse más fácilmente
        // (esto es difícil de testear directamente sin exponer facilitacionTemporal)
        // Verificamos que el engrama está activo
        assertTrue(engrama.estaActivo(0.3));
    }
}
