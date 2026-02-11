package es.jastxz.experimental;

import es.jastxz.nn.*;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoConexion;
import es.jastxz.nn.enums.TipoNeurona;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests para la clase Conexion
 * Valida plasticidad hebiana, poda sináptica, sinapsis múltiples
 */
class ConexionTest {
    
    private Neurona presináptica;
    private Neurona postsináptica;
    private Conexion conexion;
    
    @BeforeEach
    void setUp() {
        presináptica = new Neurona(1L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.REPOSO);
        postsináptica = new Neurona(2L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        conexion = new Conexion(presináptica, postsináptica, 0.5, TipoConexion.QUIMICA);
    }
    
    @Test
    @DisplayName("Conexión se inicializa correctamente")
    void testInicializacion() {
        assertEquals(presináptica, conexion.getPresinaptica());
        assertEquals(1, conexion.getPostsinápticas().size());
        assertTrue(conexion.getPostsinápticas().contains(postsináptica));
        assertEquals(0.5, conexion.getPeso(), 0.001);
        assertEquals(TipoConexion.QUIMICA, conexion.getTipo());
        assertEquals(1.0, conexion.getRecursosAsignados(), 0.001);
        assertEquals(0, conexion.getVecesActivadaJuntas());
    }
    
    @Test
    @DisplayName("Conexión se registra en neuronas pre y post")
    void testRegistroEnNeuronas() {
        // Verificar que la conexión está registrada
        assertTrue(presináptica.getAxones().contains(conexion));
        assertTrue(postsináptica.getDendritas().contains(conexion));
    }
    
    @Test
    @DisplayName("Plasticidad hebiana refuerza conexión cuando ambas activas")
    void testRefuerzoPorActivacionConjunta() {
        double pesoInicial = conexion.getPeso();
        
        // Activar ambas neuronas con inputs reales
        Neurona pre1 = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        Neurona pre2 = new Neurona(11L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        
        new Conexion(pre1, presináptica, 0.9, TipoConexion.QUIMICA);
        new Conexion(pre2, postsináptica, 0.9, TipoConexion.QUIMICA);
        
        presináptica.evaluar(1000L);
        postsináptica.evaluar(1000L);
        
        // Aplicar plasticidad
        conexion.aplicarPlasticidadHebiana(1000L, 100L);
        
        // Verificar refuerzo
        assertTrue(conexion.getPeso() > pesoInicial);
        assertEquals(1, conexion.getVecesActivadaJuntas());
    }
    
    @Test
    @DisplayName("Plasticidad hebiana debilita conexión por desuso")
    void testDebilitamientoPorDesuso() {
        // Activar inicialmente para establecer timestamp
        Neurona pre1 = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        Neurona pre2 = new Neurona(11L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        
        new Conexion(pre1, presináptica, 0.9, TipoConexion.QUIMICA);
        new Conexion(pre2, postsináptica, 0.9, TipoConexion.QUIMICA);
        
        presináptica.evaluar(1000L);
        postsináptica.evaluar(1000L);
        conexion.aplicarPlasticidadHebiana(1000L, 100L);
        
        double pesoTrasActivacion = conexion.getPeso();
        
        // Resetear neuronas
        presináptica.resetear();
        postsináptica.resetear();
        
        // Simular paso del tiempo sin activación (mucho más allá de la ventana temporal)
        conexion.aplicarPlasticidadHebiana(100000L, 100L);
        
        // Verificar debilitamiento respecto al peso tras activación
        assertTrue(conexion.getPeso() < pesoTrasActivacion);
    }
    
    @Test
    @DisplayName("Peso se mantiene en rango [0, 1]")
    void testPesoEnRango() {
        conexion.setPeso(1.5);
        assertEquals(1.0, conexion.getPeso(), 0.001);
        
        conexion.setPeso(-0.5);
        assertEquals(0.0, conexion.getPeso(), 0.001);
    }
    
    @Test
    @DisplayName("Conexión debe ser podada cuando peso muy bajo")
    void testPodaPorPesoBajo() {
        conexion.setPeso(0.03);
        assertTrue(conexion.debeSerPodada());
    }
    
    @Test
    @DisplayName("Conexión debe ser podada cuando recursos muy bajos")
    void testPodaPorRecursosBajos() {
        conexion.setRecursosAsignados(0.05);
        assertTrue(conexion.debeSerPodada());
    }
    
    @Test
    @DisplayName("Conexión NO debe ser podada cuando está saludable")
    void testNoPodaCuandoSaludable() {
        conexion.setPeso(0.7);
        conexion.setRecursosAsignados(0.8);
        assertFalse(conexion.debeSerPodada());
    }
    
    @Test
    @DisplayName("Recursos aumentan con uso frecuente")
    void testAumentoRecursosPorUso() {
        double recursosIniciales = conexion.getRecursosAsignados();
        
        // Crear neuronas para activar pre y post
        Neurona pre1 = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        Neurona pre2 = new Neurona(11L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        
        new Conexion(pre1, presináptica, 0.9, TipoConexion.QUIMICA);
        new Conexion(pre2, postsináptica, 0.9, TipoConexion.QUIMICA);
        
        // Activar repetidamente
        for (int i = 0; i < 10; i++) {
            presináptica.resetear();
            postsináptica.resetear();
            presináptica.evaluar(i * 1000L);
            postsináptica.evaluar(i * 1000L);
            conexion.aplicarPlasticidadHebiana(i * 1000L, 100L);
        }
        
        // Verificar aumento de recursos
        assertTrue(conexion.getRecursosAsignados() >= recursosIniciales);
    }
    
    @Test
    @DisplayName("Recursos disminuyen por desuso")
    void testDisminucionRecursosPorDesuso() {
        double recursosIniciales = conexion.getRecursosAsignados();
        
        // Simular desuso prolongado
        for (int i = 0; i < 10; i++) {
            conexion.aplicarPlasticidadHebiana(i * 10000L, 100L);
        }
        
        // Verificar disminución de recursos
        assertTrue(conexion.getRecursosAsignados() < recursosIniciales);
    }
    
    @Test
    @DisplayName("Conexión puede unirse a engramas")
    void testUnionAEngrama() {
        String engramaId = "engrama_test_001";
        
        conexion.unirseAEngrama(engramaId);
        
        assertTrue(conexion.getEngramasActivos().contains(engramaId));
        assertEquals(1, conexion.getEngramasActivos().size());
    }
    
    @Test
    @DisplayName("Conexión puede salir de engramas")
    void testSalidaDeEngrama() {
        String engramaId = "engrama_test_001";
        
        conexion.unirseAEngrama(engramaId);
        conexion.salirDeEngrama(engramaId);
        
        assertFalse(conexion.getEngramasActivos().contains(engramaId));
        assertEquals(0, conexion.getEngramasActivos().size());
    }
    
    @Test
    @DisplayName("Sinapsis diádica conecta 1 pre con 2 post")
    void testSinapsisDiadica() {
        Neurona post1 = new Neurona(10L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        Neurona post2 = new Neurona(11L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        
        List<Neurona> postsinápticas = Arrays.asList(post1, post2);
        Conexion diadica = new Conexion(presináptica, postsinápticas, 0.6, TipoConexion.QUIMICA);
        
        assertEquals(2, diadica.getPostsinápticas().size());
        assertTrue(diadica.getPostsinápticas().contains(post1));
        assertTrue(diadica.getPostsinápticas().contains(post2));
        
        // Verificar registro en ambas postsinápticas
        assertTrue(post1.getDendritas().contains(diadica));
        assertTrue(post2.getDendritas().contains(diadica));
    }
    
    @Test
    @DisplayName("Sinapsis triádica conecta 1 pre con 3 post")
    void testSinapsisTriadica() {
        Neurona post1 = new Neurona(10L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        Neurona post2 = new Neurona(11L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        Neurona post3 = new Neurona(12L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        
        List<Neurona> postsinápticas = Arrays.asList(post1, post2, post3);
        Conexion triadica = new Conexion(presináptica, postsinápticas, 0.6, TipoConexion.QUIMICA);
        
        assertEquals(3, triadica.getPostsinápticas().size());
        assertTrue(triadica.getPostsinápticas().contains(post1));
        assertTrue(triadica.getPostsinápticas().contains(post2));
        assertTrue(triadica.getPostsinápticas().contains(post3));
    }
    
    @Test
    @DisplayName("Plasticidad funciona con sinapsis múltiples")
    void testPlasticidadConSinapsisMultiples() {
        Neurona post1 = new Neurona(10L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        Neurona post2 = new Neurona(11L, TipoNeurona.INTER, 0.0, PotencialMemoria.REPOSO);
        
        List<Neurona> postsinápticas = Arrays.asList(post1, post2);
        Conexion diadica = new Conexion(presináptica, postsinápticas, 0.5, TipoConexion.QUIMICA);
        
        double pesoInicial = diadica.getPeso();
        
        // Activar pre y una de las post
        Neurona pre1 = new Neurona(20L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        Neurona pre2 = new Neurona(21L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        
        new Conexion(pre1, presináptica, 0.9, TipoConexion.QUIMICA);
        new Conexion(pre2, post1, 0.9, TipoConexion.QUIMICA);
        
        presináptica.evaluar(1000L);
        post1.evaluar(1000L);
        
        // Aplicar plasticidad
        diadica.aplicarPlasticidadHebiana(1000L, 100L);
        
        // Verificar refuerzo (al menos una post activa)
        assertTrue(diadica.getPeso() > pesoInicial);
    }
    
    @Test
    @DisplayName("Conexión eléctrica se diferencia de química")
    void testTipoConexionElectrica() {
        Conexion electrica = new Conexion(presináptica, postsináptica, 0.8, TipoConexion.ELECTRICA);
        
        assertEquals(TipoConexion.ELECTRICA, electrica.getTipo());
        assertEquals(TipoConexion.QUIMICA, conexion.getTipo());
        assertNotEquals(electrica.getTipo(), conexion.getTipo());
    }
    
    @Test
    @DisplayName("Múltiples activaciones incrementan contador")
    void testContadorActivaciones() {
        assertEquals(0, conexion.getVecesActivadaJuntas());
        
        // Crear neuronas para activar pre y post
        Neurona pre1 = new Neurona(10L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        Neurona pre2 = new Neurona(11L, TipoNeurona.SENSORIAL, 0.0, PotencialMemoria.PICO);
        
        new Conexion(pre1, presináptica, 0.9, TipoConexion.QUIMICA);
        new Conexion(pre2, postsináptica, 0.9, TipoConexion.QUIMICA);
        
        // Activar múltiples veces
        for (int i = 0; i < 5; i++) {
            presináptica.resetear();
            postsináptica.resetear();
            presináptica.evaluar(i * 1000L);
            postsináptica.evaluar(i * 1000L);
            conexion.aplicarPlasticidadHebiana(i * 1000L, 100L);
        }
        
        assertEquals(5, conexion.getVecesActivadaJuntas());
    }
}
