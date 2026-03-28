package es.jastxz.experimental;

import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de persistencia para RedNeuralExperimental
 * Verifica que los modelos entrenados se puedan guardar y cargar correctamente
 */
public class PersistenciaExperimentalTest {
    
    @TempDir
    Path tempDir;
    
    /**
     * Test básico de guardar y cargar
     */
    @Test
    void testGuardarYCargar() throws IOException, ClassNotFoundException {
        // Crear y entrenar una red
        RedNeuralExperimental redOriginal = new RedNeuralExperimental(new int[]{2, 4, 1}, 0.9);
        
        double[][] inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        double[][] targets = {{0.0}, {1.0}, {1.0}, {1.0}};
        
        // Entrenar un poco
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < inputs.length; j++) {
                redOriginal.entrenar(inputs[j], targets[j], 3);
            }
        }
        
        // Guardar
        String filename = tempDir.resolve("test_red_experimental.nn").toString();
        redOriginal.guardar(filename);
        
        // Cargar
        RedNeuralExperimental redCargada = RedNeuralExperimental.cargar(filename);
        
        // Verificar que son equivalentes
        assertNotNull(redCargada);
        assertEquals(redOriginal.getTotalNeuronas(), redCargada.getTotalNeuronas());
        assertEquals(redOriginal.getTotalConexiones(), redCargada.getTotalConexiones());
        
        // Verificar que producen los mismos resultados
        redOriginal.resetear();
        redCargada.resetear();
        
        for (double[] input : inputs) {
            double[] outputOriginal = redOriginal.procesar(input);
            double[] outputCargada = redCargada.procesar(input);
            
            assertEquals(outputOriginal.length, outputCargada.length);
            for (int i = 0; i < outputOriginal.length; i++) {
                assertEquals(outputOriginal[i], outputCargada[i], 0.0001, 
                    "Los outputs deberían ser idénticos");
            }
        }
    }
    
    /**
     * Test de guardar red con engramas
     */
    @Test
    void testGuardarConEngramas() throws IOException, ClassNotFoundException {
        RedNeuralExperimental redOriginal = new RedNeuralExperimental(new int[]{3, 5, 3}, 0.9);
        redOriginal.activarDeteccionEngramas(true);
        
        double[][] patrones = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        
        // Entrenar para formar engramas
        for (int i = 0; i < 200; i++) {
            for (double[] patron : patrones) {
                redOriginal.entrenar(patron, patron, 3);
            }
        }
        
        int engramasOriginales = redOriginal.getEngramas().size();
        
        // Guardar y cargar
        String filename = tempDir.resolve("test_red_con_engramas.nn").toString();
        redOriginal.guardar(filename);
        RedNeuralExperimental redCargada = RedNeuralExperimental.cargar(filename);
        
        // Verificar que los engramas se guardaron
        assertEquals(engramasOriginales, redCargada.getEngramas().size(), 
            "Los engramas deberían persistir");
    }
    
    /**
     * Test de guardar red con diferentes configuraciones
     */
    @Test
    void testGuardarConConfiguraciones() throws IOException, ClassNotFoundException {
        RedNeuralExperimental redOriginal = new RedNeuralExperimental(new int[]{4, 6, 4}, 0.8);
        
        // Activar varios sistemas
        redOriginal.activarModoPredictivo(true);
        redOriginal.activarCompeticionRecursos(true);
        redOriginal.activarDeteccionEngramas(true);
        
        // Entrenar
        double[] input = {1.0, 0.0, 1.0, 0.0};
        double[] target = {0.0, 1.0, 0.0, 1.0};
        
        for (int i = 0; i < 100; i++) {
            redOriginal.entrenar(input, target, 5);
        }
        
        // Guardar y cargar
        String filename = tempDir.resolve("test_red_configurada.nn").toString();
        redOriginal.guardar(filename);
        RedNeuralExperimental redCargada = RedNeuralExperimental.cargar(filename);
        
        // Verificar configuraciones
        assertNotNull(redCargada);
        
        // Verificar que funciona después de cargar
        redCargada.resetear();
        double[] output = redCargada.procesar(input);
        assertNotNull(output);
        assertEquals(target.length, output.length);
    }
    
    /**
     * Test de guardar red con pesos negativos (inhibición)
     */
    @Test
    void testGuardarConPesosNegativos() throws IOException, ClassNotFoundException {
        RedNeuralExperimental redOriginal = new RedNeuralExperimental(new int[]{2, 6, 1}, 0.9);
        
        // Entrenar XOR (usa inhibición)
        double[][] inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        double[][] targets = {{0.0}, {1.0}, {1.0}, {0.0}};
        
        for (int i = 0; i < 500; i++) {
            for (int j = 0; j < inputs.length; j++) {
                redOriginal.entrenar(inputs[j], targets[j], 5);
            }
        }
        
        // Contar conexiones inhibitorias antes de guardar
        long inhibitoriasOriginales = redOriginal.getConexiones().stream()
            .filter(c -> c.getPeso() < 0)
            .count();
        
        // Guardar y cargar
        String filename = tempDir.resolve("test_red_inhibicion.nn").toString();
        redOriginal.guardar(filename);
        RedNeuralExperimental redCargada = RedNeuralExperimental.cargar(filename);
        
        // Verificar que las conexiones inhibitorias se guardaron
        long inhibitoriasCargadas = redCargada.getConexiones().stream()
            .filter(c -> c.getPeso() < 0)
            .count();
        
        assertEquals(inhibitoriasOriginales, inhibitoriasCargadas, 
            "Las conexiones inhibitorias deberían persistir");
        
        // Verificar que produce resultados similares
        redOriginal.resetear();
        redCargada.resetear();
        
        double[] outputOriginal = redOriginal.procesar(inputs[0]);
        double[] outputCargada = redCargada.procesar(inputs[0]);
        
        assertEquals(outputOriginal[0], outputCargada[0], 0.0001);
    }
    
    /**
     * Test de que el archivo se crea correctamente
     */
    @Test
    void testArchivoSeCreo() throws IOException {
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 3, 1}, 0.9);
        
        String filename = tempDir.resolve("test_archivo.nn").toString();
        red.guardar(filename);
        
        // Verificar que el archivo existe
        assertTrue(tempDir.resolve("test_archivo.nn").toFile().exists(), 
            "El archivo debería existir");
        
        // Verificar que tiene contenido
        assertTrue(tempDir.resolve("test_archivo.nn").toFile().length() > 0, 
            "El archivo debería tener contenido");
    }
}
