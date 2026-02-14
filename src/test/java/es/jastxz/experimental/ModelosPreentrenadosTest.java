package es.jastxz.experimental;

import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests que demuestran el uso de modelos pre-entrenados
 * 
 * Estos modelos ya están entrenados y guardados en resources,
 * por lo que se pueden cargar y usar inmediatamente sin reentrenar.
 */
public class ModelosPreentrenadosTest {
    
    /**
     * Test de modelo OR pre-entrenado
     */
    @Test
    void testModeloORPreentrenado() throws IOException, ClassNotFoundException {
        // Cargar modelo pre-entrenado desde archivo
        RedNeuralExperimental red = RedNeuralExperimental.cargar(
            "src/main/resources/modelosExperimentales/modeloOR_experimental.nn"
        );
        
        System.out.println("\n=== Modelo OR Pre-entrenado ===");
        System.out.println("Topología: " + red.toString());
        
        // Usar directamente sin entrenar
        red.resetear();
        
        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        double[][] targets = {
            {0.0},
            {1.0},
            {1.0},
            {1.0}
        };
        
        System.out.println("Resultados:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = red.procesar(inputs[i]);
            System.out.printf("  %.0f OR %.0f = %.4f (esperado: %.0f)\n", 
                inputs[i][0], inputs[i][1], output[0], targets[i][0]);
            
            // Validar con tolerancia
            if (targets[i][0] == 0.0) {
                assertTrue(output[0] < 0.4, "Debería ser ~0.0");
            } else {
                assertTrue(output[0] > 0.6, "Debería ser ~1.0");
            }
        }
    }
    
    /**
     * Test de modelo AND pre-entrenado
     */
    @Test
    void testModeloANDPreentrenado() throws IOException, ClassNotFoundException {
        RedNeuralExperimental red = RedNeuralExperimental.cargar(
            "src/main/resources/modelosExperimentales/modeloAND_experimental.nn"
        );
        
        System.out.println("\n=== Modelo AND Pre-entrenado ===");
        
        red.resetear();
        
        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        System.out.println("Resultados:");
        for (double[] input : inputs) {
            double[] output = red.procesar(input);
            System.out.printf("  %.0f AND %.0f = %.4f\n", 
                input[0], input[1], output[0]);
        }
        
        // Verificar que al menos distingue 0,0 y 1,1
        red.resetear();
        double[] output00 = red.procesar(inputs[0]);
        double[] output11 = red.procesar(inputs[3]);
        
        assertTrue(output00[0] < 0.5, "0,0 debería dar valor bajo");
        assertTrue(output11[0] > 0.5, "1,1 debería dar valor alto");
    }
    
    /**
     * Test de modelo XOR pre-entrenado
     */
    @Test
    void testModeloXORPreentrenado() throws IOException, ClassNotFoundException {
        RedNeuralExperimental red = RedNeuralExperimental.cargar(
            "src/main/resources/modelosExperimentales/modeloXOR_experimental.nn"
        );
        
        System.out.println("\n=== Modelo XOR Pre-entrenado ===");
        
        // Contar conexiones inhibitorias
        long inhibitorias = red.getConexiones().stream()
            .filter(c -> c.getPeso() < 0)
            .count();
        System.out.println("Conexiones inhibitorias: " + inhibitorias);
        
        red.resetear();
        
        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        System.out.println("Resultados:");
        for (double[] input : inputs) {
            double[] output = red.procesar(input);
            System.out.printf("  %.0f XOR %.0f = %.4f\n", 
                input[0], input[1], output[0]);
        }
        
        // Verificar que tiene conexiones inhibitorias
        assertTrue(inhibitorias > 0, "Debería tener conexiones inhibitorias");
    }
    
    /**
     * Test de modelo de Memoria pre-entrenado
     */
    @Test
    void testModeloMemoriaPreentrenado() throws IOException, ClassNotFoundException {
        RedNeuralExperimental red = RedNeuralExperimental.cargar(
            "src/main/resources/modelosExperimentales/modeloMemoria_experimental.nn"
        );
        
        System.out.println("\n=== Modelo Memoria Pre-entrenado ===");
        System.out.println("Engramas: " + red.getEngramas().size());
        
        red.resetear();
        
        double[][] patrones = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0}
        };
        
        System.out.println("Recuperación de patrones:");
        for (int i = 0; i < patrones.length; i++) {
            double[] output = red.procesar(patrones[i]);
            System.out.printf("  Patrón %d: [%.2f, %.2f, %.2f, %.2f]\n", 
                i, output[0], output[1], output[2], output[3]);
        }
        
        // Verificar que tiene engramas
        assertTrue(red.getEngramas().size() > 0, "Debería tener engramas formados");
    }
    
    /**
     * Test de comparación: entrenar vs cargar pre-entrenado
     * Demuestra la ventaja de usar modelos pre-entrenados
     */
    @Test
    void testComparacionTiempos() throws IOException, ClassNotFoundException {
        System.out.println("\n=== Comparación: Entrenar vs Cargar ===");
        
        // Opción 1: Entrenar desde cero
        long inicioEntrenar = System.currentTimeMillis();
        RedNeuralExperimental redNueva = new RedNeuralExperimental(new int[]{2, 4, 1}, 0.9);
        
        double[][] inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
        double[][] targets = {{0.0}, {1.0}, {1.0}, {1.0}};
        
        for (int i = 0; i < 1000; i++) {
            for (int j = 0; j < inputs.length; j++) {
                redNueva.entrenar(inputs[j], targets[j], 5);
            }
        }
        long tiempoEntrenar = System.currentTimeMillis() - inicioEntrenar;
        
        // Opción 2: Cargar pre-entrenado
        long inicioCargar = System.currentTimeMillis();
        RedNeuralExperimental.cargar(
            "src/main/resources/modelosExperimentales/modeloOR_experimental.nn"
        );
        long tiempoCargar = System.currentTimeMillis() - inicioCargar;
        
        System.out.println("Tiempo entrenar desde cero: " + tiempoEntrenar + " ms");
        System.out.println("Tiempo cargar pre-entrenado: " + tiempoCargar + " ms");
        System.out.println("Mejora: " + (tiempoEntrenar / (double) tiempoCargar) + "x más rápido");
        
        // Cargar debería ser mucho más rápido
        assertTrue(tiempoCargar < tiempoEntrenar, 
            "Cargar debería ser más rápido que entrenar");
    }
}
