package es.jastxz.experimental;

import es.jastxz.nn.RedNeuralExperimental;

import java.io.IOException;

/**
 * Generador de modelos pre-entrenados para la red experimental
 * Ejecutar este main para generar los modelos y guardarlos en resources
 */
public class GeneradorModelosExperimentales {
    
    public static void main(String[] args) {
        try {
            System.out.println("=== Generando Modelos Pre-entrenados ===\n");
            
            generarModeloOR();
            generarModeloAND();
            generarModeloXOR();
            generarModeloMemoria();
            
            System.out.println("\n=== Todos los modelos generados exitosamente ===");
            System.out.println("Los modelos están en: src/main/resources/modelosExperimentales/");
            
        } catch (IOException e) {
            System.err.println("Error generando modelos: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Genera modelo pre-entrenado para función OR
     */
    private static void generarModeloOR() throws IOException {
        System.out.println("Generando modelo OR...");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 1}, 0.9);
        
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
        
        // Entrenar
        for (int epoca = 0; epoca < 1000; epoca++) {
            for (int i = 0; i < inputs.length; i++) {
                red.entrenar(inputs[i], targets[i], 5);
            }
        }
        
        // Validar
        red.resetear();
        System.out.println("Resultados OR:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = red.procesar(inputs[i]);
            System.out.printf("  %.0f,%.0f -> %.4f (esperado: %.0f)\n", 
                inputs[i][0], inputs[i][1], output[0], targets[i][0]);
        }
        
        // Guardar
        red.guardar("src/main/resources/modelosExperimentales/modeloOR_experimental.nn");
        System.out.println("✓ Modelo OR guardado\n");
    }
    
    /**
     * Genera modelo pre-entrenado para función AND
     */
    private static void generarModeloAND() throws IOException {
        System.out.println("Generando modelo AND...");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 6, 1}, 0.9);
        
        double[][] inputs = {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        };
        
        double[][] targets = {
            {0.0},
            {0.0},
            {0.0},
            {1.0}
        };
        
        // Entrenar con énfasis en casos negativos
        for (int epoca = 0; epoca < 3000; epoca++) {
            for (int i = 0; i < inputs.length; i++) {
                int repeticiones = (i < 3) ? 10 : 5;
                red.entrenar(inputs[i], targets[i], repeticiones);
            }
        }
        
        // Validar
        red.resetear();
        System.out.println("Resultados AND:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = red.procesar(inputs[i]);
            System.out.printf("  %.0f,%.0f -> %.4f (esperado: %.0f)\n", 
                inputs[i][0], inputs[i][1], output[0], targets[i][0]);
        }
        
        // Guardar
        red.guardar("src/main/resources/modelosExperimentales/modeloAND_experimental.nn");
        System.out.println("✓ Modelo AND guardado\n");
    }
    
    /**
     * Genera modelo pre-entrenado para función XOR
     */
    private static void generarModeloXOR() throws IOException {
        System.out.println("Generando modelo XOR...");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 8, 1}, 0.9);
        
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
            {0.0}
        };
        
        // Entrenar intensivamente
        for (int epoca = 0; epoca < 5000; epoca++) {
            for (int i = 0; i < inputs.length; i++) {
                int repeticiones = (i == 3) ? 15 : 10;
                red.entrenar(inputs[i], targets[i], repeticiones);
            }
            
            if (epoca % 500 == 0 && epoca > 0) {
                red.resetear();
            }
        }
        
        // Validar
        red.resetear();
        System.out.println("Resultados XOR:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = red.procesar(inputs[i]);
            System.out.printf("  %.0f,%.0f -> %.4f (esperado: %.0f)\n", 
                inputs[i][0], inputs[i][1], output[0], targets[i][0]);
        }
        
        long inhibitorias = red.getConexiones().stream()
            .filter(c -> c.getPeso() < 0)
            .count();
        System.out.println("  Conexiones inhibitorias: " + inhibitorias);
        
        // Guardar
        red.guardar("src/main/resources/modelosExperimentales/modeloXOR_experimental.nn");
        System.out.println("✓ Modelo XOR guardado\n");
    }
    
    /**
     * Genera modelo pre-entrenado para memoria asociativa
     */
    private static void generarModeloMemoria() throws IOException {
        System.out.println("Generando modelo de Memoria Asociativa...");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{4, 8, 4}, 0.9);
        red.activarDeteccionEngramas(true);
        
        double[][] patrones = {
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 1.0}
        };
        
        // Entrenar
        for (int epoca = 0; epoca < 1000; epoca++) {
            for (double[] patron : patrones) {
                red.entrenar(patron, patron, 5);
            }
        }
        
        // Validar
        red.resetear();
        System.out.println("Resultados Memoria:");
        for (int i = 0; i < patrones.length; i++) {
            double[] output = red.procesar(patrones[i]);
            System.out.printf("  Patrón %d: [%.2f, %.2f, %.2f, %.2f]\n", 
                i, output[0], output[1], output[2], output[3]);
        }
        System.out.println("  Engramas formados: " + red.getEngramas().size());
        
        // Guardar
        red.guardar("src/main/resources/modelosExperimentales/modeloMemoria_experimental.nn");
        System.out.println("✓ Modelo Memoria guardado\n");
    }
}
