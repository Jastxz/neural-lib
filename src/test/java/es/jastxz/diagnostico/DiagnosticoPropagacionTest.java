package es.jastxz.diagnostico;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.List;

/**
 * Test de diagnóstico para visualizar la propagación de señales
 * durante el entrenamiento de la red experimental
 */
public class DiagnosticoPropagacionTest {
    
    @Test
    @DisplayName("Diagnóstico: Propagación de señal en una época")
    void diagnosticarPropagacion() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("DIAGNÓSTICO: PROPAGACIÓN DE SEÑAL EN UNA ÉPOCA");
        System.out.println("=".repeat(80));
        
        // Crear red pequeña para diagnóstico
        // [5 inputs, 10 intermedias, 1 output]
        RedNeuralExperimental red = new RedNeuralExperimental(
            new int[]{5, 10, 1}, 0.7
        );
        
        // Ejemplo simple: secuencia [0.1, 0.2, 0.3, 0.4, 0.5] → 0.6
        double[] input = {0.1, 0.2, 0.3, 0.4, 0.5};
        double[] target = {0.6};
        
        System.out.println("\n--- CONFIGURACIÓN ---");
        System.out.println("Input: [0.1, 0.2, 0.3, 0.4, 0.5]");
        System.out.println("Target esperado: 0.6");
        System.out.println("Topología: [5, 10, 1]");
        System.out.println("Conexiones totales: " + red.getTotalConexiones());
        
        // PASO 1: Estado inicial
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PASO 1: ESTADO INICIAL (antes de procesar)");
        System.out.println("=".repeat(80));
        mostrarEstadoNeuronas(red);
        mostrarMuestraPesos(red, 5);
        
        // PASO 2: Procesar input (sin entrenar)
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PASO 2: DESPUÉS DE PROCESAR INPUT (sin entrenar)");
        System.out.println("=".repeat(80));
        double[] output1 = red.procesar(input);
        System.out.println("Output obtenido: " + output1[0]);
        System.out.println("Error: " + Math.abs(target[0] - output1[0]));
        mostrarEstadoNeuronas(red);
        mostrarActivacionDetallada(red);
        
        // PASO 3: Primera iteración de entrenamiento
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PASO 3: DESPUÉS DE 1 ITERACIÓN DE ENTRENAMIENTO");
        System.out.println("=".repeat(80));
        red.entrenar(input, target, 1);
        double[] output2 = red.procesar(input);
        System.out.println("Output obtenido: " + output2[0]);
        System.out.println("Error: " + Math.abs(target[0] - output2[0]));
        mostrarEstadoNeuronas(red);
        mostrarMuestraPesos(red, 5);
        
        // PASO 4: Después de 10 iteraciones
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PASO 4: DESPUÉS DE 10 ITERACIONES DE ENTRENAMIENTO");
        System.out.println("=".repeat(80));
        red.entrenar(input, target, 9); // 9 más para completar 10
        double[] output3 = red.procesar(input);
        System.out.println("Output obtenido: " + output3[0]);
        System.out.println("Error: " + Math.abs(target[0] - output3[0]));
        mostrarEstadoNeuronas(red);
        mostrarMuestraPesos(red, 5);
        
        // PASO 5: Después de 50 iteraciones
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PASO 5: DESPUÉS DE 50 ITERACIONES DE ENTRENAMIENTO");
        System.out.println("=".repeat(80));
        red.entrenar(input, target, 40); // 40 más para completar 50
        double[] output4 = red.procesar(input);
        System.out.println("Output obtenido: " + output4[0]);
        System.out.println("Error: " + Math.abs(target[0] - output4[0]));
        mostrarEstadoNeuronas(red);
        mostrarMuestraPesos(red, 5);
        
        // PASO 6: Después de 100 iteraciones
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PASO 6: DESPUÉS DE 100 ITERACIONES DE ENTRENAMIENTO");
        System.out.println("=".repeat(80));
        red.entrenar(input, target, 50); // 50 más para completar 100
        double[] output5 = red.procesar(input);
        System.out.println("Output obtenido: " + output5[0]);
        System.out.println("Error: " + Math.abs(target[0] - output5[0]));
        mostrarEstadoNeuronas(red);
        mostrarMuestraPesos(red, 5);
        
        // ANÁLISIS FINAL
        System.out.println("\n" + "=".repeat(80));
        System.out.println("ANÁLISIS FINAL");
        System.out.println("=".repeat(80));
        System.out.println("Evolución del output:");
        System.out.printf("  Inicial:        %.4f (error: %.4f)\n", output1[0], Math.abs(target[0] - output1[0]));
        System.out.printf("  Tras 1 iter:    %.4f (error: %.4f)\n", output2[0], Math.abs(target[0] - output2[0]));
        System.out.printf("  Tras 10 iter:   %.4f (error: %.4f)\n", output3[0], Math.abs(target[0] - output3[0]));
        System.out.printf("  Tras 50 iter:   %.4f (error: %.4f)\n", output4[0], Math.abs(target[0] - output4[0]));
        System.out.printf("  Tras 100 iter:  %.4f (error: %.4f)\n", output5[0], Math.abs(target[0] - output5[0]));
        System.out.printf("  Target:         %.4f\n", target[0]);
        
        System.out.println("\n¿Está convergiendo? " + 
            (Math.abs(target[0] - output5[0]) < Math.abs(target[0] - output1[0]) ? "SÍ ✓" : "NO ✗"));
        
        // Análisis de velocidad de convergencia
        double mejora1_10 = Math.abs(target[0] - output2[0]) - Math.abs(target[0] - output3[0]);
        double mejora10_50 = Math.abs(target[0] - output3[0]) - Math.abs(target[0] - output4[0]);
        double mejora50_100 = Math.abs(target[0] - output4[0]) - Math.abs(target[0] - output5[0]);
        
        System.out.println("\nVelocidad de convergencia:");
        System.out.printf("  Mejora 1→10:    %.4f (%.2f%% del error inicial)\n", 
            mejora1_10, (mejora1_10 / Math.abs(target[0] - output1[0])) * 100);
        System.out.printf("  Mejora 10→50:   %.4f (%.2f%% del error inicial)\n", 
            mejora10_50, (mejora10_50 / Math.abs(target[0] - output1[0])) * 100);
        System.out.printf("  Mejora 50→100:  %.4f (%.2f%% del error inicial)\n", 
            mejora50_100, (mejora50_100 / Math.abs(target[0] - output1[0])) * 100);
    }
    
    /**
     * Muestra el estado de activación de todas las capas
     */
    private void mostrarEstadoNeuronas(RedNeuralExperimental red) {
        System.out.println("\n--- Estado de Neuronas ---");
        
        // Capa sensorial
        List<Neurona> sensoriales = red.getCapaSensorial();
        long activasSensoriales = sensoriales.stream().filter(Neurona::estaActiva).count();
        System.out.printf("Capa Sensorial:   %d/%d activas (%.1f%%)\n", 
            activasSensoriales, sensoriales.size(), 
            (activasSensoriales * 100.0) / sensoriales.size());
        
        // Capas intermedias
        for (int i = 0; i < red.getCapasInterneuronas().size(); i++) {
            List<Neurona> capa = red.getCapasInterneuronas().get(i);
            long activas = capa.stream().filter(Neurona::estaActiva).count();
            System.out.printf("Capa Intermedia %d: %d/%d activas (%.1f%%)\n", 
                i + 1, activas, capa.size(), 
                (activas * 100.0) / capa.size());
        }
        
        // Capa motora
        List<Neurona> motoras = red.getCapaMotora();
        long activasMotoras = motoras.stream().filter(Neurona::estaActiva).count();
        System.out.printf("Capa Motora:      %d/%d activas (%.1f%%)\n", 
            activasMotoras, motoras.size(), 
            (activasMotoras * 100.0) / motoras.size());
    }
    
    /**
     * Muestra activación detallada con potenciales
     */
    private void mostrarActivacionDetallada(RedNeuralExperimental red) {
        System.out.println("\n--- Activación Detallada ---");
        
        // Mostrar neuronas sensoriales con su potencial
        System.out.println("\nNeuronas Sensoriales:");
        List<Neurona> sensoriales = red.getCapaSensorial();
        for (int i = 0; i < sensoriales.size(); i++) {
            Neurona n = sensoriales.get(i);
            System.out.printf("  S%d: %s | Potencial: %.2f | Umbral: %.2f\n",
                i, n.estaActiva() ? "ACTIVA  " : "INACTIVA",
                n.getPotencial(), n.getUmbralActivacion());
        }
        
        // Mostrar neuronas intermedias
        System.out.println("\nNeuronas Intermedias (primeras 5):");
        List<Neurona> intermedias = red.getCapasInterneuronas().get(0);
        for (int i = 0; i < Math.min(5, intermedias.size()); i++) {
            Neurona n = intermedias.get(i);
            System.out.printf("  I%d: %s | Potencial: %.2f | Umbral: %.2f\n",
                i, n.estaActiva() ? "ACTIVA  " : "INACTIVA",
                n.getPotencial(), n.getUmbralActivacion());
        }
        
        // Mostrar neurona motora
        System.out.println("\nNeurona Motora:");
        Neurona motora = red.getCapaMotora().get(0);
        System.out.printf("  M0: %s | Potencial: %.2f | Umbral: %.2f\n",
            motora.estaActiva() ? "ACTIVA  " : "INACTIVA",
            motora.getPotencial(), motora.getUmbralActivacion());
    }
    
    /**
     * Muestra una muestra de pesos de conexiones
     */
    private void mostrarMuestraPesos(RedNeuralExperimental red, int cantidad) {
        System.out.println("\n--- Muestra de Pesos (primeras " + cantidad + " conexiones) ---");
        
        List<Conexion> conexiones = red.getConexiones();
        for (int i = 0; i < Math.min(cantidad, conexiones.size()); i++) {
            Conexion c = conexiones.get(i);
            Neurona pre = c.getPresinaptica();
            
            System.out.printf("  Conexión %d: %s → %s | Peso: %+.4f | Pre activa: %s | Recursos: %.3f\n",
                i,
                pre.getTipo().toString().substring(0, 4),
                c.getPostsinapticas().get(0).getTipo().toString().substring(0, 4),
                c.getPeso(),
                pre.estaActiva() ? "SÍ" : "NO",
                c.getRecursosAsignados());
        }
        
        // Estadísticas de pesos
        double sumaPesos = 0.0;
        int positivos = 0;
        int negativos = 0;
        
        for (Conexion c : conexiones) {
            sumaPesos += c.getPeso();
            if (c.getPeso() > 0) positivos++;
            else if (c.getPeso() < 0) negativos++;
        }
        
        System.out.printf("\nEstadísticas de pesos:\n");
        System.out.printf("  Promedio: %.4f\n", sumaPesos / conexiones.size());
        System.out.printf("  Positivos: %d (%.1f%%)\n", positivos, (positivos * 100.0) / conexiones.size());
        System.out.printf("  Negativos: %d (%.1f%%)\n", negativos, (negativos * 100.0) / conexiones.size());
    }
}
