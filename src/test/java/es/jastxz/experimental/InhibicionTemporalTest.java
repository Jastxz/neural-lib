package es.jastxz.experimental;

import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.nn.Neurona;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test para verificar que la inhibición lateral funciona temporalmente
 * Las neuronas activas deberían inhibir a sus vecinas con el tiempo
 */
public class InhibicionTemporalTest {
    
    @Test
    @DisplayName("Test: Inhibición temporal entre neuronas")
    void testInhibicionTemporal() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST: INHIBICIÓN TEMPORAL ENTRE NEURONAS");
        System.out.println("=".repeat(60));
        
        // Crear red con una capa intermedia
        RedNeuralExperimental red = new RedNeuralExperimental(
            new int[]{5, 10, 1},  // 5 input, 10 hidden, 1 output
            0.7  // Densidad normal
        );
        
        System.out.println("\nConexiones totales: " + red.getTotalConexiones());
        
        // Input constante que active todas las neuronas sensoriales
        double[] input = {1.0, 1.0, 1.0, 1.0, 1.0};
        
        // IMPORTANTE: NO resetear entre procesamientos
        // Permitir que la inhibición se acumule con el tiempo
        System.out.println("\n--- Procesamiento múltiple SIN resetear (inhibición acumulativa) ---");
        for (int i = 0; i < 10; i++) {
            red.procesar(input);
            
            // Contar neuronas activas en capa intermedia
            List<Neurona> capaInter = red.getCapasInterneuronas().get(0);
            long neuronasActivas = capaInter.stream()
                .filter(Neurona::estaActiva)
                .count();
            
            System.out.printf("Iteración %d: %d/%d neuronas activas (%.1f%%)\n",
                i + 1, neuronasActivas, capaInter.size(),
                (neuronasActivas * 100.0) / capaInter.size());
            
            // Mostrar potenciales de las primeras 5 neuronas
            System.out.print("  Potenciales: ");
            for (int j = 0; j < Math.min(5, capaInter.size()); j++) {
                System.out.printf("N%d=%.3f ", j, capaInter.get(j).getPotencial());
            }
            System.out.println();
        }
        
        System.out.println("\n--- Análisis de conexiones laterales ---");
        
        // Contar conexiones laterales
        List<Neurona> capaInter = red.getCapasInterneuronas().get(0);
        int conexionesLaterales = 0;
        int conexionesInhibitorias = 0;
        int conexionesExcitatorias = 0;
        
        for (var conexion : red.getConexiones()) {
            Neurona pre = conexion.getPresinaptica();
            List<Neurona> posts = conexion.getPostsinapticas();
            
            // Verificar si es conexión lateral (pre y post en la misma capa)
            if (capaInter.contains(pre)) {
                for (Neurona post : posts) {
                    if (capaInter.contains(post)) {
                        conexionesLaterales++;
                        if (conexion.getPeso() < 0) {
                            conexionesInhibitorias++;
                        } else {
                            conexionesExcitatorias++;
                        }
                        break;
                    }
                }
            }
        }
        
        System.out.println("Total conexiones laterales: " + conexionesLaterales);
        System.out.println("  Inhibitorias: " + conexionesInhibitorias);
        System.out.println("  Excitatorias: " + conexionesExcitatorias);
        System.out.printf("  Proporción inhibitorias: %.1f%%\n",
            (conexionesInhibitorias * 100.0) / conexionesLaterales);
        
        // Calcular densidad real de conexiones laterales
        int maxConexionesLaterales = capaInter.size() * (capaInter.size() - 1);
        double densidadReal = (conexionesLaterales * 100.0) / maxConexionesLaterales;
        System.out.printf("  Densidad real: %.1f%% (esperado: 37%%)\n", densidadReal);
        
        System.out.println("\n--- Conclusión ---");
        System.out.println("Con inhibición acumulativa (sin resetear), deberíamos ver:");
        System.out.println("1. Reducción progresiva de neuronas activas");
        System.out.println("2. Convergencia a un estado estable (sparse coding)");
        System.out.println("3. Solo las neuronas más fuertes permanecen activas");
    }
    
    @Test
    @DisplayName("Test: Efecto de inhibición en una sola propagación")
    void testInhibicionEnPropagacion() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST: EFECTO DE INHIBICIÓN EN UNA PROPAGACIÓN");
        System.out.println("=".repeat(60));
        
        // Crear red pequeña para análisis detallado
        RedNeuralExperimental red = new RedNeuralExperimental(
            new int[]{3, 5, 1},  // 3 input, 5 hidden, 1 output
            0.7
        );
        
        // Input que active todas las sensoriales
        double[] input = {1.0, 1.0, 1.0};
        
        System.out.println("\n--- Antes de procesar ---");
        List<Neurona> capaInter = red.getCapasInterneuronas().get(0);
        for (int i = 0; i < capaInter.size(); i++) {
            Neurona n = capaInter.get(i);
            System.out.printf("Neurona %d: activa=%s, potencial=%.3f\n",
                i, n.estaActiva(), n.getPotencial());
        }
        
        // Procesar
        red.procesar(input);
        
        System.out.println("\n--- Después de procesar ---");
        for (int i = 0; i < capaInter.size(); i++) {
            Neurona n = capaInter.get(i);
            System.out.printf("Neurona %d: activa=%s, potencial=%.3f\n",
                i, n.estaActiva(), n.getPotencial());
        }
        
        // Contar activas
        long activas = capaInter.stream().filter(Neurona::estaActiva).count();
        System.out.printf("\nNeuronas activas: %d/%d (%.1f%%)\n",
            activas, capaInter.size(), (activas * 100.0) / capaInter.size());
        
        // Analizar conexiones laterales y su efecto
        System.out.println("\n--- Análisis de señales recibidas ---");
        for (int i = 0; i < capaInter.size(); i++) {
            Neurona neurona = capaInter.get(i);
            
            // Contar señales excitatorias e inhibitorias recibidas
            double señalExcitatoria = 0.0;
            double señalInhibitoria = 0.0;
            int conexionesExc = 0;
            int conexionesInh = 0;
            
            for (var conexion : red.getConexiones()) {
                if (conexion.getPostsinapticas().contains(neurona)) {
                    Neurona pre = conexion.getPresinaptica();
                    if (pre.estaActiva()) {
                        double señal = conexion.getPeso() * pre.getPotencial();
                        if (señal > 0) {
                            señalExcitatoria += señal;
                            conexionesExc++;
                        } else {
                            señalInhibitoria += señal;
                            conexionesInh++;
                        }
                    }
                }
            }
            
            double señalNeta = señalExcitatoria + señalInhibitoria;
            System.out.printf("N%d: exc=%.3f (%d), inh=%.3f (%d), neta=%.3f, activa=%s\n",
                i, señalExcitatoria, conexionesExc, señalInhibitoria, conexionesInh,
                señalNeta, neurona.estaActiva());
        }
    }
}
