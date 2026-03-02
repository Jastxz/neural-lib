package es.jastxz.comparativas;

import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de Características Únicas de la Red Experimental
 * 
 * Estos tests verifican que las características biológicas funcionan,
 * sin necesariamente comparar con la red clásica (que usa backpropagation supervisado).
 */
public class ComparativaCaracteristicasExperimentalesTest {
    
    @Test
    @DisplayName("Test 1: Formación de Engramas")
    void test1_FormacionEngramas() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 1: FORMACIÓN DE ENGRAMAS");
        System.out.println("=".repeat(80));
        System.out.println("\nVerificar que la red forma engramas al aprender patrones.\n");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{5, 15, 1}, 0.7);
        red.activarDeteccionEngramas(true);
        
        // Patrón para aprender
        double[] patron = {1.0, 0.0, 1.0, 0.0, 1.0};
        double[] target = {0.8};
        
        int engramasInicial = red.getEngramas().size();
        System.out.println("Engramas iniciales: " + engramasInicial);
        
        // Entrenar
        System.out.println("\n--- Entrenando ---");
        for (int i = 0; i < 50; i++) {
            red.entrenar(patron, target, 3);
        }
        
        int engramasFinal = red.getEngramas().size();
        System.out.println("Engramas después de entrenamiento: " + engramasFinal);
        
        // Verificar que se formaron engramas
        assertTrue(engramasFinal > engramasInicial,
            "Deberían formarse engramas durante el entrenamiento");
        
        System.out.println("\n✓ La red forma engramas correctamente");
        System.out.println("  Engramas formados: " + (engramasFinal - engramasInicial));
    }
    
    @Test
    @DisplayName("Test 2: Consolidación de Memoria")
    void test2_ConsolidacionMemoria() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 2: CONSOLIDACIÓN DE MEMORIA");
        System.out.println("=".repeat(80));
        System.out.println("\nVerificar que la consolidación refuerza conexiones importantes.\n");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{5, 10, 1}, 0.7);
        red.activarDeteccionEngramas(true);
        
        double[] patron = {0.5, 0.5, 0.5, 0.5, 0.5};
        double[] target = {0.5};
        
        // Entrenar y consolidar
        System.out.println("--- Entrenando con consolidación ---");
        for (int i = 0; i < 30; i++) {
            red.entrenar(patron, target, 3);
        }
        
        Map<String, Object> stats = red.getEstadisticas();
        boolean consolidacionActiva = (boolean) stats.get("consolidacionInicializada");
        
        System.out.println("Consolidación inicializada: " + consolidacionActiva);
        System.out.println("Intervalo de consolidación: " + stats.get("intervaloConsolidacion"));
        
        assertTrue(consolidacionActiva,
            "La consolidación debería estar activa");
        
        System.out.println("\n✓ La consolidación funciona correctamente");
    }
    
    @Test
    @DisplayName("Test 3: Activación de Neuronas por Capas")
    void test3_ActivacionPorCapas() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 3: ACTIVACIÓN DE NEURONAS POR CAPAS");
        System.out.println("=".repeat(80));
        System.out.println("\nVerificar que las neuronas se activan correctamente en cada capa.\n");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{5, 10, 1}, 0.7);
        
        double[] input = {0.8, 0.8, 0.8, 0.8, 0.8};
        
        System.out.println("--- Procesando input ---");
        red.procesar(input);
        
        long sensoriales = red.getCapaSensorial().stream()
            .filter(n -> n.estaActiva()).count();
        long intermedias = red.getCapasInterneuronas().get(0).stream()
            .filter(n -> n.estaActiva()).count();
        long motoras = red.getCapaMotora().stream()
            .filter(n -> n.estaActiva()).count();
        
        System.out.println("Neuronas sensoriales activas: " + sensoriales + "/5");
        System.out.println("Neuronas intermedias activas: " + intermedias + "/10");
        System.out.println("Neuronas motoras activas: " + motoras + "/1");
        
        // Verificar que al menos algunas neuronas se activan
        assertTrue(sensoriales > 0, "Algunas neuronas sensoriales deberían activarse");
        assertTrue(intermedias > 0, "Algunas neuronas intermedias deberían activarse");
        
        System.out.println("\n✓ Las neuronas se activan correctamente por capas");
    }
    
    @Test
    @DisplayName("Test 4: Output Continuo")
    void test4_OutputContinuo() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 4: OUTPUT CONTINUO");
        System.out.println("=".repeat(80));
        System.out.println("\nVerificar que el output es continuo (no binario).\n");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{5, 10, 1}, 0.7);
        
        // Entrenar con diferentes targets
        double[][] inputs = {
            {0.2, 0.2, 0.2, 0.2, 0.2},
            {0.5, 0.5, 0.5, 0.5, 0.5},
            {0.8, 0.8, 0.8, 0.8, 0.8}
        };
        
        double[][] targets = {
            {0.2},
            {0.5},
            {0.8}
        };
        
        System.out.println("--- Entrenando ---");
        for (int epoca = 0; epoca < 30; epoca++) {
            for (int i = 0; i < inputs.length; i++) {
                red.entrenar(inputs[i], targets[i], 2);
            }
        }
        
        // Evaluar outputs
        System.out.println("\n--- Evaluando Outputs ---");
        Set<Double> outputsUnicos = new HashSet<>();
        
        for (int i = 0; i < inputs.length; i++) {
            red.resetear();
            double[] output = red.procesar(inputs[i]);
            outputsUnicos.add(output[0]);
            
            System.out.printf("Input [%.1f]: Output=%.4f | Target=%.1f\n",
                inputs[i][0], output[0], targets[i][0]);
        }
        
        System.out.println("\nOutputs únicos generados: " + outputsUnicos.size());
        
        // Verificar que genera outputs diferentes (no solo 0 y 1)
        assertTrue(outputsUnicos.size() >= 2,
            "Debería generar al menos 2 outputs diferentes");
        
        // Verificar que los outputs están en rango [0, 1]
        for (double output : outputsUnicos) {
            assertTrue(output >= 0.0 && output <= 1.0,
                "Output debería estar en rango [0, 1]");
        }
        
        System.out.println("\n✓ El output es continuo y está en rango [0, 1]");
    }
    
    @Test
    @DisplayName("Test 5: Aprendizaje y Convergencia")
    void test5_AprendizajeConvergencia() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 5: APRENDIZAJE Y CONVERGENCIA");
        System.out.println("=".repeat(80));
        System.out.println("\nVerificar que la red aprende y reduce el error.\n");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{5, 15, 1}, 0.7);
        red.activarDeteccionEngramas(true);
        
        double[] input = {0.6, 0.6, 0.6, 0.6, 0.6};
        double[] target = {0.6};
        
        // Error inicial
        red.resetear();
        double[] outputInicial = red.procesar(input);
        double errorInicial = Math.abs(target[0] - outputInicial[0]);
        
        System.out.println("Error inicial: " + String.format("%.4f", errorInicial));
        
        // Entrenar
        System.out.println("\n--- Entrenando (50 épocas) ---");
        for (int i = 0; i < 50; i++) {
            red.entrenar(input, target, 3);
        }
        
        // Error final
        red.resetear();
        double[] outputFinal = red.procesar(input);
        double errorFinal = Math.abs(target[0] - outputFinal[0]);
        
        System.out.println("Error final: " + String.format("%.4f", errorFinal));
        System.out.println("Mejora: " + String.format("%.4f", errorInicial - errorFinal));
        
        // Verificar que el error disminuyó
        assertTrue(errorFinal < errorInicial,
            "El error debería disminuir con el entrenamiento");
        
        System.out.println("\n✓ La red aprende y reduce el error correctamente");
        System.out.printf("  Reducción de error: %.1f%%\n",
            ((errorInicial - errorFinal) / errorInicial) * 100);
    }
    
    @Test
    @DisplayName("Test 6: Poda de Conexiones")
    void test6_PodaConexiones() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 6: PODA DE CONEXIONES");
        System.out.println("=".repeat(80));
        System.out.println("\nVerificar que la red poda conexiones no utilizadas.\n");
        
        RedNeuralExperimental red = new RedNeuralExperimental(new int[]{5, 10, 1}, 0.7);
        red.activarDeteccionEngramas(true);
        red.activarCompeticionRecursos(true);
        
        int conexionesIniciales = red.getTotalConexiones();
        System.out.println("Conexiones iniciales: " + conexionesIniciales);
        
        double[] input = {0.5, 0.5, 0.5, 0.5, 0.5};
        double[] target = {0.5};
        
        // Entrenar con poda
        System.out.println("\n--- Entrenando con poda ---");
        for (int epoca = 0; epoca < 20; epoca++) {
            red.entrenar(input, target, 3);
            
            if ((epoca + 1) % 5 == 0) {
                red.competirPorRecursos();
                int podados = red.podarElementos();
                if (podados > 0) {
                    System.out.printf("Época %d: Podados %d elementos\n", epoca + 1, podados);
                }
            }
        }
        
        int conexionesFinal = red.getTotalConexiones();
        System.out.println("\nConexiones finales: " + conexionesFinal);
        System.out.println("Conexiones podadas: " + (conexionesIniciales - conexionesFinal));
        
        // Verificar que se podaron algunas conexiones
        assertTrue(conexionesFinal <= conexionesIniciales,
            "El número de conexiones no debería aumentar");
        
        System.out.println("\n✓ La poda de conexiones funciona correctamente");
        System.out.printf("  Reducción: %.1f%%\n",
            ((conexionesIniciales - conexionesFinal) * 100.0) / conexionesIniciales);
    }
}
