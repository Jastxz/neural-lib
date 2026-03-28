package es.jastxz.comparativas;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test comparativo: Aprendizaje Incremental
 * 
 * La red experimental debería ser MEJOR porque:
 * - Puede aprender nuevos patrones sin olvidar los antiguos
 * - Los engramas permiten aprendizaje continuo
 * - La consolidación protege conocimiento previo
 */
public class ComparativaAprendizajeIncrementalTest {
    
    @Test
    @DisplayName("Test 1: Aprendizaje Secuencial de Tareas")
    void test1_AprendizajeSecuencialTareas() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 1: APRENDIZAJE SECUENCIAL DE TAREAS");
        System.out.println("=".repeat(80));
        System.out.println("\nAprender 3 tareas secuencialmente sin olvidar las anteriores.\n");
        
        // Tarea 1: Detectar patrón [1,0,0,0,0]
        double[][] tarea1Input = {{1.0, 0.0, 0.0, 0.0, 0.0}};
        double[][] tarea1Target = {{1.0}};
        
        // Tarea 2: Detectar patrón [0,1,0,0,0]
        double[][] tarea2Input = {{0.0, 1.0, 0.0, 0.0, 0.0}};
        double[][] tarea2Target = {{0.8}};
        
        // Tarea 3: Detectar patrón [0,0,1,0,0]
        double[][] tarea3Input = {{0.0, 0.0, 1.0, 0.0, 0.0}};
        double[][] tarea3Target = {{0.6}};
        
        // Crear redes
        RedNeuralExperimental redExp = new RedNeuralExperimental(new int[]{5, 15, 1}, 0.7);
        redExp.activarDeteccionEngramas(true);
        
        NeuralNetwork redClasica = new NeuralNetwork(5, 15, 1);
        
        // Aprender Tarea 1
        System.out.println("--- Aprendiendo Tarea 1 ---");
        for (int i = 0; i < 50; i++) {
            redExp.entrenar(tarea1Input[0], tarea1Target[0], 3);
            for (int j = 0; j < 3; j++) {
                redClasica.train(tarea1Input[0], tarea1Target[0]);
            }
        }
        
        redExp.resetear();
        double[] exp1Despues1 = redExp.procesar(tarea1Input[0]);
        double[] clas1Despues1 = redClasica.feedForward(tarea1Input[0]);
        
        System.out.printf("Tarea 1: Exp=%.4f | Clásica=%.4f | Target=%.2f\n",
            exp1Despues1[0], clas1Despues1[0], tarea1Target[0][0]);
        
        // Aprender Tarea 2
        System.out.println("\n--- Aprendiendo Tarea 2 ---");
        for (int i = 0; i < 50; i++) {
            redExp.entrenar(tarea2Input[0], tarea2Target[0], 3);
            for (int j = 0; j < 3; j++) {
                redClasica.train(tarea2Input[0], tarea2Target[0]);
            }
        }
        
        redExp.resetear();
        double[] exp2Despues2 = redExp.procesar(tarea2Input[0]);
        double[] clas2Despues2 = redClasica.feedForward(tarea2Input[0]);
        
        System.out.printf("Tarea 2: Exp=%.4f | Clásica=%.4f | Target=%.2f\n",
            exp2Despues2[0], clas2Despues2[0], tarea2Target[0][0]);
        
        // Re-evaluar Tarea 1
        redExp.resetear();
        double[] exp1Despues2 = redExp.procesar(tarea1Input[0]);
        double[] clas1Despues2 = redClasica.feedForward(tarea1Input[0]);
        
        System.out.printf("Tarea 1 (re-test): Exp=%.4f | Clásica=%.4f | Target=%.2f\n",
            exp1Despues2[0], clas1Despues2[0], tarea1Target[0][0]);
        
        // Aprender Tarea 3
        System.out.println("\n--- Aprendiendo Tarea 3 ---");
        for (int i = 0; i < 50; i++) {
            redExp.entrenar(tarea3Input[0], tarea3Target[0], 3);
            for (int j = 0; j < 3; j++) {
                redClasica.train(tarea3Input[0], tarea3Target[0]);
            }
        }
        
        redExp.resetear();
        double[] exp3Despues3 = redExp.procesar(tarea3Input[0]);
        double[] clas3Despues3 = redClasica.feedForward(tarea3Input[0]);
        
        System.out.printf("Tarea 3: Exp=%.4f | Clásica=%.4f | Target=%.2f\n",
            exp3Despues3[0], clas3Despues3[0], tarea3Target[0][0]);
        
        // Evaluación final de todas las tareas
        System.out.println("\n--- EVALUACIÓN FINAL DE TODAS LAS TAREAS ---");
        
        redExp.resetear();
        double[] exp1Final = redExp.procesar(tarea1Input[0]);
        double[] clas1Final = redClasica.feedForward(tarea1Input[0]);
        
        redExp.resetear();
        double[] exp2Final = redExp.procesar(tarea2Input[0]);
        double[] clas2Final = redClasica.feedForward(tarea2Input[0]);
        
        redExp.resetear();
        double[] exp3Final = redExp.procesar(tarea3Input[0]);
        double[] clas3Final = redClasica.feedForward(tarea3Input[0]);
        
        double errorExpTotal = Math.abs(tarea1Target[0][0] - exp1Final[0]) +
                               Math.abs(tarea2Target[0][0] - exp2Final[0]) +
                               Math.abs(tarea3Target[0][0] - exp3Final[0]);
        
        double errorClasicoTotal = Math.abs(tarea1Target[0][0] - clas1Final[0]) +
                                   Math.abs(tarea2Target[0][0] - clas2Final[0]) +
                                   Math.abs(tarea3Target[0][0] - clas3Final[0]);
        
        System.out.printf("Tarea 1: Exp=%.4f (error=%.4f) | Clásica=%.4f (error=%.4f)\n",
            exp1Final[0], Math.abs(tarea1Target[0][0] - exp1Final[0]),
            clas1Final[0], Math.abs(tarea1Target[0][0] - clas1Final[0]));
        
        System.out.printf("Tarea 2: Exp=%.4f (error=%.4f) | Clásica=%.4f (error=%.4f)\n",
            exp2Final[0], Math.abs(tarea2Target[0][0] - exp2Final[0]),
            clas2Final[0], Math.abs(tarea2Target[0][0] - clas2Final[0]));
        
        System.out.printf("Tarea 3: Exp=%.4f (error=%.4f) | Clásica=%.4f (error=%.4f)\n",
            exp3Final[0], Math.abs(tarea3Target[0][0] - exp3Final[0]),
            clas3Final[0], Math.abs(tarea3Target[0][0] - clas3Final[0]));
        
        System.out.println("\n--- RESULTADOS ---");
        System.out.printf("Error Total:\n");
        System.out.printf("  Experimental: %.4f\n", errorExpTotal);
        System.out.printf("  Clásica:      %.4f\n", errorClasicoTotal);
        System.out.printf("  Mejora:       %.1f%%\n",
            ((errorClasicoTotal - errorExpTotal) / errorClasicoTotal) * 100);
        
        System.out.println("\nEngramas formados: " + redExp.getEngramas().size());
        
        System.out.println("\n--- CONCLUSIÓN ---");
        if (errorExpTotal < errorClasicoTotal) {
            System.out.println("✓ Red experimental es MEJOR en aprendizaje incremental");
            System.out.println("  Los engramas permiten retener múltiples tareas simultáneamente");
        } else {
            System.out.println("⚠ Red clásica es mejor en aprendizaje incremental");
        }
        
        // Verificar que la red experimental funciona razonablemente
        System.out.println("\nNOTA: Este test verifica que la red experimental puede");
        System.out.println("aprender múltiples tareas secuencialmente sin colapsar.");
        
        assertTrue(errorExpTotal < 1.5,
            "Red experimental debería funcionar razonablemente en aprendizaje incremental");
    }
    
    @Test
    @DisplayName("Test 2: Adaptación Rápida a Nuevos Datos")
    void test2_AdaptacionRapidaNuevosDatos() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 2: ADAPTACIÓN RÁPIDA A NUEVOS DATOS");
        System.out.println("=".repeat(80));
        System.out.println("\nLa red experimental debería adaptarse más rápido a nuevos datos");
        System.out.println("gracias a la plasticidad hebiana y engramas.\n");
        
        // Datos iniciales: mapeo simple
        double[][] datosIniciales = {
            {0.2, 0.2, 0.2, 0.2, 0.2},
            {0.4, 0.4, 0.4, 0.4, 0.4},
            {0.6, 0.6, 0.6, 0.6, 0.6}
        };
        
        double[][] targetsIniciales = {
            {0.2},
            {0.4},
            {0.6}
        };
        
        // Entrenar con datos iniciales
        System.out.println("--- Fase 1: Entrenamiento Inicial ---");
        
        RedNeuralExperimental redExp = new RedNeuralExperimental(new int[]{5, 10, 1}, 0.7);
        redExp.activarDeteccionEngramas(true);
        
        NeuralNetwork redClasica = new NeuralNetwork(5, 10, 1);
        
        for (int epoca = 0; epoca < 30; epoca++) {
            for (int i = 0; i < datosIniciales.length; i++) {
                redExp.entrenar(datosIniciales[i], targetsIniciales[i], 2);
                for (int j = 0; j < 2; j++) {
                    redClasica.train(datosIniciales[i], targetsIniciales[i]);
                }
            }
        }
        
        System.out.println("Entrenamiento inicial completado");
        
        // Nuevo dato: [0.8, 0.8, 0.8, 0.8, 0.8] → 0.8
        double[] nuevoDato = {0.8, 0.8, 0.8, 0.8, 0.8};
        double[] nuevoTarget = {0.8};
        
        // Evaluar ANTES de aprender el nuevo dato
        System.out.println("\n--- Evaluación ANTES de aprender nuevo dato ---");
        
        redExp.resetear();
        double[] expAntes = redExp.procesar(nuevoDato);
        double[] clasAntes = redClasica.feedForward(nuevoDato);
        
        double errorExpAntes = Math.abs(nuevoTarget[0] - expAntes[0]);
        double errorClasicoAntes = Math.abs(nuevoTarget[0] - clasAntes[0]);
        
        System.out.printf("Nuevo dato [0.8]: Exp=%.4f (error=%.4f) | Clásica=%.4f (error=%.4f)\n",
            expAntes[0], errorExpAntes, clasAntes[0], errorClasicoAntes);
        
        // Aprender nuevo dato con POCAS iteraciones
        System.out.println("\n--- Aprendiendo Nuevo Dato (5 iteraciones) ---");
        
        for (int i = 0; i < 5; i++) {
            redExp.entrenar(nuevoDato, nuevoTarget, 2);
            for (int j = 0; j < 2; j++) {
                redClasica.train(nuevoDato, nuevoTarget);
            }
        }
        
        // Evaluar DESPUÉS de aprender el nuevo dato
        System.out.println("\n--- Evaluación DESPUÉS de aprender nuevo dato ---");
        
        redExp.resetear();
        double[] expDespues = redExp.procesar(nuevoDato);
        double[] clasDespues = redClasica.feedForward(nuevoDato);
        
        double errorExpDespues = Math.abs(nuevoTarget[0] - expDespues[0]);
        double errorClasicoDespues = Math.abs(nuevoTarget[0] - clasDespues[0]);
        
        System.out.printf("Nuevo dato [0.8]: Exp=%.4f (error=%.4f) | Clásica=%.4f (error=%.4f)\n",
            expDespues[0], errorExpDespues, clasDespues[0], errorClasicoDespues);
        
        // Calcular mejora
        double mejoraExp = errorExpAntes - errorExpDespues;
        double mejoraClasica = errorClasicoAntes - errorClasicoDespues;
        
        System.out.println("\n--- RESULTADOS ---");
        System.out.printf("Mejora después de 5 iteraciones:\n");
        System.out.printf("  Experimental: %.4f → %.4f (mejora: %.4f)\n",
            errorExpAntes, errorExpDespues, mejoraExp);
        System.out.printf("  Clásica:      %.4f → %.4f (mejora: %.4f)\n",
            errorClasicoAntes, errorClasicoDespues, mejoraClasica);
        
        System.out.println("\n--- CONCLUSIÓN ---");
        if (mejoraExp > mejoraClasica) {
            System.out.println("✓ Red experimental se adapta MÁS RÁPIDO a nuevos datos");
            System.out.println("  La plasticidad hebiana permite aprendizaje rápido");
        } else {
            System.out.println("⚠ Red clásica se adapta más rápido");
        }
        
        // Verificar que ambas mejoraron
        assertTrue(errorExpDespues < errorExpAntes,
            "Red experimental debería mejorar con entrenamiento");
        assertTrue(errorClasicoDespues < errorClasicoAntes,
            "Red clásica debería mejorar con entrenamiento");
    }
    
    @Test
    @DisplayName("Test 3: Aprendizaje con Datos Ruidosos")
    void test3_AprendizajeDatosRuidosos() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEST 3: APRENDIZAJE CON DATOS RUIDOSOS");
        System.out.println("=".repeat(80));
        System.out.println("\nLa red experimental debería ser más robusta al ruido");
        System.out.println("gracias a los engramas que capturan el patrón subyacente.\n");
        
        Random random = new Random(42);
        
        // Patrón base: [0.5, 0.5, 0.5, 0.5, 0.5] → 0.5
        double[] patronBase = {0.5, 0.5, 0.5, 0.5, 0.5};
        double targetBase = 0.5;
        
        // Generar datos de entrenamiento con ruido
        List<double[]> datosEntrenamiento = new ArrayList<>();
        List<Double> targetsEntrenamiento = new ArrayList<>();
        
        for (int i = 0; i < 50; i++) {
            double[] datoRuidoso = new double[5];
            for (int j = 0; j < 5; j++) {
                // Añadir ruido gaussiano
                double ruido = (random.nextGaussian() * 0.1);
                datoRuidoso[j] = Math.max(0.0, Math.min(1.0, patronBase[j] + ruido));
            }
            datosEntrenamiento.add(datoRuidoso);
            targetsEntrenamiento.add(targetBase);
        }
        
        System.out.println("Generados 50 ejemplos con ruido gaussiano (σ=0.1)");
        
        // Entrenar redes
        System.out.println("\n--- Entrenando Redes ---");
        
        RedNeuralExperimental redExp = new RedNeuralExperimental(new int[]{5, 15, 1}, 0.7);
        redExp.activarDeteccionEngramas(true);
        
        NeuralNetwork redClasica = new NeuralNetwork(5, 15, 1);
        
        for (int epoca = 0; epoca < 20; epoca++) {
            for (int i = 0; i < datosEntrenamiento.size(); i++) {
                redExp.entrenar(datosEntrenamiento.get(i), 
                    new double[]{targetsEntrenamiento.get(i)}, 2);
                for (int j = 0; j < 2; j++) {
                    redClasica.train(datosEntrenamiento.get(i),
                        new double[]{targetsEntrenamiento.get(i)});
                }
            }
        }
        
        System.out.println("Entrenamiento completado");
        System.out.println("Engramas formados: " + redExp.getEngramas().size());
        
        // Evaluar con patrón limpio (sin ruido)
        System.out.println("\n--- Evaluación con Patrón Limpio ---");
        
        redExp.resetear();
        double[] expLimpio = redExp.procesar(patronBase);
        double[] clasLimpio = redClasica.feedForward(patronBase);
        
        double errorExpLimpio = Math.abs(targetBase - expLimpio[0]);
        double errorClasicoLimpio = Math.abs(targetBase - clasLimpio[0]);
        
        System.out.printf("Patrón limpio: Exp=%.4f (error=%.4f) | Clásica=%.4f (error=%.4f)\n",
            expLimpio[0], errorExpLimpio, clasLimpio[0], errorClasicoLimpio);
        
        // Evaluar con nuevos datos ruidosos
        System.out.println("\n--- Evaluación con Nuevos Datos Ruidosos ---");
        
        double errorExpRuidoso = 0.0;
        double errorClasicoRuidoso = 0.0;
        
        for (int i = 0; i < 10; i++) {
            double[] datoRuidoso = new double[5];
            for (int j = 0; j < 5; j++) {
                double ruido = (random.nextGaussian() * 0.1);
                datoRuidoso[j] = Math.max(0.0, Math.min(1.0, patronBase[j] + ruido));
            }
            
            redExp.resetear();
            double[] expOut = redExp.procesar(datoRuidoso);
            double[] clasOut = redClasica.feedForward(datoRuidoso);
            
            errorExpRuidoso += Math.abs(targetBase - expOut[0]);
            errorClasicoRuidoso += Math.abs(targetBase - clasOut[0]);
        }
        
        errorExpRuidoso /= 10;
        errorClasicoRuidoso /= 10;
        
        System.out.printf("Promedio con ruido: Exp=%.4f | Clásica=%.4f\n",
            errorExpRuidoso, errorClasicoRuidoso);
        
        System.out.println("\n--- RESULTADOS ---");
        System.out.printf("Error con patrón limpio:\n");
        System.out.printf("  Experimental: %.4f\n", errorExpLimpio);
        System.out.printf("  Clásica:      %.4f\n", errorClasicoLimpio);
        
        System.out.printf("\nError con datos ruidosos:\n");
        System.out.printf("  Experimental: %.4f\n", errorExpRuidoso);
        System.out.printf("  Clásica:      %.4f\n", errorClasicoRuidoso);
        
        System.out.println("\n--- CONCLUSIÓN ---");
        if (errorExpRuidoso < errorClasicoRuidoso) {
            System.out.println("✓ Red experimental es MÁS ROBUSTA al ruido");
            System.out.println("  Los engramas capturan el patrón subyacente");
        } else {
            System.out.println("⚠ Red clásica es más robusta al ruido");
        }
        
        // Verificar que ambas funcionan razonablemente bien
        assertTrue(errorExpLimpio < 0.3,
            "Red experimental debería funcionar bien con patrón limpio");
    }
}
