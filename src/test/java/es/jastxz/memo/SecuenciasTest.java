package es.jastxz.memo;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.RedNeuralExperimental;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test de Reconocimiento de Patrones Secuenciales
 * 
 * Compara red experimental vs red clásica en la tarea de predecir
 * el siguiente elemento de una secuencia.
 * 
 * La red experimental debería destacar porque:
 * - Los engramas memorizan transiciones entre elementos
 * - La consolidación refuerza patrones frecuentes
 * - La plasticidad hebiana asocia elementos consecutivos naturalmente
 */
public class SecuenciasTest {
    
    private static final int VENTANA = 5;  // Número de elementos para predecir el siguiente
    private static final int HIDDEN_SIZE = 30;
    
    /**
     * Test 1: Secuencia de Fibonacci
     * Secuencia: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...
     */
    @Test
    @DisplayName("Test 1: Predicción de Fibonacci")
    void test1_Fibonacci() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: PREDICCIÓN DE FIBONACCI");
        System.out.println("=".repeat(60));
        
        // Generar secuencia de Fibonacci
        List<Integer> fibonacci = generarFibonacci(50);
        System.out.println("Secuencia Fibonacci (primeros 20): " + fibonacci.subList(0, 20));
        
        // Preparar datos de entrenamiento
        List<double[]> datosEntrenamiento = prepararDatos(fibonacci, 40);
        List<double[]> datosPrueba = prepararDatos(fibonacci.subList(40, 50), 10);
        
        System.out.println("\n--- Entrenando Red Experimental ---");
        RedNeuralExperimental redExp = entrenarRedExperimental(datosEntrenamiento, 100);
        
        System.out.println("\n--- Entrenando Red Clásica ---");
        NeuralNetwork redClasica = entrenarRedClasica(datosEntrenamiento, 500);
        
        System.out.println("\n--- Evaluando Predicciones ---");
        double errorExp = evaluarPredicciones(redExp, datosPrueba, true);
        double errorClasico = evaluarPredicciones(redClasica, datosPrueba, false);
        
        System.out.println("\n--- Resultados ---");
        System.out.printf("Error Red Experimental: %.4f\n", errorExp);
        System.out.printf("Error Red Clásica: %.4f\n", errorClasico);
        
        mostrarEstadisticasExperimental(redExp);
        
        System.out.println("\n--- Conclusión ---");
        if (errorExp < errorClasico * 1.2) {
            System.out.println("✓ Red experimental tiene rendimiento comparable o mejor");
        } else {
            System.out.println("⚠ Red clásica tiene mejor rendimiento en este caso");
        }
    }
    
    /**
     * Test 2: Secuencia de Potencias de 2
     * Secuencia: 1, 2, 4, 8, 16, 32, 64, 128...
     */
    @Test
    @DisplayName("Test 2: Predicción de Potencias de 2")
    void test2_PotenciasDe2() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: PREDICCIÓN DE POTENCIAS DE 2");
        System.out.println("=".repeat(60));
        
        // Generar secuencia de potencias de 2
        List<Integer> potencias = generarPotenciasDe2(20);
        System.out.println("Secuencia Potencias de 2: " + potencias);
        
        // Normalizar (valores muy grandes)
        List<Integer> potenciasNorm = new ArrayList<>();
        for (int p : potencias) {
            potenciasNorm.add(p / 1000);  // Escalar
        }
        
        List<double[]> datosEntrenamiento = prepararDatos(potenciasNorm, 15);
        List<double[]> datosPrueba = prepararDatos(potenciasNorm.subList(15, 20), 5);
        
        System.out.println("\n--- Entrenando Red Experimental ---");
        RedNeuralExperimental redExp = entrenarRedExperimental(datosEntrenamiento, 100);
        
        System.out.println("\n--- Entrenando Red Clásica ---");
        NeuralNetwork redClasica = entrenarRedClasica(datosEntrenamiento, 500);
        
        System.out.println("\n--- Evaluando Predicciones ---");
        double errorExp = evaluarPredicciones(redExp, datosPrueba, true);
        double errorClasico = evaluarPredicciones(redClasica, datosPrueba, false);
        
        System.out.println("\n--- Resultados ---");
        System.out.printf("Error Red Experimental: %.4f\n", errorExp);
        System.out.printf("Error Red Clásica: %.4f\n", errorClasico);
        
        mostrarEstadisticasExperimental(redExp);
    }
    
    /**
     * Test 3: Secuencia Aritmética Simple
     * Secuencia: 2, 4, 6, 8, 10, 12, 14...
     */
    @Test
    @DisplayName("Test 3: Predicción de Secuencia Aritmética")
    void test3_SecuenciaAritmetica() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: PREDICCIÓN DE SECUENCIA ARITMÉTICA");
        System.out.println("=".repeat(60));
        
        // Generar secuencia aritmética (incremento de 2)
        List<Integer> aritmetica = generarSecuenciaAritmetica(2, 2, 30);
        System.out.println("Secuencia Aritmética: " + aritmetica.subList(0, 15));
        
        List<double[]> datosEntrenamiento = prepararDatos(aritmetica.subList(0, 25), 20);
        List<double[]> datosPrueba = prepararDatos(aritmetica.subList(20, 30), 5);
        
        System.out.println("Datos de entrenamiento: " + datosEntrenamiento.size());
        System.out.println("Datos de prueba: " + datosPrueba.size());
        
        System.out.println("\n--- Entrenando Red Experimental ---");
        // AUMENTADO: 200 épocas para dar tiempo a la poda y refinamiento
        RedNeuralExperimental redExp = entrenarRedExperimental(datosEntrenamiento, 200);
        
        System.out.println("\n--- Entrenando Red Clásica ---");
        NeuralNetwork redClasica = entrenarRedClasica(datosEntrenamiento, 500);
        
        System.out.println("\n--- Evaluando Predicciones ---");
        double errorExp = evaluarPredicciones(redExp, datosPrueba, true);
        double errorClasico = evaluarPredicciones(redClasica, datosPrueba, false);
        
        System.out.println("\n--- Resultados ---");
        System.out.printf("Error Red Experimental: %.4f\n", errorExp);
        System.out.printf("Error Red Clásica: %.4f\n", errorClasico);
        
        mostrarEstadisticasExperimental(redExp);
        
        // Esta secuencia es muy simple, ambas deberían hacerlo bien
        assertTrue(errorExp < 0.5, "Error experimental debe ser bajo (<0.5)");
        assertTrue(errorClasico < 0.5, "Error clásico debe ser bajo (<0.5)");
    }
    
    /**
     * Test 4: Comparación con Múltiples Secuencias
     * Entrena con varias secuencias diferentes
     */
    @Test
    @DisplayName("Test 4: Aprendizaje de Múltiples Secuencias")
    void test4_MultipleSecuencias() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: APRENDIZAJE DE MÚLTIPLES SECUENCIAS");
        System.out.println("=".repeat(60));
        
        // Generar múltiples secuencias (más largas para tener datos de prueba)
        List<Integer> fibonacci = generarFibonacci(40);
        List<Integer> aritmetica = generarSecuenciaAritmetica(1, 3, 40);
        List<Integer> cuadrados = generarCuadrados(30);
        
        System.out.println("Fibonacci: " + fibonacci.subList(0, 10));
        System.out.println("Aritmética: " + aritmetica.subList(0, 10));
        System.out.println("Cuadrados: " + cuadrados.subList(0, 10));
        
        // Combinar datos de entrenamiento
        List<double[]> datosEntrenamiento = new ArrayList<>();
        datosEntrenamiento.addAll(prepararDatos(fibonacci.subList(0, 30), 25));
        datosEntrenamiento.addAll(prepararDatos(aritmetica.subList(0, 30), 25));
        datosEntrenamiento.addAll(prepararDatos(cuadrados.subList(0, 20), 15));
        
        // Mezclar datos
        Collections.shuffle(datosEntrenamiento);
        
        System.out.println("\n--- Entrenando Red Experimental ---");
        System.out.println("Total de ejemplos: " + datosEntrenamiento.size());
        RedNeuralExperimental redExp = entrenarRedExperimental(datosEntrenamiento, 150);
        
        System.out.println("\n--- Entrenando Red Clásica ---");
        NeuralNetwork redClasica = entrenarRedClasica(datosEntrenamiento, 800);
        
        // Evaluar en cada secuencia por separado (usar más datos para prueba)
        System.out.println("\n--- Evaluando en Fibonacci ---");
        List<double[]> pruebaFib = prepararDatos(fibonacci.subList(30, 40), 5);
        double errorExpFib = evaluarPredicciones(redExp, pruebaFib, true);
        double errorClasiFib = evaluarPredicciones(redClasica, pruebaFib, false);
        
        System.out.println("\n--- Evaluando en Aritmética ---");
        List<double[]> pruebaArit = prepararDatos(aritmetica.subList(30, 40), 5);
        double errorExpArit = evaluarPredicciones(redExp, pruebaArit, true);
        double errorClasiArit = evaluarPredicciones(redClasica, pruebaArit, false);
        
        System.out.println("\n--- Evaluando en Cuadrados ---");
        List<double[]> pruebaCuad = prepararDatos(cuadrados.subList(20, 30), 5);
        double errorExpCuad = evaluarPredicciones(redExp, pruebaCuad, true);
        double errorClasiCuad = evaluarPredicciones(redClasica, pruebaCuad, false);
        
        System.out.println("\n--- Resultados Finales ---");
        System.out.println("Fibonacci:");
        System.out.printf("  Experimental: %.4f | Clásica: %.4f\n", errorExpFib, errorClasiFib);
        System.out.println("Aritmética:");
        System.out.printf("  Experimental: %.4f | Clásica: %.4f\n", errorExpArit, errorClasiArit);
        System.out.println("Cuadrados:");
        System.out.printf("  Experimental: %.4f | Clásica: %.4f\n", errorExpCuad, errorClasiCuad);
        
        mostrarEstadisticasExperimental(redExp);
        
        System.out.println("\n--- Conclusión ---");
        System.out.println("La red experimental debería formar engramas separados");
        System.out.println("para cada tipo de secuencia, permitiendo mantener");
        System.out.println("múltiples patrones sin interferencia.");
    }
    
    // ==================== MÉTODOS AUXILIARES ====================
    
    /**
     * Genera secuencia de Fibonacci
     */
    private List<Integer> generarFibonacci(int n) {
        List<Integer> fib = new ArrayList<>();
        fib.add(1);
        fib.add(1);
        for (int i = 2; i < n; i++) {
            fib.add(fib.get(i-1) + fib.get(i-2));
        }
        return fib;
    }
    
    /**
     * Genera secuencia de potencias de 2
     */
    private List<Integer> generarPotenciasDe2(int n) {
        List<Integer> potencias = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            potencias.add((int)Math.pow(2, i));
        }
        return potencias;
    }
    
    /**
     * Genera secuencia aritmética
     */
    private List<Integer> generarSecuenciaAritmetica(int inicio, int incremento, int n) {
        List<Integer> seq = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            seq.add(inicio + i * incremento);
        }
        return seq;
    }
    
    /**
     * Genera secuencia de cuadrados
     */
    private List<Integer> generarCuadrados(int n) {
        List<Integer> cuadrados = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            cuadrados.add(i * i);
        }
        return cuadrados;
    }
    
    /**
     * Prepara datos de entrenamiento con ventana deslizante
     * Entrada: [a, b, c, d, e] → Salida: f
     */
    private List<double[]> prepararDatos(List<Integer> secuencia, int numEjemplos) {
        List<double[]> datos = new ArrayList<>();
        
        if (secuencia.size() < VENTANA + 1) {
            return datos;  // No hay suficientes datos
        }
        
        // Normalizar valores
        int max = secuencia.stream().max(Integer::compare).orElse(1);
        if (max == 0) max = 1;  // Evitar división por cero
        
        int ejemplosCreados = 0;
        for (int i = 0; i + VENTANA < secuencia.size() && ejemplosCreados < numEjemplos; i++) {
            double[] ejemplo = new double[VENTANA + 1];
            
            // Input: VENTANA elementos
            for (int j = 0; j < VENTANA; j++) {
                ejemplo[j] = secuencia.get(i + j) / (double)max;
            }
            
            // Output: siguiente elemento
            ejemplo[VENTANA] = secuencia.get(i + VENTANA) / (double)max;
            
            datos.add(ejemplo);
            ejemplosCreados++;
        }
        
        return datos;
    }
    
    /**
     * Entrena red experimental
     */
    private RedNeuralExperimental entrenarRedExperimental(List<double[]> datos, int epocas) {
        // AJUSTADO: Densidad 0.7 en lugar de 0.9 para permitir más poda
        // El cerebro empieza con sobreabundancia y luego poda
        RedNeuralExperimental red = new RedNeuralExperimental(
            new int[]{VENTANA, HIDDEN_SIZE, 1}, 0.7
        );
        
        // Activar sistemas biológicos
        red.activarDeteccionEngramas(true);
        red.activarModoPredictivo(true);
        red.activarCompeticionRecursos(true);  // NUEVO: Activar competición
        
        long inicio = System.currentTimeMillis();
        int conexionesIniciales = red.getTotalConexiones();
        System.out.printf("Conexiones iniciales: %d\n", conexionesIniciales);
        
        for (int epoca = 0; epoca < epocas; epoca++) {
            Collections.shuffle(datos);
            
            for (double[] ejemplo : datos) {
                double[] input = Arrays.copyOfRange(ejemplo, 0, VENTANA);
                double[] target = {ejemplo[VENTANA]};
                
                red.entrenar(input, target, 3);
            }
            
            // NUEVO: Ejecutar competición y poda cada 10 épocas
            if ((epoca + 1) % 10 == 0) {
                red.competirPorRecursos();
                int podados = red.podarElementos();
                if (podados > 0) {
                    System.out.printf("Época %d: Podados %d elementos (quedan %d conexiones)\n", 
                        epoca + 1, podados, red.getTotalConexiones());
                }
            }
            
            if ((epoca + 1) % 50 == 0) {
                System.out.printf("Época %d/%d completada\n", epoca + 1, epocas);
            }
        }
        
        long tiempo = System.currentTimeMillis() - inicio;
        System.out.printf("Tiempo de entrenamiento: %.2f segundos\n", tiempo / 1000.0);
        
        return red;
    }
    
    /**
     * Entrena red clásica
     */
    private NeuralNetwork entrenarRedClasica(List<double[]> datos, int epocas) {
        NeuralNetwork red = new NeuralNetwork(VENTANA, HIDDEN_SIZE, 1);
        
        long inicio = System.currentTimeMillis();
        
        for (int epoca = 0; epoca < epocas; epoca++) {
            Collections.shuffle(datos);
            
            for (double[] ejemplo : datos) {
                double[] input = Arrays.copyOfRange(ejemplo, 0, VENTANA);
                double[] target = {ejemplo[VENTANA]};
                
                red.train(input, target);
            }
            
            if ((epoca + 1) % 100 == 0) {
                System.out.printf("Época %d/%d completada\n", epoca + 1, epocas);
            }
        }
        
        long tiempo = System.currentTimeMillis() - inicio;
        System.out.printf("Tiempo de entrenamiento: %.2f segundos\n", tiempo / 1000.0);
        
        return red;
    }
    
    /**
     * Evalúa predicciones y calcula error
     */
    private double evaluarPredicciones(Object red, List<double[]> datosPrueba, boolean esExperimental) {
        double errorTotal = 0.0;
        int correctas = 0;
        
        System.out.println("\nEjemplos de predicciones:");
        
        for (int i = 0; i < Math.min(5, datosPrueba.size()); i++) {
            double[] ejemplo = datosPrueba.get(i);
            double[] input = Arrays.copyOfRange(ejemplo, 0, VENTANA);
            double esperado = ejemplo[VENTANA];
            
            double predicho;
            if (esExperimental) {
                RedNeuralExperimental redExp = (RedNeuralExperimental) red;
                redExp.resetear();
                double[] output = redExp.procesar(input);
                predicho = output[0];
                
                // DEBUG: Mostrar estado de activación
                if (i == 0) {
                    long neuronasActivas = redExp.getCapaSensorial().stream().filter(n -> n.estaActiva()).count();
                    System.out.printf("  [DEBUG] Neuronas sensoriales activas: %d/%d\n", 
                        neuronasActivas, redExp.getCapaSensorial().size());
                    
                    for (int j = 0; j < redExp.getCapasInterneuronas().size(); j++) {
                        List<es.jastxz.nn.Neurona> capa = redExp.getCapasInterneuronas().get(j);
                        long activas = capa.stream().filter(n -> n.estaActiva()).count();
                        System.out.printf("  [DEBUG] Capa inter %d activas: %d/%d\n", j+1, activas, capa.size());
                    }
                    
                    long motorasActivas = redExp.getCapaMotora().stream().filter(n -> n.estaActiva()).count();
                    System.out.printf("  [DEBUG] Neuronas motoras activas: %d/%d\n", 
                        motorasActivas, redExp.getCapaMotora().size());
                }
            } else {
                NeuralNetwork redClasica = (NeuralNetwork) red;
                double[] output = redClasica.feedForward(input);
                predicho = output[0];
            }
            
            double error = Math.abs(esperado - predicho);
            errorTotal += error;
            
            if (error < 0.1) correctas++;
            
            System.out.printf("  Input: [%.2f, %.2f, %.2f, %.2f, %.2f] → Esperado: %.2f, Predicho: %.2f (error: %.4f)\n",
                input[0], input[1], input[2], input[3], input[4], esperado, predicho, error);
        }
        
        // Calcular error en todos los datos
        for (double[] ejemplo : datosPrueba) {
            double[] input = Arrays.copyOfRange(ejemplo, 0, VENTANA);
            double esperado = ejemplo[VENTANA];
            
            double predicho;
            if (esExperimental) {
                RedNeuralExperimental redExp = (RedNeuralExperimental) red;
                redExp.resetear();
                double[] output = redExp.procesar(input);
                predicho = output[0];
            } else {
                NeuralNetwork redClasica = (NeuralNetwork) red;
                double[] output = redClasica.feedForward(input);
                predicho = output[0];
            }
            
            errorTotal += Math.abs(esperado - predicho);
        }
        
        double errorPromedio = errorTotal / datosPrueba.size();
        double precision = (correctas * 100.0) / Math.min(5, datosPrueba.size());
        
        System.out.printf("\nPrecisión (error < 0.1): %.1f%% (%d/%d)\n", 
            precision, correctas, Math.min(5, datosPrueba.size()));
        
        return errorPromedio;
    }
    
    /**
     * Muestra estadísticas de la red experimental
     */
    private void mostrarEstadisticasExperimental(RedNeuralExperimental red) {
        Map<String, Object> stats = red.getEstadisticas();
        
        System.out.println("\n--- Estadísticas Red Experimental ---");
        System.out.println("Engramas formados: " + stats.get("engramasFormados"));
        System.out.println("Engramas actuales: " + stats.get("totalEngramas"));
        System.out.println("Engramas fusionados: " + stats.get("engramasFusionados"));
        System.out.println("Engramas podados: " + stats.get("engramasPodados"));
        System.out.printf("Tamaño promedio engramas: %.1f neuronas (%.1f%% de la red)\n",
            stats.get("tamañoPromedioEngramas"),
            stats.get("porcentajePromedioNeuronas"));
        System.out.println("Consolidación inicializada: " + stats.get("consolidacionInicializada"));
        System.out.println("Intervalo consolidación: " + stats.get("intervaloConsolidacion") + " iteraciones");
        System.out.printf("Tiempo promedio iteración: %.2fms\n", stats.get("tiempoPromedioIteracion"));
    }
}
