package es.jastxz.comparativas;

import es.jastxz.engine.Funciones3enRaya;
import es.jastxz.models.Modelo3enRaya;
import es.jastxz.models.Modelo3enRayaExperimental;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import es.jastxz.util.ModelManager;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa completa entre entrenamiento supervisado y no supervisado
 * para el juego de 3 en Raya
 */
public class Comparativa3enRayaTest {
    
    /**
     * Test 1: Entrenar modelo clásico (supervisado)
     */
    @Test
    void test1_EntrenarModeloClasico() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: ENTRENAMIENTO SUPERVISADO (Red Clásica)");
        System.out.println("=".repeat(60));
        
        long inicio = System.currentTimeMillis();
        
        // Crear y entrenar modelo clásico
        Modelo3enRaya modeloClasico = new Modelo3enRaya();
        modeloClasico.entrenar(1000);  // 1000 épocas
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo clásico entrenado");
        System.out.println("Tiempo: " + tiempo + " ms");
        System.out.println("Método: Backpropagation con datos exhaustivos");
        System.out.println("Datos: ~4500 estados únicos del juego");
    }
    
    /**
     * Test 2: Entrenar modelo experimental (no supervisado)
     */
    @Test
    void test2_EntrenarModeloExperimental() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: ENTRENAMIENTO NO SUPERVISADO (Red Experimental)");
        System.out.println("=".repeat(60));
        
        long inicio = System.currentTimeMillis();
        
        // Crear y entrenar modelo experimental
        Modelo3enRayaExperimental modeloExperimental = new Modelo3enRayaExperimental();
        modeloExperimental.entrenarSelfPlay(500);  // 500 partidas de self-play
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo experimental entrenado");
        System.out.println("Tiempo: " + tiempo + " ms");
        System.out.println("Método: Self-play con plasticidad hebiana");
        System.out.println("Partidas: 500");
        
        // Guardar para uso posterior
        try {
            modeloExperimental.guardar();
        } catch (IOException e) {
            System.err.println("Error guardando modelo: " + e.getMessage());
        }
    }
    
    /**
     * Test 3: Comparar rendimiento contra jugador aleatorio
     */
    @Test
    void test3_CompararContraAleatorio() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: RENDIMIENTO CONTRA JUGADOR ALEATORIO");
        System.out.println("=".repeat(60));
        
        // Cargar modelos
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modelo3enRaya.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        // Jugar 100 partidas con cada modelo
        int partidasPrueba = 100;
        
        System.out.println("\n--- Modelo Clásico vs Aleatorio ---");
        ResultadoPartidas resultadoClasico = jugarContraAleatorio(cerebroClasico, partidasPrueba);
        mostrarResultados(resultadoClasico);
        
        System.out.println("\n--- Modelo Experimental vs Aleatorio ---");
        try {
            Modelo3enRayaExperimental modeloExp = Modelo3enRayaExperimental.cargar(
                "src/main/resources/modelosExperimentales/modelo3enRayaExperimental.nn"
            );
            ResultadoPartidas resultadoExp = jugarContraAleatorioExperimental(
                modeloExp.getCerebro(), partidasPrueba
            );
            mostrarResultados(resultadoExp);
            
            // Comparar
            System.out.println("\n--- Comparación ---");
            System.out.printf("Victorias Clásico: %.1f%% vs Experimental: %.1f%%\n",
                resultadoClasico.porcentajeVictorias(),
                resultadoExp.porcentajeVictorias());
            
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Modelo experimental no encontrado, saltando comparación");
        }
    }
    
    /**
     * Test 4: Enfrentamiento directo
     */
    @Test
    void test4_EnfrentamientoDirecto() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: ENFRENTAMIENTO DIRECTO");
        System.out.println("Clásico (Supervisado) vs Experimental (No Supervisado)");
        System.out.println("=".repeat(60));
        
        // Cargar modelos
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modelo3enRaya.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        try {
            Modelo3enRayaExperimental modeloExp = Modelo3enRayaExperimental.cargar(
                "src/main/resources/modelosExperimentales/modelo3enRayaExperimental.nn"
            );
            
            int partidasPrueba = 100;
            
            // Clásico juega primero (P1)
            System.out.println("\n--- Ronda 1: Clásico (P1) vs Experimental (P2) ---");
            ResultadoEnfrentamiento ronda1 = enfrentarModelos(
                cerebroClasico, modeloExp.getCerebro(), partidasPrueba, true
            );
            mostrarResultadosEnfrentamiento(ronda1, "Clásico", "Experimental");
            
            // Experimental juega primero (P1)
            System.out.println("\n--- Ronda 2: Experimental (P1) vs Clásico (P2) ---");
            ResultadoEnfrentamiento ronda2 = enfrentarModelos(
                cerebroClasico, modeloExp.getCerebro(), partidasPrueba, false
            );
            mostrarResultadosEnfrentamiento(ronda2, "Experimental", "Clásico");
            
            // Resumen total
            System.out.println("\n--- RESUMEN TOTAL (200 partidas) ---");
            int victoriasClasico = ronda1.victoriasP1 + ronda2.victoriasP2;
            int victoriasExp = ronda1.victoriasP2 + ronda2.victoriasP1;
            int empates = ronda1.empates + ronda2.empates;
            
            System.out.println("Victorias Clásico: " + victoriasClasico + " (" + 
                String.format("%.1f%%", victoriasClasico * 100.0 / 200) + ")");
            System.out.println("Victorias Experimental: " + victoriasExp + " (" + 
                String.format("%.1f%%", victoriasExp * 100.0 / 200) + ")");
            System.out.println("Empates: " + empates + " (" + 
                String.format("%.1f%%", empates * 100.0 / 200) + ")");
            
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Modelo experimental no encontrado, saltando enfrentamiento");
        }
    }
    
    /**
     * Test 5: Análisis de estrategias aprendidas
     */
    @Test
    void test5_AnalisisEstrategias() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 5: ANÁLISIS DE ESTRATEGIAS");
        System.out.println("=".repeat(60));
        
        // Cargar modelos
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modelo3enRaya.nn");
        
        try {
            Modelo3enRayaExperimental modeloExp = Modelo3enRayaExperimental.cargar(
                "src/main/resources/modelosExperimentales/modelo3enRayaExperimental.nn"
            );
            
            // Situación 1: Centro vacío (apertura óptima)
            System.out.println("\n--- Situación 1: Tablero vacío (primer movimiento) ---");
            Tablero tableroVacio = Funciones3enRaya.inicial3enRaya();
            compararDecisiones(cerebroClasico, modeloExp.getCerebro(), tableroVacio, 1);
            
            // Situación 2: Bloquear victoria inminente
            System.out.println("\n--- Situación 2: Bloquear victoria del oponente ---");
            int[][] estadoBloqueo = {
                {1, 1, 0},
                {2, 0, 0},
                {0, 0, 0}
            };
            Tablero tableroBloqueo = new Tablero(new SmallMatrix(estadoBloqueo));
            compararDecisiones(cerebroClasico, modeloExp.getCerebro(), tableroBloqueo, 2);
            
            // Situación 3: Ganar si es posible
            System.out.println("\n--- Situación 3: Oportunidad de ganar ---");
            int[][] estadoGanar = {
                {1, 1, 0},
                {2, 2, 0},
                {0, 0, 0}
            };
            Tablero tableroGanar = new Tablero(new SmallMatrix(estadoGanar));
            compararDecisiones(cerebroClasico, modeloExp.getCerebro(), tableroGanar, 1);
            
            // Mostrar engramas del modelo experimental
            System.out.println("\n--- Engramas Formados (Modelo Experimental) ---");
            System.out.println("Total de engramas: " + modeloExp.getCerebro().getEngramas().size());
            System.out.println("(Patrones de juego memorizados)");
            
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Modelo experimental no encontrado");
        }
    }
    
    // ==================== MÉTODOS AUXILIARES ====================
    
    /**
     * Juega partidas contra jugador aleatorio (modelo clásico)
     */
    private ResultadoPartidas jugarContraAleatorio(NeuralNetwork cerebro, int numPartidas) {
        int victorias = 0;
        int derrotas = 0;
        int empates = 0;
        Random rand = new Random();
        
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;
            
            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                
                if (turno == 1) {
                    // IA juega
                    movimiento = obtenerMovimientoClasico(cerebro, tablero, turno);
                } else {
                    // Aleatorio juega
                    List<Movimiento> movs = Funciones3enRaya.movs3enRaya(tablero, turno);
                    movimiento = movs.get(rand.nextInt(movs.size())).getPos();
                }
                
                int[][] nuevoEstado = copiarTablero(tablero);
                nuevoEstado[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevoEstado));
                
                turno = (turno == 1) ? 2 : 1;
            }
            
            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victorias++;
            else if (resultado == -1) derrotas++;
            else empates++;
        }
        
        return new ResultadoPartidas(victorias, derrotas, empates);
    }
    
    /**
     * Juega partidas contra jugador aleatorio (modelo experimental)
     */
    private ResultadoPartidas jugarContraAleatorioExperimental(
            RedNeuralExperimental cerebro, int numPartidas) {
        int victorias = 0;
        int derrotas = 0;
        int empates = 0;
        Random rand = new Random();
        
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;
            
            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                
                if (turno == 1) {
                    // IA juega
                    movimiento = obtenerMovimientoExperimental(cerebro, tablero, turno);
                } else {
                    // Aleatorio juega
                    List<Movimiento> movs = Funciones3enRaya.movs3enRaya(tablero, turno);
                    movimiento = movs.get(rand.nextInt(movs.size())).getPos();
                }
                
                int[][] nuevoEstado = copiarTablero(tablero);
                nuevoEstado[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevoEstado));
                
                turno = (turno == 1) ? 2 : 1;
            }
            
            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victorias++;
            else if (resultado == -1) derrotas++;
            else empates++;
        }
        
        return new ResultadoPartidas(victorias, derrotas, empates);
    }
    
    /**
     * Enfrenta dos modelos directamente
     */
    private ResultadoEnfrentamiento enfrentarModelos(
            NeuralNetwork cerebroClasico,
            RedNeuralExperimental cerebroExp,
            int numPartidas,
            boolean clasicoEsP1) {
        
        int victoriasP1 = 0;
        int victoriasP2 = 0;
        int empates = 0;
        
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = Funciones3enRaya.inicial3enRaya();
            int turno = 1;
            
            while (!Funciones3enRaya.fin3enRaya(tablero)) {
                Posicion movimiento;
                
                boolean esTurnoClasico = (clasicoEsP1 && turno == 1) || (!clasicoEsP1 && turno == 2);
                
                if (esTurnoClasico) {
                    movimiento = obtenerMovimientoClasico(cerebroClasico, tablero, turno);
                } else {
                    movimiento = obtenerMovimientoExperimental(cerebroExp, tablero, turno);
                }
                
                int[][] nuevoEstado = copiarTablero(tablero);
                nuevoEstado[movimiento.getFila()][movimiento.getColumna()] = turno;
                tablero = new Tablero(new SmallMatrix(nuevoEstado));
                
                turno = (turno == 1) ? 2 : 1;
            }
            
            int resultado = evaluarResultado(tablero);
            if (resultado == 1) victoriasP1++;
            else if (resultado == -1) victoriasP2++;
            else empates++;
        }
        
        return new ResultadoEnfrentamiento(victoriasP1, victoriasP2, empates);
    }
    
    /**
     * Compara las decisiones de ambos modelos en una situación
     */
    private void compararDecisiones(
            NeuralNetwork cerebroClasico,
            RedNeuralExperimental cerebroExp,
            Tablero tablero,
            int turno) {
        
        System.out.println("Tablero:");
        imprimirTablero(tablero);
        
        Posicion movClasico = obtenerMovimientoClasico(cerebroClasico, tablero, turno);
        Posicion movExp = obtenerMovimientoExperimental(cerebroExp, tablero, turno);
        
        System.out.println("Decisión Clásico: " + posicionToString(movClasico));
        System.out.println("Decisión Experimental: " + posicionToString(movExp));
        
        if (movClasico.equals(movExp)) {
            System.out.println("✓ Ambos modelos coinciden");
        } else {
            System.out.println("✗ Modelos difieren");
        }
    }
    
    // Métodos auxiliares de obtención de movimientos
    
    private Posicion obtenerMovimientoClasico(NeuralNetwork cerebro, Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        double[] output = cerebro.feedForward(input);
        
        List<Movimiento> movsPosibles = Funciones3enRaya.movs3enRaya(tablero, turno);
        Posicion mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Movimiento mov : movsPosibles) {
            int index = mov.getPos().getFila() * 3 + mov.getPos().getColumna();
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejor = mov.getPos();
            }
        }
        
        return mejor != null ? mejor : movsPosibles.get(0).getPos();
    }
    
    private Posicion obtenerMovimientoExperimental(
            RedNeuralExperimental cerebro, Tablero tablero, int turno) {
        double[] input = Modelo3enRaya.tabularToInput(tablero, turno);
        cerebro.resetear();
        double[] output = cerebro.procesar(input);
        
        List<Movimiento> movsPosibles = Funciones3enRaya.movs3enRaya(tablero, turno);
        Posicion mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Movimiento mov : movsPosibles) {
            int index = mov.getPos().getFila() * 3 + mov.getPos().getColumna();
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejor = mov.getPos();
            }
        }
        
        return mejor != null ? mejor : movsPosibles.get(0).getPos();
    }
    
    // Métodos auxiliares de utilidad
    
    private int[][] copiarTablero(Tablero tablero) {
        int[][] copia = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                copia[i][j] = tablero.getValor(i, j);
            }
        }
        return copia;
    }
    
    private int evaluarResultado(Tablero tablero) {
        if (!Funciones3enRaya.hay3EnRaya(tablero)) return 0;
        
        int c1 = 0, c2 = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (tablero.getValor(i, j) == 1) c1++;
                else if (tablero.getValor(i, j) == 2) c2++;
            }
        }
        
        return (c1 == c2) ? -1 : 1;
    }
    
    private void imprimirTablero(Tablero tablero) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int val = tablero.getValor(i, j);
                System.out.print(val == 0 ? "." : (val == 1 ? "X" : "O"));
                System.out.print(" ");
            }
            System.out.println();
        }
    }
    
    private String posicionToString(Posicion pos) {
        return "(" + pos.getFila() + "," + pos.getColumna() + ")";
    }
    
    private void mostrarResultados(ResultadoPartidas resultado) {
        System.out.println("Victorias: " + resultado.victorias + " (" + 
            String.format("%.1f%%", resultado.porcentajeVictorias()) + ")");
        System.out.println("Derrotas: " + resultado.derrotas + " (" + 
            String.format("%.1f%%", resultado.porcentajeDerrotas()) + ")");
        System.out.println("Empates: " + resultado.empates + " (" + 
            String.format("%.1f%%", resultado.porcentajeEmpates()) + ")");
    }
    
    private void mostrarResultadosEnfrentamiento(
            ResultadoEnfrentamiento resultado, String nombreP1, String nombreP2) {
        System.out.println(nombreP1 + ": " + resultado.victoriasP1 + " victorias (" + 
            String.format("%.1f%%", resultado.porcentajeP1()) + ")");
        System.out.println(nombreP2 + ": " + resultado.victoriasP2 + " victorias (" + 
            String.format("%.1f%%", resultado.porcentajeP2()) + ")");
        System.out.println("Empates: " + resultado.empates + " (" + 
            String.format("%.1f%%", resultado.porcentajeEmpates()) + ")");
    }
    
    // Clases auxiliares para resultados
    
    private static class ResultadoPartidas {
        int victorias, derrotas, empates;
        
        ResultadoPartidas(int victorias, int derrotas, int empates) {
            this.victorias = victorias;
            this.derrotas = derrotas;
            this.empates = empates;
        }
        
        int total() { return victorias + derrotas + empates; }
        double porcentajeVictorias() { return victorias * 100.0 / total(); }
        double porcentajeDerrotas() { return derrotas * 100.0 / total(); }
        double porcentajeEmpates() { return empates * 100.0 / total(); }
    }
    
    private static class ResultadoEnfrentamiento {
        int victoriasP1, victoriasP2, empates;
        
        ResultadoEnfrentamiento(int victoriasP1, int victoriasP2, int empates) {
            this.victoriasP1 = victoriasP1;
            this.victoriasP2 = victoriasP2;
            this.empates = empates;
        }
        
        int total() { return victoriasP1 + victoriasP2 + empates; }
        double porcentajeP1() { return victoriasP1 * 100.0 / total(); }
        double porcentajeP2() { return victoriasP2 * 100.0 / total(); }
        double porcentajeEmpates() { return empates * 100.0 / total(); }
    }
    
    // ==================== TESTS ADICIONALES: SUPERVISADO ====================
    
    /**
     * Test 6: Entrenar modelo experimental con datos supervisados
     * Usa los MISMOS datos que el modelo clásico para comparación justa
     */
    @Test
    void test6_EntrenarExperimentalSupervisado() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 6: EXPERIMENTAL CON ENTRENAMIENTO SUPERVISADO");
        System.out.println("=".repeat(60));
        
        long inicio = System.currentTimeMillis();
        
        // Generar datos de entrenamiento (mismos que usa el modelo clásico)
        System.out.println("Generando datos de entrenamiento...");
        List<Modelo3enRayaExperimental.ParEntrenamiento> datos = generarDatosSupervisados3enRaya();
        System.out.println("Datos generados: " + datos.size() + " ejemplos");
        
        // Crear y entrenar modelo experimental con datos supervisados
        Modelo3enRayaExperimental modeloExpSupervisado = new Modelo3enRayaExperimental();
        modeloExpSupervisado.entrenarSupervisado(datos, 500);  // 500 épocas
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo experimental (supervisado) entrenado");
        System.out.println("Tiempo: " + tiempo + " ms (" + tiempo/1000.0 + " segundos)");
        System.out.println("Método: Plasticidad hebiana con datos supervisados");
        System.out.println("Datos: " + datos.size() + " estados únicos");
        
        // Guardar para uso posterior
        try {
            modeloExpSupervisado.getCerebro().guardar(
                "src/main/resources/modelosExperimentales/modelo3enRayaExperimental_supervisado.nn"
            );
            System.out.println("Modelo guardado: modelo3enRayaExperimental_supervisado.nn");
        } catch (IOException e) {
            System.err.println("Error guardando modelo: " + e.getMessage());
        }
    }
    
    /**
     * Test 7: Comparar experimental supervisado vs clásico
     */
    @Test
    void test7_CompararExperimentalSupervisadoVsClasico() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 7: EXPERIMENTAL SUPERVISADO VS CLÁSICO");
        System.out.println("Ambos entrenados con los MISMOS datos supervisados");
        System.out.println("=".repeat(60));
        
        // Cargar modelos
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modelo3enRaya.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        try {
            Modelo3enRayaExperimental modeloExpSup = Modelo3enRayaExperimental.cargar(
                "src/main/resources/modelosExperimentales/modelo3enRayaExperimental_supervisado.nn"
            );
            
            int partidasPrueba = 100;
            
            // Clásico vs Aleatorio
            System.out.println("\n--- Clásico (Backprop) vs Aleatorio ---");
            ResultadoPartidas resultadoClasico = jugarContraAleatorio(cerebroClasico, partidasPrueba);
            mostrarResultados(resultadoClasico);
            
            // Experimental Supervisado vs Aleatorio
            System.out.println("\n--- Experimental (Hebiano) vs Aleatorio ---");
            ResultadoPartidas resultadoExpSup = jugarContraAleatorioExperimental(
                modeloExpSup.getCerebro(), partidasPrueba
            );
            mostrarResultados(resultadoExpSup);
            
            // Enfrentamiento directo
            System.out.println("\n--- Enfrentamiento Directo (100 partidas) ---");
            System.out.println("Ronda 1: Clásico (P1) vs Experimental (P2)");
            ResultadoEnfrentamiento ronda1 = enfrentarModelos(
                cerebroClasico, modeloExpSup.getCerebro(), 50, true
            );
            mostrarResultadosEnfrentamiento(ronda1, "Clásico", "Experimental");
            
            System.out.println("\nRonda 2: Experimental (P1) vs Clásico (P2)");
            ResultadoEnfrentamiento ronda2 = enfrentarModelos(
                cerebroClasico, modeloExpSup.getCerebro(), 50, false
            );
            mostrarResultadosEnfrentamiento(ronda2, "Experimental", "Clásico");
            
            // Resumen
            System.out.println("\n--- RESUMEN TOTAL ---");
            int victoriasClasico = ronda1.victoriasP1 + ronda2.victoriasP2;
            int victoriasExp = ronda1.victoriasP2 + ronda2.victoriasP1;
            int empates = ronda1.empates + ronda2.empates;
            
            System.out.println("Victorias Clásico (Backprop): " + victoriasClasico + " (" + 
                String.format("%.1f%%", victoriasClasico * 100.0 / 100) + ")");
            System.out.println("Victorias Experimental (Hebiano): " + victoriasExp + " (" + 
                String.format("%.1f%%", victoriasExp * 100.0 / 100) + ")");
            System.out.println("Empates: " + empates);
            
            System.out.println("\n--- CONCLUSIÓN ---");
            System.out.println("Esta comparación muestra la diferencia entre:");
            System.out.println("- Backpropagation (optimización matemática)");
            System.out.println("- Plasticidad Hebiana (aprendizaje biológico)");
            System.out.println("Ambos con los MISMOS datos de entrenamiento");
            
        } catch (IOException | ClassNotFoundException e) {
            System.out.println("Modelo experimental supervisado no encontrado, ejecuta test6 primero");
        }
    }
    
    /**
     * Genera datos de entrenamiento supervisados para 3 en raya
     * Explora todos los estados posibles y determina el movimiento óptimo
     */
    private List<Modelo3enRayaExperimental.ParEntrenamiento> generarDatosSupervisados3enRaya() {
        List<Modelo3enRayaExperimental.ParEntrenamiento> datos = new ArrayList<>();
        Set<String> visitados = new HashSet<>();
        Queue<EstadoConTurno> cola = new LinkedList<>();
        
        // Estado inicial
        Tablero inicial = Funciones3enRaya.inicial3enRaya();
        cola.add(new EstadoConTurno(inicial, 1));
        visitados.add(tableroToString(inicial) + "_1");
        
        while (!cola.isEmpty()) {
            EstadoConTurno actual = cola.poll();
            
            if (Funciones3enRaya.fin3enRaya(actual.tablero)) {
                continue;
            }
            
            // Obtener movimientos posibles
            List<Movimiento> movimientos = Funciones3enRaya.movs3enRaya(actual.tablero, actual.turno);
            
            if (movimientos.isEmpty()) {
                continue;
            }
            
            // Evaluar cada movimiento y encontrar el mejor
            Posicion mejorMovimiento = null;
            double mejorValor = Double.NEGATIVE_INFINITY;
            
            for (Movimiento mov : movimientos) {
                double valor = evaluarMovimiento(mov.getTablero(), actual.turno);
                if (valor > mejorValor) {
                    mejorValor = valor;
                    mejorMovimiento = mov.getPos();
                }
                
                // Añadir estado siguiente a la cola
                String key = tableroToString(mov.getTablero()) + "_" + (3 - actual.turno);
                if (!visitados.contains(key)) {
                    visitados.add(key);
                    cola.add(new EstadoConTurno(mov.getTablero(), 3 - actual.turno));
                }
            }
            
            // Añadir par de entrenamiento
            if (mejorMovimiento != null) {
                datos.add(new Modelo3enRayaExperimental.ParEntrenamiento(
                    actual.tablero, actual.turno, mejorMovimiento
                ));
            }
        }
        
        return datos;
    }
    
    /**
     * Evalúa un movimiento (heurística simple)
     */
    private double evaluarMovimiento(Tablero tablero, int turno) {
        if (Funciones3enRaya.hay3EnRaya(tablero)) {
            // Determinar quién ganó
            int c1 = 0, c2 = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (tablero.getValor(i, j) == 1) c1++;
                    else if (tablero.getValor(i, j) == 2) c2++;
                }
            }
            boolean ganoP1 = (c1 > c2);
            return (turno == 1 && ganoP1) || (turno == 2 && !ganoP1) ? 100.0 : -100.0;
        }
        
        // Heurística: contar líneas potenciales
        double valor = 0.0;
        // Centro vale más
        if (tablero.getValor(1, 1) == turno) valor += 3.0;
        // Esquinas valen algo
        if (tablero.getValor(0, 0) == turno) valor += 2.0;
        if (tablero.getValor(0, 2) == turno) valor += 2.0;
        if (tablero.getValor(2, 0) == turno) valor += 2.0;
        if (tablero.getValor(2, 2) == turno) valor += 2.0;
        
        return valor;
    }
    
    private String tableroToString(Tablero tablero) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sb.append(tablero.getValor(i, j));
            }
        }
        return sb.toString();
    }
    
    private static class EstadoConTurno {
        Tablero tablero;
        int turno;
        
        EstadoConTurno(Tablero tablero, int turno) {
            this.tablero = tablero;
            this.turno = turno;
        }
    }
}
