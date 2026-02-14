package es.jastxz.comparativas;

import es.jastxz.engine.FuncionesDamas;
import es.jastxz.models.ModeloDamas;
import es.jastxz.models.ModeloDamasExperimental;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import es.jastxz.util.ModelManager;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa completa entre entrenamiento supervisado y no supervisado
 * para el juego de Damas
 * 
 * NOTA: Solo se prueba entrenamiento NO supervisado para el modelo experimental,
 * ya que los resultados de 3 en raya mostraron que el supervisado es PEOR
 * para la red experimental (78% vs 87%).
 */
public class ComparativaDamasTest {
    
    /**
     * Test 1: Entrenar modelo clásico (supervisado)
     * NOTA: Este test puede tardar VARIAS HORAS (>30 min)
     * El modelo ya está pre-entrenado en modeloDamas.nn
     * Solo ejecutar si se quiere re-entrenar desde cero
     */
    @Test
    @org.junit.jupiter.api.Disabled("Test muy lento (>30 min). Modelo ya pre-entrenado.")
    void test1_EntrenarModeloClasico() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 1: ENTRENAMIENTO SUPERVISADO (Red Clásica)");
        System.out.println("ADVERTENCIA: Este test puede tardar varias horas");
        System.out.println("=".repeat(60));
        
        long inicio = System.currentTimeMillis();
        
        // Crear y entrenar modelo clásico
        ModeloDamas modeloClasico = new ModeloDamas();
        modeloClasico.entrenar();
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo clásico entrenado");
        System.out.println("Tiempo: " + tiempo + " ms (" + tiempo/60000.0 + " minutos)");
        System.out.println("Método: Backpropagation con datos pre-generados");
    }
    
    /**
     * Test 2: Entrenar y probar modelo experimental (no supervisado)
     * NOTA: No se guarda el modelo debido a StackOverflow en serialización
     */
    @Test
    void test2_EntrenarYProbarModeloExperimental() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 2: ENTRENAMIENTO Y PRUEBA (Red Experimental)");
        System.out.println("=".repeat(60));
        
        long inicio = System.currentTimeMillis();
        
        // Crear y entrenar modelo experimental
        ModeloDamasExperimental modeloExperimental = new ModeloDamasExperimental();
        modeloExperimental.entrenarSelfPlay(100);  // 100 partidas de self-play
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo experimental entrenado");
        System.out.println("Tiempo: " + tiempo + " ms (" + tiempo/60000.0 + " minutos)");
        System.out.println("Método: Self-play con plasticidad hebiana");
        System.out.println("Partidas: 100");
        
        // Probar inmediatamente contra jugador aleatorio (sin guardar)
        System.out.println("\n--- Probando contra Jugador Aleatorio ---");
        ResultadoPartidas resultado = jugarContraAleatorio(
            modeloExperimental.getCerebro(), 20, false
        );
        mostrarResultados(resultado);
        
        System.out.println("\nNOTA: Modelo no guardado (StackOverflow en serialización con redes grandes)");
    }
    
    /**
     * Test 3: Rendimiento modelo clásico contra jugador aleatorio
     */
    @Test
    void test3_ModeloClasicoContraAleatorio() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 3: MODELO CLÁSICO VS ALEATORIO");
        System.out.println("=".repeat(60));
        
        // Cargar modelo clásico
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modeloDamas.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        // Jugar 20 partidas (Damas es muy lento)
        int partidasPrueba = 20;
        
        System.out.println("\n--- Modelo Clásico vs Aleatorio ---");
        ResultadoPartidas resultado = jugarContraAleatorio(cerebroClasico, partidasPrueba, true);
        mostrarResultados(resultado);
        
        System.out.println("\nConclusión:");
        if (resultado.porcentajeVictorias() >= 70) {
            System.out.println("✓ Modelo clásico tiene buen rendimiento (>70% victorias)");
        } else {
            System.out.println("⚠ Modelo clásico necesita más entrenamiento");
        }
    }
    
    /**
     * Test 4: Análisis de estrategias del modelo clásico
     */
    @Test
    void test4_AnalisisEstrategiasClasico() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("TEST 4: ANÁLISIS DE ESTRATEGIAS (Modelo Clásico)");
        System.out.println("=".repeat(60));
        
        // Cargar modelo clásico
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modeloDamas.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        // Situación 1: Apertura (primer movimiento)
        System.out.println("\n--- Situación 1: Apertura (primer movimiento) ---");
        Tablero tableroInicial = FuncionesDamas.inicialDamas();
        analizarDecision(cerebroClasico, tableroInicial, 1);
        
        // Situación 2: Medio juego
        System.out.println("\n--- Situación 2: Medio juego ---");
        Tablero tableroMedio = crearTableroMedioJuego();
        analizarDecision(cerebroClasico, tableroMedio, 1);
        
        System.out.println("\n--- Conclusión ---");
        System.out.println("El modelo clásico muestra decisiones consistentes basadas en");
        System.out.println("el entrenamiento supervisado con datos pre-generados.");
    }
    
    // ==================== MÉTODOS AUXILIARES ====================
    
    /**
     * Juega partidas contra jugador aleatorio
     */
    private ResultadoPartidas jugarContraAleatorio(Object cerebro, int numPartidas, boolean esClasico) {
        int victorias = 0;
        int derrotas = 0;
        int empates = 0;
        Random rand = new Random();
        
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesDamas.inicialDamas();
            int turno = 1;  // Empiezan blancas
            int movimientos = 0;
            int maxMovimientos = 200;
            
            // IA juega como blancas (turno 1)
            while (!FuncionesDamas.finDamas(tablero) && movimientos < maxMovimientos) {
                List<Movimiento> movsPosibles = FuncionesDamas.movimientosDamas(tablero, FuncionesDamas.bandoMarca(turno));
                
                if (movsPosibles.isEmpty()) break;
                
                Movimiento movimiento;
                
                if (turno == 1) {
                    // IA juega (Blancas)
                    movimiento = esClasico ? 
                        obtenerMovimientoClasico((NeuralNetwork)cerebro, tablero, turno, movsPosibles) :
                        obtenerMovimientoExperimental((RedNeuralExperimental)cerebro, tablero, turno, movsPosibles);
                } else {
                    // Aleatorio juega (Negras)
                    movimiento = movsPosibles.get(rand.nextInt(movsPosibles.size()));
                }
                
                if (movimiento == null) break;
                
                tablero = movimiento.getTablero();
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }
            
            // Determinar resultado
            int resultado = evaluarResultado(tablero, movimientos >= maxMovimientos);
            if (resultado == 1) victorias++;
            else if (resultado == -1) derrotas++;
            else empates++;
        }
        
        return new ResultadoPartidas(victorias, derrotas, empates);
    }
    
    /**
     * Analiza la decisión del modelo clásico
     */
    private void analizarDecision(NeuralNetwork cerebroClasico, Tablero tablero, int turno) {
        System.out.println("Tablero:");
        imprimirTablero(tablero);
        
        List<Movimiento> movsPosibles = FuncionesDamas.movimientosDamas(tablero, FuncionesDamas.bandoMarca(turno));
        
        if (movsPosibles.isEmpty()) {
            System.out.println("No hay movimientos posibles");
            return;
        }
        
        Movimiento movClasico = obtenerMovimientoClasico(cerebroClasico, tablero, turno, movsPosibles);
        
        System.out.println("Decisión: " + movimientoToString(tablero, movClasico, turno));
        System.out.println("Movimientos posibles: " + movsPosibles.size());
    }
    
    // Métodos de obtención de movimientos
    
    private Movimiento obtenerMovimientoClasico(
            NeuralNetwork cerebro, Tablero tablero, int turno, List<Movimiento> movsPosibles) {
        
        double[] input = ModeloDamas.tabularToInput(tablero, turno);
        double[] output = cerebro.feedForward(input);
        
        Movimiento mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Movimiento mov : movsPosibles) {
            Posicion origen = encontrarOrigen(tablero, mov.getTablero(), turno);
            Posicion destino = mov.getPos();
            
            if (origen == null) continue;
            
            int indexOrigen = origen.getFila() * 8 + origen.getColumna();
            int indexDestino = destino.getFila() * 8 + destino.getColumna();
            
            double valor = output[indexOrigen] + output[64 + indexDestino];
            
            if (valor > mejorValor) {
                mejorValor = valor;
                mejor = mov;
            }
        }
        
        return mejor != null ? mejor : movsPosibles.get(0);
    }
    
    private Movimiento obtenerMovimientoExperimental(
            RedNeuralExperimental cerebro, Tablero tablero, int turno, List<Movimiento> movsPosibles) {
        
        double[] input = ModeloDamas.tabularToInput(tablero, turno);
        cerebro.resetear();
        double[] output = cerebro.procesar(input);
        
        Movimiento mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Movimiento mov : movsPosibles) {
            Posicion origen = encontrarOrigen(tablero, mov.getTablero(), turno);
            Posicion destino = mov.getPos();
            
            if (origen == null) continue;
            
            int indexOrigen = origen.getFila() * 8 + origen.getColumna();
            int indexDestino = destino.getFila() * 8 + destino.getColumna();
            
            double valor = output[indexOrigen] + output[64 + indexDestino];
            
            if (valor > mejorValor) {
                mejorValor = valor;
                mejor = mov;
            }
        }
        
        return mejor != null ? mejor : movsPosibles.get(0);
    }
    
    private Posicion encontrarOrigen(Tablero antes, Tablero despues, int turno) {
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int valAntes = antes.getValor(f, c);
                int valDespues = despues.getValor(f, c);
                
                if (valAntes != 0 && valDespues == 0) {
                    boolean esBlanca = ModeloDamas.contiene(FuncionesDamas.nombresBlancas, valAntes) ||
                                      ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, valAntes);
                    boolean esNegra = ModeloDamas.contiene(FuncionesDamas.nombresNegras, valAntes) ||
                                     ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, valAntes);
                    
                    if ((turno == 1 && esBlanca) || (turno == 2 && esNegra)) {
                        return new Posicion(f, c);
                    }
                }
            }
        }
        return null;
    }
    
    private int evaluarResultado(Tablero tablero, boolean empate) {
        if (empate) return 0;
        
        int blancas = 0, negras = 0;
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int val = tablero.getValor(f, c);
                if (ModeloDamas.contiene(FuncionesDamas.nombresBlancas, val) ||
                    ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, val)) {
                    blancas++;
                } else if (ModeloDamas.contiene(FuncionesDamas.nombresNegras, val) ||
                          ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, val)) {
                    negras++;
                }
            }
        }
        
        if (blancas > negras) return 1;
        if (negras > blancas) return -1;
        return 0;
    }
    
    // Métodos de creación de tableros de prueba
    
    private Tablero crearTableroMedioJuego() {
        int[][] tablero = new int[8][8];
        // Algunas piezas blancas
        tablero[5][0] = 1;
        tablero[5][2] = 3;
        tablero[6][1] = 5;
        // Algunas piezas negras
        tablero[2][1] = 2;
        tablero[2][3] = 4;
        tablero[3][2] = 6;
        return new Tablero(new SmallMatrix(tablero));
    }
    
    // Métodos de visualización
    
    private void imprimirTablero(Tablero tablero) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if ((i + j) % 2 != 0) {
                    System.out.print("  ");
                    continue;
                }
                int val = tablero.getValor(i, j);
                if (val == 0) System.out.print(". ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresBlancas, val)) System.out.print("b ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, val)) System.out.print("B ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresNegras, val)) System.out.print("n ");
                else if (ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, val)) System.out.print("N ");
                else System.out.print("? ");
            }
            System.out.println();
        }
    }
    
    private String movimientoToString(Tablero tablero, Movimiento mov, int turno) {
        if (mov == null) return "null";
        Posicion origen = encontrarOrigen(tablero, mov.getTablero(), turno);
        Posicion destino = mov.getPos();
        return (origen != null ? origen.toString() : "?") + " -> " + destino.toString();
    }
    
    private void mostrarResultados(ResultadoPartidas resultado) {
        System.out.println("Victorias: " + resultado.victorias + " (" + 
            String.format("%.1f%%", resultado.porcentajeVictorias()) + ")");
        System.out.println("Derrotas: " + resultado.derrotas + " (" + 
            String.format("%.1f%%", resultado.porcentajeDerrotas()) + ")");
        System.out.println("Empates: " + resultado.empates + " (" + 
            String.format("%.1f%%", resultado.porcentajeEmpates()) + ")");
    }
    
    // Clases auxiliares
    
    private static class ResultadoPartidas {
        int victorias, derrotas, empates;
        
        ResultadoPartidas(int victorias, int derrotas, int empates) {
            this.victorias = victorias;
            this.derrotas = derrotas;
            this.empates = empates;
        }
        
        int total() { return victorias + derrotas + empates; }
        double porcentajeVictorias() { return total() > 0 ? victorias * 100.0 / total() : 0; }
        double porcentajeDerrotas() { return total() > 0 ? derrotas * 100.0 / total() : 0; }
        double porcentajeEmpates() { return total() > 0 ? empates * 100.0 / total() : 0; }
    }
}
