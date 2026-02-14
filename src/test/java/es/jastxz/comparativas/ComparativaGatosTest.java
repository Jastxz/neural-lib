package es.jastxz.comparativas;

import es.jastxz.engine.FuncionesGato;
import es.jastxz.models.ModeloGatos;
import es.jastxz.models.ModeloGatosExperimental;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import es.jastxz.util.ModelManager;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comparativa completa entre entrenamiento supervisado y no supervisado
 * para el juego de Gatos
 */
public class ComparativaGatosTest {
    
    private static final Posicion POS_MOUSE_INICIAL = new Posicion(0, 2);
    
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
        ModeloGatos modeloClasico = new ModeloGatos();
        modeloClasico.entrenar();
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo clásico entrenado");
        System.out.println("Tiempo: " + tiempo + " ms (" + tiempo/1000.0 + " segundos)");
        System.out.println("Método: Backpropagation con simulación Minimax");
        System.out.println("Datos: 100 partidas completas simuladas");
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
        ModeloGatosExperimental modeloExperimental = new ModeloGatosExperimental();
        modeloExperimental.entrenarSelfPlay(300);  // 300 partidas de self-play
        
        long tiempo = System.currentTimeMillis() - inicio;
        
        System.out.println("\n✓ Modelo experimental entrenado");
        System.out.println("Tiempo: " + tiempo + " ms (" + tiempo/1000.0 + " segundos)");
        System.out.println("Método: Self-play con plasticidad hebiana");
        System.out.println("Partidas: 300");
        
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
        
        // Cargar modelo clásico
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modeloGatos.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        // Jugar 50 partidas con cada modelo (Gatos es más lento que 3 en raya)
        int partidasPrueba = 50;
        
        System.out.println("\n--- Modelo Clásico vs Aleatorio ---");
        ResultadoPartidas resultadoClasico = jugarContraAleatorio(cerebroClasico, partidasPrueba, true);
        mostrarResultados(resultadoClasico);
        
        System.out.println("\n--- Modelo Experimental vs Aleatorio ---");
        try {
            ModeloGatosExperimental modeloExp = ModeloGatosExperimental.cargar(
                "src/main/resources/modelosExperimentales/modeloGatosExperimental.nn"
            );
            ResultadoPartidas resultadoExp = jugarContraAleatorio(
                modeloExp.getCerebro(), partidasPrueba, false
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
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modeloGatos.nn");
        assertNotNull(cerebroClasico, "Modelo clásico debe existir");
        
        try {
            ModeloGatosExperimental modeloExp = ModeloGatosExperimental.cargar(
                "src/main/resources/modelosExperimentales/modeloGatosExperimental.nn"
            );
            
            int partidasPrueba = 40;  // 20 por lado
            
            // Clásico juega como Ratón
            System.out.println("\n--- Ronda 1: Clásico (Ratón) vs Experimental (Gatos) ---");
            ResultadoEnfrentamiento ronda1 = enfrentarModelos(
                cerebroClasico, modeloExp.getCerebro(), partidasPrueba/2, true, true
            );
            mostrarResultadosEnfrentamiento(ronda1, "Clásico", "Experimental");
            
            // Clásico juega como Gatos
            System.out.println("\n--- Ronda 2: Clásico (Gatos) vs Experimental (Ratón) ---");
            ResultadoEnfrentamiento ronda2 = enfrentarModelos(
                cerebroClasico, modeloExp.getCerebro(), partidasPrueba/2, false, true
            );
            mostrarResultadosEnfrentamiento(ronda2, "Clásico", "Experimental");
            
            // Resumen total
            System.out.println("\n--- RESUMEN TOTAL (" + partidasPrueba + " partidas) ---");
            int victoriasClasico = ronda1.victoriasP1 + ronda2.victoriasP2;
            int victoriasExp = ronda1.victoriasP2 + ronda2.victoriasP1;
            
            System.out.println("Victorias Clásico: " + victoriasClasico + " (" + 
                String.format("%.1f%%", victoriasClasico * 100.0 / partidasPrueba) + ")");
            System.out.println("Victorias Experimental: " + victoriasExp + " (" + 
                String.format("%.1f%%", victoriasExp * 100.0 / partidasPrueba) + ")");
            
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
        NeuralNetwork cerebroClasico = ModelManager.loadModel("modeloGatos.nn");
        
        try {
            ModeloGatosExperimental modeloExp = ModeloGatosExperimental.cargar(
                "src/main/resources/modelosExperimentales/modeloGatosExperimental.nn"
            );
            
            // Situación 1: Apertura (primer movimiento del ratón)
            System.out.println("\n--- Situación 1: Apertura (primer movimiento Ratón) ---");
            Tablero tableroInicial = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
            compararDecisiones(cerebroClasico, modeloExp.getCerebro(), tableroInicial, 1, true);
            
            // Situación 2: Ratón cerca de la meta
            System.out.println("\n--- Situación 2: Ratón cerca de la victoria (fila 6) ---");
            int[][] estadoCerca = crearTableroRatonCerca();
            Tablero tableroCerca = new Tablero(new SmallMatrix(estadoCerca));
            compararDecisiones(cerebroClasico, modeloExp.getCerebro(), tableroCerca, 1, true);
            
            // Situación 3: Gatos intentando encerrar
            System.out.println("\n--- Situación 3: Gatos intentando encerrar ---");
            int[][] estadoEncerrar = crearTableroGatosEncerrar();
            Tablero tableroEncerrar = new Tablero(new SmallMatrix(estadoEncerrar));
            compararDecisiones(cerebroClasico, modeloExp.getCerebro(), tableroEncerrar, 2, true);
            
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
     * Juega partidas contra jugador aleatorio
     */
    private ResultadoPartidas jugarContraAleatorio(Object cerebro, int numPartidas, boolean esClasico) {
        int victorias = 0;
        int derrotas = 0;
        Random rand = new Random();
        
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
            int turno = 1;  // Empieza el ratón
            int movimientos = 0;
            int maxMovimientos = 100;
            
            // IA juega como ratón (turno 1)
            while (!FuncionesGato.finGato(tablero) && movimientos < maxMovimientos) {
                Posicion movimiento;
                
                if (turno == 1) {
                    // IA juega (Ratón)
                    movimiento = esClasico ? 
                        obtenerMovimientoClasico((NeuralNetwork)cerebro, tablero, turno) :
                        obtenerMovimientoExperimental((RedNeuralExperimental)cerebro, tablero, turno);
                } else {
                    // Aleatorio juega (Gatos)
                    movimiento = obtenerMovimientoAleatorio(tablero, turno, rand);
                }
                
                if (movimiento == null) break;
                
                tablero = aplicarMovimiento(tablero, movimiento, turno);
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }
            
            // Determinar resultado
            boolean ratonGano = !FuncionesGato.ratonEncerrado(tablero, encontrarRaton(tablero));
            if (ratonGano) victorias++;
            else derrotas++;
        }
        
        return new ResultadoPartidas(victorias, derrotas, 0);
    }
    
    /**
     * Enfrenta dos modelos directamente
     */
    private ResultadoEnfrentamiento enfrentarModelos(
            NeuralNetwork cerebroClasico,
            RedNeuralExperimental cerebroExp,
            int numPartidas,
            boolean clasicoEsRaton,
            boolean esClasico) {
        
        int victoriasP1 = 0;
        int victoriasP2 = 0;
        
        for (int i = 0; i < numPartidas; i++) {
            Tablero tablero = FuncionesGato.inicialGato(POS_MOUSE_INICIAL);
            int turno = 1;
            int movimientos = 0;
            int maxMovimientos = 100;
            
            while (!FuncionesGato.finGato(tablero) && movimientos < maxMovimientos) {
                Posicion movimiento;
                
                boolean esTurnoClasico = (clasicoEsRaton && turno == 1) || (!clasicoEsRaton && turno == 2);
                
                if (esTurnoClasico) {
                    movimiento = obtenerMovimientoClasico(cerebroClasico, tablero, turno);
                } else {
                    movimiento = obtenerMovimientoExperimental(cerebroExp, tablero, turno);
                }
                
                if (movimiento == null) break;
                
                tablero = aplicarMovimiento(tablero, movimiento, turno);
                turno = (turno == 1) ? 2 : 1;
                movimientos++;
            }
            
            // Determinar resultado
            boolean ratonGano = !FuncionesGato.ratonEncerrado(tablero, encontrarRaton(tablero));
            
            if (clasicoEsRaton) {
                if (ratonGano) victoriasP1++;
                else victoriasP2++;
            } else {
                if (ratonGano) victoriasP2++;
                else victoriasP1++;
            }
        }
        
        return new ResultadoEnfrentamiento(victoriasP1, victoriasP2, 0);
    }
    
    /**
     * Compara las decisiones de ambos modelos
     */
    private void compararDecisiones(
            NeuralNetwork cerebroClasico,
            RedNeuralExperimental cerebroExp,
            Tablero tablero,
            int turno,
            boolean esClasico) {
        
        System.out.println("Tablero:");
        imprimirTablero(tablero);
        
        Posicion movClasico = obtenerMovimientoClasico(cerebroClasico, tablero, turno);
        Posicion movExp = obtenerMovimientoExperimental(cerebroExp, tablero, turno);
        
        System.out.println("Decisión Clásico: " + posicionToString(movClasico));
        System.out.println("Decisión Experimental: " + posicionToString(movExp));
        
        if (movClasico != null && movClasico.equals(movExp)) {
            System.out.println("✓ Ambos modelos coinciden");
        } else {
            System.out.println("✗ Modelos difieren");
        }
    }
    
    // Métodos de obtención de movimientos
    
    private Posicion obtenerMovimientoClasico(NeuralNetwork cerebro, Tablero tablero, int turno) {
        double[] input = ModeloGatos.tabularToInput(tablero, turno);
        double[] output = cerebro.feedForward(input);
        
        List<Posicion> movsPosibles = generarMovimientosLegales(tablero, turno);
        Posicion mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Posicion pos : movsPosibles) {
            int index = pos.getFila() * 8 + pos.getColumna();
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejor = pos;
            }
        }
        
        return mejor;
    }
    
    private Posicion obtenerMovimientoExperimental(
            RedNeuralExperimental cerebro, Tablero tablero, int turno) {
        double[] input = ModeloGatos.tabularToInput(tablero, turno);
        cerebro.resetear();
        double[] output = cerebro.procesar(input);
        
        List<Posicion> movsPosibles = generarMovimientosLegales(tablero, turno);
        Posicion mejor = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Posicion pos : movsPosibles) {
            int index = pos.getFila() * 8 + pos.getColumna();
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejor = pos;
            }
        }
        
        return mejor;
    }
    
    private Posicion obtenerMovimientoAleatorio(Tablero tablero, int turno, Random rand) {
        List<Posicion> movsPosibles = generarMovimientosLegales(tablero, turno);
        if (movsPosibles.isEmpty()) return null;
        return movsPosibles.get(rand.nextInt(movsPosibles.size()));
    }
    
    private List<Posicion> generarMovimientosLegales(Tablero tablero, int turno) {
        List<Posicion> movimientos = new ArrayList<>();
        
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int val = tablero.getValor(f, c);
                boolean esMio = false;
                
                if (turno == 1 && val == FuncionesGato.nombreRaton) {
                    esMio = true;
                } else if (turno == 2 && val != 0 && val != FuncionesGato.nombreRaton) {
                    esMio = true;
                }
                
                if (esMio) {
                    int[][] dirs;
                    if (turno == 1) {
                        dirs = new int[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
                    } else {
                        dirs = new int[][]{{-1, -1}, {-1, 1}};
                    }
                    
                    for (int[] d : dirs) {
                        int nf = f + d[0];
                        int nc = c + d[1];
                        
                        if (nf >= 0 && nf < 8 && nc >= 0 && nc < 8 && 
                            (nf + nc) % 2 == 0 && tablero.getValor(nf, nc) == 0) {
                            movimientos.add(new Posicion(nf, nc));
                        }
                    }
                }
            }
        }
        
        return movimientos;
    }
    
    private Tablero aplicarMovimiento(Tablero tablero, Posicion destino, int turno) {
        Posicion origen = encontrarOrigen(tablero, destino, turno);
        if (origen == null) return tablero;
        
        int[][] nuevoEstado = new int[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                nuevoEstado[i][j] = tablero.getValor(i, j);
            }
        }
        
        int pieza = tablero.getValor(origen.getFila(), origen.getColumna());
        nuevoEstado[destino.getFila()][destino.getColumna()] = pieza;
        nuevoEstado[origen.getFila()][origen.getColumna()] = 0;
        
        return new Tablero(new SmallMatrix(nuevoEstado));
    }
    
    private Posicion encontrarOrigen(Tablero tablero, Posicion destino, int turno) {
        int[][] dirs = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        
        for (int[] d : dirs) {
            int f = destino.getFila() + d[0];
            int c = destino.getColumna() + d[1];
            
            if (f >= 0 && f < 8 && c >= 0 && c < 8) {
                int val = tablero.getValor(f, c);
                
                if (turno == 1 && val == FuncionesGato.nombreRaton) {
                    return new Posicion(f, c);
                } else if (turno == 2 && val != 0 && val != FuncionesGato.nombreRaton) {
                    return new Posicion(f, c);
                }
            }
        }
        
        return null;
    }
    
    private Posicion encontrarRaton(Tablero tablero) {
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                if (tablero.getValor(f, c) == FuncionesGato.nombreRaton) {
                    return new Posicion(f, c);
                }
            }
        }
        return null;
    }
    
    // Métodos de creación de tableros de prueba
    
    private int[][] crearTableroRatonCerca() {
        int[][] tablero = new int[8][8];
        // Ratón en fila 6, cerca de la meta
        tablero[6][2] = FuncionesGato.nombreRaton;
        // Algunos gatos
        tablero[7][1] = 1;
        tablero[7][3] = 3;
        tablero[7][5] = 5;
        return tablero;
    }
    
    private int[][] crearTableroGatosEncerrar() {
        int[][] tablero = new int[8][8];
        // Ratón en posición media
        tablero[3][4] = FuncionesGato.nombreRaton;
        // Gatos rodeando
        tablero[4][3] = 1;
        tablero[4][5] = 3;
        tablero[5][2] = 5;
        tablero[5][6] = 7;
        return tablero;
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
                else if (val == FuncionesGato.nombreRaton) System.out.print("R ");
                else System.out.print("G ");
            }
            System.out.println();
        }
    }
    
    private String posicionToString(Posicion pos) {
        return pos == null ? "null" : "(" + pos.getFila() + "," + pos.getColumna() + ")";
    }
    
    private void mostrarResultados(ResultadoPartidas resultado) {
        System.out.println("Victorias: " + resultado.victorias + " (" + 
            String.format("%.1f%%", resultado.porcentajeVictorias()) + ")");
        System.out.println("Derrotas: " + resultado.derrotas + " (" + 
            String.format("%.1f%%", resultado.porcentajeDerrotas()) + ")");
    }
    
    private void mostrarResultadosEnfrentamiento(
            ResultadoEnfrentamiento resultado, String nombreP1, String nombreP2) {
        System.out.println(nombreP1 + ": " + resultado.victoriasP1 + " victorias (" + 
            String.format("%.1f%%", resultado.porcentajeP1()) + ")");
        System.out.println(nombreP2 + ": " + resultado.victoriasP2 + " victorias (" + 
            String.format("%.1f%%", resultado.porcentajeP2()) + ")");
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
        double porcentajeVictorias() { return victorias * 100.0 / total(); }
        double porcentajeDerrotas() { return derrotas * 100.0 / total(); }
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
    }
}
