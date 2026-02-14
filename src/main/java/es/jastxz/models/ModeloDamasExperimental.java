package es.jastxz.models;

import es.jastxz.engine.FuncionesDamas;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.Tablero;

import java.io.IOException;
import java.util.*;

/**
 * Modelo de Damas usando RedNeuralExperimental
 * 
 * Entrenamiento NO supervisado:
 * - Aprende jugando contra sí mismo (self-play)
 * - Usa plasticidad hebiana para reforzar movimientos ganadores
 * - Forma engramas de patrones de juego
 * 
 * Juego de Damas (simplificado):
 * - Tablero 8x8 (solo casillas negras)
 * - Blancas (1,3,5,7): empiezan abajo, mueven hacia arriba
 * - Negras (2,4,6,8): empiezan arriba, mueven hacia abajo
 * - Objetivo: capturar todas las piezas del oponente
 * - Coronación: al llegar al otro lado, se convierte en reina
 */
public class ModeloDamasExperimental {
    
    private RedNeuralExperimental cerebro;
    private String nombreModelo = "modeloDamasExperimental.nn";
    
    // Estadísticas de entrenamiento
    private int partidasJugadas = 0;
    private int victoriasBlancas = 0;
    private int victoriasNegras = 0;
    private int empates = 0;
    
    /**
     * Constructor con red nueva
     */
    public ModeloDamasExperimental() {
        // Topología: 65 inputs (8x8 + turno), 256 ocultas, 128 outputs (64 origen + 64 destino)
        cerebro = new RedNeuralExperimental(new int[]{65, 256, 128}, 0.85);
        
        // Activar sistemas biológicos
        cerebro.activarModoPredictivo(true);
        cerebro.activarDeteccionEngramas(true);
        cerebro.activarCompeticionRecursos(true);
    }
    
    /**
     * Constructor con red existente
     */
    public ModeloDamasExperimental(RedNeuralExperimental cerebro) {
        this.cerebro = cerebro;
    }
    
    /**
     * Entrenamiento NO supervisado mediante self-play
     */
    public void entrenarSelfPlay(int numPartidas) {
        System.out.println("=== Entrenamiento Self-Play Damas (No Supervisado) ===");
        System.out.println("Partidas a jugar: " + numPartidas);
        
        for (int partida = 0; partida < numPartidas; partida++) {
            jugarPartidaEntrenamiento();
            
            // Cada 20 partidas: consolidar y mostrar progreso
            if ((partida + 1) % 20 == 0) {
                consolidar();
                mostrarProgreso(partida + 1, numPartidas);
            }
        }
        
        System.out.println("\n=== Entrenamiento Completado ===");
        mostrarEstadisticas();
    }
    
    /**
     * Juega una partida completa para entrenamiento
     */
    private void jugarPartidaEntrenamiento() {
        Tablero tablero = FuncionesDamas.inicialDamas();
        int turno = 1; // Empiezan blancas
        
        List<EstadoJuego> historial = new ArrayList<>();
        int maxMovimientos = 200;
        int movimientos = 0;
        
        // Jugar hasta el final
        while (!FuncionesDamas.finDamas(tablero) && movimientos < maxMovimientos) {
            // Obtener movimiento de la red
            MovimientoDamas movimiento = obtenerMovimiento(tablero, turno);
            
            if (movimiento == null) {
                // No hay movimientos válidos
                break;
            }
            
            // Validar que el movimiento tiene origen válido
            if (movimiento.origen == null) {
                break;
            }
            
            // Guardar estado antes del movimiento
            historial.add(new EstadoJuego(tablero, turno, movimiento));
            
            // Aplicar movimiento
            tablero = aplicarMovimiento(tablero, movimiento, turno);
            
            // Cambiar turno
            turno = (turno == 1) ? 2 : 1;
            movimientos++;
        }
        
        // Determinar resultado
        int resultado = evaluarResultado(tablero, movimientos >= maxMovimientos);
        
        // Actualizar estadísticas
        partidasJugadas++;
        if (resultado == 1) victoriasBlancas++;
        else if (resultado == -1) victoriasNegras++;
        else empates++;
        
        // Reforzar movimientos basándose en el resultado
        reforzarMovimientos(historial, resultado);
    }
    
    /**
     * Obtiene el mejor movimiento según la red neuronal
     */
    private MovimientoDamas obtenerMovimiento(Tablero tablero, int turno) {
        List<Movimiento> movimientosPosibles = FuncionesDamas.movimientosDamas(tablero, FuncionesDamas.bandoMarca(turno));
        
        if (movimientosPosibles.isEmpty()) {
            return null;
        }
        
        // Convertir tablero a input
        double[] input = ModeloDamas.tabularToInput(tablero, turno);
        
        // Procesar con la red
        cerebro.resetear();
        double[] output = cerebro.procesar(input);
        
        // Interpretar output: primeros 64 = origen, últimos 64 = destino
        MovimientoDamas mejorMovimiento = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Movimiento mov : movimientosPosibles) {
            Posicion origen = encontrarOrigen(tablero, mov.getTablero(), turno);
            Posicion destino = mov.getPos();
            
            if (origen == null) continue;
            
            int indexOrigen = origen.getFila() * 8 + origen.getColumna();
            int indexDestino = destino.getFila() * 8 + destino.getColumna();
            
            // Valor combinado de origen y destino
            double valor = output[indexOrigen] + output[64 + indexDestino];
            
            if (valor > mejorValor) {
                mejorValor = valor;
                mejorMovimiento = new MovimientoDamas(origen, destino);
            }
        }
        
        // Si no encontró ninguno válido, elegir aleatorio
        if (mejorMovimiento == null) {
            Random rand = new Random();
            Movimiento movAleatorio = movimientosPosibles.get(rand.nextInt(movimientosPosibles.size()));
            Posicion origen = encontrarOrigen(tablero, movAleatorio.getTablero(), turno);
            mejorMovimiento = new MovimientoDamas(origen, movAleatorio.getPos());
        }
        
        return mejorMovimiento;
    }
    
    /**
     * Encuentra la posición de origen comparando tableros
     */
    private Posicion encontrarOrigen(Tablero antes, Tablero despues, int turno) {
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int valAntes = antes.getValor(f, c);
                int valDespues = despues.getValor(f, c);
                
                // Si había una pieza y ya no está
                if (valAntes != 0 && valDespues == 0) {
                    // Verificar que es del turno correcto
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
    
    /**
     * Aplica un movimiento al tablero
     */
    private Tablero aplicarMovimiento(Tablero tablero, MovimientoDamas movimiento, int turno) {
        // Buscar el movimiento correspondiente en la lista de movimientos válidos
        List<Movimiento> movsPosibles = FuncionesDamas.movimientosDamas(tablero, FuncionesDamas.bandoMarca(turno));
        
        for (Movimiento mov : movsPosibles) {
            Posicion origen = encontrarOrigen(tablero, mov.getTablero(), turno);
            if (origen != null && origen.equals(movimiento.origen) && 
                mov.getPos().equals(movimiento.destino)) {
                return mov.getTablero();
            }
        }
        
        // Si no se encontró, devolver tablero sin cambios
        return tablero;
    }
    
    /**
     * Refuerza los movimientos basándose en el resultado
     */
    private void reforzarMovimientos(List<EstadoJuego> historial, int resultado) {
        for (int i = 0; i < historial.size(); i++) {
            EstadoJuego estado = historial.get(i);
            
            // Validar que el movimiento tiene origen y destino válidos
            if (estado.movimiento.origen == null || estado.movimiento.destino == null) {
                continue;
            }
            
            // Calcular recompensa para este movimiento
            double recompensa = calcularRecompensa(estado.turno, resultado, i, historial.size());
            
            // Crear target: reforzar origen y destino del movimiento
            double[] target = new double[128];
            int indexOrigen = estado.movimiento.origen.getFila() * 8 + estado.movimiento.origen.getColumna();
            int indexDestino = estado.movimiento.destino.getFila() * 8 + estado.movimiento.destino.getColumna();
            target[indexOrigen] = recompensa;
            target[64 + indexDestino] = recompensa;
            
            // Entrenar la red con este ejemplo
            double[] input = ModeloDamas.tabularToInput(estado.tablero, estado.turno);
            cerebro.entrenar(input, target, 2);
        }
    }
    
    /**
     * Calcula la recompensa para un movimiento
     */
    private double calcularRecompensa(int turnoJugador, int resultado, int indiceMov, int totalMovs) {
        boolean gano = (turnoJugador == 1 && resultado == 1) || (turnoJugador == 2 && resultado == -1);
        
        if (gano) {
            // Recompensa positiva, mayor para movimientos finales
            double factor = (double) (indiceMov + 1) / totalMovs;
            return 0.5 + 0.5 * factor;  // [0.5, 1.0]
        } else if (resultado == 0) {
            // Empate
            return 0.5;
        } else {
            // Perdió
            return 0.0;
        }
    }
    
    /**
     * Evalúa el resultado final
     */
    private int evaluarResultado(Tablero tablero, boolean empate) {
        if (empate) {
            return 0;
        }
        
        // Contar piezas
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
    
    /**
     * Consolida el aprendizaje
     */
    private void consolidar() {
        cerebro.iniciarConsolidacion();
        cerebro.consolidar();
        cerebro.finalizarConsolidacion();
        
        cerebro.competirPorRecursos();
        cerebro.podarElementos();
    }
    
    /**
     * Muestra progreso del entrenamiento
     */
    private void mostrarProgreso(int partidasActuales, int partidasTotales) {
        double porcentaje = (partidasActuales * 100.0) / partidasTotales;
        System.out.printf("Progreso: %d/%d (%.1f%%) - Blancas: %d, Negras: %d, Empates: %d, Engramas: %d\n",
            partidasActuales, partidasTotales, porcentaje,
            victoriasBlancas, victoriasNegras, empates,
            cerebro.getEngramas().size());
    }
    
    /**
     * Muestra estadísticas finales
     */
    private void mostrarEstadisticas() {
        System.out.println("Partidas jugadas: " + partidasJugadas);
        System.out.println("Victorias Blancas: " + victoriasBlancas + " (" + 
            String.format("%.1f%%", victoriasBlancas * 100.0 / partidasJugadas) + ")");
        System.out.println("Victorias Negras: " + victoriasNegras + " (" + 
            String.format("%.1f%%", victoriasNegras * 100.0 / partidasJugadas) + ")");
        System.out.println("Empates: " + empates + " (" + 
            String.format("%.1f%%", empates * 100.0 / partidasJugadas) + ")");
        System.out.println("Engramas formados: " + cerebro.getEngramas().size());
        System.out.println("Conexiones totales: " + cerebro.getTotalConexiones());
    }
    
    /**
     * Guarda el modelo entrenado
     */
    public void guardar() throws IOException {
        cerebro.guardar("src/main/resources/modelosExperimentales/" + nombreModelo);
        System.out.println("Modelo guardado: " + nombreModelo);
    }
    
    /**
     * Carga un modelo pre-entrenado
     */
    public static ModeloDamasExperimental cargar(String ruta) throws IOException, ClassNotFoundException {
        RedNeuralExperimental cerebro = RedNeuralExperimental.cargar(ruta);
        return new ModeloDamasExperimental(cerebro);
    }
    
    // Getters
    public RedNeuralExperimental getCerebro() { return cerebro; }
    public int getPartidasJugadas() { return partidasJugadas; }
    public int getVictoriasBlancas() { return victoriasBlancas; }
    public int getVictoriasNegras() { return victoriasNegras; }
    public int getEmpates() { return empates; }
    
    /**
     * Clase interna para guardar estados del juego
     */
    private static class EstadoJuego {
        Tablero tablero;
        int turno;
        MovimientoDamas movimiento;
        
        EstadoJuego(Tablero tablero, int turno, MovimientoDamas movimiento) {
            this.tablero = tablero;
            this.turno = turno;
            this.movimiento = movimiento;
        }
    }
    
    /**
     * Clase para representar un movimiento de damas (origen + destino)
     */
    private static class MovimientoDamas {
        Posicion origen;
        Posicion destino;
        
        MovimientoDamas(Posicion origen, Posicion destino) {
            this.origen = origen;
            this.destino = destino;
        }
    }
}
