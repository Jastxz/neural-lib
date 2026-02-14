package es.jastxz.models;

import es.jastxz.engine.Funciones3enRaya;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.tipos.Tablero;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;

import java.io.IOException;
import java.util.*;

/**
 * Modelo de 3 en Raya usando RedNeuralExperimental
 * 
 * Entrenamiento NO supervisado:
 * - Aprende jugando contra sí mismo (self-play)
 * - Usa plasticidad hebiana para reforzar movimientos ganadores
 * - Forma engramas de patrones de juego
 * - Consolida estrategias durante "sueño"
 */
public class Modelo3enRayaExperimental {
    
    private RedNeuralExperimental cerebro;
    private String nombreModelo = "modelo3enRayaExperimental.nn";
    
    // Estadísticas de entrenamiento
    private int partidasJugadas = 0;
    private int victoriasP1 = 0;
    private int victoriasP2 = 0;
    private int empates = 0;
    
    /**
     * Constructor con red nueva
     */
    public Modelo3enRayaExperimental() {
        // Topología: 10 inputs (9 casillas + turno), 30 ocultas, 9 outputs (movimientos)
        cerebro = new RedNeuralExperimental(new int[]{10, 30, 9}, 0.9);
        
        // Activar sistemas biológicos
        cerebro.activarModoPredictivo(true);
        cerebro.activarDeteccionEngramas(true);
        cerebro.activarCompeticionRecursos(true);
    }
    
    /**
     * Constructor con red existente
     */
    public Modelo3enRayaExperimental(RedNeuralExperimental cerebro) {
        this.cerebro = cerebro;
    }
    
    /**
     * Entrenamiento NO supervisado mediante self-play
     * La red aprende jugando contra sí misma
     */
    public void entrenarSelfPlay(int numPartidas) {
        System.out.println("=== Entrenamiento Self-Play (No Supervisado) ===");
        System.out.println("Partidas a jugar: " + numPartidas);
        
        for (int partida = 0; partida < numPartidas; partida++) {
            jugarPartidaEntrenamiento();
            
            // Cada 100 partidas: consolidar y mostrar progreso
            if ((partida + 1) % 100 == 0) {
                consolidar();
                mostrarProgreso(partida + 1, numPartidas);
            }
        }
        
        System.out.println("\n=== Entrenamiento Completado ===");
        mostrarEstadisticas();
    }
    
    /**
     * Entrenamiento SUPERVISADO con datos pre-generados
     * Usa los mismos datos que el modelo clásico para comparación justa
     * 
     * @param datosEntrenamiento Lista de pares (tablero, movimiento óptimo)
     * @param epocas Número de veces que se repasa todo el dataset
     */
    public void entrenarSupervisado(List<ParEntrenamiento> datosEntrenamiento, int epocas) {
        System.out.println("=== Entrenamiento Supervisado (Red Experimental) ===");
        System.out.println("Datos de entrenamiento: " + datosEntrenamiento.size());
        System.out.println("Épocas: " + epocas);
        
        for (int epoca = 1; epoca <= epocas; epoca++) {
            double errorTotal = 0.0;
            
            // Mezclar datos para cada época
            List<ParEntrenamiento> datosMezclados = new ArrayList<>(datosEntrenamiento);
            Collections.shuffle(datosMezclados);
            
            for (ParEntrenamiento par : datosMezclados) {
                // Convertir tablero a input
                double[] input = tabularToInput(par.tablero, par.turno);
                
                // Crear target: 1.0 para el movimiento óptimo, 0.0 para el resto
                double[] target = new double[9];
                int index = par.movimientoOptimo.getFila() * 3 + par.movimientoOptimo.getColumna();
                target[index] = 1.0;
                
                // Entrenar con este ejemplo
                cerebro.entrenar(input, target, 5);
                
                // Calcular error para monitoreo
                cerebro.resetear();
                double[] output = cerebro.procesar(input);
                double error = 0.0;
                for (int i = 0; i < 9; i++) {
                    error += Math.pow(target[i] - output[i], 2);
                }
                errorTotal += error;
            }
            
            // Consolidar cada 100 épocas
            if (epoca % 100 == 0) {
                consolidar();
                double errorPromedio = errorTotal / datosEntrenamiento.size();
                System.out.printf("Época %d/%d - Error promedio: %.4f\n", 
                    epoca, epocas, errorPromedio);
            }
        }
        
        System.out.println("\n=== Entrenamiento Supervisado Completado ===");
        System.out.println("Engramas formados: " + cerebro.getEngramas().size());
        System.out.println("Conexiones totales: " + cerebro.getTotalConexiones());
    }
    
    /**
     * Juega una partida completa para entrenamiento
     * Ambos jugadores usan la misma red (self-play)
     */
    private void jugarPartidaEntrenamiento() {
        Tablero tablero = Funciones3enRaya.inicial3enRaya();
        int turno = 1;
        
        List<EstadoJuego> historial = new ArrayList<>();
        
        // Jugar hasta el final
        while (!Funciones3enRaya.fin3enRaya(tablero)) {
            // Obtener movimiento de la red
            Posicion movimiento = obtenerMovimiento(tablero, turno);
            
            if (movimiento == null) {
                // No hay movimientos válidos (no debería pasar)
                break;
            }
            
            // Guardar estado antes del movimiento
            historial.add(new EstadoJuego(tablero, turno, movimiento));
            
            // Aplicar movimiento
            int[][] nuevoEstado = copiarTablero(tablero);
            nuevoEstado[movimiento.getFila()][movimiento.getColumna()] = turno;
            tablero = new Tablero(new SmallMatrix(nuevoEstado));
            
            // Cambiar turno
            turno = (turno == 1) ? 2 : 1;
        }
        
        // Determinar resultado
        int resultado = evaluarResultado(tablero);
        
        // Actualizar estadísticas
        partidasJugadas++;
        if (resultado == 1) victoriasP1++;
        else if (resultado == -1) victoriasP2++;
        else empates++;
        
        // Reforzar movimientos basándose en el resultado
        reforzarMovimientos(historial, resultado);
    }
    
    /**
     * Obtiene el mejor movimiento según la red neuronal
     */
    private Posicion obtenerMovimiento(Tablero tablero, int turno) {
        List<Movimiento> movimientosPosibles = Funciones3enRaya.movs3enRaya(tablero, turno);
        
        if (movimientosPosibles.isEmpty()) {
            return null;
        }
        
        // Convertir tablero a input
        double[] input = tabularToInput(tablero, turno);
        
        // Procesar con la red
        cerebro.resetear();
        double[] output = cerebro.procesar(input);
        
        // Encontrar el mejor movimiento válido
        Posicion mejorMovimiento = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Movimiento mov : movimientosPosibles) {
            Posicion pos = mov.getPos();
            int index = pos.getFila() * 3 + pos.getColumna();
            
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejorMovimiento = pos;
            }
        }
        
        // Si no encontró ninguno válido, elegir aleatorio
        if (mejorMovimiento == null) {
            Random rand = new Random();
            mejorMovimiento = movimientosPosibles.get(rand.nextInt(movimientosPosibles.size())).getPos();
        }
        
        return mejorMovimiento;
    }
    
    /**
     * Refuerza los movimientos basándose en el resultado de la partida
     * Aprendizaje por refuerzo simple
     */
    private void reforzarMovimientos(List<EstadoJuego> historial, int resultado) {
        for (int i = 0; i < historial.size(); i++) {
            EstadoJuego estado = historial.get(i);
            
            // Calcular recompensa para este movimiento
            double recompensa = calcularRecompensa(estado.turno, resultado, i, historial.size());
            
            // Crear target: reforzar el movimiento realizado
            double[] target = new double[9];
            int index = estado.movimiento.getFila() * 3 + estado.movimiento.getColumna();
            target[index] = recompensa;
            
            // Entrenar la red con este ejemplo
            double[] input = tabularToInput(estado.tablero, estado.turno);
            cerebro.entrenar(input, target, 3);
        }
    }
    
    /**
     * Calcula la recompensa para un movimiento
     * Movimientos que llevan a victoria reciben recompensa positiva
     * Movimientos que llevan a derrota reciben recompensa negativa
     */
    private double calcularRecompensa(int turnoJugador, int resultado, int indiceMov, int totalMovs) {
        // Determinar si este jugador ganó, perdió o empató
        boolean gano = (turnoJugador == 1 && resultado == 1) || (turnoJugador == 2 && resultado == -1);
        boolean perdio = (turnoJugador == 1 && resultado == -1) || (turnoJugador == 2 && resultado == 1);
        
        if (gano) {
            // Recompensa positiva, mayor para movimientos finales
            double factor = (double) (indiceMov + 1) / totalMovs;
            return 0.5 + 0.5 * factor;  // [0.5, 1.0]
        } else if (perdio) {
            // Penalización negativa
            return 0.0;
        } else {
            // Empate: recompensa neutral
            return 0.5;
        }
    }
    
    /**
     * Evalúa el resultado final del tablero
     * @return 1 si gana P1, -1 si gana P2, 0 si empate
     */
    private int evaluarResultado(Tablero tablero) {
        if (!Funciones3enRaya.hay3EnRaya(tablero)) {
            return 0; // Empate
        }
        
        // Determinar quién ganó contando piezas
        int c1 = 0, c2 = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (tablero.getValor(i, j) == 1) c1++;
                else if (tablero.getValor(i, j) == 2) c2++;
            }
        }
        
        // Si hay 3 en raya y c1 == c2, P2 hizo el último movimiento -> P2 ganó
        // Si c1 > c2, P1 hizo el último movimiento -> P1 ganó
        return (c1 == c2) ? -1 : 1;
    }
    
    /**
     * Consolida el aprendizaje (simula "sueño")
     */
    private void consolidar() {
        cerebro.iniciarConsolidacion();
        cerebro.consolidar();
        cerebro.finalizarConsolidacion();
        
        // Competir y podar ocasionalmente
        cerebro.competirPorRecursos();
        cerebro.podarElementos();
    }
    
    /**
     * Convierte tablero a input para la red
     */
    private static double[] tabularToInput(Tablero tablero, int turno) {
        SmallMatrix tableroSmallMatrix = tablero.getMatrix();
        double[] inputs = new double[10];
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double valor = tableroSmallMatrix.get(i, j);
                int index = i * 3 + j;
                
                if (valor == 0) inputs[index] = 0.0;
                else if (valor == 1) inputs[index] = 1.0;
                else if (valor == 2) inputs[index] = -1.0;
            }
        }
        
        // Agregar turno
        inputs[9] = (turno == 1) ? 1.0 : -1.0;
        
        return inputs;
    }
    
    /**
     * Copia un tablero
     */
    private int[][] copiarTablero(Tablero tablero) {
        int[][] copia = new int[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                copia[i][j] = tablero.getValor(i, j);
            }
        }
        return copia;
    }
    
    /**
     * Muestra progreso del entrenamiento
     */
    private void mostrarProgreso(int partidasActuales, int partidasTotales) {
        double porcentaje = (partidasActuales * 100.0) / partidasTotales;
        System.out.printf("Progreso: %d/%d (%.1f%%) - P1: %d, P2: %d, Empates: %d, Engramas: %d\n",
            partidasActuales, partidasTotales, porcentaje,
            victoriasP1, victoriasP2, empates,
            cerebro.getEngramas().size());
    }
    
    /**
     * Muestra estadísticas finales
     */
    private void mostrarEstadisticas() {
        System.out.println("Partidas jugadas: " + partidasJugadas);
        System.out.println("Victorias P1: " + victoriasP1 + " (" + 
            String.format("%.1f%%", victoriasP1 * 100.0 / partidasJugadas) + ")");
        System.out.println("Victorias P2: " + victoriasP2 + " (" + 
            String.format("%.1f%%", victoriasP2 * 100.0 / partidasJugadas) + ")");
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
    public static Modelo3enRayaExperimental cargar(String ruta) throws IOException, ClassNotFoundException {
        RedNeuralExperimental cerebro = RedNeuralExperimental.cargar(ruta);
        return new Modelo3enRayaExperimental(cerebro);
    }
    
    // Getters
    public RedNeuralExperimental getCerebro() { return cerebro; }
    public int getPartidasJugadas() { return partidasJugadas; }
    public int getVictoriasP1() { return victoriasP1; }
    public int getVictoriasP2() { return victoriasP2; }
    public int getEmpates() { return empates; }
    
    /**
     * Clase interna para guardar estados del juego
     */
    private static class EstadoJuego {
        Tablero tablero;
        int turno;
        Posicion movimiento;
        
        EstadoJuego(Tablero tablero, int turno, Posicion movimiento) {
            this.tablero = tablero;
            this.turno = turno;
            this.movimiento = movimiento;
        }
    }
    
    /**
     * Clase para datos de entrenamiento supervisado
     */
    public static class ParEntrenamiento {
        public Tablero tablero;
        public int turno;
        public Posicion movimientoOptimo;
        
        public ParEntrenamiento(Tablero tablero, int turno, Posicion movimientoOptimo) {
            this.tablero = tablero;
            this.turno = turno;
            this.movimientoOptimo = movimientoOptimo;
        }
    }
}
