package es.jastxz.models;

import es.jastxz.engine.FuncionesGato;
import es.jastxz.nn.RedNeuralExperimental;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;

import java.io.IOException;
import java.util.*;

/**
 * Modelo de Gatos usando RedNeuralExperimental
 * 
 * Entrenamiento NO supervisado:
 * - Aprende jugando contra sí mismo (self-play)
 * - Usa plasticidad hebiana para reforzar movimientos ganadores
 * - Forma engramas de patrones de juego
 * - Consolida estrategias durante "sueño"
 * 
 * Juego de Gatos:
 * - Tablero 8x8 (solo casillas blancas, patrón ajedrez)
 * - Ratón (9): empieza en (0,2), quiere llegar a fila 7, mueve en 4 diagonales
 * - Gatos (1,3,5,7): empiezan en filas 5-7, quieren encerrar al ratón, mueven solo hacia arriba
 * - Gana Ratón: llega a fila 7
 * - Ganan Gatos: encierran al ratón (sin movimientos válidos)
 */
public class ModeloGatosExperimental {
    
    private RedNeuralExperimental cerebro;
    private String nombreModelo = "modeloGatosExperimental.nn";
    private Posicion posMouseInicial = new Posicion(0, 2);
    
    // Estadísticas de entrenamiento
    private int partidasJugadas = 0;
    private int victoriasRaton = 0;
    private int victoriasGatos = 0;
    
    /**
     * Constructor con red nueva
     */
    public ModeloGatosExperimental() {
        // Topología: 65 inputs (8x8 + turno), 80 ocultas, 64 outputs (movimientos)
        cerebro = new RedNeuralExperimental(new int[]{65, 80, 64}, 0.9);
        
        // Activar sistemas biológicos
        cerebro.activarModoPredictivo(true);
        cerebro.activarDeteccionEngramas(true);
        cerebro.activarCompeticionRecursos(true);
    }
    
    /**
     * Constructor con red existente
     */
    public ModeloGatosExperimental(RedNeuralExperimental cerebro) {
        this.cerebro = cerebro;
    }
    
    /**
     * Entrenamiento NO supervisado mediante self-play
     */
    public void entrenarSelfPlay(int numPartidas) {
        System.out.println("=== Entrenamiento Self-Play Gatos (No Supervisado) ===");
        System.out.println("Partidas a jugar: " + numPartidas);
        
        for (int partida = 0; partida < numPartidas; partida++) {
            jugarPartidaEntrenamiento();
            
            // Cada 50 partidas: consolidar y mostrar progreso
            if ((partida + 1) % 50 == 0) {
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
        System.out.println("=== Entrenamiento Supervisado Gatos (Red Experimental) ===");
        System.out.println("Datos de entrenamiento: " + datosEntrenamiento.size());
        System.out.println("Épocas: " + epocas);
        
        for (int epoca = 1; epoca <= epocas; epoca++) {
            double errorTotal = 0.0;
            
            // Mezclar datos para cada época
            List<ParEntrenamiento> datosMezclados = new ArrayList<>(datosEntrenamiento);
            Collections.shuffle(datosMezclados);
            
            for (ParEntrenamiento par : datosMezclados) {
                // Convertir tablero a input
                double[] input = ModeloGatos.tabularToInput(par.tablero, par.turno);
                
                // Crear target: 1.0 para el movimiento óptimo, 0.0 para el resto
                double[] target = new double[64];
                int index = par.movimientoOptimo.getFila() * 8 + par.movimientoOptimo.getColumna();
                target[index] = 1.0;
                
                // Entrenar con este ejemplo
                cerebro.entrenar(input, target, 3);
                
                // Calcular error para monitoreo
                cerebro.resetear();
                double[] output = cerebro.procesar(input);
                double error = 0.0;
                for (int i = 0; i < 64; i++) {
                    error += Math.pow(target[i] - output[i], 2);
                }
                errorTotal += error;
            }
            
            // Consolidar cada 50 épocas
            if (epoca % 50 == 0) {
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
     */
    private void jugarPartidaEntrenamiento() {
        Tablero tablero = FuncionesGato.inicialGato(posMouseInicial);
        int turno = 1; // Empieza el ratón
        
        List<EstadoJuego> historial = new ArrayList<>();
        int maxMovimientos = 100;
        int movimientos = 0;
        
        // Jugar hasta el final
        while (!FuncionesGato.finGato(tablero) && movimientos < maxMovimientos) {
            // Obtener movimiento de la red
            Posicion movimiento = obtenerMovimiento(tablero, turno);
            
            if (movimiento == null) {
                // No hay movimientos válidos
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
        boolean ratonGano = !FuncionesGato.ratonEncerrado(tablero, encontrarRaton(tablero));
        
        // Actualizar estadísticas
        partidasJugadas++;
        if (ratonGano) victoriasRaton++;
        else victoriasGatos++;
        
        // Reforzar movimientos basándose en el resultado
        reforzarMovimientos(historial, ratonGano);
    }
    
    /**
     * Obtiene el mejor movimiento según la red neuronal
     */
    private Posicion obtenerMovimiento(Tablero tablero, int turno) {
        List<Posicion> movimientosPosibles = generarMovimientosLegales(tablero, turno);
        
        if (movimientosPosibles.isEmpty()) {
            return null;
        }
        
        // Convertir tablero a input
        double[] input = ModeloGatos.tabularToInput(tablero, turno);
        
        // Procesar con la red
        cerebro.resetear();
        double[] output = cerebro.procesar(input);
        
        // Encontrar el mejor movimiento válido
        Posicion mejorMovimiento = null;
        double mejorValor = Double.NEGATIVE_INFINITY;
        
        for (Posicion pos : movimientosPosibles) {
            int index = pos.getFila() * 8 + pos.getColumna();
            
            if (output[index] > mejorValor) {
                mejorValor = output[index];
                mejorMovimiento = pos;
            }
        }
        
        // Si no encontró ninguno válido, elegir aleatorio
        if (mejorMovimiento == null) {
            Random rand = new Random();
            mejorMovimiento = movimientosPosibles.get(rand.nextInt(movimientosPosibles.size()));
        }
        
        return mejorMovimiento;
    }
    
    /**
     * Genera movimientos legales para el turno actual
     */
    private List<Posicion> generarMovimientosLegales(Tablero tablero, int turno) {
        List<Posicion> movimientos = new ArrayList<>();
        
        for (int f = 0; f < 8; f++) {
            for (int c = 0; c < 8; c++) {
                int val = tablero.getValor(f, c);
                boolean esMio = false;
                
                if (turno == 1 && val == FuncionesGato.nombreRaton) {
                    esMio = true;
                } else if (turno == 2 && val != 0 && val != FuncionesGato.nombreRaton) {
                    esMio = true; // Es un gato
                }
                
                if (esMio) {
                    // Generar movimientos adyacentes
                    int[][] dirs;
                    if (turno == 1) {
                        // Ratón: 4 direcciones diagonales
                        dirs = new int[][]{{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
                    } else {
                        // Gatos: solo hacia arriba (fila disminuye)
                        dirs = new int[][]{{-1, -1}, {-1, 1}};
                    }
                    
                    for (int[] d : dirs) {
                        int nf = f + d[0];
                        int nc = c + d[1];
                        
                        // Verificar límites y casilla válida
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
    
    /**
     * Aplica un movimiento al tablero
     */
    private Tablero aplicarMovimiento(Tablero tablero, Posicion destino, int turno) {
        // Encontrar la pieza que se mueve (debe estar adyacente al destino)
        Posicion origen = encontrarOrigen(tablero, destino, turno);
        
        if (origen == null) {
            return tablero; // No debería pasar
        }
        
        // Crear nuevo tablero
        int[][] nuevoEstado = copiarTablero(tablero);
        int pieza = tablero.getValor(origen.getFila(), origen.getColumna());
        nuevoEstado[destino.getFila()][destino.getColumna()] = pieza;
        nuevoEstado[origen.getFila()][origen.getColumna()] = 0;
        
        return new Tablero(new SmallMatrix(nuevoEstado));
    }
    
    /**
     * Encuentra la pieza de origen que se mueve al destino
     */
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
    
    /**
     * Refuerza los movimientos basándose en el resultado
     */
    private void reforzarMovimientos(List<EstadoJuego> historial, boolean ratonGano) {
        for (int i = 0; i < historial.size(); i++) {
            EstadoJuego estado = historial.get(i);
            
            // Calcular recompensa para este movimiento
            double recompensa = calcularRecompensa(estado.turno, ratonGano, i, historial.size());
            
            // Crear target: reforzar el movimiento realizado
            double[] target = new double[64];
            int index = estado.movimiento.getFila() * 8 + estado.movimiento.getColumna();
            target[index] = recompensa;
            
            // Entrenar la red con este ejemplo
            double[] input = ModeloGatos.tabularToInput(estado.tablero, estado.turno);
            cerebro.entrenar(input, target, 2);
        }
    }
    
    /**
     * Calcula la recompensa para un movimiento
     */
    private double calcularRecompensa(int turnoJugador, boolean ratonGano, int indiceMov, int totalMovs) {
        boolean gano = (turnoJugador == 1 && ratonGano) || (turnoJugador == 2 && !ratonGano);
        
        if (gano) {
            // Recompensa positiva, mayor para movimientos finales
            double factor = (double) (indiceMov + 1) / totalMovs;
            return 0.5 + 0.5 * factor;  // [0.5, 1.0]
        } else {
            // Penalización
            return 0.0;
        }
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
     * Encuentra la posición del ratón
     */
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
    
    /**
     * Copia un tablero
     */
    private int[][] copiarTablero(Tablero tablero) {
        int[][] copia = new int[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
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
        System.out.printf("Progreso: %d/%d (%.1f%%) - Ratón: %d, Gatos: %d, Engramas: %d\n",
            partidasActuales, partidasTotales, porcentaje,
            victoriasRaton, victoriasGatos,
            cerebro.getEngramas().size());
    }
    
    /**
     * Muestra estadísticas finales
     */
    private void mostrarEstadisticas() {
        System.out.println("Partidas jugadas: " + partidasJugadas);
        System.out.println("Victorias Ratón: " + victoriasRaton + " (" + 
            String.format("%.1f%%", victoriasRaton * 100.0 / partidasJugadas) + ")");
        System.out.println("Victorias Gatos: " + victoriasGatos + " (" + 
            String.format("%.1f%%", victoriasGatos * 100.0 / partidasJugadas) + ")");
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
    public static ModeloGatosExperimental cargar(String ruta) throws IOException, ClassNotFoundException {
        RedNeuralExperimental cerebro = RedNeuralExperimental.cargar(ruta);
        return new ModeloGatosExperimental(cerebro);
    }
    
    // Getters
    public RedNeuralExperimental getCerebro() { return cerebro; }
    public int getPartidasJugadas() { return partidasJugadas; }
    public int getVictoriasRaton() { return victoriasRaton; }
    public int getVictoriasGatos() { return victoriasGatos; }
    
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
