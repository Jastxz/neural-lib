package org.javig.util;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.javig.engine.FuncionesDamas;
import org.javig.engine.Minimax;
import org.javig.models.ModeloDamas;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.Tablero;

public class GeneradorDatosDamas {

    private static final Posicion posInicial = new Posicion(4, 2);
    private static final int PIECES_ENDGAME_THRESHOLD = 7;
    private static final int MAX_MOVES_SIMULATION = 100;

    private static GeneradorDatosDamas instance;

    private GeneradorDatosDamas() {
    }

    public static GeneradorDatosDamas getInstance() {
        if (instance == null) {
            instance = new GeneradorDatosDamas();
        }
        return instance;
    }

    /**
     * Genera aperturas explorando todos los movimientos hasta cierta profundidad.
     * 
     * @param profundidadMoves profundidad de movimientos reales a explorar
     */
    public List<TrainingPair> generarAperturas(int profundidadMoves) {
        System.out.println("Generando aperturas (profundidad " + profundidadMoves + " de movimientos reales)...");
        List<TrainingPair> pairs = new ArrayList<>();
        Set<String> visited = new HashSet<>();

        // Estado inicial
        Tablero inicial = FuncionesDamas.inicialDamas();
        // Cola para BFS
        List<TableroState> frontier = new ArrayList<>();
        frontier.add(new TableroState(inicial, 1, 0, posInicial)); // Turno 1 (blancas/rojas), move 0

        long startTime = System.currentTimeMillis();
        while (!frontier.isEmpty()) {
            TableroState current = frontier.remove(0);

            if (current.depth >= profundidadMoves)
                continue;

            // Generar mejor movimiento con Minimax para aprender "buenos" movimientos

            TrainingPair pair = procesarEstado(current.tablero, current.turno, current.posicion);
            if (pair != null) {
                pairs.add(pair);
            }

            // Expandir hijos para seguir explorando el árbol de aperturas
            List<Movimiento> hijos = FuncionesDamas.movimientosDamas(current.tablero,
                    FuncionesDamas.bandoMarca(current.turno));
            for (Movimiento mov : hijos) {
                int nextTurn = (current.turno == 1) ? 2 : 1;
                String hash = mov.getTablero().toString() + nextTurn; // Simplificado

                if (!visited.contains(hash)) {
                    visited.add(hash);
                    frontier.add(new TableroState(mov.getTablero(), nextTurn, current.depth + 1, mov.getPos()));
                }
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Entradas generadas: " + pairs.size());
        System.out.println("Tiempo total: " + (endTime - startTime) / 60000 + " minutos, "
                + (endTime - startTime) / 3600000 + " horas");

        return pairs;
    }

    // Wrapper para estado en BFS
    private static class TableroState {
        Tablero tablero;
        int turno;
        int depth;
        Posicion posicion;

        public TableroState(Tablero t, int tu, int d, Posicion p) {
            tablero = t;
            turno = tu;
            depth = d;
            posicion = p;
        }
    }

    /**
     * Genera medio juego mediante simulaciones aleatorias (Monte Carlo) hasta
     * llegar a estados intermedios.
     * 
     * @param numPartidas Número de partidas a simular
     */
    public List<TrainingPair> generarMedioJuego(int numPartidas) {
        System.out.println("Generando medio juego (" + numPartidas + " partidas)...");
        // Usamos una lista sincronizada o collect al final
        return IntStream.range(0, numPartidas).parallel()
                .mapToObj(i -> {
                    long startTime = System.currentTimeMillis();
                    List<TrainingPair> pairs = simularPartidaMedioJuego();
                    long endTime = System.currentTimeMillis();
                    System.out.println("Partida medio juego generada en " + (endTime - startTime) / 1000 + " segundos");
                    System.out.println("Medio juego generadas: " + pairs.size());
                    return pairs;
                })
                .flatMap(List::stream)
                .collect(Collectors.toList());
    }

    private List<TrainingPair> simularPartidaMedioJuego() {
        List<TrainingPair> pairs = new ArrayList<>();
        Tablero tablero = FuncionesDamas.inicialDamas();
        int turno = 1;
        int moves = 0;
        Posicion posicion = posInicial;

        // Simular movimientos semi-aleatorios para llegar a medio juego
        // O usar un modelo básico/minimax de baja profundidad para avanzar rápido
        while (!FuncionesDamas.finDamas(tablero) && moves < MAX_MOVES_SIMULATION) {
            // Criterio de medio juego: Ni apertura (<10 moves) ni final (pocas piezas)
            int piezas = contarPiezas(tablero);
            boolean esMedioJuego = moves > 10 && piezas > PIECES_ENDGAME_THRESHOLD;

            if (esMedioJuego) {
                // Si estamos en medio juego, analizamos con ALTA calidad para entrenar
                TrainingPair pair = procesarEstado(tablero, turno, posicion);
                if (pair != null)
                    pairs.add(pair);
            }

            // Avanzar estado (usando algo rápido, random o profundidad baja)
            // Aquí usamos random para diversidad, o profundidad 1
            List<Movimiento> invalidMovies = FuncionesDamas.movimientosDamas(tablero,
                    FuncionesDamas.bandoMarca(turno));
            if (invalidMovies.isEmpty())
                break;

            Movimiento nextMove = invalidMovies.get(ThreadLocalRandom.current().nextInt(invalidMovies.size()));
            tablero = nextMove.getTablero();
            turno = (turno == 1) ? 2 : 1;
            posicion = nextMove.getPos();
            moves++;
        }
        return pairs;
    }

    /**
     * Genera finales explorando exhaustivamente desde semillas aleatorias con 7
     * piezas.
     */
    public List<TrainingPair> generarFinales(int samples) {
        System.out.println("Generando finales (BFS desde " + samples + " semillas de 4 piezas)...");
        Set<String> visited = new HashSet<>(); // Global visited set to avoid duplicates across seeds
        List<TrainingPair> allPairs = new ArrayList<>();
        int maxPairs = 100000;

        long startTime = System.currentTimeMillis();

        // Sequential seeds with internal BFS is safer for memory and logic.
        for (int i = 0; i < samples; i++) {
            // Generar semilla con EXACTAMENTE 4 piezas
            RandomBoardState seed = generarTableroAleatorio(4);

            // Cola BFS local
            List<TableroState> frontier = new ArrayList<>();
            frontier.add(new TableroState(seed.tablero, seed.turno, 0, seed.posicion));

            int maxMoves = 50;
            int numPieces = 4;
            int numPiecesBegin = 4;

            long startBoardTime = System.currentTimeMillis();
            while (!frontier.isEmpty()) {
                TableroState current = frontier.remove(0);

                // Check Global Visited
                String hash = current.tablero.toString() + current.turno;
                if (visited.contains(hash))
                    continue;
                visited.add(hash);

                // Process State (Generate Target)
                TrainingPair pair = procesarEstado(current.tablero, current.turno, current.posicion);
                if (pair != null) {
                    allPairs.add(pair);
                } else {
                    continue;
                }

                // Expand Children
                List<Movimiento> hijos = FuncionesDamas.movimientosDamas(current.tablero,
                        FuncionesDamas.bandoMarca(current.turno));
                for (Movimiento mov : hijos) {
                    int nextTurn = (current.turno == 1) ? 2 : 1;
                    // Optimization: Don't add to frontier if already visited
                    String childHash = mov.getTablero().toString() + nextTurn;
                    if (!visited.contains(childHash)) {
                        frontier.add(new TableroState(mov.getTablero(), nextTurn, current.depth + 1, mov.getPos()));
                    }
                }

                // Limitamos el número de movimientos posibles por tablero final y
                // actualizamos el número de piezas restantes y el de movimientos si
                // se han eliminado piezas.
                maxMoves--;
                if (numPiecesBegin > numPieces) {
                    numPiecesBegin--;
                    maxMoves += 10;
                }

                // Safety break for massive trees
                if (maxMoves <= 0)
                    break; // Hard limit per batch
            }
            long endBoardTime = System.currentTimeMillis();
            System.out.println("Tiempo de para generar los finales del tablero: "
                    + (endBoardTime - startBoardTime) / 1000 + " segundos");

            if (allPairs.size() > maxPairs) {
                System.out.println("Max pairs limit reached.");
                break;
            }
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Finales generados: " + allPairs.size());
        System.out.println("Tiempo total: " + (endTime - startTime) / 60000 + " minutos");

        return allPairs;
    }

    private RandomBoardState generarTableroAleatorio(int totalPiezas) {
        Tablero vacio = tableroDamasVacio();

        // Nombres de piezas
        List<Integer> blancas = arrayAlist(FuncionesDamas.nombresBlancas);
        List<Integer> negras = arrayAlist(FuncionesDamas.nombresNegras);

        // Distribute pieces randomly between White and Black
        int numBlancas = ThreadLocalRandom.current().nextInt(1, totalPiezas); // 1 to total-1
        int numNegras = totalPiezas - numBlancas;

        // Seleccionar piezas aleatorias
        List<Integer> piezasAColocar = new ArrayList<>();
        // Shuffle manual o simple selection
        for (int i = 0; i < numBlancas; i++)
            piezasAColocar.add(blancas.remove(ThreadLocalRandom.current().nextInt(blancas.size())));
        for (int i = 0; i < numNegras; i++)
            piezasAColocar.add(negras.remove(ThreadLocalRandom.current().nextInt(negras.size())));

        // Obtener casillas válidas
        List<Posicion> validas = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                if (ModeloDamas.esCasillaBlanca(i, j)) {
                    validas.add(new Posicion(i, j));
                }
            }
        }

        List<Posicion> posiciones = new ArrayList<>();
        // Colocar piezas
        for (Integer pieza : piezasAColocar) {
            if (validas.isEmpty())
                break;
            int idx = ThreadLocalRandom.current().nextInt(validas.size());
            Posicion p = validas.remove(idx);
            posiciones.add(p);
            vacio.setValue(p.getFila(), p.getColumna(), pieza);
        }

        // Turno aleatorio
        int turno = ThreadLocalRandom.current().nextBoolean() ? 1 : 2;
        // Posicion dummy para iniciar (necesaria para Mundo, aunque no sepamos cual
        // movió anterior)
        // En endgames random, el "último movimiento" no existe, cogemos una pieza
        // válida cualquiera.
        Posicion posPieza = posiciones.isEmpty() ? posInicial
                : posiciones.get(ThreadLocalRandom.current().nextInt(posiciones.size()));

        return new RandomBoardState(vacio, turno, posPieza);
    }

    // Wrapper class to pass data
    private static class RandomBoardState {
        final Tablero tablero;
        final int turno;
        final Posicion posicion;

        RandomBoardState(Tablero t, int tu, Posicion p) {
            tablero = t;
            turno = tu;
            posicion = p;
        }
    }

    /**
     * Ejecuta Minimax (Negamax) con alta profundidad para encontrar el mejor
     * movimiento desde el estado actual.
     */
    private TrainingPair procesarEstado(Tablero estado, int turno, Posicion posicion) {
        // Configuramos un Mundo para Minimax
        int turnoMinimax = (turno == 1) ? 2 : 1;
        Mundo m = new Mundo(new Movimiento(estado, posicion), Util.juegoDamas, 4, 5, turnoMinimax,
                turnoMinimax, true);

        Tablero bestState = Minimax.negamax(m);
        if (bestState == null) {
            System.out.println("No se encontró movimiento válido");
            return null;
        } else if (estado.equals(bestState)) {
            System.out.println("Uno de los lados ya no tiene movimientos válidos");
            return null;
        }

        // Inferir el movimiento (Origen -> Destino)
        Posicion dest = ModeloDamas.obtenerMovimiento(estado, bestState, turno);
        Posicion source = inferirFuente(estado, bestState, turno);

        if (dest != null && source != null) {
            double[] input = ModeloDamas.tabularToInput(estado, turno);
            double[] target = new double[128];

            // Output encoding:
            // 0-63: Source Probability
            // 64-127: Destination Probability
            target[source.getFila() * 8 + source.getColumna()] = 1.0;
            target[64 + dest.getFila() * 8 + dest.getColumna()] = 1.0;

            return new TrainingPair(input, target);
        }
        System.out.println("No se pudo inferir origen/destino correctamente");
        return null;
    }

    /**
     * Deduce la pieza de origen comparando tableros.
     * La casilla origen es aquella que tenía una pieza del bando 'turno'
     * y ahora está VACÍA.
     */
    private Posicion inferirFuente(Tablero inicial, Tablero fin, int turno) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int vIni = inicial.getValor(i, j);
                int vFin = fin.getValor(i, j);

                // Buscamos casilla que tenía pieza propia y ahora es 0
                if (vIni != 0 && vFin == 0) {
                    if (perteneceAlBando(vIni, turno)) {
                        // Esta es candidata a ser la fuente.
                        return new Posicion(i, j);
                    }
                }
            }
        }
        return null;
    }

    private boolean perteneceAlBando(int pieza, int turno) {
        boolean esBlancas = (turno == 1); // Asumiendo turno 1=Blancas
        boolean piezaBlanca = ModeloDamas.contiene(FuncionesDamas.nombresBlancas, pieza)
                || ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, pieza);
        if (esBlancas && piezaBlanca)
            return true;

        boolean piezaNegra = ModeloDamas.contiene(FuncionesDamas.nombresNegras, pieza)
                || ModeloDamas.contiene(FuncionesDamas.nombresReinasNegras, pieza);
        if (!esBlancas && piezaNegra)
            return true;

        return false;
    }

    private int contarPiezas(Tablero t) {
        int c = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                if (t.getValor(i, j) != 0)
                    c++;
        return c;
    }

    private Tablero tableroDamasVacio() {
        Tablero t = new Tablero(8, 8);
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                if (!ModeloDamas.esCasillaBlanca(i, j)) {
                    t.setValue(i, j, -1);
                }
        return t;
    }

    private List<Integer> arrayAlist(int[] array) {
        List<Integer> piezas = new ArrayList<>(array.length);
        for (int i = 0; i < array.length; i++)
            piezas.add(array[i]);
        return piezas;
    }

    public static void guardarEnFichero(List<TrainingPair> datos, String fichero) {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(fichero))) {
            dos.writeInt(datos.size());
            for (TrainingPair pair : datos) {
                for (double d : pair.getInput())
                    dos.writeDouble(d);
                for (double d : pair.getOutput())
                    dos.writeDouble(d);
            }
            System.out.println("Guardado " + fichero);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
