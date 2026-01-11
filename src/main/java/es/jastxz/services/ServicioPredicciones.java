package es.jastxz.services;

import es.jastxz.engine.FuncionesDamas;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;

import es.jastxz.models.Modelo3enRaya;
import es.jastxz.models.ModeloDamas;
import es.jastxz.models.ModeloGatos;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.util.ModelManager;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ServicioPredicciones {

    private static Map<String, NeuralNetwork> modelCache = new HashMap<>();

    /**
     * Predice el siguiente movimiento para 3 en Raya.
     * 
     * @param datosTablero Matriz de enteros representando el tablero.
     * @param turno        Turno del jugador (no siempre necesario, pero útil para
     *                     contexto).
     * @return Movimiento con el nuevo estado y el movimiento realizado.
     */
    public Movimiento predecir3enRaya(int[][] datosTablero, int turno) {
        Modelo3enRaya modelo = new Modelo3enRaya();
        NeuralNetwork nn = getOrLoadModel(modelo.getNombreModelo());
        if (nn == null)
            throw new RuntimeException("No se pudo cargar el modelo de 3 en Raya.");

        Tablero tableroActual = new Tablero(new SmallMatrix(datosTablero));
        Posicion p = ModelManager.getMejorMovimiento(ModelManager.predictIndex(nn,
                Modelo3enRaya.tabularToInput(tableroActual, turno)));

        Tablero nuevoTablero = ModelManager.obtenerTablero(tableroActual, p, turno);
        return new Movimiento(nuevoTablero, p);
    }

    public Movimiento predecirGatos(int[][] datosTablero, int turno) {
        ModeloGatos modelo = new ModeloGatos();
        NeuralNetwork nn = getOrLoadModel(modelo.getNombreModelo());
        if (nn == null)
            throw new RuntimeException("No se pudo cargar el modelo de Gatos.");

        Tablero tableroActual = new Tablero(new SmallMatrix(datosTablero));
        int turnoReal = turno == 2 ? 1 : 2;
        double[] inputs = ModeloGatos.tabularToInput(tableroActual, turnoReal);
        double[] output = nn.feedForward(inputs);

        // Create list of candidates (Indices) sorted by score
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < output.length; i++) {
            candidates.add(i);
        }
        candidates.sort((a, b) -> Double.compare(output[b], output[a]));

        for (int idx : candidates) {
            Posicion candidatePos = ModelManager.getMejorMovimiento(idx, 8);

            // Check if target is empty (Cats rule: Can't land on occupied square)
            if (tableroActual.getValor(candidatePos.getFila(), candidatePos.getColumna()) != 0) {
                continue;
            }

            // Check for source piece (Distance = 1)
            // We need to find a piece belonging to the current turn's mark adjacent to
            // candidatePos
            Posicion source = findSourcePiece(tableroActual, candidatePos, turnoReal);

            if (source != null) {
                // Execute Move
                Tablero nuevoTablero = new Tablero(tableroActual.getMatrix().copia());
                int pieceValue = tableroActual.getValor(source.getFila(), source.getColumna());
                nuevoTablero.setValue(source, 0); // Clear old
                nuevoTablero.setValue(candidatePos, pieceValue); // Set new using actual piece ID

                return new Movimiento(nuevoTablero, candidatePos);
            }
        }

        return null;
    }

    public Movimiento predecirDamas(int[][] datosTablero, int turno) {
        ModeloDamas modelo = new ModeloDamas();
        NeuralNetwork nn = getOrLoadModel(modelo.getNombreModelo());
        if (nn == null)
            throw new RuntimeException("No se pudo cargar el modelo de Damas.");

        Tablero tableroActual = new Tablero(new SmallMatrix(datosTablero));
        double[] input = ModeloDamas.tabularToInput(tableroActual, turno);
        double[] output = nn.feedForward(input);

        // Get legal moves once to validate candidates
        String bandoStr = FuncionesDamas.bandoMarca(turno);
        List<Movimiento> legalMoves = new ArrayList<>();
        try {
            legalMoves = FuncionesDamas.movimientosDamas(tableroActual, bandoStr);
        } catch (Exception e) {
            System.err.println("Warning: FuncionesDamas failed to generate moves. Forfeiting turn.");
            return null;
        }

        // Approach: Score all legal moves. Pick best.
        double bestScore = -Double.MAX_VALUE;
        Movimiento bestMove = null;

        for (Movimiento leg : legalMoves) {
            // Determine Source of this legal move
            Posicion dest = leg.getPos();
            Posicion src = inferSourceFromMove(tableroActual, leg.getTablero(), turno);

            if (src != null) {
                int srcIdx = src.getFila() * 8 + src.getColumna();
                int destIdx = dest.getFila() * 8 + dest.getColumna();

                // Score = SourceProb + DestScore (or SourceProb * DestProb)
                // output has 128 elements.
                // Source Prob at srcIdx. Dest Prob at 64 + destIdx.
                double score = output[srcIdx] + output[64 + destIdx];

                if (score > bestScore) {
                    bestScore = score;
                    bestMove = leg;
                }
            }
        }

        if (bestMove != null) {
            return bestMove;
        } else {
            System.err.println("Warning: No legal moves found. Forfeiting turn.");
            return null;
        }
    }

    // --- HELPERS ---

    private Posicion findSourcePiece(Tablero tablero, Posicion dest, int marca) {
        // Search 3x3 area around dest for a piece of 'marca'
        int startRow = Math.max(0, dest.getFila() - 1);
        int endRow = Math.min(7, dest.getFila() + 1);
        int startCol = Math.max(0, dest.getColumna() - 1);
        int endCol = Math.min(7, dest.getColumna() + 1);

        for (int r = startRow; r <= endRow; r++) {
            for (int c = startCol; c <= endCol; c++) {
                if (r == dest.getFila() && c == dest.getColumna())
                    continue;

                int val = tablero.getValor(r, c);
                // Check if value matches mark.

                boolean isMatch = (val == marca);
                if (!isMatch && marca != 9 && val > 0 && val % 2 != 0 && val != 9) {
                    isMatch = true;
                }

                if (isMatch) {
                    return new Posicion(r, c);
                }
            }
        }
        return null;
    }

    private Posicion inferSourceFromMove(Tablero antes, Tablero despues, int turno) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int vIni = antes.getValor(i, j);
                int vFin = despues.getValor(i, j);
                if (vIni != 0 && vFin == 0) {
                    // Check if piece belonged to me
                    boolean esBlancas = (turno == 1);
                    boolean isWhitePiece = ModeloDamas.contiene(FuncionesDamas.nombresBlancas, vIni)
                            || ModeloDamas.contiene(FuncionesDamas.nombresReinasBlancas, vIni);
                    if ((esBlancas && isWhitePiece) || (!esBlancas && !isWhitePiece)) { // (black & black)
                        return new Posicion(i, j);
                    }
                }
            }
        }
        return null;
    }

    private NeuralNetwork getOrLoadModel(String modelName) {
        if (modelCache.containsKey(modelName)) {
            return modelCache.get(modelName);
        }
        NeuralNetwork nn = ModelManager.loadModel(modelName);
        if (nn != null) {
            modelCache.put(modelName, nn);
        }
        return nn;
    }

}
