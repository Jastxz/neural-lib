package org.javig;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.javig.engine.FuncionesGato;
import org.javig.engine.Minimax;
import org.javig.models.ModeloGatos;
import org.javig.nn.NeuralNetwork;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.ModelManager;
import org.javig.util.ResultadoSimulacion;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class ModeloGatosTest {
    ModeloGatos modeloBasico;
    ModeloGatos modeloNormal;
    ModeloGatos modeloAvanzado;
    ModeloGatos modeloExperto;

    @BeforeEach
    void setUp() {
        modeloBasico = new ModeloGatos();
        modeloBasico.setProfundidad(modeloBasico.getProfundidadBasico());

        modeloNormal = new ModeloGatos(modeloBasico.getPosiblesEstados());
        modeloNormal.setProfundidad(modeloNormal.getProfundidadNormal());

        modeloAvanzado = new ModeloGatos(modeloBasico.getPosiblesEstados());
        modeloAvanzado.setProfundidad(modeloAvanzado.getProfundidadAvanzado());

        modeloExperto = new ModeloGatos(modeloBasico.getPosiblesEstados());
        modeloExperto.setProfundidad(modeloExperto.getProfundidadExperto());
    }

    @Test
    public void testEntrenar() {
        // Validation that training runs without error

        long start = System.currentTimeMillis();
        // Basico
        modeloBasico.entrenar(modeloBasico.getNombreModeloBasico());
        System.out.println("Entrenamiento finalizado para " + modeloBasico.getNombreModeloBasico() + "\n");

        // Normal
        modeloNormal.entrenar(modeloNormal.getNombreModeloNormal());
        System.out.println("Entrenamiento finalizado para " + modeloNormal.getNombreModeloNormal() + "\n");

        // Avanzado
        modeloAvanzado.entrenar(modeloAvanzado.getNombreModeloAvanzado());
        System.out.println("Entrenamiento finalizado para " + modeloAvanzado.getNombreModeloAvanzado() + "\n");

        // Experto
        modeloExperto.entrenar(modeloExperto.getNombreModeloExperto());
        System.out.println("Entrenamiento finalizado para " + modeloExperto.getNombreModeloExperto() + "\n");
        long end = System.currentTimeMillis();
        System.out.println("Tiempo total de entrenamiento: " + (end - start) / 3600000.0 + " horas");
    }

    @Test
    public void testModeloVsMinimax() {
        NeuralNetwork loadedModelBasico = ModelManager.loadModel(modeloBasico.getNombreModeloBasico());
        NeuralNetwork loadedModelNormal = ModelManager.loadModel(modeloNormal.getNombreModeloNormal());
        NeuralNetwork loadedModelAvanzado = ModelManager.loadModel(modeloAvanzado.getNombreModeloAvanzado());
        NeuralNetwork loadedModelExperto = ModelManager.loadModel(modeloExperto.getNombreModeloExperto());

        int numPartidas = 100;

        // Run simulations for each model sequentially, but parallelize games within
        // each model
        simulaModeloVsMinimax("Básico", loadedModelBasico, modeloBasico,
                numPartidas);

        simulaModeloVsMinimax("Normal", loadedModelNormal, modeloNormal,
                numPartidas);
        simulaModeloVsMinimax("Avanzado", loadedModelAvanzado, modeloAvanzado,
                numPartidas);

        simulaModeloVsMinimax("Experto", loadedModelExperto, modeloExperto,
                numPartidas);
    }

    private ResultadoSimulacion simulaModeloVsMinimax(String nombreModelo, NeuralNetwork loadedModel,
            ModeloGatos modelo, int numPartidas) {
        // Use parallelStream to parallelize games for this model
        long start = System.currentTimeMillis();
        List<String> resultados = IntStream.range(0, numPartidas)
                .parallel()
                .mapToObj(i -> {
                    Mundo mundo = new Mundo(modelo.getMundo());
                    if (i % 2 == 0) {
                        mundo.setMarca(FuncionesGato.marcaMaquinaGato(mundo.getMarca()));
                        mundo.setTurno(2);
                    }
                    String ladoModelo = i % 2 == 0 ? "Gatos" : "Ratón";
                    String ladoMinimax = i % 2 == 0 ? "Ratón" : "Gatos";
                    mundo = simulaPartida(mundo, loadedModel, modelo);
                    String ganador = FuncionesGato.ratonEncerrado(mundo.getMovimiento().getTablero(),
                            mundo.getMovimiento().getPos()) ? "Gatos" : "Ratón";

                    System.out.println("[" + nombreModelo + "] Partida " + i + " terminada - Modelo: " +
                            ladoModelo + ", Ganador: " + ganador);

                    // Return result as "modelo" or "minimax"
                    if (ganador.equals(ladoModelo)) {
                        return "modelo";
                    } else if (ganador.equals(ladoMinimax)) {
                        return "minimax";
                    }
                    return "empate";
                })
                .toList();
        long end = System.currentTimeMillis();
        System.out.println("Tiempo total de simulación: " + (end - start) / 3600000.0 + " horas");
        // Count results
        int ganaModelo = (int) resultados.stream().filter(r -> r.equals("modelo")).count();
        int ganaMinimax = (int) resultados.stream().filter(r -> r.equals("minimax")).count();
        System.out.println("\n=== Resultados Modelo " + nombreModelo + " ===");
        System.out.println("Veces que gana el modelo: " + ganaModelo);
        System.out.println("Veces que gana Minimax: " + ganaMinimax);
        System.out.println("Porcentaje de victoria del modelo: " +
                String.format("%.2f%%", ganaModelo / (double) numPartidas * 100.0));

        return new ResultadoSimulacion(nombreModelo, ganaModelo, ganaMinimax);
    }

    private Mundo simulaPartida(Mundo mundo, NeuralNetwork loadedModel, ModeloGatos modelo) {
        int moves = 0;
        int maxMoves = 100; // Prevent infinite games
        while (moves < maxMoves) {
            moves++;
            // 1. Minimax Turn
            Tablero movimientoMinimax = Minimax.negamax(mundo);
            if (FuncionesGato.finGato(movimientoMinimax)) {
                mundo.getMovimiento().setTablero(movimientoMinimax); // Ensure final state is set
                break;
            }
            mundo.getMovimiento().setTablero(movimientoMinimax);
            mundo.setProfundidad(8);

            // 2. Model Turn
            // Get model predictions
            int turno = mundo.getTurno() == 2 ? 1 : 2;
            double[] inputs = modelo.tabularToInput(movimientoMinimax, turno);
            double[] output = loadedModel.feedForward(inputs);

            // Create list of candidates (Indices) sorted by score
            List<Integer> candidates = new ArrayList<>();
            for (int i = 0; i < output.length; i++) {
                candidates.add(i);
            }
            candidates.sort((a, b) -> Double.compare(output[b], output[a]));

            boolean moveMade = false;
            for (int idx : candidates) {
                Posicion candidatePos = ModelManager.getMejorMovimiento(idx, 8);

                // Check if target is empty (Cats rule: Can't land on occupied square)
                if (movimientoMinimax.getValor(candidatePos.getFila(), candidatePos.getColumna()) != 0) {
                    continue;
                }

                // Check for source piece (Distance = 1)
                // We need to find a piece belonging to the current turn's mark adjacent to
                // candidatePos
                Posicion source = findSourcePiece(movimientoMinimax, candidatePos, mundo.getMarca());

                if (source != null) {
                    // Execute Move
                    Tablero nuevoTablero = new Tablero(movimientoMinimax.getMatrix());
                    int pieceValue = movimientoMinimax.getValor(source.getFila(), source.getColumna());
                    nuevoTablero.setValue(source, 0); // Clear old
                    nuevoTablero.setValue(candidatePos, pieceValue); // Set new using actual piece ID

                    mundo.setMovimiento(new Movimiento(nuevoTablero, candidatePos));
                    moveMade = true;
                    break;
                }
            }

            if (!moveMade) {
                System.out.println("Model could not find a valid move. Forfeiting.");
                break;
            }

            if (FuncionesGato.finGato(mundo.getMovimiento().getTablero())) {
                break;
            }
        }
        if (moves >= maxMoves) {
            System.out.println("Partida terminada forzosamente por límite de movimientos (" + maxMoves + ")");
            System.out.println("Tablero final: \n" + mundo.getMovimiento().getTablero());
        }
        return mundo;
    }

    @Test
    public void testMovimientoGatosNoSeBloquea() {
        // Test that the model can successfully make a move as cats without getting
        // stuck
        // Board state:
        // [ 0, -1, 0, -1, 0, -1, 0, -1 ]
        // [ -1, 0, -1, 0, -1, 0, -1, 0 ]
        // [ 9, -1, 0, -1, 0, -1, 0, -1 ]
        // [ -1, 5, -1, 0, -1, 0, -1, 0 ]
        // [ 0, -1, 0, -1, 0, -1, 0, -1 ]
        // [ -1, 0, -1, 0, -1, 0, -1, 0 ]
        // [ 0, -1, 0, -1, 0, -1, 0, -1 ]
        // [ -1, 1, -1, 3, -1, 0, -1, 7 ]

        // Create the board state (-1 represents invalid squares in the checkerboard
        // pattern)
        int[][] boardMatrix = {
                { 0, -1, 0, -1, 0, -1, 0, -1 },
                { -1, 0, -1, 0, -1, 0, -1, 0 },
                { 9, -1, 0, -1, 0, -1, 0, -1 },
                { -1, 5, -1, 0, -1, 0, -1, 0 },
                { 0, -1, 0, -1, 0, -1, 0, -1 },
                { -1, 0, -1, 0, -1, 0, -1, 0 },
                { 0, -1, 0, -1, 0, -1, 0, -1 },
                { -1, 1, -1, 3, -1, 0, -1, 7 }
        };

        Tablero tablero = new Tablero(new SmallMatrix(boardMatrix));

        // Load the trained model
        ModeloGatos modelo = new ModeloGatos();
        NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModeloNormal());

        // Get model predictions for cats' move
        double[] inputs = modelo.tabularToInput(tablero, 2);
        double[] output = loadedModel.feedForward(inputs);

        // Create list of candidates sorted by score
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < output.length; i++) {
            candidates.add(i);
        }
        candidates.sort((a, b) -> Double.compare(output[b], output[a]));

        // Try to find a valid move
        boolean moveMade = false;
        Posicion selectedMove = null;
        for (int idx : candidates) {
            Posicion candidatePos = ModelManager.getMejorMovimiento(idx, 8);

            // Check if target is empty
            if (tablero.getValor(candidatePos.getFila(), candidatePos.getColumna()) != 0) {
                continue;
            }

            // Check for source piece (any cat piece adjacent to candidatePos)
            Posicion source = findSourcePiece(tablero, candidatePos, FuncionesGato.nombresGatos[0]);

            if (source != null) {
                selectedMove = candidatePos;
                moveMade = true;
                System.out.println("Modelo encontró movimiento válido: " + source + " -> " + candidatePos);
                break;
            }
        }

        // Assert that a move was successfully found
        if (!moveMade) {
            System.err.println("ERROR: El modelo no pudo encontrar un movimiento válido jugando como gatos");
            System.err.println("Estado del tablero:");
            System.err.println(tablero);
        }

        // The test passes if a valid move was found
        assert moveMade : "El modelo debería ser capaz de realizar un movimiento válido como gatos sin bloquearse";
        assert selectedMove != null : "Debería haberse seleccionado una posición de destino válida";
    }

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

}
