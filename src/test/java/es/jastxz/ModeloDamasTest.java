package es.jastxz;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.javig.engine.FuncionesDamas;
import org.javig.engine.Minimax;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.junit.jupiter.api.Test;

import es.jastxz.models.ModeloDamas;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.util.GeneradorDatosDamas;
import es.jastxz.util.ModelManager;
import es.jastxz.util.TrainingPair;

public class ModeloDamasTest {

    @Test
    public void testGeneraciónDatosApertura() {
        GeneradorDatosDamas generador = GeneradorDatosDamas.getInstance();
        List<TrainingPair> datosApertura = generador.generarAperturas(7);
        GeneradorDatosDamas.guardarEnFichero(datosApertura,
                "src/main/resources/datosDamas/aperturas.dat");
        System.out.println("Aperturas generadas\n\n");
    }

    @Test
    public void testGeneraciónDatosMedioJuego() {
        GeneradorDatosDamas generador = GeneradorDatosDamas.getInstance();
        long startTime = System.currentTimeMillis();
        List<TrainingPair> datosMedioJuego = generador.generarMedioJuego(1000);
        GeneradorDatosDamas.guardarEnFichero(datosMedioJuego,
                "src/main/resources/datosDamas/medioJuego.dat");
        long endTime = System.currentTimeMillis();
        System.out.println("Entradas medio juego generadas: " + datosMedioJuego.size());
        System.out.println("Medio juego generado en " + (endTime - startTime) / 60000 + " minutos, "
                + (endTime - startTime) / 3600000 + " horas\n\n");
    }

    @Test
    public void testGeneraciónDatosFinales() {
        GeneradorDatosDamas generador = GeneradorDatosDamas.getInstance();
        long startTime = System.currentTimeMillis();
        List<TrainingPair> datosFinal = generador.generarFinales(1000000);
        GeneradorDatosDamas.guardarEnFichero(datosFinal, "src/main/resources/datosDamas/finales.dat");
        long endTime = System.currentTimeMillis();
        System.out.println("Entradas finales generadas: " + datosFinal.size());
        System.out.println("Finales generados en " + (endTime - startTime) / 60000 + " minutos, "
                + (endTime - startTime) / 3600000 + " horas\n\n");
    }

    @Test
    public void testEntrenamiento() {
        ModeloDamas modelo = new ModeloDamas();
        modelo.entrenar();
    }

    @Test
    public void testModeloVsMinimax() {
        ModeloDamas modelo = new ModeloDamas();
        NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModelo());

        int numPartidas = 100;

        // Use parallelStream to parallelize games for this model
        long start = System.currentTimeMillis();
        List<String> resultados = IntStream.range(0, numPartidas)
                .parallel()
                .mapToObj(i -> {
                    try {
                        // Deep copy Mundo to ensure thread safety
                        Mundo original = modelo.getMundo();
                        // Assuming Mundo copy constructor is shallow or unknown, we rebuild it
                        // partially
                        // But Movimiento -> Tablero needs deep copy
                        Tablero tCopy = new Tablero(original.getMovimiento().getTablero().getMatrix());
                        Movimiento movCopy = new Movimiento(tCopy, original.getMovimiento().getPos());

                        Mundo mundo = new Mundo(movCopy, original.getJuego(), original.getDificultad(),
                                original.getProfundidad(), original.getMarca(),
                                original.getTurno(), true);

                        // Override for test setup
                        if (i % 2 == 0) {
                            mundo.setMarca(FuncionesDamas.marcaMaquinaDamas(mundo.getMarca()));
                            mundo.setTurno(2);
                        }
                        String ladoModelo = i % 2 == 0 ? "N" : "B";
                        String ladoMinimax = i % 2 == 0 ? "B" : "N";
                        mundo = simulaPartida(mundo, loadedModel, modelo);
                        Tablero tablero = mundo.getMovimiento().getTablero();

                        if (tablero == null) {
                            System.err.println("FATAL: Tablero is null in game " + i);
                            return "error";
                        }

                        List<Integer>[] piezasVivas = piezasVivas(tablero);
                        String ganador = FuncionesDamas.bandoGanador(tablero, piezasVivas);

                        System.out.println("Partida " + i + " terminada - Modelo: " +
                                ladoModelo + ", Ganador: " + ganador);

                        // Return result as "modelo" or "minimax"
                        if (ganador.equals(ladoModelo)) {
                            return "modelo";
                        } else if (ganador.equals(ladoMinimax)) {
                            return "minimax";
                        }
                        return "empate";
                    } catch (Exception e) {
                        System.err.println("Exception in game " + i);
                        e.printStackTrace();
                        return "error";
                    }
                })
                .toList();
        long end = System.currentTimeMillis();
        System.out.println("Tiempo total de simulación: " + (end - start) / 3600000.0 + " horas");
        // Count results
        int ganaModelo = (int) resultados.stream().filter(r -> r.equals("modelo")).count();
        int ganaMinimax = (int) resultados.stream().filter(r -> r.equals("minimax")).count();
        int empate = (int) resultados.stream().filter(r -> r.equals("empate")).count();
        System.out.println("\n=== Resultados Modelo " + " ===");
        System.out.println("Veces que gana el modelo: " + ganaModelo);
        System.out.println("Veces que gana Minimax: " + ganaMinimax);
        System.out.println("Veces que empata: " + empate);
        System.out.println("Porcentaje de victoria del modelo: " +
                String.format("%.2f%%", ganaModelo / (double) numPartidas * 100.0));
        System.out.println("Porcentaje de empate: " +
                String.format("%.2f%%", empate / (double) numPartidas * 100.0));
    }

    private Mundo simulaPartida(Mundo mundo, NeuralNetwork loadedModel, ModeloDamas modelo) {
        int moves = 0;
        int maxMoves = 100; // Prevent infinite games

        // Determinar quién empieza basado en la configuración inicial del mundo
        // Si el mundo dice turno 1, empiezan blancas.
        // Asumimos bucle: Minimax juega -> Modelo juega.
        // Pero esto depende de quién sea Minimax.
        // En testModeloVsMinimax:
        // Case 1 (Pares): i%2==0. mundo.setTurno(2). Minimax es blancas? NO.
        // Dice: ladoModelo="N", ladoMinimax="B". Minimax=Blancas.
        // Si turno=2 (Negras), y Minimax=Blancas (1), entonces NO LE TOCA A MINIMAX.
        // El bucle actual FUERZA a Minimax a mover primero.

        // Corrección: El bucle debe respetar el turno.
        // El Mundo pasado ya tiene configurado el 'turno' inicial y la 'marca'
        // (identidad del Modelo).

        int turnoActual = mundo.getTurno();
        int marcaModelo = mundo.getMarca(); // Identidad del Modelo
        int marcaMinimax = (marcaModelo == 1) ? 2 : 1; // Identidad de Minimax

        while (moves < maxMoves) {
            moves++;

            if (FuncionesDamas.finDamas(mundo.getMovimiento().getTablero())) {
                break;
            }

            if (turnoActual == marcaMinimax) {
                // --- Turno Minimax ---
                // Para Negamax, 'marca' en mundo suele ser "quien maximiza".
                // Si llamamos a negamax(mundo), Minimax iterará asumiendo que
                // 'mundo.getMarca()' es el jugador MAX.
                // Queremos que Minimax maximice para SÍ MISMO.
                mundo.setMarca(marcaMinimax);
                mundo.setTurno(turnoActual);

                Tablero movimientoMinimax = Minimax.negamax(mundo);
                if (movimientoMinimax == null)
                    break; // No moves

                mundo.getMovimiento().setTablero(movimientoMinimax);

                // Cambio de turno
                turnoActual = (turnoActual == 1) ? 2 : 1;

            } else {
                // --- Turno Modelo ---
                // Generar input para el modelo
                Tablero tableroActual = mundo.getMovimiento().getTablero();
                double[] inputs = ModeloDamas.tabularToInput(tableroActual, turnoActual);
                double[] output = loadedModel.feedForward(inputs);

                // Parse 128 outputs: 0-63 Source, 64-127 Dest
                // Logic updated to iterate legal moves directly, so these are currently unused.
                // List<Integer> sortedSources = getSortedIndices(output, 0, 64);
                // List<Integer> sortedDests = getSortedIndices(output, 64, 128);

                boolean moveMade = false;

                // Get legal moves once to validate candidates
                String bandoStr = FuncionesDamas.bandoMarca(turnoActual);
                // Wrap in try-catch due to library instability
                List<Movimiento> legalMoves = new ArrayList<>();
                try {
                    legalMoves = FuncionesDamas.movimientosDamas(tableroActual, bandoStr);
                } catch (Exception e) {
                    System.err.println("Warning: FuncionesDamas failed to generate moves. Forfeiting turn.");
                    break;
                }

                // Optimization: Iterate through best pairs (heuristically top 5x5 or similar)
                // Or just iterate through legal moves and pick the one with highest Combined
                // Score.
                // Approach: Score all legal moves. Pick best.

                double bestScore = -Double.MAX_VALUE;
                Movimiento bestMove = null;

                for (Movimiento leg : legalMoves) {
                    // Determine Source of this legal move
                    Posicion dest = leg.getPos();
                    Posicion src = inferSourceFromMove(tableroActual, leg.getTablero(), turnoActual);

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
                    mundo.setMovimiento(new Movimiento(bestMove.getTablero(), bestMove.getPos()));
                    moveMade = true;
                }

                if (!moveMade) {
                    System.out.println("Model could not find a valid move. Forfeiting.");
                    break;
                }

                // Cambio de turno
                turnoActual = (turnoActual == 1) ? 2 : 1;
            }
        }

        if (moves >= maxMoves) {
            System.out.println("Partida terminada forzosamente por límite de movimientos (" + maxMoves + ")");
        }

        // Restaurar marca original del mundo por si acaso
        mundo.setMarca(marcaModelo);
        return mundo;
    }

    private static List<Integer>[] piezasVivas(Tablero t) {
        SmallMatrix tablero = t.getMatrix();
        List<Integer> damasB = Arrays.stream(FuncionesDamas.nombresBlancas).filter(n -> tablero.getPosicion(n) != null)
                .boxed()
                .toList();
        List<Integer> reinasB = Arrays.stream(FuncionesDamas.nombresReinasBlancas)
                .filter(n -> tablero.getPosicion(n) != null).boxed()
                .toList();
        List<Integer> damasN = Arrays.stream(FuncionesDamas.nombresNegras).filter(n -> tablero.getPosicion(n) != null)
                .boxed()
                .toList();
        List<Integer> reinasN = Arrays.stream(FuncionesDamas.nombresReinasNegras)
                .filter(n -> tablero.getPosicion(n) != null).boxed()
                .toList();
        List<Integer> blancas = new ArrayList<>();
        blancas.addAll(damasB);
        blancas.addAll(reinasB);
        List<Integer> negras = new ArrayList<>();
        negras.addAll(damasN);
        negras.addAll(reinasN);
        if (blancas.isEmpty() && negras.isEmpty()) {
            throw new IllegalStateException("Error en piezasVivas: No hay piezas en el tablero.");
        }
        @SuppressWarnings("unchecked")
        List<Integer>[] result = (List<Integer>[]) new List<?>[2];
        result[0] = blancas;
        result[1] = negras;
        return result;
    }

    // Re-implemented inference logic locally to avoid dependency on
    // GeneradorDatosDamas
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
}
