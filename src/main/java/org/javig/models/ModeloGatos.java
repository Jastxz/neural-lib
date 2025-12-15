package org.javig.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.javig.engine.FuncionesGato;
import org.javig.engine.Minimax;
import org.javig.nn.NeuralNetwork;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.DataContainer;
import org.javig.util.ModelManager;
import org.javig.util.NeuralNetworkTrainer;
import org.javig.util.Util;

public class ModeloGatos {
    private NeuralNetwork cerebro;
    private String nombreModelo = "modeloGatos.nn";
    private Mundo mundo;
    private Posicion posMouseInicial = new Posicion(0, 4); // Posición inicial típica del ratón si no se define otra
    private Tablero tablero = FuncionesGato.inicialGato(posMouseInicial);
    private Movimiento movimientoInicial = new Movimiento(tablero, posMouseInicial);
    private String juego = Util.juegoGato;
    private int dificultad = 4; // Ajustado para un tiempo razonable en Gatos
    private int profundidad = 4;
    private int marcaRatón = FuncionesGato.nombreRaton;
    private int turno = 1;
    private boolean esMaquina = true;

    public ModeloGatos() {
        // Input: 8x8 = 64
        // Hidden: 128 (arbitrario, suficiente para capturar lógica básica)
        // Output: 8x8 = 64 (probabilidad de mover a cada casilla)
        cerebro = new NeuralNetwork(64, 128, 64);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public ModeloGatos(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marcaRatón, turno, esMaquina);
    }

    public void entrenar() {
        System.out.println("Generando datos de entrenamiento via simulación Minimax...");
        // Generamos un número de partidas para cubrir estados diversos
        DataContainer data = generateTrainingData(500);
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        // Entrenar
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, 100, 10);

        // Guardar
        ModelManager.saveModel(cerebro, nombreModelo);

        // Verificar carga
        cerebro = ModelManager.loadModel(nombreModelo);

        System.out.println("\nEntrenamiento completado.");
    }

    /**
     * Genera datos simulando partidas completas usando Minimax para ambos bandos.
     * 
     * @param numGames Número de partidas a simular.
     */
    public DataContainer generateTrainingData(int numGames) {
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        long startSimulationTime = System.currentTimeMillis();
        for (int i = 0; i < numGames; i++) {
            // Aleatorizar posición inicial del ratón para variedad (fila 0, cualquier
            // columna negra)
            // Las columnas negras en fila 0 son 0, 2, 4, 6
            int colRandom = new int[] { 0, 2, 4, 6 }[(int) (Math.random() * 4)];
            Posicion inicioRaton = new Posicion(0, colRandom);

            Tablero currentTablero = FuncionesGato.inicialGato(inicioRaton);
            int currentBand = FuncionesGato.nombreRaton; // Empieza el ratón por defecto o según reglas
            int currentTurn = 2; // Turno 2 porque buscamos el movimiento del ratón y Minimax asume turno 1

            // Límite de movimientos para evitar bucles infinitos en simulación
            int moves = 0;
            long startTime = System.currentTimeMillis();
            while (!FuncionesGato.finGato(currentTablero) && moves < 200) {
                // Configurar mundo actua para Minimax
                // Minimax necesita saber de quién es el turno para maximizar su jugada
                // Creamos un dummy movimiento solo para pasar el tablero
                Movimiento movDummy = new Movimiento(currentTablero, new Posicion(0, 0));
                mundo.setMovimiento(movDummy);
                mundo.setMarca(FuncionesGato.marcaMaquinaGato(currentBand)); // A Minimax hay que pasarle la marca del
                                                                             // bando contrario al que está jugando
                mundo.setTurno(currentTurn);

                // Llamada al oráculo (Minimax)
                Tablero bestNextState = Minimax.negamax(mundo);

                if (bestNextState == null)
                    break;

                // Identificar el movimiento realizado (diferencia entre currentTablero y
                // bestNextState)
                Posicion moveDest = obtenerMovimiento(currentTablero, bestNextState, currentBand);

                if (moveDest != null) {
                    // Guardar par (Estado -> Acción)
                    inputs.add(tabularToInput(currentTablero));

                    double[] target = new double[64];
                    int idx = moveDest.getFila() * 8 + moveDest.getColumna();
                    target[idx] = 1.0;
                    outputs.add(target);
                }

                // Avanzar estado
                currentTablero = bestNextState;
                // Cambiar turno: Si era Ratón, ahora toca a LOS Gatos (representados por uno de
                // ellos genéricamente o lógica de control)
                // En Minimax para Gatos, FuncionesGato maneja "el bando de los gatos".
                currentBand = FuncionesGato.marcaMaquinaGato(currentBand);
                currentTurn = (currentTurn == 1) ? 2 : 1;
                moves++;
                if (moves % 10 == 0) {
                    long endTime = System.currentTimeMillis() - startTime;
                    System.out
                            .println("Tiempo de movimiento acumulado: " + endTime + " ms, " + endTime / 1000.0 + " s");
                }
            }
            if (i % 10 == 0) {
                System.out.println("Partida simulada " + i + "/" + numGames);
                long endSimulationTime = System.currentTimeMillis() - startSimulationTime;
                System.out.println("Tiempo de simulación acumulado: " + endSimulationTime + " ms, "
                        + endSimulationTime / 1000.0 + " s");
            }
        }

        return new DataContainer(
                inputs.toArray(new double[0][]),
                outputs.toArray(new double[0][]));
    }

    // Sobrecarga para mantener firma por si acaso, aunque no recomendable usar
    // generateTrainingData(Mundo) directamente
    public DataContainer generateTrainingData(Mundo m) {
        return generateTrainingData(1);
    }

    public double[] tabularToInput(Tablero tablero) {
        SmallMatrix matrix = tablero.getMatrix();
        double[] inputs = new double[64];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double val = matrix.get(i, j);
                int idx = i * 8 + j;

                // Normalización:
                // 0 (Vacío) -> 0.0
                // 9 (Ratón) -> 1.0
                // 1,3,5,7 (Gatos) -> -1.0
                if (val == 0) {
                    inputs[idx] = 0.0;
                } else if (val == FuncionesGato.nombreRaton) {
                    inputs[idx] = 1.0;
                } else {
                    // Asumimos que cualquier otro valor positivo es un gato
                    inputs[idx] = -1.0;
                }
            }
        }
        return inputs;
    }

    /**
     * Deduce la posición de destino comparando el tablero antes y después.
     */
    public Posicion obtenerMovimiento(Tablero inicial, Tablero fin, int marcaQueMovio) {
        // En Gatos, una pieza se mueve de A -> B.
        // A (origen) tendrá valor 0 en 'fin' (o diferente si fue captura, pero aquí no
        // hay capturas).
        // B (destino) tendrá valor 'marca' en 'fin' y 0 en 'inicial'.

        // Buscamos la casilla que ahora tiene una pieza y antes estaba vacía.
        // Ojo: Si mueven los gatos, hay 4 gatos. Necesitamos ver cuál cambió.

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int vIni = inicial.getValor(i, j);
                int vFin = fin.getValor(i, j);

                // Si antes estaba vacío y ahora tiene la marca (o uno de los gatos)
                if (vIni == 0 && vFin != 0) {
                    // Verificamos si la pieza que apareció corresponde al bando que movió
                    boolean esMismaMarca = (vFin == marcaQueMovio);
                    boolean esGato = Arrays.stream(FuncionesGato.nombresGatos).anyMatch(x -> x == vFin);

                    if ((esMismaMarca && esGato) || (esMismaMarca && vFin == FuncionesGato.nombreRaton)) {
                        return new Posicion(i, j);
                    }
                }
            }
        }
        return null;
    }

    public Mundo getMundo() {
        return mundo;
    }

    public String getNombreModelo() {
        return nombreModelo;
    }

    public void setNombreModelo(String nombreModelo) {
        this.nombreModelo = nombreModelo;
    }
}
