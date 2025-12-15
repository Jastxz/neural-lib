package org.javig.models;

import java.util.ArrayList;
import java.util.List;

import org.javig.engine.Funciones3enRaya;
import org.javig.engine.Minimax;
import org.javig.tipos.Mundo;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.Util;
import org.javig.nn.NeuralNetwork;
import org.javig.util.DataContainer;
import org.javig.util.FormatClassToString;
import org.javig.util.ModelManager;
import org.javig.util.NeuralNetworkTrainer;

public class Modelo3enRaya {

    private NeuralNetwork cerebro;
    private String nombreModelo = "modelo3enRaya.nn";
    private Mundo mundo;
    private Posicion posInicial = new Posicion(0, 0);
    private Tablero tablero = Funciones3enRaya.inicial3enRaya();
    private Movimiento movimientoInicial = new Movimiento(tablero, posInicial);
    private String juego = Util.juego3enRaya;
    private int dificultad = 2;
    private int profundidad = 9;
    private int marca = 1;
    private int turno = 1;
    private boolean esMaquina = true;

    public Modelo3enRaya() {
        cerebro = new NeuralNetwork(9, 15, 9);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public Modelo3enRaya(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public void entrenar() {
        // 1. Preparar datos - Generar datos dinámicamente
        System.out.println("Generando datos de entrenamiento exhaustivos...");
        DataContainer data = generateTrainingData(mundo);
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        // 2. Entrenar usando la Clase de Utilidad
        // Entrena por 100 épocas, registrando progreso cada 10 épocas
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, 100, 10);

        // 3. Guardar el Modelo
        ModelManager.saveModel(cerebro, nombreModelo);

        // 4. Cargar el Modelo (Simulador)
        cerebro = ModelManager.loadModel(nombreModelo);

        // 5. Predecir usando el modelo cargado
        System.out.println("\nPredicciones del modelo cargado:");
        for (int i = 0; i < training_inputs.length; i += 200) {
            double[] output = cerebro.feedForward(training_inputs[i]);
            System.out.println(
                    "Entrada: " + FormatClassToString.formatDoubleArray(training_inputs[i]) + " -> Salida: "
                            + ModelManager.getMejorMovimiento(output));
        }
    }

    public DataContainer generateTrainingData(Mundo mundo) {
        List<double[]> inputs = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();

        List<int[][]> todasLasPosibilidades = generarTodasLasPosibilidades();

        for (int[][] estado : todasLasPosibilidades) {
            Tablero tablero = new Tablero(new SmallMatrix(estado));

            // 1. Obtener el movimiento IDEAL de Minimax (Maestro)
            // Debemos asegurar que 'mundo' tenga el estado actual del tablero
            // para que Minimax calcule correctamente
            mundo.getMovimiento().setTablero(tablero);

            // Nota: Minimax.negamax devuelve el tablero resultante, no el movimiento.
            // Calcula el mejor movimiento para quien sea el turno en 'mundo'
            Tablero bestNextState = Minimax.negamax(mundo);

            if (bestNextState == null) {
                break;
            }

            // 2. Registrar Datos de Entrenamiento: (EstadoActual -> MovimientoIdeal)
            Posicion idealMove = obtenerMovimiento(tablero, bestNextState);
            if (idealMove != null) {
                double[] input = tabularToInput(tablero);
                inputs.add(input);

                double[] target = new double[9];
                int index = idealMove.getFila() * 3 + idealMove.getColumna();
                // El objetivo (target) debe ser 1.0 (Activación alta) para que predictIndex lo
                // elija.
                target[index] = 1.0;
                outputs.add(target);
            }
        }

        double[][] training_inputs = inputs.toArray(new double[inputs.size()][]);
        double[][] training_outputs = outputs.toArray(new double[outputs.size()][]);

        return new DataContainer(training_inputs, training_outputs);
    }

    public double[] tabularToInput(Tablero tablero) {
        SmallMatrix tableroSmallMatrix = tablero.getMatrix();
        double[] inputs = new double[9];
        for (int i = 0; i < tableroSmallMatrix.getRows(); i++) {
            for (int j = 0; j < tableroSmallMatrix.getCols(); j++) {
                double valor = tableroSmallMatrix.get(i, j);
                int index = i * tableroSmallMatrix.getCols() + j;

                // Normalizar entradas para la Red Neuronal (0, 1, -1)
                // Asumiendo que las entradas del Tablero son standard 0, 1, 2
                if (valor == 0)
                    inputs[index] = 0.0;
                else if (valor == 1)
                    inputs[index] = 1.0;
                else if (valor == 2)
                    inputs[index] = -1.0; // J2 es -1
            }
        }

        return inputs;
    }

    public List<int[][]> generarTodasLasPosibilidades() {
        List<int[][]> resultado = new ArrayList<>();
        int tamano = 3;
        int totalCeldas = tamano * tamano; // 9
        int totalCombinaciones = (int) Math.pow(3, totalCeldas);

        for (int i = 0; i < totalCombinaciones; i++) {
            int[][] matriz = new int[tamano][tamano];
            int valor = i;
            int count1 = 0;
            int count2 = 0;

            for (int fila = 0; fila < tamano; fila++) {
                for (int col = 0; col < tamano; col++) {
                    int resto = valor % 3;
                    // Usar valores estándar del juego 0, 1, 2
                    matriz[fila][col] = resto;

                    if (matriz[fila][col] == 1)
                        count1++;
                    else if (matriz[fila][col] == 2)
                        count2++;

                    valor /= 3;
                }
            }

            // Validar estado
            Tablero tablero = new Tablero(new SmallMatrix(matriz));
            // ¿Queremos estados donde sea el turno de J2? Mapeamos 2 -> -1 en
            // tabularToInput.
            // J2 mueve cuando J1 tiene 1 pieza más (asumiendo que J1 empieza).
            if (count1 == count2 + 1 && !Funciones3enRaya.fin3enRaya(tablero)) {
                resultado.add(matriz);
            }
        }

        return resultado;
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

    public Posicion obtenerMovimiento(Tablero inicial, Tablero fin) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (inicial.getValor(i, j) == 0 && fin.getValor(i, j) != 0) {
                    return new Posicion(i, j);
                }
            }
        }
        return null;
    }

}
