package es.jastxz.models;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.javig.engine.FuncionesDamas;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.Util;

import es.jastxz.nn.NeuralNetwork;
import es.jastxz.util.DataContainer;
import es.jastxz.util.ModelManager;
import es.jastxz.util.NeuralNetworkTrainer;
import es.jastxz.util.TrainingPair;

public class ModeloDamas {
    private NeuralNetwork cerebro;
    private String nombreModelo = "modeloDamas.nn";
    private Mundo mundo;
    private Tablero tablero = FuncionesDamas.inicialDamas();
    private Posicion posInicial = new Posicion(4, 2);
    private Movimiento movimientoInicial = new Movimiento(tablero, posInicial);
    private String juego = Util.juegoDamas;
    private int dificultad = 4; // Ajustado para un tiempo razonable en Damas
    private int profundidad = 5;
    private int marca = 1;
    private int turno = 1;
    private boolean esMaquina = true;
    private List<Movimiento> posiblesEstados = new ArrayList<>();

    public ModeloDamas() {
        // Input: 8x8 + 1 (turno) = 65
        // Hidden: 512, 256 (Increased capacity for tactical depth)
        // Output: 64 (Source) + 64 (Dest) = 128
        cerebro = new NeuralNetwork(65, 512, 256, 128);
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public ModeloDamas(NeuralNetwork cerebro) {
        this.cerebro = cerebro;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public ModeloDamas(List<Movimiento> posiblesEstados) {
        cerebro = new NeuralNetwork(65, 512, 256, 128);
        this.posiblesEstados = posiblesEstados;
        mundo = new Mundo(movimientoInicial, juego, dificultad, profundidad, marca, turno, esMaquina);
    }

    public void entrenar() {
        System.out.println("Generando datos de entrenamiento via simulación Minimax...");
        long start = System.currentTimeMillis();
        // Generamos datos de entrenamiento
        DataContainer data = generateTrainingData();
        // Generamos un número de partidas para cubrir estados diversos
        double[][] training_inputs = data.getInputs();
        double[][] training_outputs = data.getOutputs();
        System.out.println("Datos generados exitosamente. Total de muestras: " + training_inputs.length);

        // Entrenar
        NeuralNetworkTrainer.train(cerebro, training_inputs, training_outputs, 10, 1);

        // Guardar
        ModelManager.saveModel(cerebro, nombreModelo);

        // Verificar carga
        cerebro = ModelManager.loadModel(nombreModelo);

        System.out.println("\nEntrenamiento completado.");
        long end = System.currentTimeMillis();
        long total = end - start;
        System.out.println("Tiempo de entrenamiento: " + total / 60000.0 + " minutos, " + total / 3600000.0 + " horas");
    }

    /**
     * Carga los datos de entrenamiento desde los archivos .dat
     */
    public DataContainer generateTrainingData() {
        String dataDir = "src/main/resources/datosDamas";
        File dir = new File(dataDir);
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("Directorio de datos no encontrado: " + dataDir);
            return new DataContainer(new double[0][], new double[0][]);
        }

        List<TrainingPair> allPairs = new ArrayList<>();
        File[] files = dir.listFiles((d, name) -> name.endsWith(".dat"));
        if (files != null) {
            for (File f : files) {
                allPairs.addAll(cargarDatosDeFichero(f));
            }
        }

        System.out.println("Cargados " + allPairs.size() + " pares de entrenamiento desde " + dataDir);

        double[][] inputs = new double[allPairs.size()][];
        double[][] outputs = new double[allPairs.size()][];
        for (int i = 0; i < allPairs.size(); i++) {
            double[] rawInput = allPairs.get(i).getInput();
            // Transform data on the fly to match Relative Encoding
            // Raw Data: Input[64] = Turn (1 or 2). Board: White=1, Black=-1.
            // Requirement: Me=1, Enemy=-1.

            double turn = rawInput[64];
            if (turn == 2.0) { // Black's Turn
                // We need to flip signs:
                // Currently White(1) is Enemy -> Needs to be -1.
                // Currently Black(-1) is Me -> Needs to be 1.
                // Multiplying by -1 achieves this.
                for (int j = 0; j < 64; j++) {
                    rawInput[j] = -rawInput[j];
                }
            }
            // Normalize turn input to 1.0
            rawInput[64] = 1.0;

            inputs[i] = rawInput;
            outputs[i] = allPairs.get(i).getOutput();
        }

        return new DataContainer(inputs, outputs);
    }

    private List<TrainingPair> cargarDatosDeFichero(File f) {
        List<TrainingPair> pairs = new ArrayList<>();
        try (DataInputStream dis = new DataInputStream(new FileInputStream(f))) {
            int count = dis.readInt();
            for (int i = 0; i < count; i++) {
                double[] input = new double[65];
                for (int j = 0; j < 65; j++)
                    input[j] = dis.readDouble();
                double[] output = new double[128];
                for (int j = 0; j < 128; j++)
                    output[j] = dis.readDouble();
                pairs.add(new TrainingPair(input, output));
            }
            System.out.println("Leído " + f.getName() + ": " + count + " muestras.");
        } catch (IOException e) {
            System.err.println("Error leyendo fichero " + f.getName() + ": " + e.getMessage());
        }
        return pairs;
    }

    public static double[] tabularToInput(Tablero tablero, int turno) {
        SmallMatrix matrix = tablero.getMatrix();
        double[] inputs = new double[65];
        boolean soyBlancas = (turno == 1);

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double val = matrix.get(i, j);
                int idx = i * 8 + j;
                int pieza = (int) val;

                // Relative Encoding:
                // My Pieces -> 1.0
                // Enemy Pieces -> -1.0
                // Empty -> 0.0
                if (pieza == 0) {
                    inputs[idx] = 0.0;
                } else if (contiene(FuncionesDamas.nombresBlancas, pieza)
                        || contiene(FuncionesDamas.nombresReinasBlancas, pieza)) {
                    // Piece is White
                    inputs[idx] = soyBlancas ? 1.0 : -1.0;
                } else if (contiene(FuncionesDamas.nombresNegras, pieza)
                        || contiene(FuncionesDamas.nombresReinasNegras, pieza)) {
                    // Piece is Black
                    inputs[idx] = soyBlancas ? -1.0 : 1.0;
                }
            }
        }
        // Normalize Turn: Always 1.0 from the network's perspective ("My Turn")
        inputs[64] = 1.0;
        return inputs;
    }

    public static boolean esCasillaBlanca(int fila, int col) {
        return (fila + col) % 2 == 0;
    }

    /**
     * Deduce la posición de destino comparando el tablero antes y después.
     */
    public static Posicion obtenerMovimiento(Tablero inicial, Tablero fin, int marcaQueMovio) {
        // Determinar si movieron blancas o negras
        boolean movioBlancas = marcaQueMovio == 1;

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                int vIni = inicial.getValor(i, j);
                int vFin = fin.getValor(i, j);

                // Si antes estaba vacío y ahora tiene una pieza
                if (vIni == 0 && vFin != 0) {
                    // Verificar si la pieza que apareció pertenece al bando correcto
                    boolean esBlancas = contiene(FuncionesDamas.nombresBlancas, vFin)
                            || contiene(FuncionesDamas.nombresReinasBlancas, vFin);
                    boolean esNegras = contiene(FuncionesDamas.nombresNegras, vFin)
                            || contiene(FuncionesDamas.nombresReinasNegras, vFin);

                    if ((movioBlancas && esBlancas) || (!movioBlancas && esNegras)) {
                        return new Posicion(i, j);
                    } else {
                        System.out.println("Movimiento inválido: " + vIni + " -> " + vFin);
                        System.out.println("Mueven: " + (movioBlancas ? "Blancas" : "Negras") + "; La pieza es: "
                                + (esBlancas ? "Blancas" : "Negras"));
                        System.out.println("Pieza: " + vFin);
                        System.out.println("Tablero inicial: \n" + inicial);
                        System.out.println("Tablero final: \n" + fin);
                        System.exit(0);
                    }
                }
            }
        }

        return null;
    }

    public static boolean contiene(int[] array, int valor) {
        for (int val : array) {
            if (val == valor) {
                return true;
            }
        }
        return false;
    }

    public String getNombreModelo() {
        return nombreModelo;
    }

    public void setNombreModelo(String nombreModelo) {
        this.nombreModelo = nombreModelo;
    }

    public Mundo getMundo() {
        return mundo;
    }

    public void setMundo(Mundo mundo) {
        this.mundo = mundo;
    }

    public int getProfundidad() {
        return profundidad;
    }

    public void setProfundidad(int profundidad) {
        this.profundidad = profundidad;
    }

    public List<Movimiento> getPosiblesEstados() {
        return posiblesEstados;
    }
}
