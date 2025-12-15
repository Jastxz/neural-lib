package org.javig;

import org.javig.engine.Funciones3enRaya;
import org.javig.engine.FuncionesGato;
import org.javig.engine.Minimax;
import org.javig.models.Modelo3enRaya;
import org.javig.models.ModeloGatos;
import org.javig.nn.NeuralNetwork;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.Tablero;
import org.javig.util.DataContainer;
import org.javig.util.ModelManager;
import org.junit.jupiter.api.Test;

public class ModeloGatosTest {

    @Test
    public void testDataGeneration() {
        System.out.println("Validating Data Generation Logic...");
        ModeloGatos modelo = new ModeloGatos();
        // Generate a small amount of data to ensure no exceptions
        DataContainer data = modelo.generateTrainingData(5);

        if (data.getInputs().length > 0) {
            System.out.println("Success: Generated " + data.getInputs().length + " training samples.");
        } else {
            System.out.println("Warning: Generated 0 samples (might be short games or errors).");
        }

        assert data.getInputs().length == data.getOutputs().length;
        assert data.getInputs()[0].length == 64; // Check input size
    }

    @Test
    public void testEntrenar() {
        // Validation that training runs without error
        ModeloGatos modelo = new ModeloGatos();
        modelo.entrenar();
        System.out.println("Entrenamiento completado");
    }

    @Test
    public void testModeloVsMinimax() {
        for (int i = 0; i < 100; i++) {
            ModeloGatos modelo = new ModeloGatos();
            NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModelo());
            Mundo mundo = modelo.getMundo();
            if (i % 2 == 0) {
                mundo.setMarca(FuncionesGato.marcaMaquinaGato(mundo.getMarca()));
                mundo.setTurno(2);
            }
            mundo = simulaPartida(mundo, loadedModel, modelo);
            System.out.println("Partida " + i + " terminada");
            String ganador = FuncionesGato.ratonEncerrado(mundo.getMovimiento().getTablero(),
                    mundo.getMovimiento().getPos()) ? "Gatos"
                            : "Ratón";
            System.out.println("Ganador: " + ganador);
        }
    }

    private Mundo simulaPartida(Mundo mundo, NeuralNetwork loadedModel, ModeloGatos modelo) {
        while (true) {
            Tablero movimientoMinimax = Minimax.negamax(mundo);
            if (FuncionesGato.finGato(movimientoMinimax)) {
                break;
            }
            Posicion posicionModelo = ModelManager.getMejorMovimiento(ModelManager.predictIndex(loadedModel,
                    modelo.tabularToInput(movimientoMinimax)));
            Tablero movimientoModelo = ModelManager.obtenerTablero(movimientoMinimax, posicionModelo, mundo.getMarca());
            if (FuncionesGato.finGato(movimientoModelo)) {
                break;
            }
            mundo.setMovimiento(new Movimiento(movimientoModelo, posicionModelo));
        }
        return mundo;
    }
}
