package org.javig;

import org.javig.engine.Funciones3enRaya;
import org.javig.engine.Minimax;
import org.javig.models.Modelo3enRaya;
import org.javig.nn.NeuralNetwork;
import org.javig.tipos.Movimiento;
import org.javig.tipos.Mundo;
import org.javig.tipos.Posicion;
import org.javig.tipos.SmallMatrix;
import org.javig.tipos.Tablero;
import org.javig.util.ModelManager;
import org.junit.jupiter.api.Test;

public class Modelo3enRayaTest {

    @Test
    public void testEntrenar() {
        // Validation that training runs without error
        Modelo3enRaya modelo = new Modelo3enRaya();
        modelo.entrenar();
        System.out.println("Entrenamiento completado");
    }

    @Test
    public void testEjecutarModelo() {
        Modelo3enRaya modelo = new Modelo3enRaya();

        // Cargar modelo entrenado
        NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModelo());

        // Verificamos que todos eligen la misma casilla ganadora/bloqueadora en este
        // caso determinista
        // Nota: En este estado (1,2) es el único movimiento correcto.
        int[][] data = {
                { 2, 1, 1 },
                { 0, 2, 0 },
                { 0, 0, 1 } };
        SmallMatrix matrix = new SmallMatrix(data);
        Tablero tablero = new Tablero(matrix);
        double[] input = modelo.tabularToInput(tablero);
        int moveIndex = ModelManager.predictIndex(loadedModel, input);
        System.out.println(ModelManager.getMejorMovimiento(moveIndex));
        assert ModelManager.getMejorMovimiento(moveIndex).equals(new Posicion(1, 2));
    }

    @Test
    public void testModeloVsMinimax() {

        for (int i = 0; i < 100; i++) {
            Modelo3enRaya modelo = new Modelo3enRaya();
            NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModelo());
            Mundo mundo = modelo.getMundo();
            mundo.setMarca(Funciones3enRaya.marcaMaquina3enRaya(mundo.getMarca()));
            mundo = simulaPartida(mundo, loadedModel, modelo);
            assert !Funciones3enRaya.hay3EnRaya(mundo.getMovimiento().getTablero());
            System.out.println("Partida " + i + " terminada");
        }

    }

    private Mundo simulaPartida(Mundo mundo, NeuralNetwork loadedModel, Modelo3enRaya modelo) {
        while (true) {
            Tablero movimientoMinimax = Minimax.negamax(mundo);
            if (Funciones3enRaya.fin3enRaya(movimientoMinimax)) {
                break;
            }
            Posicion posicionModelo = ModelManager.getMejorMovimiento(ModelManager.predictIndex(loadedModel,
                    modelo.tabularToInput(movimientoMinimax)));
            Tablero movimientoModelo = modelo.obtenerTablero(movimientoMinimax, posicionModelo);
            if (Funciones3enRaya.fin3enRaya(movimientoModelo)) {
                break;
            }
            mundo.setMovimiento(new Movimiento(movimientoModelo, posicionModelo));
        }
        return mundo;
    }
}
