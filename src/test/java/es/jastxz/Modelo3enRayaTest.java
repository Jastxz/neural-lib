package es.jastxz;

import es.jastxz.engine.Funciones3enRaya;
import es.jastxz.engine.Minimax;
import es.jastxz.tipos.Movimiento;
import es.jastxz.tipos.Mundo;
import es.jastxz.tipos.Posicion;
import es.jastxz.tipos.SmallMatrix;
import es.jastxz.tipos.Tablero;
import org.junit.jupiter.api.Test;

import es.jastxz.models.Modelo3enRaya;
import es.jastxz.nn.NeuralNetwork;
import es.jastxz.util.ModelManager;

public class Modelo3enRayaTest {

    @Test
    public void testEntrenar() {
        // Validation that training runs without error
        Modelo3enRaya modelo = new Modelo3enRaya();
        modelo.entrenar(600);
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
        double[] input = Modelo3enRaya.tabularToInput(tablero, 2);
        int moveIndex = ModelManager.predictIndex(loadedModel, input);
        System.out.println(ModelManager.getMejorMovimiento(moveIndex));
        assert ModelManager.getMejorMovimiento(moveIndex).equals(new Posicion(1, 2));
    }

    @Test
    public void testModeloVsMinimax() {
        int winsModelo = 0;
        int winsMinimax = 0;
        int ties = 0;
        Modelo3enRaya modelo = new Modelo3enRaya();
        NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModelo());

        for (int i = 0; i < 100; i++) {
            Mundo mundo = modelo.getMundo();
            mundo.setMarca(Funciones3enRaya.marcaMaquina3enRaya(mundo.getMarca()));

            // Simula partida tracking winner
            int resultado = simulaPartida(mundo, loadedModel, modelo);

            if (resultado == 1) {
                winsMinimax++;
            } else if (resultado == 2) {
                winsModelo++;
            } else {
                ties++;
            }

            System.out.println("Partida " + i + " terminada. Resultado: "
                    + (resultado == 0 ? "Empate" : (resultado == 1 ? "Minimax" : "Modelo")));
        }

        System.out.println("Resultados Finales:");
        System.out.println("Gana Modelo: " + winsModelo);
        System.out.println("Gana Minimax: " + winsMinimax);
        System.out.println("Empates: " + ties);
    }

    @Test
    public void testOptimizarModelo() {
        int conteoPartidas = 1;
        int neuronasOcultas = 18;
        int epocas = 300;
        boolean empateTotal = false;

        while (!empateTotal && neuronasOcultas < 100 && epocas < 1000) {
            NeuralNetwork nn = new NeuralNetwork(10, neuronasOcultas, 9);
            Modelo3enRaya modelo = new Modelo3enRaya(nn);
            modelo.entrenar(epocas);
            NeuralNetwork loadedModel = ModelManager.loadModel(modelo.getNombreModelo());

            int winsModelo = 0;
            int winsMinimax = 0;
            int ties = 0;
            for (int k = 0; k < 100; k++) {
                Mundo mundo = modelo.getMundo();
                mundo.setMarca(Funciones3enRaya.marcaMaquina3enRaya(mundo.getMarca()));

                // Simula partida tracking winner
                int resultado = simulaPartida(mundo, loadedModel, modelo);

                if (resultado == 1) {
                    winsMinimax++;
                } else if (resultado == 2) {
                    winsModelo++;
                } else {
                    ties++;
                }
            }

            if (ties == 100) {
                empateTotal = true;
            } else if (winsModelo == 100) {
                empateTotal = true;
            } else if (ties + winsModelo == 100) {
                empateTotal = true;
            } else if (ties + winsModelo > 90) {
                ModelManager.saveModel(loadedModel,
                        modelo.getNombreModelo() + "_neur" + neuronasOcultas + "_epoc" + epocas);
            }

            System.out.println("Resultados iteración " + conteoPartidas + ":");
            System.out.println("Gana Modelo: " + winsModelo);
            System.out.println("Gana Minimax: " + winsMinimax);
            System.out.println("Empates: " + ties);
            System.out.println("Neuronas: " + neuronasOcultas);
            System.out.println("Epocas: " + epocas);
            System.out.println("------------------------------------\n");

            conteoPartidas++;
            neuronasOcultas++;
            if (conteoPartidas % 10 == 0) {
                epocas += 100;
            }

        }
        System.out.println("Partidas totales: " + conteoPartidas);
    }

    private int simulaPartida(Mundo mundo, NeuralNetwork loadedModel, Modelo3enRaya modelo) {
        while (true) {
            Tablero movimientoMinimax = Minimax.negamax(mundo);
            if (Funciones3enRaya.fin3enRaya(movimientoMinimax)) {
                if (Funciones3enRaya.hay3EnRaya(movimientoMinimax)) {
                    return 1; // Win Minimax
                }
                return 0; // Tie
            }

            Posicion posicionModelo = ModelManager.getMejorMovimiento(ModelManager.predictIndex(loadedModel,
                    Modelo3enRaya.tabularToInput(movimientoMinimax, mundo.getMarca())));
            Tablero movimientoModelo = ModelManager.obtenerTablero(movimientoMinimax, posicionModelo, mundo.getMarca());

            if (Funciones3enRaya.fin3enRaya(movimientoModelo)) {
                if (Funciones3enRaya.hay3EnRaya(movimientoModelo)) {
                    return 2; // Win Modelo
                }
                return 0; // Tie
            }
            mundo.setMovimiento(new Movimiento(movimientoModelo, posicionModelo));
        }
    }
}
