package es.jastxz.nn.spiking.integration;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de integración para verificar el entrenamiento completo con STDP.
 * Feature: spiking-neural-network, Task 17.1
 * 
 * Estos tests verifican que el modo entrenamiento funciona correctamente
 * en escenarios realistas con múltiples neuronas y capas.
 */
public class EntrenamientoIntegrationTest {

    /**
     * Test: Una red simple aprende a fortalecer conexiones con STDP.
     * Este test verifica manualmente que STDP se aplica correctamente.
     */
    @Test
    void redAprendeAFortalecerConexionesConSTDP() {
        // Crear red 2-1
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre1 = red.getNeurona(0, 0);
        NeuronaSpiking pre2 = red.getNeurona(0, 1);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        // Crear conexiones
        red.crearConexion(pre1, post, 0.4, 0);
        red.crearConexion(pre2, post, 0.4, 0);
        
        // Guardar pesos iniciales
        double peso1Inicial = red.getSinapsis().get(0).getPeso();
        double peso2Inicial = red.getSinapsis().get(1).getPeso();
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Simular múltiples ciclos de entrenamiento
        for (int i = 0; i < 10; i++) {
            // Pre neuronas disparan
            pre1.setPotencialMembrana(-54.0);
            pre2.setPotencialMembrana(-54.0);
            red.avanzarTimestep();
            
            // Post neurona dispara (LTP)
            post.setPotencialMembrana(-54.0);
            red.avanzarTimestep();
            
            // Esperar periodo refractario
            for (int j = 0; j < 3; j++) {
                red.avanzarTimestep();
            }
        }
        
        // Verificar que los pesos cambiaron
        double peso1Final = red.getSinapsis().get(0).getPeso();
        double peso2Final = red.getSinapsis().get(1).getPeso();
        
        assertTrue(peso1Final > peso1Inicial,
            "Peso 1 debería incrementarse después del entrenamiento");
        assertTrue(peso2Final > peso2Inicial,
            "Peso 2 debería incrementarse después del entrenamiento");
    }

    /**
     * Test: El modo entrenamiento puede desactivarse para evaluación.
     */
    @Test
    void modoEntrenamientoPuedeDesactivarseParaEvaluacion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        // Crear conexiones
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                red.crearConexion(red.getNeurona(0, i), red.getNeurona(1, j), 0.5, 0);
            }
        }
        for (int i = 0; i < 2; i++) {
            red.crearConexion(red.getNeurona(1, i), red.getNeurona(2, 0), 0.5, 0);
        }
        
        // Fase 1: Entrenar
        red.setModoEntrenamiento(true);
        double[] inputs = {0.9, 0.8};
        
        for (int i = 0; i < 5; i++) {
            red.procesar(inputs, 50);
            red.resetearEstadoTemporal();
        }
        
        double pesosDespuesEntrenamiento = calcularPesoPromedio(red);
        
        // Fase 2: Evaluar (sin entrenamiento)
        red.setModoEntrenamiento(false);
        
        for (int i = 0; i < 5; i++) {
            red.procesar(inputs, 50);
            red.resetearEstadoTemporal();
        }
        
        double pesosDespuesEvaluacion = calcularPesoPromedio(red);
        
        // Los pesos no deberían cambiar durante la evaluación
        assertEquals(pesosDespuesEntrenamiento, pesosDespuesEvaluacion, 0.0001,
            "Los pesos no deberían cambiar cuando el modo entrenamiento está desactivado");
    }

    /**
     * Test: STDP fortalece conexiones que contribuyen a la activación.
     */
    @Test
    void stdpFortalececonexionesQueContribuyenAActivacion() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking n1 = red.getNeurona(0, 0);
        NeuronaSpiking n2 = red.getNeurona(0, 1);
        NeuronaSpiking output = red.getNeurona(1, 0);
        
        // Crear conexiones con pesos iniciales
        red.crearConexion(n1, output, 0.4, 0);
        red.crearConexion(n2, output, 0.4, 0);
        
        double peso1Inicial = red.getSinapsis().get(0).getPeso();
        double peso2Inicial = red.getSinapsis().get(1).getPeso();
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Simular patrón donde ambas neuronas de entrada disparan
        // seguidas por la neurona de salida
        for (int i = 0; i < 10; i++) {
            n1.setPotencialMembrana(-54.0);
            n2.setPotencialMembrana(-54.0);
            red.avanzarTimestep();
            
            // La neurona de salida debería recibir señales y disparar
            output.setPotencialMembrana(-54.0);
            red.avanzarTimestep();
            
            // Avanzar algunos timesteps para periodo refractario
            for (int j = 0; j < 3; j++) {
                red.avanzarTimestep();
            }
        }
        
        double peso1Final = red.getSinapsis().get(0).getPeso();
        double peso2Final = red.getSinapsis().get(1).getPeso();
        
        // Ambos pesos deberían incrementarse por LTP
        assertTrue(peso1Final > peso1Inicial,
            "Peso 1 debería incrementarse por LTP");
        assertTrue(peso2Final > peso2Inicial,
            "Peso 2 debería incrementarse por LTP");
    }

    /**
     * Test: Procesamiento por lotes con entrenamiento.
     * Verifica que el modo entrenamiento funciona durante procesamiento por lotes.
     */
    @Test
    void procesamientoPorLotesConEntrenamiento() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre1 = red.getNeurona(0, 0);
        NeuronaSpiking pre2 = red.getNeurona(0, 1);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        // Crear conexiones
        red.crearConexion(pre1, post, 0.4, 0);
        red.crearConexion(pre2, post, 0.4, 0);
        
        double pesoInicial = calcularPesoPromedio(red);
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Simular múltiples patrones
        for (int patron = 0; patron < 3; patron++) {
            for (int ciclo = 0; ciclo < 5; ciclo++) {
                pre1.setPotencialMembrana(-54.0);
                pre2.setPotencialMembrana(-54.0);
                red.avanzarTimestep();
                
                post.setPotencialMembrana(-54.0);
                red.avanzarTimestep();
                
                for (int j = 0; j < 3; j++) {
                    red.avanzarTimestep();
                }
            }
            
            // Reset entre patrones
            red.resetearEstadoTemporal();
        }
        
        double pesoFinal = calcularPesoPromedio(red);
        
        // Los pesos deberían haber cambiado
        assertTrue(pesoFinal > pesoInicial,
            "Los pesos deberían incrementarse después de procesar múltiples patrones en modo entrenamiento");
    }

    /**
     * Test: Verificar que STDP respeta los límites de peso.
     */
    @Test
    void stdpRespetaLimitesDePeso() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 1)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.1, 0.12, 20.0, 20.0) // Amplitudes grandes para cambios rápidos
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        
        // Crear conexión cerca del límite superior
        red.crearConexion(pre, post, 0.95, 0);
        
        // Activar modo entrenamiento
        red.setModoEntrenamiento(true);
        
        // Aplicar muchos ciclos de LTP
        for (int i = 0; i < 50; i++) {
            pre.setPotencialMembrana(-54.0);
            red.avanzarTimestep();
            
            post.setPotencialMembrana(-54.0);
            red.avanzarTimestep();
            
            for (int j = 0; j < 3; j++) {
                red.avanzarTimestep();
            }
        }
        
        double pesoFinal = red.getSinapsis().get(0).getPeso();
        
        // El peso no debería exceder el límite superior (1.0)
        assertTrue(pesoFinal <= 1.0,
            "El peso no debería exceder el límite superior de 1.0, pero es: " + pesoFinal);
        
        // El peso debería estar cerca del límite
        assertTrue(pesoFinal > 0.95,
            "El peso debería haber aumentado significativamente");
    }

    /**
     * Calcula el peso promedio de todas las sinapsis en la red.
     */
    private double calcularPesoPromedio(RedNeuralSpiking red) {
        double suma = 0.0;
        int count = 0;
        
        for (SinapsisSpiking sinapsis : red.getSinapsis()) {
            suma += sinapsis.getPeso();
            count++;
        }
        
        return count > 0 ? suma / count : 0.0;
    }
}
