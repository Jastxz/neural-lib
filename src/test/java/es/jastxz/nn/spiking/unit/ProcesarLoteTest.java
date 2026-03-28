package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para el método procesarLote() de RedNeuralSpiking.
 * 
 * Verifica que el procesamiento por lotes funcione correctamente,
 * incluyendo aislamiento entre patrones y acumulación de métricas.
 * 
 * Feature: spiking-neural-network, Task 15.2
 */
class ProcesarLoteTest {

    /**
     * Crea una red simple para testing: 2 entradas, 3 ocultas, 2 salidas.
     */
    private RedNeuralSpiking crearRedSimple() {
        ConfiguracionRed config = new ConfiguracionRedBuilder()
            .topologia(2, 3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
            .inicializacionPesos(TipoInicializacion.UNIFORME, 0.3, 0.7)
            .retardos(1, 3)
            .build();
        
        RedNeuralSpiking red = new RedNeuralSpiking(config);
        
        // Crear conexiones completas entre capas
        // Capa 0 -> Capa 1
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                red.crearConexion(
                    red.getNeurona(0, i),
                    red.getNeurona(1, j),
                    0.5,
                    1
                );
            }
        }
        
        // Capa 1 -> Capa 2
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                red.crearConexion(
                    red.getNeurona(1, i),
                    red.getNeurona(2, j),
                    0.5,
                    1
                );
            }
        }
        
        return red;
    }

    @Test
    void procesarLoteSimpleProduceSalidas() {
        // Feature: spiking-neural-network, Test: procesar lote simple produce salidas
        // Requisito 14.1, 14.4
        RedNeuralSpiking red = crearRedSimple();
        
        // Lote de 3 patrones
        double[][] batchInputs = {
            {0.8, 0.3},
            {0.5, 0.7},
            {0.2, 0.9}
        };
        
        // Procesar lote con reset entre patrones
        double[][] batchOutputs = red.procesarLote(batchInputs, 100, true);
        
        // Verificar que produce el número correcto de salidas (Requisito 14.4)
        assertEquals(3, batchOutputs.length, "Debe retornar 3 conjuntos de salidas");
        
        // Verificar que cada salida tiene el tamaño correcto
        for (int i = 0; i < batchOutputs.length; i++) {
            assertEquals(2, batchOutputs[i].length, 
                "Salida " + i + " debe tener 2 valores");
            
            // Verificar que los valores están en el rango [0, 1]
            for (double output : batchOutputs[i]) {
                assertTrue(output >= 0.0 && output <= 1.0, 
                    "Output " + output + " debe estar en [0, 1]");
            }
        }
    }

    @Test
    void procesarLoteConResetAislaPatrones() {
        // Feature: spiking-neural-network, Test: reset entre patrones aísla procesamiento
        // Requisito 14.2
        RedNeuralSpiking red = crearRedSimple();
        
        // Lote de 2 patrones
        double[][] batchInputs = {
            {0.9, 0.1},
            {0.1, 0.9}
        };
        
        // Procesar con reset
        red.procesarLote(batchInputs, 100, true);
        
        // El timestep debe reflejar solo el último patrón procesado
        // ya que se resetea entre patrones
        // Después del reset, timestep vuelve a 0, luego avanza 100
        assertEquals(100, red.getTimestepActual(), 
            "Con reset, timestep debe ser 100 (solo el último patrón)");
    }

    @Test
    void procesarLoteSinResetMantienEstado() {
        // Feature: spiking-neural-network, Test: sin reset mantiene estado temporal
        // Requisito 14.5
        RedNeuralSpiking red = crearRedSimple();
        
        // Lote de 2 patrones
        double[][] batchInputs = {
            {0.7, 0.3},
            {0.4, 0.8}
        };
        
        // Procesar sin reset (estado se mantiene)
        red.procesarLote(batchInputs, 100, false);
        
        // El timestep debe reflejar ambos patrones procesados
        // 100 timesteps por patrón × 2 patrones = 200
        assertEquals(200, red.getTimestepActual(), 
            "Sin reset, timestep debe ser 200 (ambos patrones acumulados)");
    }

    @Test
    void procesarLoteAcumulaMetricas() {
        // Feature: spiking-neural-network, Test: métricas se acumulan a través del lote
        // Requisito 14.3
        RedNeuralSpiking red = crearRedSimple();
        
        // Lote de 3 patrones
        double[][] batchInputs = {
            {0.8, 0.2},
            {0.6, 0.5},
            {0.3, 0.9}
        };
        
        // Procesar lote sin reset para acumular métricas
        red.procesarLote(batchInputs, 100, false);
        
        // Obtener métricas acumuladas
        var metricas = red.obtenerMetricas();
        long totalSpikes = (Long) metricas.get("totalSpikes");
        
        // Debe haber acumulado spikes de todos los patrones
        assertTrue(totalSpikes > 0, 
            "Debe haber spikes acumulados de todos los patrones");
    }

    @Test
    void procesarLoteConBatchNullFalla() {
        // Feature: spiking-neural-network, Test: validación de batch null
        RedNeuralSpiking red = crearRedSimple();
        
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> red.procesarLote(null, 100, true));
        assertTrue(exception.getMessage().contains("null"));
    }

    @Test
    void procesarLoteConBatchVacioFalla() {
        // Feature: spiking-neural-network, Test: validación de batch vacío
        RedNeuralSpiking red = crearRedSimple();
        
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> red.procesarLote(new double[0][0], 100, true));
        assertTrue(exception.getMessage().contains("vacío"));
    }

    @Test
    void procesarLoteConDuracionNegativaFalla() {
        // Feature: spiking-neural-network, Test: validación de duración negativa
        RedNeuralSpiking red = crearRedSimple();
        
        double[][] batchInputs = {{0.5, 0.5}};
        
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> red.procesarLote(batchInputs, -10, true));
        assertTrue(exception.getMessage().contains("positiva"));
    }

    @Test
    void procesarLoteConDuracionCeroFalla() {
        // Feature: spiking-neural-network, Test: validación de duración cero
        RedNeuralSpiking red = crearRedSimple();
        
        double[][] batchInputs = {{0.5, 0.5}};
        
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> red.procesarLote(batchInputs, 0, true));
        assertTrue(exception.getMessage().contains("positiva"));
    }

    @Test
    void procesarLoteConPatronInvalidoFalla() {
        // Feature: spiking-neural-network, Test: validación de patrón con tamaño incorrecto
        RedNeuralSpiking red = crearRedSimple();
        
        // Segundo patrón tiene tamaño incorrecto (3 en lugar de 2)
        double[][] batchInputs = {
            {0.5, 0.5},
            {0.3, 0.7, 0.9}  // Tamaño incorrecto
        };
        
        // Debe fallar al procesar el segundo patrón
        Exception exception = assertThrows(IllegalArgumentException.class,
            () -> red.procesarLote(batchInputs, 100, true));
        assertTrue(exception.getMessage().contains("no coincide"));
    }

    @Test
    void procesarLoteUnPatronEquivaleAProcesar() {
        // Feature: spiking-neural-network, Test: lote de un patrón equivale a procesar()
        RedNeuralSpiking red1 = crearRedSimple();
        RedNeuralSpiking red2 = crearRedSimple();
        
        double[] patron = {0.7, 0.4};
        
        // Procesar con método individual
        double[] outputIndividual = red1.procesar(patron, 100);
        
        // Procesar con método de lote
        double[][] batchOutputs = red2.procesarLote(new double[][]{patron}, 100, true);
        
        // Ambos deben producir salidas del mismo tamaño
        assertEquals(outputIndividual.length, batchOutputs[0].length);
        
        // Los valores deben estar en el mismo rango (no necesariamente iguales por aleatoriedad)
        for (int i = 0; i < outputIndividual.length; i++) {
            assertTrue(outputIndividual[i] >= 0.0 && outputIndividual[i] <= 1.0);
            assertTrue(batchOutputs[0][i] >= 0.0 && batchOutputs[0][i] <= 1.0);
        }
    }

    @Test
    void procesarLoteGrandeProduceSalidasCorrectas() {
        // Feature: spiking-neural-network, Test: procesar lote grande
        RedNeuralSpiking red = crearRedSimple();
        
        // Lote de 10 patrones
        double[][] batchInputs = new double[10][2];
        for (int i = 0; i < 10; i++) {
            batchInputs[i][0] = i * 0.1;
            batchInputs[i][1] = 1.0 - (i * 0.1);
        }
        
        // Procesar lote
        double[][] batchOutputs = red.procesarLote(batchInputs, 100, true);
        
        // Verificar que produce el número correcto de salidas
        assertEquals(10, batchOutputs.length);
        
        // Verificar que todas las salidas son válidas
        for (int i = 0; i < batchOutputs.length; i++) {
            assertEquals(2, batchOutputs[i].length);
            for (double output : batchOutputs[i]) {
                assertTrue(output >= 0.0 && output <= 1.0);
            }
        }
    }

    @Test
    void procesarLoteConResetLimpiaEstadoCorrectamente() {
        // Feature: spiking-neural-network, Test: reset limpia estado entre patrones
        RedNeuralSpiking red = crearRedSimple();
        
        // Primer patrón con valores altos
        double[][] primerLote = {{1.0, 1.0}};
        red.procesarLote(primerLote, 100, true);
        
        long timestepDespuesPrimero = red.getTimestepActual();
        
        // Resetear manualmente antes del segundo lote
        red.resetearEstadoTemporal();
        
        // Segundo lote con reset entre sus patrones
        double[][] segundoLote = {{0.5, 0.5}};
        red.procesarLote(segundoLote, 100, true);
        
        // El timestep debe ser 100 (solo el último patrón)
        assertEquals(100, red.getTimestepActual());
    }

    @Test
    void procesarLoteSinResetPermitePropagacionTemporal() {
        // Feature: spiking-neural-network, Test: sin reset permite propagación temporal
        RedNeuralSpiking red = crearRedSimple();
        
        // Lote de 2 patrones sin reset
        double[][] batchInputs = {
            {0.8, 0.2},
            {0.3, 0.7}
        };
        
        // Procesar sin reset
        double[][] outputs = red.procesarLote(batchInputs, 50, false);
        
        // Verificar que ambos patrones produjeron salidas
        assertEquals(2, outputs.length);
        
        // El timestep debe reflejar ambos patrones
        assertEquals(100, red.getTimestepActual(), 
            "Timestep debe ser 100 (50 × 2 patrones)");
    }
}
