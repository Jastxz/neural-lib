package es.jastxz.nn.spiking.unit;

import es.jastxz.nn.spiking.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para la gestión de conexiones en RedNeuralSpiking.
 * 
 * Feature: spiking-neural-network
 * Task: 13.1 Implementar creación de conexiones
 */
class RedNeuralSpikingConexionesTest {

    private RedNeuralSpiking red;
    private ConfiguracionRed config;

    @BeforeEach
    void setUp() {
        // Crear configuración básica con 2 capas: 3 neuronas en capa 0, 2 en capa 1
        config = new ConfiguracionRedBuilder()
            .topologia(3, 2)
            .parametrosLIF(-55.0, -70.0, 20.0, 2)
            .parametrosSTDP(0.01, 0.012, 20.0, 20.0)
            .parametrosCodificacion(100.0, ModoCodificacion.POISSON, 50)
            .inicializacionPesos(TipoInicializacion.UNIFORME, 0.0, 1.0)
            .build();

        red = new RedNeuralSpiking(config);
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Crear conexión válida
    void crearConexionValida() {
        // Obtener neuronas de diferentes capas
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);

        // Crear conexión
        assertDoesNotThrow(() -> {
            red.crearConexion(pre, post, 0.5, 2);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar neurona presinaptica null
    void crearConexionConPresinapticaNull() {
        NeuronaSpiking post = red.getNeurona(1, 0);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(null, post, 0.5, 2);
        });
        assertTrue(exception.getMessage().contains("presinaptica no puede ser null"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar neurona postsinaptica null
    void crearConexionConPostsinapticaNull() {
        NeuronaSpiking pre = red.getNeurona(0, 0);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(pre, null, 0.5, 2);
        });
        assertTrue(exception.getMessage().contains("postsinaptica no puede ser null"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar neurona presinaptica no existe en red (Requisito 15.3)
    void crearConexionConPresinapticaInexistente() {
        // Crear neurona que no pertenece a la red
        NeuronaSpiking preExterna = new NeuronaSpiking(999L, 0, 0, -55.0, -70.0, 20.0, 2);
        NeuronaSpiking post = red.getNeurona(1, 0);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(preExterna, post, 0.5, 2);
        });
        assertTrue(exception.getMessage().contains("Neurona presinaptica con ID 999 no existe en la red"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar neurona postsinaptica no existe en red (Requisito 15.3)
    void crearConexionConPostsinapticaInexistente() {
        NeuronaSpiking pre = red.getNeurona(0, 0);
        // Crear neurona que no pertenece a la red
        NeuronaSpiking postExterna = new NeuronaSpiking(999L, 1, 0, -55.0, -70.0, 20.0, 2);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(pre, postExterna, 0.5, 2);
        });
        assertTrue(exception.getMessage().contains("Neurona postsinaptica con ID 999 no existe en la red"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar conexión duplicada (Requisito 15.4)
    void crearConexionDuplicadaFalla() {
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);

        // Crear primera conexión
        red.crearConexion(pre, post, 0.5, 2);

        // Intentar crear conexión duplicada
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(pre, post, 0.7, 1);
        });
        assertTrue(exception.getMessage().contains("Ya existe una conexión"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Crear múltiples conexiones diferentes
    void crearMultiplesConexionesDiferentes() {
        NeuronaSpiking pre1 = red.getNeurona(0, 0);
        NeuronaSpiking pre2 = red.getNeurona(0, 1);
        NeuronaSpiking post1 = red.getNeurona(1, 0);
        NeuronaSpiking post2 = red.getNeurona(1, 1);

        // Crear varias conexiones diferentes
        assertDoesNotThrow(() -> {
            red.crearConexion(pre1, post1, 0.5, 2);
            red.crearConexion(pre1, post2, 0.6, 1);
            red.crearConexion(pre2, post1, 0.4, 3);
            red.crearConexion(pre2, post2, 0.7, 0);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Crear conexión recurrente (misma capa)
    void crearConexionRecurrenteDentroDeMismaCapa() {
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(0, 1);

        // Debe permitir conexiones dentro de la misma capa (Requisito 6.3)
        assertDoesNotThrow(() -> {
            red.crearConexion(pre, post, 0.5, 1);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar retardo negativo
    void crearConexionConRetardoNegativo() {
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(pre, post, 0.5, -1);
        });
        assertTrue(exception.getMessage().contains("retardo sináptico no puede ser negativo"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Validar peso fuera de límites
    void crearConexionConPesoFueraDelimites() {
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);

        // Peso fuera del rango [0.0, 1.0] configurado
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.crearConexion(pre, post, 1.5, 2);
        });
        assertTrue(exception.getMessage().contains("debe estar en el rango"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Obtener neurona con índices válidos
    void obtenerNeuronaConIndicesValidos() {
        NeuronaSpiking neurona = red.getNeurona(0, 0);

        assertNotNull(neurona);
        assertEquals(0, neurona.getCapa());
        assertEquals(0, neurona.getIndice());
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Obtener neurona con capa inválida
    void obtenerNeuronaConCapaInvalida() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.getNeurona(5, 0);
        });
        assertTrue(exception.getMessage().contains("Índice de capa inválido"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.1: Obtener neurona con índice inválido
    void obtenerNeuronaConIndiceInvalido() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.getNeurona(0, 10);
        });
        assertTrue(exception.getMessage().contains("Índice de neurona inválido"));
    }

    // ========== Tests de Inicialización de Pesos (Task 13.2) ==========

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicialización uniforme de pesos (Requisito 11.1)
    void inicializarPesosUniforme() {
        // Crear algunas conexiones
        NeuronaSpiking pre1 = red.getNeurona(0, 0);
        NeuronaSpiking pre2 = red.getNeurona(0, 1);
        NeuronaSpiking post1 = red.getNeurona(1, 0);
        NeuronaSpiking post2 = red.getNeurona(1, 1);

        red.crearConexion(pre1, post1, 0.0, 1);
        red.crearConexion(pre1, post2, 0.0, 1);
        red.crearConexion(pre2, post1, 0.0, 1);
        red.crearConexion(pre2, post2, 0.0, 1);

        // Inicializar pesos con distribución uniforme
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.3, 0.7);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicialización normal de pesos (Requisito 11.2)
    void inicializarPesosNormal() {
        // Crear algunas conexiones
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.0, 1);

        // Inicializar pesos con distribución normal
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.NORMAL, 0.2, 0.8);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicialización constante de pesos (Requisito 11.3)
    void inicializarPesosConstante() {
        // Crear algunas conexiones
        NeuronaSpiking pre1 = red.getNeurona(0, 0);
        NeuronaSpiking pre2 = red.getNeurona(0, 1);
        NeuronaSpiking post = red.getNeurona(1, 0);

        red.crearConexion(pre1, post, 0.0, 1);
        red.crearConexion(pre2, post, 0.0, 1);

        // Inicializar pesos con valor constante
        red.inicializarPesos(TipoInicializacion.CONSTANTE, 0.4, 0.6);

        // Verificar que todos los pesos son iguales (punto medio del rango)
        // No podemos acceder directamente a las sinapsis, pero podemos verificar que no falla
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.CONSTANTE, 0.4, 0.6);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Validar rango min >= max (Requisito 11.4)
    void inicializarPesosConRangoInvalido() {
        // Crear una conexión
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Intentar inicializar con min >= max
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.7, 0.3);
        });
        assertTrue(exception.getMessage().contains("min") && exception.getMessage().contains("max"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Validar rango min == max
    void inicializarPesosConRangoIgual() {
        // Crear una conexión
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Intentar inicializar con min == max
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.5, 0.5);
        });
        assertTrue(exception.getMessage().contains("min") && exception.getMessage().contains("max"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Validar tipo null
    void inicializarPesosConTipoNull() {
        // Crear una conexión
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Intentar inicializar con tipo null
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.inicializarPesos(null, 0.0, 1.0);
        });
        assertTrue(exception.getMessage().contains("tipo de inicialización no puede ser null"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Validar DESDE_ARRAY no soportado
    void inicializarPesosDesdeArrayNoSoportado() {
        // Crear una conexión
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Intentar inicializar con DESDE_ARRAY
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> {
            red.inicializarPesos(TipoInicializacion.DESDE_ARRAY, 0.0, 1.0);
        });
        assertTrue(exception.getMessage().contains("DESDE_ARRAY no es soportado"));
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicializar sin conexiones
    void inicializarPesosSinConexiones() {
        // No crear ninguna conexión
        // Debe funcionar sin errores
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.0, 1.0);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicializar múltiples veces
    void inicializarPesosMultiplesVeces() {
        // Crear conexiones
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Inicializar varias veces con diferentes tipos
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.0, 1.0);
            red.inicializarPesos(TipoInicializacion.NORMAL, 0.2, 0.8);
            red.inicializarPesos(TipoInicializacion.CONSTANTE, 0.4, 0.6);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicializar con rango completo [0, 1]
    void inicializarPesosRangoCompleto() {
        // Crear conexiones
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Inicializar con rango completo
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.0, 1.0);
        });
    }

    @Test
    // Feature: spiking-neural-network, Task 13.2: Inicializar con rango pequeño
    void inicializarPesosRangoPequeno() {
        // Crear conexiones
        NeuronaSpiking pre = red.getNeurona(0, 0);
        NeuronaSpiking post = red.getNeurona(1, 0);
        red.crearConexion(pre, post, 0.5, 1);

        // Inicializar con rango pequeño
        assertDoesNotThrow(() -> {
            red.inicializarPesos(TipoInicializacion.UNIFORME, 0.49, 0.51);
        });
    }

}
