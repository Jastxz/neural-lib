package es.jastxz.nn.nube;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests de integración para el flujo completo del Método de la Nube Aleatoria
 * usando el dataset XOR como problema de referencia.
 *
 * <p>Valida: Requisitos 5.4, 8.1, 8.2, 8.3, 8.4</p>
 */
class IntegracionNubeTest {

    // XOR dataset con codificación one-hot: [not XOR, XOR]
    private static final double[][] XOR_ENTRADAS = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    private static final double[][] XOR_OBJETIVOS = {
        {1, 0}, {0, 1}, {0, 1}, {1, 0}
    };

    /**
     * Flujo completo: configurar → ejecutar → verificar informe coherente.
     * Usa umbral bajo y nube pequeña para ejecución rápida.
     */
    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void flujoCompletoConXOR() {
        ConfiguracionNube config = new ConfiguracionNubeBuilder()
            .tamañoNube(5)
            .topologiaInicial(2, 4, 2)
            .umbralAcierto(0.25)
            .neuronasEliminar(1)
            .epocasRefinamiento(500)
            .tasaAprendizaje(0.5)
            .semilla(42L)
            .build();

        MotorNube motor = new MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS);
        InformeNube informe = motor.ejecutar();

        // Verificar estructura del informe
        assertNotNull(informe);
        assertTrue(informe.totalRedesEvaluadas() > 0,
            "Debe haber evaluado al menos una red");
        assertTrue(informe.tiempoEjecucionMs() >= 0,
            "El tiempo de ejecución no puede ser negativo");

        if (informe.exitoso()) {
            assertNotNull(informe.mejorRed(),
                "Si exitoso, debe haber una mejor red");
            assertNotNull(informe.topologiaFinal(),
                "Si exitoso, debe haber topología final");
            assertTrue(informe.precision() > 0.0,
                "Si exitoso, la precisión debe ser > 0");
            // La topología debe empezar con 2 (entrada) y terminar con 2 (salida)
            assertEquals(2, informe.topologiaFinal()[0],
                "La capa de entrada debe tener 2 neuronas");
            assertEquals(2, informe.topologiaFinal()[informe.topologiaFinal().length - 1],
                "La capa de salida debe tener 2 neuronas");
        }
    }

    /**
     * Verifica que con un umbral imposible (1.0), ninguna red aleatoria pasa
     * y el informe indica fracaso.
     */
    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void flujoSinRedViable() {
        ConfiguracionNube config = new ConfiguracionNubeBuilder()
            .tamañoNube(3)
            .topologiaInicial(2, 3, 2)
            .umbralAcierto(1.0)
            .neuronasEliminar(1)
            .epocasRefinamiento(10)
            .tasaAprendizaje(0.1)
            .semilla(42L)
            .build();

        MotorNube motor = new MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS);
        InformeNube informe = motor.ejecutar();

        assertNotNull(informe);
        assertFalse(informe.exitoso(),
            "Con umbral 1.0 (comparación estricta >), ninguna red debería pasar");
        assertNull(informe.mejorRed(),
            "Sin red viable, mejorRed debe ser null");
        assertNull(informe.topologiaFinal(),
            "Sin red viable, topologiaFinal debe ser null");
        assertEquals(0.0, informe.precision(),
            "Sin red viable, la precisión debe ser 0.0");
    }

    /**
     * Verifica que una política de eliminación personalizada se aplica
     * correctamente en lugar de la política por defecto (Requisito 5.4).
     */
    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void politicaPersonalizada() {
        // Política custom: siempre elimina de la primera capa oculta
        PoliticaEliminacion customPolicy = (topologia, n) -> {
            int[] nueva = Arrays.copyOf(topologia, topologia.length);
            // Buscar primera capa oculta con neuronas > 0
            for (int i = 1; i < nueva.length - 1; i++) {
                if (nueva[i] > 0) {
                    nueva[i] = Math.max(0, nueva[i] - n);
                    break;
                }
            }
            // Si todas las capas ocultas quedaron en 0, retornar null
            for (int i = 1; i < nueva.length - 1; i++) {
                if (nueva[i] > 0) return nueva;
            }
            return null;
        };

        ConfiguracionNube config = new ConfiguracionNubeBuilder()
            .tamañoNube(3)
            .topologiaInicial(2, 4, 2)
            .umbralAcierto(0.25)
            .neuronasEliminar(1)
            .epocasRefinamiento(100)
            .tasaAprendizaje(0.5)
            .semilla(42L)
            .build();

        MotorNube motor = new MotorNube(config, XOR_ENTRADAS, XOR_OBJETIVOS, customPolicy);
        InformeNube informe = motor.ejecutar();

        assertNotNull(informe, "El informe no debe ser null con política personalizada");
        assertTrue(informe.totalRedesEvaluadas() > 0,
            "Debe haber evaluado redes con la política personalizada");
    }

    /**
     * Verifica reproducibilidad: dos ejecuciones con la misma semilla y datos
     * producen el mismo resultado.
     */
    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void reproducibilidad() {
        ConfiguracionNube config1 = new ConfiguracionNubeBuilder()
            .tamañoNube(3)
            .topologiaInicial(2, 4, 2)
            .umbralAcierto(0.25)
            .neuronasEliminar(1)
            .epocasRefinamiento(100)
            .tasaAprendizaje(0.5)
            .semilla(42L)
            .build();

        MotorNube motor1 = new MotorNube(config1, XOR_ENTRADAS, XOR_OBJETIVOS);
        InformeNube informe1 = motor1.ejecutar();

        ConfiguracionNube config2 = new ConfiguracionNubeBuilder()
            .tamañoNube(3)
            .topologiaInicial(2, 4, 2)
            .umbralAcierto(0.25)
            .neuronasEliminar(1)
            .epocasRefinamiento(100)
            .tasaAprendizaje(0.5)
            .semilla(42L)
            .build();

        MotorNube motor2 = new MotorNube(config2, XOR_ENTRADAS, XOR_OBJETIVOS);
        InformeNube informe2 = motor2.ejecutar();

        assertEquals(informe1.exitoso(), informe2.exitoso(),
            "Ambas ejecuciones deben tener el mismo resultado de éxito");
        assertEquals(informe1.precision(), informe2.precision(), 1e-10,
            "La precisión debe ser idéntica con la misma semilla");
        assertEquals(informe1.totalReducciones(), informe2.totalReducciones(),
            "El número de reducciones debe ser idéntico");
        assertArrayEquals(informe1.topologiaFinal(), informe2.topologiaFinal(),
            "La topología final debe ser idéntica");
    }
}
