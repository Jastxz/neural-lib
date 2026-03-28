package es.jastxz.nn.genetico;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests unitarios para {@link ConfiguracionAG} y {@link ConfiguracionAGBuilder}.
 *
 * <p>Verifica valores por defecto del builder y validaciones del constructor.</p>
 */
class ConfiguracionAGTest {

    // ==================== Valores por defecto ====================

    @Test
    void builderProduceValoresPorDefecto() {
        ConfiguracionAG config = new ConfiguracionAGBuilder().build();

        assertEquals(30, config.tamañoPoblacion());
        assertEquals(50, config.maxGeneraciones());
        assertEquals(10, config.generacionesEstancamiento());
        assertEquals(0.8, config.probabilidadCruce());
        assertEquals(2, config.puntosCorte());
        assertEquals(0.1, config.probabilidadMutacion());
        assertEquals(3, config.tamañoTorneo());
        assertEquals(2, config.numElites());
        assertEquals(0.15, config.porcentajeNoElite());
        assertEquals(0.5, config.pesoPrecision());
        assertEquals(0.3, config.pesoEnergia());
        assertEquals(0.2, config.pesoTamanio());
        assertEquals(512, config.limiteTopologico());
        assertEquals(10, config.epocasBenchmark());
        assertEquals(1, config.repeticionesBenchmark());
    }

    // ==================== Validación de pesos ====================

    @Test
    void pesosQueNoSumanUnoLanzanExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .pesosFitness(0.5, 0.5, 0.5)
                        .build());
    }

    @Test
    void pesosQueSumanUnoSonValidos() {
        assertDoesNotThrow(() ->
                new ConfiguracionAGBuilder()
                        .pesosFitness(0.6, 0.2, 0.2)
                        .build());
    }

    @Test
    void pesosConToleranciaAceptable() {
        // Pesos que suman 1.0 dentro de tolerancia 1e-6
        assertDoesNotThrow(() ->
                new ConfiguracionAGBuilder()
                        .pesosFitness(0.33, 0.33, 0.34)
                        .build());
    }

    // ==================== Validación de torneo ====================

    @Test
    void torneoMayorQuePoblacionLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .tamañoPoblacion(10)
                        .tamañoTorneo(11)
                        .build());
    }

    @Test
    void torneoIgualAPoblacionEsValido() {
        assertDoesNotThrow(() ->
                new ConfiguracionAGBuilder()
                        .tamañoPoblacion(10)
                        .tamañoTorneo(10)
                        .build());
    }

    // ==================== Validación de puntosCorte ====================

    @Test
    void puntosCorteEnCeroLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .puntosCorte(0)
                        .build());
    }

    @Test
    void puntosCorteEnCincoLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .puntosCorte(5)
                        .build());
    }

    @Test
    void puntosCorteEnRangoValidoNoLanzaExcepcion() {
        for (int p = 1; p <= 4; p++) {
            int puntos = p;
            assertDoesNotThrow(() ->
                    new ConfiguracionAGBuilder()
                            .puntosCorte(puntos)
                            .build());
        }
    }

    // ==================== Validación de numElites ====================

    @Test
    void numElitesIgualAPoblacionLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .tamañoPoblacion(10)
                        .numElites(10)
                        .build());
    }

    @Test
    void numElitesMayorQuePoblacionLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .tamañoPoblacion(10)
                        .numElites(15)
                        .build());
    }

    // ==================== Validación de limiteTopologico ====================

    @Test
    void limiteTopologicoCeroLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .limiteTopologico(0)
                        .build());
    }

    @Test
    void limiteTopologicoNegativoLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .limiteTopologico(-1)
                        .build());
    }

    @Test
    void limiteTopologicoPositivoEsValido() {
        assertDoesNotThrow(() ->
                new ConfiguracionAGBuilder()
                        .limiteTopologico(256)
                        .build());
    }

    // ==================== Validación de porcentajeNoElite ====================

    @Test
    void porcentajeNoEliteNegativoLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .porcentajeNoElite(-0.1)
                        .build());
    }

    @Test
    void porcentajeNoEliteMayorQueUnoLanzaExcepcion() {
        assertThrows(IllegalArgumentException.class, () ->
                new ConfiguracionAGBuilder()
                        .porcentajeNoElite(1.1)
                        .build());
    }

    // ==================== Builder fluido ====================

    @Test
    void builderFluidoPermiteEncadenar() {
        ConfiguracionAG config = new ConfiguracionAGBuilder()
                .tamañoPoblacion(50)
                .maxGeneraciones(100)
                .generacionesEstancamiento(20)
                .probabilidadCruce(0.9)
                .puntosCorte(3)
                .probabilidadMutacion(0.05)
                .tamañoTorneo(5)
                .numElites(3)
                .porcentajeNoElite(0.2)
                .pesosFitness(0.4, 0.3, 0.3)
                .limiteTopologico(1024)
                .epocasBenchmark(20)
                .repeticionesBenchmark(3)
                .semilla(42L)
                .build();

        assertEquals(50, config.tamañoPoblacion());
        assertEquals(100, config.maxGeneraciones());
        assertEquals(20, config.generacionesEstancamiento());
        assertEquals(0.9, config.probabilidadCruce());
        assertEquals(3, config.puntosCorte());
        assertEquals(0.05, config.probabilidadMutacion());
        assertEquals(5, config.tamañoTorneo());
        assertEquals(3, config.numElites());
        assertEquals(0.2, config.porcentajeNoElite());
        assertEquals(0.4, config.pesoPrecision());
        assertEquals(0.3, config.pesoEnergia());
        assertEquals(0.3, config.pesoTamanio());
        assertEquals(1024, config.limiteTopologico());
        assertEquals(20, config.epocasBenchmark());
        assertEquals(3, config.repeticionesBenchmark());
        assertEquals(42L, config.semilla());
    }
}
