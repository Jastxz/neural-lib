package es.jastxz.nn.genetico;

import java.util.List;

/**
 * Resultado del proceso evolutivo del algoritmo genético.
 *
 * <p>Contiene el mejor individuo encontrado, estadísticas de la evolución
 * y la configuración utilizada.</p>
 *
 * @param mejorIndividuo   mejor individuo encontrado en toda la evolución
 * @param generacionMejor  generación donde se encontró el mejor individuo
 * @param totalGeneraciones número total de generaciones ejecutadas
 * @param motivoParada     razón de parada: "max_generaciones" o "estancamiento"
 * @param historial        estadísticas por generación
 * @param configuracionAG  configuración del AG utilizada
 */
public record InformeEvolucion(
    Individuo mejorIndividuo,
    int generacionMejor,
    int totalGeneraciones,
    String motivoParada,
    List<EstadisticaGeneracion> historial,
    ConfiguracionAG configuracionAG
) {

    /**
     * Imprime el informe por System.out en formato legible.
     *
     * <p>Incluye: mejor individuo (fitness y generación), motivo de parada,
     * total de generaciones, historial de fitness por generación,
     * configuración del AG y límite topológico.</p>
     */
    public void imprimir() {
        String separador = "=".repeat(60);
        System.out.println(separador);
        System.out.println("         INFORME DE EVOLUCIÓN DEL AG");
        System.out.println(separador);

        // Mejor individuo
        System.out.println();
        System.out.println("--- Mejor Individuo ---");
        System.out.printf("  Fitness:              %.6f%n", mejorIndividuo.fitness());
        System.out.printf("  Encontrado en gen.:   %d%n", generacionMejor);

        // Motivo de parada y generaciones
        System.out.println();
        System.out.println("--- Ejecución ---");
        System.out.printf("  Total generaciones:   %d%n", totalGeneraciones);
        System.out.printf("  Motivo de parada:     %s%n", motivoParada);

        // Historial de fitness por generación
        System.out.println();
        System.out.println("--- Historial de Fitness por Generación ---");
        System.out.printf("  %-6s  %-12s  %-12s  %-12s%n",
                "Gen", "Mejor", "Promedio", "Peor");
        System.out.println("  " + "-".repeat(48));
        for (EstadisticaGeneracion est : historial) {
            System.out.printf("  %-6d  %-12.6f  %-12.6f  %-12.6f%n",
                    est.numero(), est.mejorFitness(),
                    est.fitnessPromedio(), est.peorFitness());
        }

        // Configuración del AG
        System.out.println();
        System.out.println("--- Configuración del AG ---");
        System.out.printf("  Tamaño población:     %d%n", configuracionAG.tamañoPoblacion());
        System.out.printf("  Máx. generaciones:    %d%n", configuracionAG.maxGeneraciones());
        System.out.printf("  Gen. estancamiento:   %d%n", configuracionAG.generacionesEstancamiento());
        System.out.printf("  Prob. cruce:          %.2f%n", configuracionAG.probabilidadCruce());
        System.out.printf("  Puntos de corte:      %d%n", configuracionAG.puntosCorte());
        System.out.printf("  Prob. mutación:       %.2f%n", configuracionAG.probabilidadMutacion());
        System.out.printf("  Tamaño torneo:        %d%n", configuracionAG.tamañoTorneo());
        System.out.printf("  Núm. élites:          %d%n", configuracionAG.numElites());
        System.out.printf("  %% no-élite:           %.2f%n", configuracionAG.porcentajeNoElite());
        System.out.printf("  Peso precisión:       %.2f%n", configuracionAG.pesoPrecision());
        System.out.printf("  Peso energía:         %.2f%n", configuracionAG.pesoEnergia());
        System.out.printf("  Peso tamaño:          %.2f%n", configuracionAG.pesoTamanio());
        System.out.printf("  Límite topológico:    %d%n", configuracionAG.limiteTopologico());
        System.out.printf("  Épocas benchmark:     %d%n", configuracionAG.epocasBenchmark());
        System.out.printf("  Repeticiones bench.:  %d%n", configuracionAG.repeticionesBenchmark());
        System.out.printf("  Semilla:              %d%n", configuracionAG.semilla());

        System.out.println();
        System.out.println(separador);
    }
}
