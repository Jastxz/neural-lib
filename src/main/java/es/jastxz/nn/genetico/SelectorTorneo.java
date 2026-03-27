package es.jastxz.nn.genetico;

import java.util.*;

/**
 * Selección de padres por torneo con diversidad garantizada.
 *
 * <p>Implementa selección por torneo de tamaño k configurable, garantizando
 * que al menos un porcentaje mínimo de padres seleccionados provengan de
 * individuos no-élite para mantener diversidad genética.</p>
 *
 * <p>El algoritmo:</p>
 * <ol>
 *   <li>Ordena la población por fitness descendente</li>
 *   <li>Identifica élites (primeros numElites individuos)</li>
 *   <li>Calcula parejas necesarias y parejas no-élite mínimas</li>
 *   <li>Selecciona parejas no-élite del subconjunto no-élite</li>
 *   <li>Selecciona parejas restantes por torneo general</li>
 * </ol>
 */
public class SelectorTorneo {

    private static final int MAX_REINTENTOS = 100;

    private final int tamañoTorneo;
    private final int numElites;
    private final double porcentajeNoElite;
    private final Random random;

    /**
     * @param tamañoTorneo      número de individuos por torneo (default 3)
     * @param numElites         número de individuos élite a preservar (default 2)
     * @param porcentajeNoElite fracción mínima de padres no-élite (default 0.15)
     * @param random            generador de números aleatorios
     */
    public SelectorTorneo(int tamañoTorneo, int numElites,
                          double porcentajeNoElite, Random random) {
        this.tamañoTorneo = tamañoTorneo;
        this.numElites = numElites;
        this.porcentajeNoElite = porcentajeNoElite;
        this.random = random;
    }

    /**
     * Selecciona parejas de padres para cruce.
     *
     * @param poblacion lista de individuos evaluados
     * @return lista de parejas de padres
     */
    public List<Pareja> seleccionarParejas(List<Individuo> poblacion) {
        // 1. Ordenar población por fitness descendente
        List<Individuo> ordenada = new ArrayList<>(poblacion);
        ordenada.sort(Comparator.naturalOrder());

        // 2. Identificar élites
        int numElitesEfectivo = Math.min(numElites, ordenada.size());
        Set<Individuo> elites = new LinkedHashSet<>(ordenada.subList(0, numElitesEfectivo));
        List<Individuo> noElites = new ArrayList<>();
        for (Individuo ind : ordenada) {
            if (!elites.contains(ind)) {
                noElites.add(ind);
            }
        }

        // 3. Calcular número de parejas
        int numParejas = (ordenada.size() - numElitesEfectivo) / 2;
        if (numParejas <= 0) {
            return List.of();
        }

        // 4. Calcular parejas no-élite mínimas
        int parejasNoElite = (int) Math.ceil(numParejas * porcentajeNoElite);

        List<Pareja> parejas = new ArrayList<>();

        // 5. Seleccionar parejas no-élite del subconjunto no-élite
        for (int i = 0; i < parejasNoElite && noElites.size() >= 2; i++) {
            Individuo padre1 = noElites.get(random.nextInt(noElites.size()));
            Individuo padre2;
            do {
                padre2 = noElites.get(random.nextInt(noElites.size()));
            } while (padre2 == padre1);
            parejas.add(new Pareja(padre1, padre2));
        }

        // 6. Seleccionar parejas restantes por torneo
        int parejasRestantes = numParejas - parejas.size();
        for (int i = 0; i < parejasRestantes; i++) {
            Individuo padre1 = ejecutarTorneo(ordenada);
            Individuo padre2 = padre1;
            int intentos = 0;
            while (padre2 == padre1 && intentos < MAX_REINTENTOS) {
                padre2 = ejecutarTorneo(ordenada);
                intentos++;
            }
            // If tournament is deterministic (k=N), pick a different individual
            if (padre2 == padre1) {
                padre2 = seleccionarDistinto(ordenada, padre1);
            }
            parejas.add(new Pareja(padre1, padre2));
        }

        return parejas;
    }

    /**
     * Ejecuta un torneo: selecciona k individuos aleatorios y retorna el de mayor fitness.
     *
     * @param poblacion lista de individuos ordenada por fitness
     * @return ganador del torneo (individuo con mayor fitness entre los k seleccionados)
     */
    private Individuo ejecutarTorneo(List<Individuo> poblacion) {
        int k = Math.min(tamañoTorneo, poblacion.size());
        Individuo mejor = null;
        Set<Integer> indices = new HashSet<>();
        while (indices.size() < k) {
            indices.add(random.nextInt(poblacion.size()));
        }
        for (int idx : indices) {
            Individuo candidato = poblacion.get(idx);
            if (mejor == null || candidato.fitness() > mejor.fitness()) {
                mejor = candidato;
            }
        }
        return mejor;
    }

    /**
     * Selects a different individual from the population when tournament
     * is deterministic (k = population size).
     */
    private Individuo seleccionarDistinto(List<Individuo> poblacion, Individuo excluido) {
        // Population is sorted by fitness descending; pick the second-best
        for (Individuo ind : poblacion) {
            if (ind != excluido) {
                return ind;
            }
        }
        // Fallback: should not happen with population size >= 4
        return poblacion.get(1);
    }

    /**
     * Pareja de padres seleccionados para cruce.
     *
     * @param padre1 primer padre
     * @param padre2 segundo padre
     */
    public record Pareja(Individuo padre1, Individuo padre2) {

        /**
         * Retorna el padre con mayor fitness.
         *
         * @return padre con mayor fitness
         */
        public Individuo mejorPadre() {
            return padre1.fitness() >= padre2.fitness() ? padre1 : padre2;
        }

        /**
         * Retorna el padre con menor fitness.
         *
         * @return padre con menor fitness
         */
        public Individuo peorPadre() {
            return padre1.fitness() >= padre2.fitness() ? padre2 : padre1;
        }
    }
}
