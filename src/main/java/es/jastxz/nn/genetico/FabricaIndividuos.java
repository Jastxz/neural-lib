package es.jastxz.nn.genetico;

import es.jastxz.nn.spiking.*;

import java.util.*;

/**
 * Fábrica responsable de generar individuos aleatorios válidos y convertir
 * cromosomas a {@link ConfiguracionRed}.
 *
 * <p>Garantiza que todo individuo producido cumple:</p>
 * <ul>
 *   <li>Todos los genes dentro de sus rangos definidos</li>
 *   <li>umbralDisparo &gt; potencialReposo</li>
 *   <li>neuronasTotal ≤ limiteTopologico</li>
 * </ul>
 */
public class FabricaIndividuos {

    private final int dimensionEntrada;
    private final int dimensionSalida;
    private final int limiteTopologico;
    private final Random random;

    /**
     * @param dimensionEntrada  número de neuronas de entrada
     * @param dimensionSalida   número de neuronas de salida
     * @param limiteTopologico  máximo de neuronas totales permitidas
     * @param random            generador de números aleatorios
     */
    public FabricaIndividuos(int dimensionEntrada, int dimensionSalida,
                             int limiteTopologico, Random random) {
        this.dimensionEntrada = dimensionEntrada;
        this.dimensionSalida = dimensionSalida;
        this.limiteTopologico = limiteTopologico;
        this.random = random;
    }

    /**
     * Genera un individuo aleatorio con genes dentro de los rangos definidos.
     * Garantiza: umbralDisparo &gt; potencialReposo y neuronasTotal ≤ limiteTopologico.
     *
     * @return individuo sin evaluar con configuración válida
     */
    public Individuo generarAleatorio() {
        // --- TOPOLOGIA ---
        int capasOcultas = randomInt(1, 10);
        List<Gen<?>> genesTopologia = new ArrayList<>();
        genesTopologia.add(new Gen.GenEntero("capasOcultas", capasOcultas, 1, 10));
        for (int i = 0; i < capasOcultas; i++) {
            int neuronas = randomInt(1, 512);
            genesTopologia.add(new Gen.GenEntero("neuronasCapa_" + i, neuronas, 1, 512));
        }

        // --- LIF (con restricción umbralDisparo > potencialReposo) ---
        double umbralDisparo;
        double potencialReposo;
        do {
            umbralDisparo = randomDouble(-60.0, -40.0);
            potencialReposo = randomDouble(-80.0, -60.0);
        } while (umbralDisparo <= potencialReposo);

        double constanteDecaimiento = randomDouble(5.0, 50.0);
        int duracionRefractario = randomInt(1, 10);
        List<Gen<?>> genesLIF = List.of(
                new Gen.GenReal("umbralDisparo", umbralDisparo, -60.0, -40.0),
                new Gen.GenReal("potencialReposo", potencialReposo, -80.0, -60.0),
                new Gen.GenReal("constanteDecaimiento", constanteDecaimiento, 5.0, 50.0),
                new Gen.GenEntero("duracionRefractario", duracionRefractario, 1, 10)
        );

        // --- STDP ---
        List<Gen<?>> genesSTDP = List.of(
                new Gen.GenReal("amplitudLTP", randomDouble(0.001, 0.1), 0.001, 0.1),
                new Gen.GenReal("amplitudLTD", randomDouble(0.001, 0.1), 0.001, 0.1),
                new Gen.GenReal("tauLTP", randomDouble(5.0, 50.0), 5.0, 50.0),
                new Gen.GenReal("tauLTD", randomDouble(5.0, 50.0), 5.0, 50.0)
        );

        // --- CODIFICACION ---
        ModoCodificacion[] modos = ModoCodificacion.values();
        ModoCodificacion modo = modos[random.nextInt(modos.length)];
        List<Gen<?>> genesCodificacion = List.of(
                new Gen.GenReal("frecuenciaMaxima", randomDouble(10.0, 500.0), 10.0, 500.0),
                new Gen.GenEnum<>("modoCodificacion", modo, ModoCodificacion.class),
                new Gen.GenEntero("ventanaDecodificacion", randomInt(10, 200), 10, 200)
        );

        // --- REGULACION ---
        boolean homeostasis = random.nextBoolean();
        boolean inhibicion = random.nextBoolean();
        List<Gen<?>> genesRegulacion = List.of(
                new Gen.GenBooleano("homeostasisActiva", homeostasis),
                new Gen.GenReal("tasaDisparoObjetivo", randomDouble(1.0, 50.0), 1.0, 50.0),
                new Gen.GenReal("tasaAjusteHomeostasis", randomDouble(0.001, 0.1), 0.001, 0.1),
                new Gen.GenBooleano("inhibicionLateralActiva", inhibicion),
                new Gen.GenEntero("radioInhibicion", randomInt(1, 5), 1, 5),
                new Gen.GenReal("fuerzaInhibicion", randomDouble(0.1, 2.0), 0.1, 2.0)
        );

        // --- COMPETICION (WTA) ---
        boolean wtaActivo = random.nextBoolean();
        boolean wtaCapaSalida = random.nextBoolean();
        boolean wtaCapasOcultas = random.nextBoolean();
        List<Gen<?>> genesCompeticion = List.of(
                new Gen.GenBooleano("wtaActivo", wtaActivo),
                new Gen.GenBooleano("wtaCapaSalida", wtaCapaSalida),
                new Gen.GenBooleano("wtaCapasOcultas", wtaCapasOcultas),
                new Gen.GenEntero("radioWTA", randomInt(0, 10), 0, 10),
                new Gen.GenReal("fuerzaWTA", randomDouble(0.5, 5.0), 0.5, 5.0),
                new Gen.GenReal("umbralActivacionWTA", randomDouble(0.0, 0.5), 0.0, 0.5)
        );

        // Ensamblar cromosoma
        var bloques = new EnumMap<BloqueFuncional, List<Gen<?>>>(BloqueFuncional.class);
        bloques.put(BloqueFuncional.TOPOLOGIA, genesTopologia);
        bloques.put(BloqueFuncional.LIF, genesLIF);
        bloques.put(BloqueFuncional.STDP, genesSTDP);
        bloques.put(BloqueFuncional.CODIFICACION, genesCodificacion);
        bloques.put(BloqueFuncional.REGULACION, genesRegulacion);
        bloques.put(BloqueFuncional.COMPETICION, genesCompeticion);
        Cromosoma cromosoma = new Cromosoma(bloques);

        // Reparar topología si excede el límite
        if (cromosoma.neuronasTotal(dimensionEntrada, dimensionSalida) > limiteTopologico) {
            cromosoma = repararTopologia(cromosoma);
        }

        ConfiguracionRed config = construirConfiguracion(cromosoma);
        return Individuo.sinEvaluar(cromosoma, config);
    }

    /**
     * Convierte un Cromosoma a ConfiguracionRed usando ConfiguracionRedBuilder.
     *
     * @param cromosoma cromosoma a convertir
     * @return configuración de red construida
     * @throws IllegalArgumentException si la configuración resultante es inválida
     */
    public ConfiguracionRed construirConfiguracion(Cromosoma cromosoma) {
        List<Gen<?>> topologia = cromosoma.genesDeBloque(BloqueFuncional.TOPOLOGIA);
        List<Gen<?>> lif = cromosoma.genesDeBloque(BloqueFuncional.LIF);
        List<Gen<?>> stdp = cromosoma.genesDeBloque(BloqueFuncional.STDP);
        List<Gen<?>> codificacion = cromosoma.genesDeBloque(BloqueFuncional.CODIFICACION);
        List<Gen<?>> regulacion = cromosoma.genesDeBloque(BloqueFuncional.REGULACION);
        List<Gen<?>> competicion = cromosoma.genesDeBloque(BloqueFuncional.COMPETICION);

        // Topología: entrada + neuronasPorCapa[0..n-1] + salida
        int capasOcultas = ((Number) topologia.get(0).valor()).intValue();
        int[] capas = new int[capasOcultas + 2];
        capas[0] = dimensionEntrada;
        for (int i = 0; i < capasOcultas; i++) {
            capas[i + 1] = ((Number) topologia.get(i + 1).valor()).intValue();
        }
        capas[capasOcultas + 1] = dimensionSalida;

        // LIF
        double umbralDisparo = ((Number) lif.get(0).valor()).doubleValue();
        double potencialReposo = ((Number) lif.get(1).valor()).doubleValue();
        double constanteDecaimiento = ((Number) lif.get(2).valor()).doubleValue();
        int duracionRefractario = ((Number) lif.get(3).valor()).intValue();

        // STDP
        double amplitudLTP = ((Number) stdp.get(0).valor()).doubleValue();
        double amplitudLTD = ((Number) stdp.get(1).valor()).doubleValue();
        double tauLTP = ((Number) stdp.get(2).valor()).doubleValue();
        double tauLTD = ((Number) stdp.get(3).valor()).doubleValue();

        // Codificación
        double frecuenciaMaxima = ((Number) codificacion.get(0).valor()).doubleValue();
        @SuppressWarnings("unchecked")
        ModoCodificacion modoCodificacion = ((Gen.GenEnum<ModoCodificacion>) codificacion.get(1)).valor();
        int ventanaDecodificacion = ((Number) codificacion.get(2).valor()).intValue();

        // Regulación
        boolean homeostasisActiva = (Boolean) regulacion.get(0).valor();
        double tasaDisparoObjetivo = ((Number) regulacion.get(1).valor()).doubleValue();
        double tasaAjusteHomeostasis = ((Number) regulacion.get(2).valor()).doubleValue();
        boolean inhibicionLateralActiva = (Boolean) regulacion.get(3).valor();
        int radioInhibicion = ((Number) regulacion.get(4).valor()).intValue();
        double fuerzaInhibicion = ((Number) regulacion.get(5).valor()).doubleValue();

        // Competición (WTA) — valores por defecto si el bloque no existe
        boolean wtaActivo = false;
        boolean wtaCapaSalida = false;
        boolean wtaCapasOcultas = false;
        int radioWTA = 0;
        double fuerzaWTA = 2.0;
        double umbralActivacionWTA = 0.1;
        if (!competicion.isEmpty()) {
            wtaActivo = (Boolean) competicion.get(0).valor();
            wtaCapaSalida = (Boolean) competicion.get(1).valor();
            wtaCapasOcultas = (Boolean) competicion.get(2).valor();
            radioWTA = ((Number) competicion.get(3).valor()).intValue();
            fuerzaWTA = ((Number) competicion.get(4).valor()).doubleValue();
            umbralActivacionWTA = ((Number) competicion.get(5).valor()).doubleValue();
        }

        return new ConfiguracionRedBuilder()
                .topologia(capas)
                .parametrosLIF(umbralDisparo, potencialReposo, constanteDecaimiento, duracionRefractario)
                .parametrosSTDP(amplitudLTP, amplitudLTD, tauLTP, tauLTD)
                .parametrosCodificacion(frecuenciaMaxima, modoCodificacion, ventanaDecodificacion)
                .homeostasis(homeostasisActiva, tasaDisparoObjetivo, tasaAjusteHomeostasis)
                .inhibicionLateral(inhibicionLateralActiva, radioInhibicion, fuerzaInhibicion)
                .inicializacionPesos(TipoInicializacion.UNIFORME, 0.3, 3.0)
                .parametrosNormalizacion(TipoNormalizacion.L2, 1.0)
                .retardos(1, 1)
                .duracionTimestep(1.0)
                .wta(wtaActivo, wtaCapaSalida, wtaCapasOcultas,
                        radioWTA, fuerzaWTA, umbralActivacionWTA)
                .build();
    }

    /**
     * Repara un cromosoma que excede el límite topológico reduciendo
     * proporcionalmente las neuronas por capa oculta.
     *
     * <p>Algoritmo:</p>
     * <ol>
     *   <li>Calcular neuronasOcultas = sum(neuronasPorCapa)</li>
     *   <li>maxOcultas = limiteTopologico - entrada - salida</li>
     *   <li>Si neuronasOcultas &gt; maxOcultas:
     *       factor = maxOcultas / neuronasOcultas;
     *       para cada capa: nuevasNeuronas = max(1, round(neuronas * factor))</li>
     *   <li>Ajustar última capa si la suma aún excede el límite</li>
     * </ol>
     *
     * @param cromosoma cromosoma a reparar
     * @return nuevo cromosoma con topología reparada
     */
    public Cromosoma repararTopologia(Cromosoma cromosoma) {
        List<Gen<?>> topologia = cromosoma.genesDeBloque(BloqueFuncional.TOPOLOGIA);
        int capasOcultas = ((Number) topologia.get(0).valor()).intValue();

        int neuronasOcultas = 0;
        for (int i = 1; i <= capasOcultas; i++) {
            neuronasOcultas += ((Number) topologia.get(i).valor()).intValue();
        }

        int maxOcultas = limiteTopologico - dimensionEntrada - dimensionSalida;
        if (maxOcultas < capasOcultas) {
            maxOcultas = capasOcultas; // al menos 1 neurona por capa
        }

        if (neuronasOcultas <= maxOcultas) {
            return cromosoma;
        }

        double factor = (double) maxOcultas / neuronasOcultas;
        List<Gen<?>> nuevaTopologia = new ArrayList<>();
        nuevaTopologia.add(topologia.get(0)); // capasOcultas sin cambio

        int sumaReparada = 0;
        for (int i = 1; i <= capasOcultas; i++) {
            int original = ((Number) topologia.get(i).valor()).intValue();
            int nuevas = Math.max(1, (int) Math.round(original * factor));
            sumaReparada += nuevas;
            nuevaTopologia.add(new Gen.GenEntero(
                    topologia.get(i).nombre(), nuevas, 1, 512));
        }

        // Ajustar capas desde la última si la suma aún excede el límite
        int exceso = sumaReparada - maxOcultas;
        for (int i = capasOcultas; i >= 1 && exceso > 0; i--) {
            int valor = ((Number) nuevaTopologia.get(i).valor()).intValue();
            int reduccion = Math.min(exceso, valor - 1); // mantener al menos 1
            if (reduccion > 0) {
                nuevaTopologia.set(i, new Gen.GenEntero(
                        nuevaTopologia.get(i).nombre(), valor - reduccion, 1, 512));
                exceso -= reduccion;
            }
        }

        return cromosoma.conBloque(BloqueFuncional.TOPOLOGIA, nuevaTopologia);
    }

    /**
     * Genera un individuo aleatorio con topología fija.
     *
     * <p>Los hiperparámetros (LIF, STDP, codificación, regulación) se generan
     * aleatoriamente, pero la topología se fija a la proporcionada.</p>
     *
     * @param topologiaFija array con tamaños de cada capa (entrada + ocultas + salida)
     * @return individuo sin evaluar con topología fija y parámetros aleatorios
     */
    public Individuo generarConTopologiaFija(int[] topologiaFija) {
        // Generar un individuo aleatorio normal
        Individuo base = generarAleatorio();

        // Reemplazar el bloque TOPOLOGIA con la topología fija
        int capasOcultas = topologiaFija.length - 2; // sin entrada ni salida
        List<Gen<?>> genesTopologia = new ArrayList<>();
        genesTopologia.add(new Gen.GenEntero("capasOcultas", capasOcultas, 1, 10));
        for (int i = 0; i < capasOcultas; i++) {
            int neuronas = topologiaFija[i + 1]; // saltar capa de entrada
            genesTopologia.add(new Gen.GenEntero("neuronasCapa_" + i, neuronas, 1, 512));
        }

        Cromosoma cromosoma = base.cromosoma().conBloque(BloqueFuncional.TOPOLOGIA, genesTopologia);
        ConfiguracionRed config = construirConfiguracion(cromosoma);
        return Individuo.sinEvaluar(cromosoma, config);
    }

    // ==================== Helpers privados ====================

    private int randomInt(int min, int max) {
        return min + random.nextInt(max - min + 1);
    }

    private double randomDouble(double min, double max) {
        return min + random.nextDouble() * (max - min);
    }
}
