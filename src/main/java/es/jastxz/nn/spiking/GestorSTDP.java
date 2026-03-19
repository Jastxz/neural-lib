package es.jastxz.nn.spiking;

import java.util.List;

/**
 * Gestor de plasticidad STDP (Spike-Timing-Dependent Plasticity).
 * 
 * <p>Implementa el mecanismo de aprendizaje basado en diferencias temporales entre spikes
 * pre y post-sinápticos. Aplica Long-Term Potentiation (LTP) cuando el spike pre precede
 * al post, y Long-Term Depression (LTD) cuando el spike post precede al pre.</p>
 * 
 * <p>Las fórmulas utilizadas son:</p>
 * <ul>
 *   <li>LTP (dt > 0): dw = A_LTP * exp(-dt/tau_LTP)</li>
 *   <li>LTD (dt < 0): dw = -A_LTD * exp(dt/tau_LTD)</li>
 * </ul>
 * 
 * <p>Donde dt = timestampPost - timestampPre</p>
 * 
 * @author jastxz
 * @version 1.0
 * @since 1.0
 */
public class GestorSTDP implements java.io.Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Amplitud de Long-Term Potentiation (refuerzo sináptico).
     * Debe ser mayor que 0.
     */
    private final double amplitudLTP;
    
    /**
     * Amplitud de Long-Term Depression (debilitamiento sináptico).
     * Debe ser mayor que 0.
     */
    private final double amplitudLTD;
    
    /**
     * Constante de tiempo para LTP en milisegundos.
     * Debe ser mayor que 0.
     */
    private final double tauLTP;
    
    /**
     * Constante de tiempo para LTD en milisegundos.
     * Debe ser mayor que 0.
     */
    private final double tauLTD;
    
    /**
     * Construye un nuevo gestor STDP con los parámetros especificados.
     * 
     * @param amplitudLTP amplitud de Long-Term Potentiation (debe ser > 0)
     * @param amplitudLTD amplitud de Long-Term Depression (debe ser > 0)
     * @param tauLTP constante de tiempo para LTP en ms (debe ser > 0)
     * @param tauLTD constante de tiempo para LTD en ms (debe ser > 0)
     * @throws IllegalArgumentException si algún parámetro es <= 0
     */
    public GestorSTDP(double amplitudLTP, double amplitudLTD, double tauLTP, double tauLTD) {
        if (amplitudLTP <= 0) {
            throw new IllegalArgumentException(
                "Amplitud LTP debe ser positiva, recibido: " + amplitudLTP
            );
        }
        if (amplitudLTD <= 0) {
            throw new IllegalArgumentException(
                "Amplitud LTD debe ser positiva, recibido: " + amplitudLTD
            );
        }
        if (tauLTP <= 0) {
            throw new IllegalArgumentException(
                "Tau LTP debe ser positivo, recibido: " + tauLTP
            );
        }
        if (tauLTD <= 0) {
            throw new IllegalArgumentException(
                "Tau LTD debe ser positivo, recibido: " + tauLTD
            );
        }
        
        this.amplitudLTP = amplitudLTP;
        this.amplitudLTD = amplitudLTD;
        this.tauLTP = tauLTP;
        this.tauLTD = tauLTD;
    }
    
    /**
     * Calcula el cambio de peso sináptico basado en la diferencia temporal.
     * 
     * <p>Fórmulas:</p>
     * <ul>
     *   <li>Si dt > 0 (pre antes que post): dw = A_LTP * exp(-dt/tau_LTP) [LTP]</li>
     *   <li>Si dt < 0 (post antes que pre): dw = -A_LTD * exp(dt/tau_LTD) [LTD]</li>
     *   <li>Si dt = 0: dw = 0 (sin cambio)</li>
     * </ul>
     * 
     * @param dt diferencia temporal en milisegundos (timestampPost - timestampPre)
     * @param esLTP true si es LTP (dt > 0), false si es LTD (dt < 0)
     * @return cambio de peso a aplicar (positivo para LTP, negativo para LTD)
     */
    public double calcularCambioPeso(long dt, boolean esLTP) {
        if (dt == 0) {
            return 0.0;
        }
        
        if (esLTP) {
            // LTP: pre antes que post (dt > 0)
            // dw = A_LTP * exp(-dt/tau_LTP)
            return amplitudLTP * Math.exp(-dt / tauLTP);
        } else {
            // LTD: post antes que pre (dt < 0)
            // dw = -A_LTD * exp(dt/tau_LTD)
            // Nota: dt es negativo, por lo que dt/tau_LTD es negativo
            // exp(dt/tau_LTD) = exp(-|dt|/tau_LTD)
            return -amplitudLTD * Math.exp(dt / tauLTD);
        }
    }
    
    /**
     * Aplica STDP a una sinapsis individual basándose en los timestamps de spikes.
     * 
     * <p>Calcula la diferencia temporal dt = timestampPost - timestampPre y aplica
     * LTP o LTD según el signo de dt. El peso se actualiza respetando los límites
     * configurados en la sinapsis.</p>
     * 
     * <p>Si alguno de los timestamps es null (no ha habido spike), no se aplica STDP.</p>
     * 
     * @param sinapsis la sinapsis a la que aplicar STDP
     * @param timestampActual timestamp actual de la simulación (no utilizado actualmente)
     * @throws IllegalArgumentException si sinapsis es null
     */
    public void aplicarSTDP(SinapsisSpiking sinapsis, long timestampActual) {
        if (sinapsis == null) {
            throw new IllegalArgumentException("La sinapsis no puede ser null");
        }
        
        Long timestampPre = sinapsis.getTimestampUltimoSpikePresinaptico();
        Long timestampPost = sinapsis.getTimestampUltimoSpikePostsinaptico();
        
        // Si no hay spikes en ambas neuronas, no aplicar STDP
        if (timestampPre == null || timestampPost == null) {
            return;
        }
        
        // Calcular diferencia temporal
        long dt = timestampPost - timestampPre;
        
        // Si dt = 0, no hay cambio
        if (dt == 0) {
            return;
        }
        
        // Determinar si es LTP o LTD
        boolean esLTP = dt > 0;
        
        // Calcular cambio de peso
        double cambioPeso = calcularCambioPeso(dt, esLTP);
        
        // Aplicar cambio de peso (ajustarPeso ya respeta los límites)
        sinapsis.ajustarPeso(cambioPeso);
    }
    
    /**
     * Aplica STDP a todas las sinapsis de una lista.
     * 
     * <p>Itera sobre todas las sinapsis aplicando STDP individualmente a cada una.</p>
     * 
     * @param sinapsis lista de sinapsis a las que aplicar STDP
     * @param timestampActual timestamp actual de la simulación
     * @throws IllegalArgumentException si la lista de sinapsis es null
     */
    public void aplicarSTDPACapa(List<SinapsisSpiking> sinapsis, long timestampActual) {
        if (sinapsis == null) {
            throw new IllegalArgumentException("La lista de sinapsis no puede ser null");
        }
        
        for (SinapsisSpiking sinap : sinapsis) {
            aplicarSTDP(sinap, timestampActual);
        }
    }
    
    /**
     * Obtiene la amplitud de Long-Term Potentiation.
     * 
     * @return amplitud LTP
     */
    public double getAmplitudLTP() {
        return amplitudLTP;
    }
    
    /**
     * Obtiene la amplitud de Long-Term Depression.
     * 
     * @return amplitud LTD
     */
    public double getAmplitudLTD() {
        return amplitudLTD;
    }
    
    /**
     * Obtiene la constante de tiempo para LTP.
     * 
     * @return tau LTP en milisegundos
     */
    public double getTauLTP() {
        return tauLTP;
    }
    
    /**
     * Obtiene la constante de tiempo para LTD.
     * 
     * @return tau LTD en milisegundos
     */
    public double getTauLTD() {
        return tauLTD;
    }
}
