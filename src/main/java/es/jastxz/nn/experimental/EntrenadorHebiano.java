package es.jastxz.nn.experimental;

import java.util.List;
import java.io.Serializable;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;

/**
 * Maneja el entrenamiento mediante plasticidad hebiana
 * Implementa el principio: "neuronas que se activan juntas, se conectan"
 * (pg. 46-47 Eagleman)
 */
public class EntrenadorHebiano implements Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Aplica plasticidad hebiana a todas las conexiones de la red
     * Refuerza conexiones donde pre y post están activas
     * Debilita conexiones por desuso
     */
    public void aplicarPlasticidadHebianaGlobal(List<Conexion> conexiones, 
                                                long timestampActual) {
        long ventanaTemporal = 100L; // Ventana temporal para STDP
        
        for (Conexion conexion : conexiones) {
            Neurona pre = conexion.getPresinaptica();
            List<Neurona> post = conexion.getPostsinapticas();
            conexion.aplicarPlasticidadHebiana(pre, post, timestampActual, ventanaTemporal);
        }
    }
    
    /**
     * Modula el aprendizaje basándose en el error de predicción
     * Implementa supervisión débil: solo se transmite el error (pg. 64 Eagleman)
     * 
     * REFACTORIZADO: Ajusta SOLO pesos de conexiones (conocimiento en sinapsis)
     * NO ajusta valorAlmacenado de neuronas (conceptualmente incorrecto)
     * 
     * MEJORA 1: Backpropagation solo entre neuronas coactivas
     * MEJORA 2: No ajustar conexiones congeladas (estables)
     * HÍBRIDO A+B: Tasa de aprendizaje decreciente pasada como parámetro
     * 
     * Principio biológico:
     * - Neuronas: Procesadores (no almacenan conocimiento)
     * - Sinapsis: Memoria (almacenan conocimiento en pesos)
     * - Hebbian: "Neuronas que se activan juntas, se conectan"
     * 
     * @param capaMotora Lista de neuronas motoras
     * @param capasInterneuronas Capas intermedias
     * @param errores Diferencia entre target y output para cada neurona motora
     * @param todasConexiones Lista de todas las conexiones de la red
     * @param tasaAprendizaje Tasa de aprendizaje (decreciente con iteraciones)
     */
    public void modularAprendizajePorError(List<Neurona> capaMotora, 
                                           List<List<Neurona>> capasInterneuronas,
                                           double[] errores,
                                           List<Conexion> todasConexiones,
                                           double tasaAprendizaje) {
        double lambda = 0.001;  // Factor de regularización L2
        
        // Pre-build lookup: neurona motora → índice en array de errores
        java.util.Map<Neurona, Integer> motoraIndice = new java.util.IdentityHashMap<>(capaMotora.size());
        for (int i = 0; i < capaMotora.size(); i++) {
            motoraIndice.put(capaMotora.get(i), i);
        }
        
        for (Conexion conexion : todasConexiones) {
            if (conexion.estaCongelada()) {
                continue;
            }
            
            List<Neurona> postsinapticas = conexion.getPostsinapticas();
            
            // Buscar si alguna postsináptica es motora usando el mapa O(1)
            for (Neurona post : postsinapticas) {
                Integer idx = motoraIndice.get(post);
                if (idx != null) {
                    Neurona pre = conexion.getPresinaptica();
                    double error = errores[idx];
                    
                    if (pre.estaActiva() && post.estaActiva()) {
                        double ajustePeso = error * tasaAprendizaje;
                        double pesoActual = conexion.getPeso();
                        
                        if (pesoActual < 0) {
                            double nuevoPeso = pesoActual - ajustePeso;
                            conexion.setPeso(Math.min(-0.1, Math.max(-1.0, nuevoPeso)));
                        } else {
                            double penalizacion = lambda * pesoActual;
                            double nuevoPeso = pesoActual + ajustePeso - penalizacion;
                            conexion.setPeso(Math.max(0.0, Math.min(1.0, nuevoPeso)));
                        }
                        
                        conexion.actualizarCongelacion();
                    }
                    break;
                }
            }
        }
        
        // Propagar señal de error hacia atrás (modulación)
        propagarErrorHaciaAtras(capaMotora, capasInterneuronas, errores, tasaAprendizaje, lambda, todasConexiones);
    }
    
    /**
     * Sobrecarga para compatibilidad: usa tasa de aprendizaje fija
     */
    public void modularAprendizajePorError(List<Neurona> capaMotora, 
                                           List<List<Neurona>> capasInterneuronas,
                                           double[] errores,
                                           List<Conexion> todasConexiones) {
        modularAprendizajePorError(capaMotora, capasInterneuronas, errores, todasConexiones, 0.1);
    }
    
    /**
     * Propaga señal de error hacia capas anteriores
     * Permite que capas intermedias ajusten su conocimiento
     * 
     * REFACTORIZADO: Ajusta SOLO pesos de conexiones (conocimiento en sinapsis)
     * NO ajusta valorAlmacenado de neuronas (conceptualmente incorrecto)
     * 
     * MEJORA 1: Backpropagation solo entre neuronas coactivas
     * MEJORA 2: No ajustar conexiones congeladas (estables)
     * 
     * @param capaMotora Capa de salida
     * @param capasInterneuronas Capas intermedias
     * @param erroresMotora Errores de la capa motora
     * @param tasaAprendizaje Tasa de aprendizaje
     * @param lambda Factor de regularización L2
     * @param todasConexiones Lista de todas las conexiones de la red
     */
    private void propagarErrorHaciaAtras(List<Neurona> capaMotora,
                                        List<List<Neurona>> capasInterneuronas,
                                        double[] erroresMotora,
                                        double tasaAprendizaje,
                                        double lambda,
                                        List<Conexion> todasConexiones) {
        if (capasInterneuronas.isEmpty()) {
            return;
        }
        
        // Pre-build lookup: neurona motora → índice
        java.util.Map<Neurona, Integer> motoraIndice = new java.util.IdentityHashMap<>(capaMotora.size());
        for (int i = 0; i < capaMotora.size(); i++) {
            motoraIndice.put(capaMotora.get(i), i);
        }
        
        // Pre-build lookup: neurona intermedia activa → set para O(1) check
        List<Neurona> ultimaInter = capasInterneuronas.get(capasInterneuronas.size() - 1);
        java.util.Set<Neurona> interActivasSet = java.util.Collections.newSetFromMap(
            new java.util.IdentityHashMap<>(ultimaInter.size()));
        for (Neurona n : ultimaInter) {
            if (n.estaActiva()) interActivasSet.add(n);
        }
        
        if (interActivasSet.isEmpty()) return;
        
        // Single pass over connections
        for (Conexion conexion : todasConexiones) {
            if (conexion.estaCongelada()) continue;
            
            Neurona pre = conexion.getPresinaptica();
            if (!interActivasSet.contains(pre)) continue;
            
            List<Neurona> postsinapticas = conexion.getPostsinapticas();
            
            for (Neurona post : postsinapticas) {
                Integer j = motoraIndice.get(post);
                if (j != null && post.estaActiva()) {
                    double ajustePeso = erroresMotora[j] * tasaAprendizaje * 0.5;
                    double pesoActual = conexion.getPeso();
                    
                    if (pesoActual < 0) {
                        if (ajustePeso < 0) {
                            double nuevoPeso = pesoActual + ajustePeso;
                            conexion.setPeso(Math.min(-0.1, Math.max(-1.0, nuevoPeso)));
                        }
                    } else {
                        double penalizacion = lambda * pesoActual;
                        double nuevoPeso = pesoActual + ajustePeso - penalizacion;
                        conexion.setPeso(Math.max(0.0, Math.min(1.0, nuevoPeso)));
                    }
                    
                    conexion.actualizarCongelacion();
                    break;
                }
            }
        }
    }
}
