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
        double lambda = 0.001;  // Factor de regularización L2 (reducido 10x para evitar colapso de pesos)
        
        // REFACTORIZADO: Ajustar SOLO pesos de conexiones que llegan a neuronas motoras
        // El conocimiento está en las sinapsis, no en las neuronas
        for (Conexion conexion : todasConexiones) {
            // MEJORA 2: Saltar conexiones congeladas
            if (conexion.estaCongelada()) {
                continue;
            }
            
            List<Neurona> postsinapticas = conexion.getPostsinapticas();
            
            // Verificar si alguna postsináptica es motora
            for (int i = 0; i < capaMotora.size(); i++) {
                if (postsinapticas.contains(capaMotora.get(i))) {
                    Neurona pre = conexion.getPresinaptica();
                    Neurona post = capaMotora.get(i);
                    double error = errores[i];
                    
                    // MEJORA 1: Solo ajustar si AMBAS neuronas están activas
                    // Principio hebiano: "neuronas que se activan juntas, se conectan"
                    if (pre.estaActiva() && post.estaActiva()) {
                        double ajustePeso = error * tasaAprendizaje;
                        double pesoActual = conexion.getPeso();
                        
                        // PROTECCIÓN TOTAL: Pesos negativos (inhibitorios) permanecen negativos
                        // Principio biológico: neuronas inhibitorias son estables
                        if (pesoActual < 0) {
                            // Conexión inhibitoria: solo permitir ajustes negativos (más inhibición)
                            double nuevoPeso = pesoActual - ajustePeso;
                            conexion.setPeso(Math.min(-0.1, Math.max(-1.0, nuevoPeso)));
                        } else {
                            // Conexión excitatoria: ajuste normal con regularización
                            double penalizacion = lambda * pesoActual;
                            double nuevoPeso = pesoActual + ajustePeso - penalizacion;
                            conexion.setPeso(Math.max(0.0, Math.min(1.0, nuevoPeso)));
                        }
                        
                        // Actualizar estado de congelación
                        conexion.actualizarCongelacion();
                    }
                    break; // Ya procesamos esta conexión
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
        // Si no hay capas intermedias, no hay nada que propagar
        if (capasInterneuronas.isEmpty()) {
            return;
        }
        
        // Propagar error a última capa intermedia
        List<Neurona> ultimaInter = capasInterneuronas.get(capasInterneuronas.size() - 1);
        
        for (int i = 0; i < ultimaInter.size(); i++) {
            Neurona neuronaInter = ultimaInter.get(i);
            
            // MEJORA 1: Solo ajustar si la neurona intermedia está activa
            if (!neuronaInter.estaActiva()) {
                continue;
            }
            
            // REFACTORIZADO: Ajustar SOLO pesos de conexiones hacia capa motora
            for (Conexion conexion : todasConexiones) {
                // MEJORA 2: Saltar conexiones congeladas
                if (conexion.estaCongelada()) {
                    continue;
                }
                
                if (conexion.getPresinaptica() == neuronaInter) {
                    List<Neurona> postsinapticas = conexion.getPostsinapticas();
                    
                    for (int j = 0; j < capaMotora.size(); j++) {
                        Neurona neuronaMotora = capaMotora.get(j);
                        
                        if (postsinapticas.contains(neuronaMotora)) {
                            // MEJORA 1: Solo ajustar si la neurona motora también está activa
                            if (neuronaMotora.estaActiva()) {
                                double ajustePeso = erroresMotora[j] * tasaAprendizaje * 0.5;
                                double pesoActual = conexion.getPeso();
                                
                                // PROTECCIÓN TOTAL: Pesos negativos permanecen negativos
                                if (pesoActual < 0) {
                                    // Conexión inhibitoria: solo permitir ajustes negativos
                                    if (ajustePeso < 0) {
                                        // Ajuste negativo: permitir sin regularización
                                        double nuevoPeso = pesoActual + ajustePeso;
                                        conexion.setPeso(Math.min(-0.1, Math.max(-1.0, nuevoPeso)));
                                    }
                                } else {
                                    // Conexión excitatoria: ajuste normal con regularización
                                    double penalizacion = lambda * pesoActual;
                                    double nuevoPeso = pesoActual + ajustePeso - penalizacion;
                                    conexion.setPeso(Math.max(0.0, Math.min(1.0, nuevoPeso)));
                                }
                                
                                // Actualizar estado de congelación
                                conexion.actualizarCongelacion();
                            }
                            break; // Ya ajustamos esta conexión
                        }
                    }
                }
            }
        }
    }
}
