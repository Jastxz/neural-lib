package es.jastxz.nn.experimental;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.enums.PotencialMemoria;
import es.jastxz.nn.enums.TipoNeurona;

import java.util.List;
import java.io.Serializable;

/**
 * Maneja la propagación de señales en la red neuronal
 * Implementa propagación feed-forward y feedback
 * 
 * REFACTORIZADO: Ahora itera sobre conexiones en lugar de neuronas
 * Más eficiente O(n) vs O(n²) y más natural biológicamente
 */
public class PropagadorSeñal implements Serializable {
    private static final long serialVersionUID = 1L;
    
    /**
     * Establece los valores de entrada en la capa sensorial
     * 
     * REFACTORIZADO: Activa neuronas sensoriales basándose en input
     * NO usa valorAlmacenado (conceptualmente incorrecto)
     * 
     * MEJORADO: Usa umbrales personalizados de cada neurona para selectividad
     * Las neuronas sensoriales se activan si el input supera SU umbral específico
     */
    public void establecerInputs(List<Neurona> capaSensorial, double[] inputs, long timestamp) {
        for (int i = 0; i < inputs.length; i++) {
            Neurona neurona = capaSensorial.get(i);
            
            // MEJORA: Usar umbral personalizado de la neurona
            // Esto permite selectividad: solo neuronas relevantes se activan
            // Cada neurona tiene su propio umbral (típicamente 0.17-0.48)
            if (Math.abs(inputs[i]) > neurona.getUmbralActivacion()) {
                neurona.activar(timestamp);
            } else {
                // Si el input no supera el umbral, resetear la neurona
                neurona.resetear();
            }
        }
    }
    
    /**
     * Propagación feed-forward: de sensorial hacia motora
     * REFACTORIZADO: Propagación en dos pasadas para respetar orden de capas
     * 
     * CORRECCIÓN CRÍTICA: Las neuronas intermedias deben activarse ANTES de
     * propagar hacia las neuronas motoras. Solución: doble pasada.
     * 
     * Pasada 1: Propagar sensorial → intermedia, evaluar intermedias
     * Pasada 2: Propagar intermedia → motora, evaluar motoras
     * 
     * @param conexiones Lista de todas las conexiones de la red
     * @param todasNeuronas Lista de todas las neuronas
     * @param timestamp Timestamp actual
     */
    public void propagarHaciaAdelante(List<Conexion> conexiones, 
                                      List<Neurona> todasNeuronas,
                                      long timestamp) {
        long ventanaTemporal = 100L; // Ventana temporal para STDP
        
        // ========== PASADA 1: SENSORIAL → INTERMEDIA ==========
        
        // Fase 1.1: Propagar señales hacia neuronas intermedias
        for (Conexion conexion : conexiones) {
            Neurona pre = conexion.getPresinaptica();
            
            // Solo procesar si la neurona presináptica está activa
            if (!pre.estaActiva()) {
                continue;
            }
            
            // Verificar si esta conexión va hacia neuronas intermedias
            boolean vaHaciaIntermedia = false;
            for (Neurona post : conexion.getPostsinapticas()) {
                if (post.getTipo() == TipoNeurona.INTER) {
                    vaHaciaIntermedia = true;
                    break;
                }
            }
            
            // Solo propagar si va hacia intermedias
            if (!vaHaciaIntermedia) {
                continue;
            }
            
            double señal = conexion.getPeso() * pre.getPotencial();
            
            // Enviar señal a neuronas intermedias
            for (Neurona post : conexion.getPostsinapticas()) {
                if (post.getTipo() == TipoNeurona.INTER) {
                    post.recibirSeñal(señal);
                    
                    // Aplicar plasticidad hebiana si ambas están activas
                    if (post.estaActiva()) {
                        double pesoActual = conexion.getPeso();
                        
                        if (pesoActual > 0) {
                            double tasaRefuerzo = 0.01;
                            double lambda = 0.005;
                            double penalizacion = lambda * pesoActual;
                            double nuevoPeso = pesoActual + tasaRefuerzo - penalizacion;
                            conexion.setPeso(Math.max(0.0, Math.min(1.0, nuevoPeso)));
                        }
                        
                        conexion.setRecursosAsignados(
                            Math.min(1.0, conexion.getRecursosAsignados() + 0.005)
                        );
                    }
                }
            }
        }
        
        // Fase 1.2: Evaluar activación de neuronas intermedias
        for (Neurona neurona : todasNeuronas) {
            if (neurona.getTipo() == TipoNeurona.INTER) {
                neurona.evaluarActivacion(timestamp);
            }
        }
        
        // ========== PASADA 2: INTERMEDIA → MOTORA ==========
        
        // Fase 2.1: Propagar señales hacia neuronas motoras
        for (Conexion conexion : conexiones) {
            Neurona pre = conexion.getPresinaptica();
            
            // Solo procesar si la neurona presináptica está activa
            if (!pre.estaActiva()) {
                // DESACTIVADO: Debilitamiento por desuso durante entrenamiento supervisado
                // El debilitamiento por desuso es útil para consolidación, pero durante
                // entrenamiento supervisado causa colapso de pesos y divergencia.
                // La regularización L2 ya controla el crecimiento excesivo de pesos.
                
                // // Debilitar por desuso
                // long tiempoSinUso = timestamp - conexion.getTimestampUltimaActivacion();
                // if (tiempoSinUso > ventanaTemporal) {
                //     double pesoActual = conexion.getPeso();
                //     
                //     if (pesoActual > 0) {
                //         double tasaDebilitamiento = 0.015;
                //         double nuevoPeso = pesoActual * (1.0 - tasaDebilitamiento);
                //         conexion.setPeso(nuevoPeso);
                //     }
                //     
                //     conexion.setRecursosAsignados(
                //         Math.max(0.0, conexion.getRecursosAsignados() - 0.01)
                //     );
                // }
                continue;
            }
            
            // Verificar si esta conexión va hacia neuronas motoras
            boolean vaHaciaMotora = false;
            for (Neurona post : conexion.getPostsinapticas()) {
                if (post.getTipo() == TipoNeurona.MOTORA) {
                    vaHaciaMotora = true;
                    break;
                }
            }
            
            // Solo propagar si va hacia motoras
            if (!vaHaciaMotora) {
                continue;
            }
            
            double señal = conexion.getPeso() * pre.getPotencial();
            
            // Enviar señal a neuronas motoras
            for (Neurona post : conexion.getPostsinapticas()) {
                if (post.getTipo() == TipoNeurona.MOTORA) {
                    post.recibirSeñal(señal);
                    
                    // Aplicar plasticidad hebiana si ambas están activas
                    if (post.estaActiva()) {
                        double pesoActual = conexion.getPeso();
                        
                        if (pesoActual > 0) {
                            double tasaRefuerzo = 0.01;
                            double lambda = 0.005;
                            double penalizacion = lambda * pesoActual;
                            double nuevoPeso = pesoActual + tasaRefuerzo - penalizacion;
                            conexion.setPeso(Math.max(0.0, Math.min(1.0, nuevoPeso)));
                        }
                        
                        conexion.setRecursosAsignados(
                            Math.min(1.0, conexion.getRecursosAsignados() + 0.005)
                        );
                    }
                }
            }
        }
        
        // Fase 2.2: Evaluar activación de neuronas motoras
        for (Neurona neurona : todasNeuronas) {
            if (neurona.getTipo() == TipoNeurona.MOTORA) {
                neurona.evaluarActivacion(timestamp);
            }
        }
    }
    
    /**
     * Propagación feedback: de motora hacia sensorial
     * Implementa retroalimentación para refinamiento (pg. 61 Eagleman)
     * 
     * REFACTORIZADO: Ajusta SOLO pesos de conexiones feedback
     * NO ajusta valorAlmacenado de neuronas (conceptualmente incorrecto)
     * 
     * @param conexionesFeedback Lista de conexiones feedback (de posterior a anterior)
     * @param todasNeuronas Lista de todas las neuronas
     */
    public void propagarHaciaAtras(List<Conexion> conexionesFeedback,
                                   List<Neurona> todasNeuronas) {
        // Fase 1: Propagar señales feedback a través de conexiones
        // El feedback modula ligeramente los pesos (refinamiento)
        for (Conexion conexion : conexionesFeedback) {
            Neurona pre = conexion.getPresinaptica();
            
            // Si la neurona presináptica está activa, propagar feedback
            if (pre.estaActiva() && conexion.getPeso() > 0) {
                double feedbackSeñal = conexion.getPeso() * pre.getPotencial() * 0.01; // Factor de modulación muy pequeño
                
                // Ajustar peso de la conexión feedback basándose en activación
                // Esto implementa refinamiento hebiano bidireccional
                double ajustePeso = feedbackSeñal * 0.1;
                double nuevoPeso = conexion.getPeso() + ajustePeso;
                conexion.setPeso(Math.max(-1.0, Math.min(1.0, nuevoPeso)));
            }
        }
    }
    
    /**
     * Extrae los valores de salida de la capa motora
     * 
     * MEJORADO: Output continuo basado en potencial acumulado
     * 
     * Principio biológico:
     * - El output representa la frecuencia de disparo de la neurona
     * - Proporcional al potencial integrado (suma de señales)
     * - Sigmoide para mapear a rango [0,1] de forma suave
     * 
     * Ventajas:
     * - Output continuo (puede alcanzar valores intermedios como 0.6)
     * - Diferenciable (gradientes suaves para aprendizaje)
     * - Compatible con red clásica (ambas usan sigmoide)
     * - Biológicamente defendible (frecuencia de disparo)
     * 
     * @param capaMotora Lista de neuronas motoras
     * @param conexiones Lista de todas las conexiones de la red
     * @return Array de valores de salida continuos [0,1]
     */
    public double[] getOutputs(List<Neurona> capaMotora, List<Conexion> conexiones) {
        double[] outputs = new double[capaMotora.size()];
        
        // Pre-build index: neurona motora → incoming connections with active pre
        java.util.Map<Neurona, List<Conexion>> entrantes = new java.util.IdentityHashMap<>();
        for (Conexion c : conexiones) {
            if (!c.getPresinaptica().estaActiva()) continue;
            for (Neurona post : c.getPostsinapticas()) {
                entrantes.computeIfAbsent(post, k -> new java.util.ArrayList<>()).add(c);
            }
        }
        
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neuronaMotora = capaMotora.get(i);
            List<Conexion> conns = entrantes.get(neuronaMotora);
            
            if (conns == null || conns.isEmpty()) {
                outputs[i] = 0.0;
                continue;
            }
            
            double potencialAcumulado = 0.0;
            for (Conexion c : conns) {
                potencialAcumulado += c.getPeso() * c.getPresinaptica().getPotencial();
            }
            
            double potencialPromedio = potencialAcumulado / conns.size();
            double x = (potencialPromedio / 40.0) * 10.0 - 5.0;
            outputs[i] = 1.0 / (1.0 + Math.exp(-x));
        }
        
        return outputs;
    }
}
