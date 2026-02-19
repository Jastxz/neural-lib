package es.jastxz.nn.experimental;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.enums.PotencialMemoria;

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
     * Las neuronas sensoriales se activan si el input supera un umbral mínimo
     */
    public void establecerInputs(List<Neurona> capaSensorial, double[] inputs, long timestamp) {
        for (int i = 0; i < inputs.length; i++) {
            Neurona neurona = capaSensorial.get(i);
            
            // Activar neurona si el input es significativo (>0.05)
            // Umbral bajo para permitir que inputs pequeños también activen neuronas
            if (Math.abs(inputs[i]) > 0.05) {
                neurona.activar(timestamp);
            } else {
                // Si el input es muy bajo, resetear la neurona
                neurona.resetear();
            }
        }
    }
    
    /**
     * Propagación feed-forward: de sensorial hacia motora
     * REFACTORIZADO: Itera sobre conexiones en lugar de neuronas
     * 
     * @param conexiones Lista de todas las conexiones de la red
     * @param todasNeuronas Lista de todas las neuronas (para resetear potencial acumulado)
     * @param timestamp Timestamp actual
     */
    public void propagarHaciaAdelante(List<Conexion> conexiones, 
                                      List<Neurona> todasNeuronas,
                                      long timestamp) {
        // Fase 1: Propagar señales a través de conexiones
        for (Conexion conexion : conexiones) {
            Neurona pre = conexion.getPresinaptica();
            
            // Si la neurona presináptica está activa, propagar señal
            if (pre.estaActiva()) {
                double señal = conexion.getPeso() * pre.getPotencial();
                
                // Enviar señal a todas las neuronas postsinápticas
                for (Neurona post : conexion.getPostsinapticas()) {
                    post.recibirSeñal(señal);
                }
            }
        }
        
        // Fase 2: Evaluar activación de todas las neuronas
        for (Neurona neurona : todasNeuronas) {
            // No evaluar neuronas sensoriales (ya están activadas por inputs)
            if (neurona.getTipo() != es.jastxz.nn.enums.TipoNeurona.SENSORIAL) {
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
            if (pre.estaActiva()) {
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
     * REFACTORIZADO: Calcula output basándose en pesos de conexiones (conocimiento)
     * en lugar de valorAlmacenado (que es conceptualmente incorrecto).
     * 
     * Principio biológico:
     * - Neuronas: Procesadores (integran y disparan)
     * - Sinapsis: Memoria (almacenan conocimiento en sus pesos)
     * - Output: Suma ponderada de conexiones activas
     * 
     * @param capaMotora Lista de neuronas motoras
     * @param conexiones Lista de todas las conexiones de la red
     * @return Array de valores de salida
     */
    public double[] getOutputs(List<Neurona> capaMotora, List<Conexion> conexiones) {
        double[] outputs = new double[capaMotora.size()];
        
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neuronaMotora = capaMotora.get(i);
            
            // Calcular output basándose en conexiones que llegan a esta neurona motora
            double sumaConexiones = 0.0;
            int contadorConexiones = 0;
            
            for (Conexion c : conexiones) {
                // Verificar si esta conexión llega a la neurona motora
                if (c.getPostsinapticas().contains(neuronaMotora)) {
                    Neurona pre = c.getPresinaptica();
                    
                    // Solo considerar conexiones desde neuronas activas
                    if (pre.estaActiva()) {
                        // El conocimiento está en el peso de la conexión
                        sumaConexiones += c.getPeso();
                        contadorConexiones++;
                    }
                }
            }
            
            // Output: promedio de pesos de conexiones activas
            // Normalizado entre 0 y 1 (asumiendo pesos en [-1, 1])
            if (contadorConexiones > 0) {
                double promedio = sumaConexiones / contadorConexiones;
                // Normalizar de [-1, 1] a [0, 1]
                outputs[i] = (promedio + 1.0) / 2.0;
            } else {
                // Si no hay conexiones activas, output es 0
                outputs[i] = 0.0;
            }
        }
        
        return outputs;
    }
}
