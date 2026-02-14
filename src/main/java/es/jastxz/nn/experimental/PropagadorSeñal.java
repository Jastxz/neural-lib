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
     */
    public void establecerInputs(List<Neurona> capaSensorial, double[] inputs, long timestamp) {
        for (int i = 0; i < inputs.length; i++) {
            Neurona neurona = capaSensorial.get(i);
            
            // Establecer valor almacenado
            neurona.setValorAlmacenado(inputs[i]);
            
            // Si el input es significativo, activar la neurona directamente
            if (Math.abs(inputs[i]) > 0.1) {
                // Activar directamente (simula input sensorial)
                neurona.activar(timestamp);
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
     * REFACTORIZADO: Itera sobre conexiones feedback
     * 
     * @param conexionesFeedback Lista de conexiones feedback (de posterior a anterior)
     * @param todasNeuronas Lista de todas las neuronas
     */
    public void propagarHaciaAtras(List<Conexion> conexionesFeedback,
                                   List<Neurona> todasNeuronas) {
        // Fase 1: Propagar señales feedback a través de conexiones
        for (Conexion conexion : conexionesFeedback) {
            Neurona pre = conexion.getPresinaptica();
            
            // Si la neurona presináptica está activa, propagar feedback
            if (pre.estaActiva()) {
                double feedbackSeñal = conexion.getPeso() * pre.getPotencial() * 0.1; // Factor de modulación
                
                // Enviar feedback a todas las neuronas postsinápticas
                for (Neurona post : conexion.getPostsinapticas()) {
                    // Ajustar valor almacenado basándose en feedback
                    double nuevoValor = post.getValorAlmacenado() + feedbackSeñal;
                    post.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
                }
            }
        }
    }
    
    /**
     * Extrae los valores de salida de la capa motora
     */
    public double[] getOutputs(List<Neurona> capaMotora) {
        double[] outputs = new double[capaMotora.size()];
        
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neurona = capaMotora.get(i);
            
            // Combinar potencial y valor almacenado
            // Si está activa, usar potencial normalizado
            if (neurona.estaActiva()) {
                outputs[i] = neurona.getPotencial() / PotencialMemoria.PICO.getValor();
            } else {
                // Si no está activa, usar valor almacenado
                outputs[i] = neurona.getValorAlmacenado();
            }
        }
        
        return outputs;
    }
}
