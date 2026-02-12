package es.jastxz.nn.experimental;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.enums.PotencialMemoria;

import java.util.List;

/**
 * Maneja la propagación de señales en la red neuronal
 * Implementa propagación feed-forward y feedback
 */
public class PropagadorSeñal {
    
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
     * Cada capa evalúa sus neuronas basándose en inputs de la capa anterior
     */
    public void propagarHaciaAdelante(List<List<Neurona>> capasInterneuronas, 
                                      List<Neurona> capaMotora, 
                                      long timestamp) {
        // Evaluar capas de interneuronas
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona neurona : capa) {
                neurona.evaluar(timestamp);
            }
        }
        
        // Evaluar capa motora
        for (Neurona neurona : capaMotora) {
            neurona.evaluar(timestamp);
        }
    }
    
    /**
     * Propagación feedback: de motora hacia sensorial
     * Implementa retroalimentación para refinamiento (pg. 61 Eagleman)
     * 
     * En esta versión simplificada, el feedback modula la activación
     * de capas anteriores basándose en la actividad de capas posteriores
     */
    public void propagarHaciaAtras(List<Neurona> capaMotora, 
                                   List<List<Neurona>> capasInterneuronas) {
        // Feedback desde capa motora
        if (!capasInterneuronas.isEmpty()) {
            aplicarFeedbackACapa(capaMotora, capasInterneuronas.get(capasInterneuronas.size() - 1));
        }
        
        // Feedback entre capas de interneuronas (de posterior a anterior)
        for (int i = capasInterneuronas.size() - 1; i > 0; i--) {
            aplicarFeedbackACapa(capasInterneuronas.get(i), capasInterneuronas.get(i - 1));
        }
    }
    
    /**
     * Aplica feedback de una capa posterior a una anterior
     * Modula la activación basándose en conexiones feedback
     */
    private void aplicarFeedbackACapa(List<Neurona> capaOrigen, List<Neurona> capaDestino) {
        for (Neurona destino : capaDestino) {
            double feedbackTotal = 0.0;
            int contadorFeedback = 0;
            
            // Buscar conexiones feedback desde capaOrigen
            for (Conexion axon : destino.getAxones()) {
                // Si alguna postsináptica está en capaOrigen y activa
                for (Neurona post : axon.getPostsinápticas()) {
                    if (capaOrigen.contains(post) && post.estaActiva()) {
                        feedbackTotal += axon.getPeso() * post.getPotencial();
                        contadorFeedback++;
                    }
                }
            }
            
            // Aplicar modulación por feedback
            if (contadorFeedback > 0) {
                double modulacion = feedbackTotal / contadorFeedback;
                // Ajustar valor almacenado basándose en feedback
                double nuevoValor = destino.getValorAlmacenado() + (modulacion * 0.1);
                destino.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
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
