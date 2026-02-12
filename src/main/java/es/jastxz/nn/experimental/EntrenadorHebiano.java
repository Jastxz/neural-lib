package es.jastxz.nn.experimental;

import java.util.List;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;

/**
 * Maneja el entrenamiento mediante plasticidad hebiana
 * Implementa el principio: "neuronas que se activan juntas, se conectan"
 * (pg. 46-47 Eagleman)
 */
public class EntrenadorHebiano {
    
    /**
     * Aplica plasticidad hebiana a todas las conexiones de la red
     * Refuerza conexiones donde pre y post están activas
     * Debilita conexiones por desuso
     */
    public void aplicarPlasticidadHebianaGlobal(List<Conexion> conexiones, 
                                                long timestampActual) {
        long ventanaTemporal = 100L; // Ventana temporal para STDP
        
        for (Conexion conexion : conexiones) {
            conexion.aplicarPlasticidadHebiana(timestampActual, ventanaTemporal);
        }
    }
    
    /**
     * Modula el aprendizaje basándose en el error de predicción
     * Implementa supervisión débil: solo se transmite el error (pg. 64 Eagleman)
     * 
     * @param errores Diferencia entre target y output para cada neurona motora
     */
    public void modularAprendizajePorError(List<Neurona> capaMotora, 
                                           List<List<Neurona>> capasInterneuronas,
                                           double[] errores) {
        // Ajustar valores almacenados de neuronas motoras basándose en error
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neurona = capaMotora.get(i);
            double error = errores[i];
            
            // Ajustar valor almacenado proporcionalmente al error
            double ajuste = error * 0.1; // Tasa de aprendizaje
            double nuevoValor = neurona.getValorAlmacenado() + ajuste;
            neurona.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
        }
        
        // Propagar señal de error hacia atrás (modulación)
        propagarErrorHaciaAtras(capaMotora, capasInterneuronas, errores);
    }
    
    /**
     * Propaga señal de error hacia capas anteriores
     * Permite que capas intermedias ajusten su conocimiento
     */
    private void propagarErrorHaciaAtras(List<Neurona> capaMotora,
                                        List<List<Neurona>> capasInterneuronas,
                                        double[] erroresMotora) {
        // Si no hay capas intermedias, no hay nada que propagar
        if (capasInterneuronas.isEmpty()) {
            return;
        }
        
        // Propagar error a última capa intermedia
        List<Neurona> ultimaInter = capasInterneuronas.get(capasInterneuronas.size() - 1);
        
        for (int i = 0; i < ultimaInter.size(); i++) {
            Neurona neuronaInter = ultimaInter.get(i);
            double errorAcumulado = 0.0;
            int contadorConexiones = 0;
            
            // Acumular error de todas las neuronas motoras conectadas
            for (Conexion axon : neuronaInter.getAxones()) {
                for (int j = 0; j < capaMotora.size(); j++) {
                    if (axon.getPostsinápticas().contains(capaMotora.get(j))) {
                        errorAcumulado += erroresMotora[j] * axon.getPeso();
                        contadorConexiones++;
                    }
                }
            }
            
            // Ajustar valor almacenado si hay conexiones
            if (contadorConexiones > 0) {
                double errorPromedio = errorAcumulado / contadorConexiones;
                double ajuste = errorPromedio * 0.05; // Tasa menor para capas intermedias
                double nuevoValor = neuronaInter.getValorAlmacenado() + ajuste;
                neuronaInter.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
            }
        }
    }
}
