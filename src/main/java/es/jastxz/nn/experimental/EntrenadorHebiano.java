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
     * MEJORADO: Permite crear conexiones inhibitorias (pesos negativos) cuando es necesario
     * 
     * @param capaMotora Lista de neuronas motoras
     * @param capasInterneuronas Capas intermedias
     * @param errores Diferencia entre target y output para cada neurona motora
     * @param todasConexiones Lista de todas las conexiones de la red
     */
    public void modularAprendizajePorError(List<Neurona> capaMotora, 
                                           List<List<Neurona>> capasInterneuronas,
                                           double[] errores,
                                           List<Conexion> todasConexiones) {
        double tasaAprendizaje = 0.5;  // Tasa más agresiva para aprendizaje supervisado
        
        // Ajustar valores almacenados de neuronas motoras basándose en error
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neurona = capaMotora.get(i);
            double error = errores[i];
            
            // Ajustar valor almacenado proporcionalmente al error
            double nuevoValor = neurona.getValorAlmacenado() + error * tasaAprendizaje;
            neurona.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
        }
        
        // MEJORADO: Iterar sobre conexiones (más eficiente O(n) vs O(n²))
        // Ajustar pesos de conexiones que llegan a neuronas motoras
        for (Conexion conexion : todasConexiones) {
            List<Neurona> postsinapticas = conexion.getPostsinapticas();
            
            // Verificar si alguna postsináptica es motora
            for (int i = 0; i < capaMotora.size(); i++) {
                if (postsinapticas.contains(capaMotora.get(i))) {
                    Neurona pre = conexion.getPresinaptica();
                    double error = errores[i];
                    
                    if (pre.estaActiva()) {
                        // Regla delta mejorada: permite pesos negativos
                        double ajustePeso = error * tasaAprendizaje * 0.8;
                        double nuevoPeso = conexion.getPeso() + ajustePeso;
                        
                        // Permitir pesos negativos para inhibición
                        conexion.setPeso(Math.max(-1.0, Math.min(1.0, nuevoPeso)));
                    }
                    break; // Ya procesamos esta conexión
                }
            }
        }
        
        // Propagar señal de error hacia atrás (modulación)
        propagarErrorHaciaAtras(capaMotora, capasInterneuronas, errores, tasaAprendizaje, todasConexiones);
    }
    
    /**
     * Propaga señal de error hacia capas anteriores
     * Permite que capas intermedias ajusten su conocimiento
     * MEJORADO: Permite crear conexiones inhibitorias
     * 
     * @param capaMotora Capa de salida
     * @param capasInterneuronas Capas intermedias
     * @param erroresMotora Errores de la capa motora
     * @param tasaAprendizaje Tasa de aprendizaje
     * @param todasConexiones Lista de todas las conexiones de la red
     */
    private void propagarErrorHaciaAtras(List<Neurona> capaMotora,
                                        List<List<Neurona>> capasInterneuronas,
                                        double[] erroresMotora,
                                        double tasaAprendizaje,
                                        List<Conexion> todasConexiones) {
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
            // Iterar sobre conexiones donde esta neurona intermedia es presináptica
            for (Conexion conexion : todasConexiones) {
                if (conexion.getPresinaptica() == neuronaInter) {
                    List<Neurona> postsinapticas = conexion.getPostsinapticas();
                    
                    for (int j = 0; j < capaMotora.size(); j++) {
                        if (postsinapticas.contains(capaMotora.get(j))) {
                            errorAcumulado += erroresMotora[j] * conexion.getPeso();
                            contadorConexiones++;
                        }
                    }
                }
            }
            
            // Ajustar valor almacenado y pesos si hay conexiones
            if (contadorConexiones > 0) {
                double errorPromedio = errorAcumulado / contadorConexiones;
                
                // Ajustar valor almacenado
                double nuevoValor = neuronaInter.getValorAlmacenado() + errorPromedio * tasaAprendizaje * 0.5;
                neuronaInter.setValorAlmacenado(Math.max(-1.0, Math.min(1.0, nuevoValor)));
                
                // MEJORADO: Ajustar pesos de conexiones hacia capa motora (permitir negativos)
                for (Conexion conexion : todasConexiones) {
                    if (conexion.getPresinaptica() == neuronaInter && neuronaInter.estaActiva()) {
                        List<Neurona> postsinapticas = conexion.getPostsinapticas();
                        
                        for (int j = 0; j < capaMotora.size(); j++) {
                            if (postsinapticas.contains(capaMotora.get(j))) {
                                double ajustePeso = erroresMotora[j] * tasaAprendizaje * 0.6;
                                double nuevoPeso = conexion.getPeso() + ajustePeso;
                                // Permitir pesos negativos para inhibición
                                conexion.setPeso(Math.max(-1.0, Math.min(1.0, nuevoPeso)));
                                break; // Ya ajustamos esta conexión
                            }
                        }
                    }
                }
            }
        }
    }
}
