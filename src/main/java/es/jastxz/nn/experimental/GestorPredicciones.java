package es.jastxz.nn.experimental;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;
import es.jastxz.nn.enums.PotencialMemoria;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.io.Serializable;

/**
 * Gestiona el sistema de predicción y codificación predictiva
 * Implementa el principio de Eagleman (pg. 64): "solo se transmite el error de predicción"
 */
public class GestorPredicciones implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private boolean modoPredictivo;
    private Map<Integer, double[]> prediccionesPorCapa;
    private double[] erroresPrediccion;
    private Random random;
    
    public GestorPredicciones(int tamanoCapaMotora) {
        this.modoPredictivo = false;
        this.prediccionesPorCapa = new HashMap<>();
        this.erroresPrediccion = new double[tamanoCapaMotora];
        this.random = new Random();
    }
    
    public void activar(boolean activar, List<List<Neurona>> capasInterneuronas, 
                       List<Neurona> capaMotora) {
        this.modoPredictivo = activar;
        
        if (activar) {
            inicializarPredicciones(capasInterneuronas, capaMotora);
        } else {
            prediccionesPorCapa.clear();
        }
    }
    
    public boolean estaActivo() {
        return modoPredictivo;
    }
    
    public double[] getErroresPrediccion() {
        return erroresPrediccion.clone();
    }
    
    public int getNumeroCapasConPredicciones() {
        return prediccionesPorCapa.size();
    }
    
    public void inicializarPredicciones(List<List<Neurona>> capasInterneuronas, 
                                       List<Neurona> capaMotora) {
        prediccionesPorCapa.clear();
        
        // Predicciones para capas intermedias
        for (int i = 0; i < capasInterneuronas.size(); i++) {
            int tamano = capasInterneuronas.get(i).size();
            double[] prediccion = new double[tamano];
            for (int j = 0; j < tamano; j++) {
                prediccion[j] = randomValue() * 0.1;
            }
            prediccionesPorCapa.put(i, prediccion);
        }
        
        // Predicción para capa motora
        double[] prediccionMotora = new double[capaMotora.size()];
        for (int i = 0; i < prediccionMotora.length; i++) {
            prediccionMotora[i] = randomValue() * 0.1;
        }
        prediccionesPorCapa.put(capasInterneuronas.size(), prediccionMotora);
    }
    
    public void calcularPredicciones(List<List<Neurona>> capasInterneuronas, 
                                     List<Neurona> capaMotora,
                                     List<Conexion> todasConexiones) {
        // Predicción de capas intermedias
        for (int i = 0; i < capasInterneuronas.size(); i++) {
            List<Neurona> capaActual = capasInterneuronas.get(i);
            double[] prediccion = new double[capaActual.size()];
            
            for (int j = 0; j < capaActual.size(); j++) {
                Neurona neurona = capaActual.get(j);
                
                double suma = 0.0;
                int contador = 0;
                
                // Buscar conexiones que llegan a esta neurona
                for (Conexion conexion : todasConexiones) {
                    if (conexion.getPostsinapticas().contains(neurona)) {
                        Neurona pre = conexion.getPresinaptica();
                        if (pre.estaActiva()) {
                            double potencialNormalizado = pre.getPotencial() / 
                                PotencialMemoria.PICO.getValor();
                            suma += conexion.getPeso() * potencialNormalizado;
                            contador++;
                        }
                    }
                }
                
                prediccion[j] = contador > 0 ? suma / contador : 0.0;
            }
            
            prediccionesPorCapa.put(i, prediccion);
        }
        
        // Predicción de capa motora
        double[] prediccionMotora = new double[capaMotora.size()];
        for (int i = 0; i < capaMotora.size(); i++) {
            Neurona neurona = capaMotora.get(i);
            
            double suma = 0.0;
            int contador = 0;
            
            // Buscar conexiones que llegan a esta neurona motora
            for (Conexion conexion : todasConexiones) {
                if (conexion.getPostsinapticas().contains(neurona)) {
                    Neurona pre = conexion.getPresinaptica();
                    if (pre.estaActiva()) {
                        double potencialNormalizado = pre.getPotencial() / 
                            PotencialMemoria.PICO.getValor();
                        suma += conexion.getPeso() * potencialNormalizado;
                        contador++;
                    }
                }
            }
            
            prediccionMotora[i] = contador > 0 ? suma / contador : 0.0;
        }
        prediccionesPorCapa.put(capasInterneuronas.size(), prediccionMotora);
    }
    
    public void calcularErroresPrediccion(List<List<Neurona>> capasInterneuronas, 
                                         List<Neurona> capaMotora) {
        int indiceMotora = capasInterneuronas.size();
        double[] prediccionMotora = prediccionesPorCapa.get(indiceMotora);
        
        if (prediccionMotora != null) {
            for (int i = 0; i < capaMotora.size(); i++) {
                Neurona neurona = capaMotora.get(i);
                double activacionReal = neurona.estaActiva() ? 
                    neurona.getPotencial() / PotencialMemoria.PICO.getValor() : 
                    neurona.getValorAlmacenado();
                
                erroresPrediccion[i] = activacionReal - prediccionMotora[i];
            }
        }
    }
    
    public void ajustarModeloPredictivo(List<List<Neurona>> capasInterneuronas, 
                                       List<Neurona> capaMotora) {
        double tasaAprendizaje = 0.15;
        
        for (int indiceCapa : prediccionesPorCapa.keySet()) {
            double[] prediccion = prediccionesPorCapa.get(indiceCapa);
            
            List<Neurona> capa;
            if (indiceCapa < capasInterneuronas.size()) {
                capa = capasInterneuronas.get(indiceCapa);
            } else {
                capa = capaMotora;
            }
            
            for (int i = 0; i < Math.min(prediccion.length, capa.size()); i++) {
                Neurona neurona = capa.get(i);
                double activacionReal = neurona.estaActiva() ? 
                    neurona.getPotencial() / PotencialMemoria.PICO.getValor() : 
                    neurona.getValorAlmacenado();
                
                double error = activacionReal - prediccion[i];
                prediccion[i] += error * tasaAprendizaje;
                prediccion[i] = Math.max(-1.0, Math.min(1.0, prediccion[i]));
            }
        }
    }
    
    private double randomValue() {
        return (random.nextDouble() * 2.0) - 1.0;
    }
}
