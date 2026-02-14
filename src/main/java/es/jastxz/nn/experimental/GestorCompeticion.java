package es.jastxz.nn.experimental;

import es.jastxz.nn.Conexion;
import es.jastxz.nn.Neurona;

import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;

/**
 * Gestiona la competición por recursos y poda de elementos
 * Implementa el principio de Eagleman (pg. 18-19, 229-230): "supervivencia del más útil"
 */
public class GestorCompeticion implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private boolean competicionActiva;
    private long ultimaCompeticion;
    
    public GestorCompeticion() {
        this.competicionActiva = false;
        this.ultimaCompeticion = 0L;
    }
    
    public void activar(boolean activar, long timestampActual) {
        this.competicionActiva = activar;
        if (activar) {
            this.ultimaCompeticion = timestampActual;
        }
    }
    
    public boolean estaActiva() {
        return competicionActiva;
    }
    
    public void competir(List<Neurona> capaSensorial,
                        List<List<Neurona>> capasInterneuronas,
                        List<Neurona> capaMotora,
                        List<Conexion> conexiones,
                        long timestampActual) {
        if (!competicionActiva) {
            return;
        }
        
        long ventanaTemporal = 20L;
        
        // Competición entre neuronas
        competirNeuronas(capaSensorial, ventanaTemporal, timestampActual, 0.03, 0.02);
        
        for (List<Neurona> capa : capasInterneuronas) {
            competirNeuronas(capa, ventanaTemporal, timestampActual, 0.03, 0.025);
        }
        
        competirNeuronas(capaMotora, ventanaTemporal, timestampActual, 0.03, 0.02);
        
        // Competición entre conexiones
        competirConexiones(conexiones, ventanaTemporal, timestampActual);
        
        ultimaCompeticion = timestampActual;
    }
    
    private void competirNeuronas(List<Neurona> capa, long ventanaTemporal, 
                                 long timestampActual, double ganancia, double perdida) {
        for (Neurona neurona : capa) {
            if (timestampActual - neurona.getTimestampUltimaActivacion() < ventanaTemporal) {
                neurona.setRecursosAsignados(
                    Math.min(1.0, neurona.getRecursosAsignados() + ganancia)
                );
            } else {
                neurona.setRecursosAsignados(
                    Math.max(0.0, neurona.getRecursosAsignados() - perdida)
                );
            }
        }
    }
    
    private void competirConexiones(List<Conexion> conexiones, long ventanaTemporal, 
                                   long timestampActual) {
        for (Conexion conexion : conexiones) {
            long ultimaActivacion = conexion.getTimestampUltimaActivacion();
            
            if (timestampActual - ultimaActivacion < ventanaTemporal) {
                conexion.setRecursosAsignados(
                    Math.min(1.0, conexion.getRecursosAsignados() + 0.025)
                );
            } else {
                conexion.setRecursosAsignados(
                    Math.max(0.0, conexion.getRecursosAsignados() - 0.03)
                );
            }
        }
    }
    
    public int podarElementos(List<Conexion> conexiones) {
        List<Conexion> conexionesAPodar = new ArrayList<>();
        
        for (Conexion conexion : conexiones) {
            if (conexion.debeSerPodada()) {
                conexionesAPodar.add(conexion);
            }
        }
        
        // Eliminar conexiones podadas
        conexionesAPodar.stream().forEach(c -> conexiones.remove(c));
        
        return conexionesAPodar.size();
    }
}
