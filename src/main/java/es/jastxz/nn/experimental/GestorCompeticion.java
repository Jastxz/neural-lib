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
        
        // AJUSTADO: Ventana temporal más larga para dar tiempo a que se usen las conexiones
        // Con avance de 10L por iteración, necesitamos ventana generosa
        // 500L = ~50 iteraciones de margen antes de penalizar
        long ventanaTemporal = 500L;
        
        // Competición entre neuronas (más conservadora)
        competirNeuronas(capaSensorial, ventanaTemporal, timestampActual, 0.02, 0.01);
        
        for (List<Neurona> capa : capasInterneuronas) {
            competirNeuronas(capa, ventanaTemporal, timestampActual, 0.02, 0.015);
        }
        
        competirNeuronas(capaMotora, ventanaTemporal, timestampActual, 0.02, 0.01);
        
        // Competición entre conexiones (más conservadora)
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
                // Ganancia más suave
                conexion.setRecursosAsignados(
                    Math.min(1.0, conexion.getRecursosAsignados() + 0.015)
                );
            } else {
                // Pérdida más suave para dar más oportunidades
                conexion.setRecursosAsignados(
                    Math.max(0.0, conexion.getRecursosAsignados() - 0.015)
                );
            }
        }
    }
    
    public int podarElementos(List<Conexion> conexiones) {
        List<Conexion> conexionesAPodar = new ArrayList<>();
        
        // Calcular cuántas conexiones podemos podar como máximo
        // Nunca podar más del 50% de las conexiones en una sola poda
        // Y siempre mantener al menos 20 conexiones mínimas
        int maxPodar = Math.max(0, Math.min(
            conexiones.size() / 2,  // Máximo 50%
            conexiones.size() - 20  // Mantener al menos 20
        ));
        
        for (Conexion conexion : conexiones) {
            if (conexion.debeSerPodada()) {
                conexionesAPodar.add(conexion);
            }
        }
        
        // Si vamos a podar demasiadas, solo podar las peores
        if (conexionesAPodar.size() > maxPodar) {
            // Ordenar por recursos (las de menos recursos primero)
            conexionesAPodar.sort((c1, c2) -> 
                Double.compare(c1.getRecursosAsignados(), c2.getRecursosAsignados())
            );
            // Solo podar las maxPodar peores
            conexionesAPodar = conexionesAPodar.subList(0, maxPodar);
        }
        
        // Eliminar conexiones podadas
        conexionesAPodar.stream().forEach(c -> conexiones.remove(c));
        
        return conexionesAPodar.size();
    }
}
