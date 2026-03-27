package es.jastxz.nn.experimental;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Gestiona la consolidación adaptativa basada en tiempo de procesamiento
 * 
 * Implementa consolidación dinámica que se ajusta según la complejidad del procesamiento:
 * - Procesamiento rápido → consolidación menos frecuente
 * - Procesamiento lento → consolidación más frecuente
 * 
 * Fórmula: intervalo = max(10, min(100, tiempoPromedioMs / 2))
 * 
 * Basado en el principio biológico de que el sueño/consolidación es más necesario
 * después de períodos de aprendizaje intenso.
 */
public class GestorConsolidacionAdaptativa implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Configuración
    private static final int VENTANA_MEDICION = 10;
    private static final int INTERVALO_MIN = 10;
    private static final int INTERVALO_MAX = 100;
    private static final int INTERVALO_INICIAL = 50;
    
    // Estado
    private Deque<Long> tiemposProcesamiento; // Ventana deslizante (ArrayDeque: O(1) add/remove)
    private long sumaTiempos; // Running sum para evitar recalcular
    private int contadorIteraciones;
    private int intervaloActual;
    private boolean inicializado;
    
    /**
     * Constructor
     */
    public GestorConsolidacionAdaptativa() {
        this.tiemposProcesamiento = new ArrayDeque<>(VENTANA_MEDICION);
        this.sumaTiempos = 0L;
        this.contadorIteraciones = 0;
        this.intervaloActual = INTERVALO_INICIAL;
        this.inicializado = false;
    }
    
    /**
     * Registra el tiempo de una iteración
     * Actualiza la ventana deslizante y ajusta el intervalo si es necesario
     * 
     * @param tiempoMs Tiempo en milisegundos
     */
    public void registrarTiempo(long tiempoMs) {
        tiemposProcesamiento.addLast(tiempoMs);
        sumaTiempos += tiempoMs;
        
        // Mantener ventana deslizante de tamaño VENTANA_MEDICION
        if (tiemposProcesamiento.size() > VENTANA_MEDICION) {
            sumaTiempos -= tiemposProcesamiento.removeFirst();
        }
        
        contadorIteraciones++;
        
        // Inicializar después de las primeras VENTANA_MEDICION iteraciones
        if (!inicializado && tiemposProcesamiento.size() >= VENTANA_MEDICION) {
            inicializado = true;
            actualizarIntervalo();
        }
        
        // Actualizar intervalo cada 10 iteraciones después de inicializar
        if (inicializado && contadorIteraciones % 10 == 0) {
            actualizarIntervalo();
        }
    }
    
    /**
     * Calcula el intervalo óptimo basándose en tiempos de procesamiento
     * Fórmula: intervalo = max(10, min(100, tiempoPromedioMs / 2))
     */
    private void actualizarIntervalo() {
        if (tiemposProcesamiento.isEmpty()) {
            return;
        }
        
        double promedioMs = (double) sumaTiempos / tiemposProcesamiento.size();
        
        // Aplicar fórmula: max(10, min(100, promedioMs / 2))
        int nuevoIntervalo = (int) Math.max(INTERVALO_MIN, 
                                Math.min(INTERVALO_MAX, promedioMs / 2.0));
        
        this.intervaloActual = nuevoIntervalo;
    }
    
    /**
     * Verifica si debe consolidar en esta iteración
     * 
     * @return true si debe consolidar, false en caso contrario
     */
    public boolean debeConsolidar() {
        if (!inicializado) {
            // Durante inicialización, consolidar cada INTERVALO_INICIAL iteraciones
            return contadorIteraciones > 0 && contadorIteraciones % INTERVALO_INICIAL == 0;
        }
        
        // Después de inicializar, usar intervalo adaptativo
        return contadorIteraciones % intervaloActual == 0;
    }
    
    /**
     * Obtiene el intervalo actual de consolidación
     * 
     * @return Número de iteraciones entre consolidaciones
     */
    public int getIntervaloActual() {
        return intervaloActual;
    }
    
    /**
     * Obtiene el tiempo promedio de procesamiento
     * 
     * @return Tiempo promedio en milisegundos, 0.0 si no hay datos
     */
    public double getTiempoPromedioMs() {
        if (tiemposProcesamiento.isEmpty()) {
            return 0.0;
        }
        
        return (double) sumaTiempos / tiemposProcesamiento.size();
    }
    
    /**
     * Verifica si el gestor está inicializado
     * 
     * @return true si ya se midieron las primeras 10 iteraciones
     */
    public boolean estaInicializado() {
        return inicializado;
    }
    
    /**
     * Obtiene el contador de iteraciones
     * 
     * @return Número total de iteraciones registradas
     */
    public int getContadorIteraciones() {
        return contadorIteraciones;
    }
    
    /**
     * Obtiene el tamaño de la ventana de medición
     * 
     * @return Número de iteraciones en la ventana deslizante
     */
    public int getTamañoVentana() {
        return tiemposProcesamiento.size();
    }
}
