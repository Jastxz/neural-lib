package es.jastxz.nn.experimental;

import es.jastxz.nn.Engrama;
import es.jastxz.nn.Neurona;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.Serializable;

/**
 * Gestiona la detección y formación de engramas
 * Implementa el principio de Campillo (pg. 84-85): "conjunto de neuronas que funcionan como bits"
 */
public class GestorEngramas implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private boolean deteccionActiva;
    private int contadorEngramas;
    private Map<String, Engrama> engramas;
    
    public GestorEngramas() {
        this.deteccionActiva = false;
        this.contadorEngramas = 0;
        this.engramas = new HashMap<>();
    }
    
    public void activarDeteccion(boolean activar) {
        this.deteccionActiva = activar;
    }
    
    public boolean esDeteccionActiva() {
        return deteccionActiva;
    }
    
    public Map<String, Engrama> getEngramas() {
        return new HashMap<>(engramas);
    }
    
    /**
     * Obtiene el mapa interno de engramas (para consolidación)
     * CUIDADO: Modificar este mapa afecta directamente al gestor
     */
    public Map<String, Engrama> getEngramasInterno() {
        return engramas;
    }
    
    /**
     * Elimina un engrama por ID
     */
    public void eliminarEngrama(String id) {
        engramas.remove(id);
        // No necesitamos desregistrar de neuronas - sin referencias bidireccionales
    }
    
    public void formarEngrama(String id, List<Neurona> participantes, long timestamp) {
        if (id == null || id.isEmpty()) {
            throw new IllegalArgumentException("El ID del engrama no puede ser null o vacío");
        }
        
        if (participantes == null || participantes.isEmpty()) {
            throw new IllegalArgumentException("Debe haber al menos una neurona participante");
        }
        
        Engrama engrama = new Engrama(id, participantes, timestamp);
        engramas.put(id, engrama);
        // No registrar en neuronas - sin referencias bidireccionales
    }
    
    public void activarEngrama(String id, long timestamp) {
        Engrama engrama = engramas.get(id);
        
        if (engrama == null) {
            throw new IllegalArgumentException("No existe un engrama con ID: " + id);
        }
        
        engrama.activar(timestamp);
    }
    
    /**
     * Detecta y forma engramas basándose en patrones de activación
     * MEJORADO: Detecta patrones más diversos y no solo por capa
     */
    public void detectarYFormarEngramas(List<List<Neurona>> capasInterneuronas, long timestamp) {
        if (!deteccionActiva) {
            return;
        }
        
        // Recolectar todas las neuronas activas de todas las capas
        List<Neurona> todasNeuronasActivas = new ArrayList<>();
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona neurona : capa) {
                if (neurona.estaActiva()) {
                    todasNeuronasActivas.add(neurona);
                }
            }
        }
        
        // Si hay suficientes neuronas activas, considerar formar un engrama
        if (todasNeuronasActivas.size() >= 2) {
            // Verificar si ya existe un engrama similar
            boolean yaExiste = false;
            for (Engrama engramaExistente : engramas.values()) {
                // Reducir umbral de solapamiento de 0.7 a 0.5 para permitir más variación
                if (engramaExistente.contieneNeuronas(todasNeuronasActivas, 0.5)) {
                    yaExiste = true;
                    // Reforzar el engrama existente
                    engramaExistente.activar(timestamp);
                    break;
                }
            }
            
            // Si no existe, formar nuevo engrama
            if (!yaExiste) {
                String id = "auto_" + contadorEngramas++;
                formarEngrama(id, todasNeuronasActivas, timestamp);
            }
        }
        
        // ADICIONAL: Detectar patrones por capa también (para patrones locales)
        for (int i = 0; i < capasInterneuronas.size(); i++) {
            List<Neurona> capa = capasInterneuronas.get(i);
            
            List<Neurona> neuronasActivasCapa = new ArrayList<>();
            for (Neurona neurona : capa) {
                if (neurona.estaActiva()) {
                    neuronasActivasCapa.add(neurona);
                }
            }
            
            // Formar engramas locales si hay suficientes neuronas activas
            if (neuronasActivasCapa.size() >= 3) {
                boolean yaExisteLocal = false;
                for (Engrama engramaExistente : engramas.values()) {
                    if (engramaExistente.contieneNeuronas(neuronasActivasCapa, 0.6)) {
                        yaExisteLocal = true;
                        engramaExistente.activar(timestamp);
                        break;
                    }
                }
                
                if (!yaExisteLocal) {
                    String id = "local_capa" + i + "_" + contadorEngramas++;
                    formarEngrama(id, neuronasActivasCapa, timestamp);
                }
            }
        }
    }
}
