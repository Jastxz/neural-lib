package es.jastxz.nn.experimental;

import es.jastxz.nn.Engrama;
import es.jastxz.nn.Neurona;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gestiona la detección y formación de engramas
 * Implementa el principio de Campillo (pg. 84-85): "conjunto de neuronas que funcionan como bits"
 */
public class GestorEngramas {
    
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
        Engrama engrama = engramas.remove(id);
        if (engrama != null) {
            // Desregistrar de neuronas
            for (Neurona neurona : engrama.getNeuronas()) {
                neurona.abandonarEngrama(id);
            }
        }
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
        
        for (Neurona neurona : participantes) {
            neurona.unirseAEngrama(id);
        }
    }
    
    public void activarEngrama(String id, long timestamp) {
        Engrama engrama = engramas.get(id);
        
        if (engrama == null) {
            throw new IllegalArgumentException("No existe un engrama con ID: " + id);
        }
        
        engrama.activar(timestamp);
    }
    
    public void detectarYFormarEngramas(List<List<Neurona>> capasInterneuronas, long timestamp) {
        if (!deteccionActiva) {
            return;
        }
        
        for (int i = 0; i < capasInterneuronas.size(); i++) {
            List<Neurona> capa = capasInterneuronas.get(i);
            
            List<Neurona> neuronasActivas = new ArrayList<>();
            for (Neurona neurona : capa) {
                if (neurona.estaActiva()) {
                    neuronasActivas.add(neurona);
                }
            }
            
            if (neuronasActivas.size() >= 2) {
                boolean yaExiste = false;
                for (Engrama engramaExistente : engramas.values()) {
                    if (engramaExistente.contieneNeuronas(neuronasActivas, 0.7)) {
                        yaExiste = true;
                        break;
                    }
                }
                
                if (!yaExiste) {
                    String id = "auto_" + contadorEngramas++;
                    formarEngrama(id, neuronasActivas, timestamp);
                }
            }
        }
    }
}
