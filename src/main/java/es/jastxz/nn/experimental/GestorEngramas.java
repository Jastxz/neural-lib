package es.jastxz.nn.experimental;

import es.jastxz.nn.Engrama;
import es.jastxz.nn.Neurona;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.io.Serializable;

/**
 * Gestiona la detección y formación de engramas
 * Implementa el principio de Campillo (pg. 84-85): "conjunto de neuronas que funcionan como bits"
 * 
 * MEJORAS:
 * - Límite del 22% de neuronas totales por engrama
 * - Clustering y fusión durante consolidación
 * - Métricas detalladas de formación
 */
public class GestorEngramas implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Límite de neuronas por engrama (22% del total)
    private static final double LIMITE_PORCENTAJE_NEURONAS = 0.22;
    
    private boolean deteccionActiva;
    private int contadorEngramas;
    private Map<String, Engrama> engramas;
    private int totalNeuronasRed;
    
    // Contadores para métricas
    private int engramasFormados;
    private int engramasFusionados;
    private int engramasPodados;
    
    public GestorEngramas(int totalNeuronasRed) {
        this.totalNeuronasRed = totalNeuronasRed;
        this.deteccionActiva = false;
        this.contadorEngramas = 0;
        this.engramas = new HashMap<>();
        this.engramasFormados = 0;
        this.engramasFusionados = 0;
        this.engramasPodados = 0;
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
        engramasFormados++; // Incrementar contador
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
     * MEJORADO: Detecta patrones más diversos considerando combinaciones únicas
     * LÍMITE: Máximo 22% de neuronas totales por engrama
     */
    public void detectarYFormarEngramas(List<List<Neurona>> capasInterneuronas, long timestamp) {
        if (!deteccionActiva) {
            return;
        }
        
        // Calcular límite de neuronas por engrama (22% del total)
        int limiteNeuronas = (int) (totalNeuronasRed * LIMITE_PORCENTAJE_NEURONAS);
        
        // Recolectar todas las neuronas activas de todas las capas
        List<Neurona> todasNeuronasActivas = new ArrayList<>();
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona neurona : capa) {
                if (neurona.estaActiva()) {
                    todasNeuronasActivas.add(neurona);
                }
            }
        }
        
        // Si hay suficientes neuronas activas, considerar formar un engrama global
        if (todasNeuronasActivas.size() >= 2) {
            // Si supera el límite, dividir en chunks
            if (todasNeuronasActivas.size() > limiteNeuronas) {
                List<List<Neurona>> chunks = dividirEnChunks(todasNeuronasActivas, limiteNeuronas);
                for (List<Neurona> chunk : chunks) {
                    formarEngramaConLimite(chunk, timestamp);
                }
            } else {
                formarEngramaConLimite(todasNeuronasActivas, timestamp);
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
            if (neuronasActivasCapa.size() >= 2) {
                // Verificar límite también para engramas locales
                if (neuronasActivasCapa.size() > limiteNeuronas) {
                    List<List<Neurona>> chunks = dividirEnChunks(neuronasActivasCapa, limiteNeuronas);
                    for (List<Neurona> chunk : chunks) {
                        formarEngramaLocalConLimite(chunk, i, timestamp);
                    }
                } else {
                    formarEngramaLocalConLimite(neuronasActivasCapa, i, timestamp);
                }
            }
        }
    }
    
    /**
     * Divide una lista de neuronas en chunks del tamaño especificado
     */
    private List<List<Neurona>> dividirEnChunks(List<Neurona> neuronas, int tamañoChunk) {
        List<List<Neurona>> chunks = new ArrayList<>();
        
        for (int i = 0; i < neuronas.size(); i += tamañoChunk) {
            int fin = Math.min(i + tamañoChunk, neuronas.size());
            chunks.add(new ArrayList<>(neuronas.subList(i, fin)));
        }
        
        return chunks;
    }
    
    /**
     * Forma un engrama verificando similitud con existentes
     * No forma si ya existe uno muy similar (>80%)
     */
    private void formarEngramaConLimite(List<Neurona> neuronas, long timestamp) {
        // Verificar similitud con engramas existentes
        for (Engrama engramaExistente : engramas.values()) {
            double similitud = calcularSimilitud(engramaExistente.getNeuronas(), neuronas);
            
            // Si la similitud es muy alta (>80%), reforzar existente en lugar de crear nuevo
            if (similitud > 0.8) {
                engramaExistente.activar(timestamp);
                return;
            }
        }
        
        // Formar nuevo engrama
        String id = "auto_" + contadorEngramas++;
        formarEngrama(id, neuronas, timestamp);
        engramasFormados++;
    }
    
    /**
     * Forma un engrama local (por capa) verificando similitud
     */
    private void formarEngramaLocalConLimite(List<Neurona> neuronas, int indiceCapa, long timestamp) {
        // Verificar similitud con engramas existentes
        for (Engrama engramaExistente : engramas.values()) {
            double similitud = calcularSimilitud(engramaExistente.getNeuronas(), neuronas);
            
            // Para patrones locales, usar umbral más alto (85%)
            if (similitud > 0.85) {
                engramaExistente.activar(timestamp);
                return;
            }
        }
        
        // Formar nuevo engrama local
        String id = "local_capa" + indiceCapa + "_" + contadorEngramas++;
        formarEngrama(id, neuronas, timestamp);
        engramasFormados++;
    }
    
    /**
     * Calcula la similitud bidireccional entre dos conjuntos de neuronas
     * Considera tanto el solapamiento como el tamaño de los conjuntos
     * 
     * @param conjunto1 Primer conjunto de neuronas
     * @param conjunto2 Segundo conjunto de neuronas
     * @return Valor entre 0.0 (totalmente diferentes) y 1.0 (idénticos)
     */
    private double calcularSimilitud(List<Neurona> conjunto1, List<Neurona> conjunto2) {
        if (conjunto1.isEmpty() || conjunto2.isEmpty()) {
            return 0.0;
        }
        
        // Contar neuronas comunes
        long neuronasComunes = conjunto1.stream()
            .filter(conjunto2::contains)
            .count();
        
        // Calcular similitud como promedio de ambas direcciones
        double similitud1 = (double) neuronasComunes / conjunto1.size();
        double similitud2 = (double) neuronasComunes / conjunto2.size();
        
        // Usar el promedio para considerar ambas direcciones
        return (similitud1 + similitud2) / 2.0;
    }
    
    /**
     * Optimiza engramas mediante clustering y fusión
     * Llamado durante consolidación para reducir redundancia
     * 
     * Proceso:
     * 1. Identificar engramas muy similares para fusión (>90%)
     * 2. Fusionar engramas similares
     * 3. Aplicar clustering a engramas grandes (>15% de neuronas)
     * 4. Podar engramas con baja relevancia (<0.05)
     * 
     * AJUSTADO: Umbral de poda reducido para mantener más engramas
     */
    public void optimizarEngramas() {
        List<Engrama> listaEngramas = new ArrayList<>(engramas.values());
        
        // 1. Identificar engramas muy similares para fusión (>90%)
        List<ParFusion> candidatosFusion = identificarCandidatosFusion(listaEngramas, 0.90);
        
        // 2. Fusionar engramas similares
        for (ParFusion par : candidatosFusion) {
            fusionarEngramas(par.engrama1, par.engrama2);
        }
        
        // 3. Aplicar clustering a engramas grandes (>15% similitud interna)
        for (Engrama engrama : new ArrayList<>(engramas.values())) {
            if (engrama.getNeuronas().size() > totalNeuronasRed * 0.15) {
                aplicarClusteringInterno(engrama);
            }
        }
        
        // 4. Podar engramas con baja relevancia
        // AJUSTADO: Umbral reducido a 0.05 (antes 0.15) para mantener más engramas
        List<String> idsAPodar = new ArrayList<>();
        for (Map.Entry<String, Engrama> entry : engramas.entrySet()) {
            if (entry.getValue().getRelevancia() < 0.05) {
                idsAPodar.add(entry.getKey());
            }
        }
        for (String id : idsAPodar) {
            eliminarEngrama(id);
            engramasPodados++;
        }
    }
    
    /**
     * Identifica pares de engramas candidatos para fusión
     * 
     * @param engramas Lista de engramas a analizar
     * @param umbral Umbral de similitud mínimo (0.0 a 1.0)
     * @return Lista de pares ordenados por similitud descendente
     */
    private List<ParFusion> identificarCandidatosFusion(List<Engrama> engramas, double umbral) {
        List<ParFusion> candidatos = new ArrayList<>();
        
        for (int i = 0; i < engramas.size(); i++) {
            for (int j = i + 1; j < engramas.size(); j++) {
                Engrama e1 = engramas.get(i);
                Engrama e2 = engramas.get(j);
                
                double similitud = calcularSimilitud(e1.getNeuronas(), e2.getNeuronas());
                if (similitud >= umbral) {
                    candidatos.add(new ParFusion(e1, e2, similitud));
                }
            }
        }
        
        // Ordenar por similitud descendente
        candidatos.sort((a, b) -> Double.compare(b.similitud, a.similitud));
        
        return candidatos;
    }
    
    /**
     * Fusiona dos engramas en uno solo
     * Respeta el límite del 22% de neuronas
     * 
     * @param e1 Primer engrama
     * @param e2 Segundo engrama
     */
    private void fusionarEngramas(Engrama e1, Engrama e2) {
        // Verificar que ambos engramas aún existen
        if (!engramas.containsKey(e1.getId()) || !engramas.containsKey(e2.getId())) {
            return; // Ya fueron fusionados o eliminados
        }
        
        // Crear conjunto unido de neuronas
        Set<Neurona> neuronasUnidas = new HashSet<>();
        neuronasUnidas.addAll(e1.getNeuronasParticipantes());
        neuronasUnidas.addAll(e2.getNeuronasParticipantes());
        
        // Verificar límite del 22%
        int limite = (int) (totalNeuronasRed * LIMITE_PORCENTAJE_NEURONAS);
        if (neuronasUnidas.size() > limite) {
            return; // No fusionar si supera el límite
        }
        
        // Crear nuevo engrama fusionado
        String nuevoId = "fusion_" + contadorEngramas++;
        Engrama fusionado = new Engrama(nuevoId, new ArrayList<>(neuronasUnidas), 
                                        System.currentTimeMillis());
        
        // Heredar propiedades (promedio de fuerza, máximo de relevancia)
        double fuerzaPromedio = (e1.getFuerza() + e2.getFuerza()) / 2.0;
        double relevanciaMax = Math.max(e1.getRelevancia(), e2.getRelevancia());
        fusionado.setFuerza(fuerzaPromedio);
        fusionado.setRelevancia(relevanciaMax);
        
        // Reemplazar engramas antiguos
        engramas.put(nuevoId, fusionado);
        engramas.remove(e1.getId());
        engramas.remove(e2.getId());
        
        engramasFusionados++;
    }
    
    /**
     * Aplica clustering interno a un engrama grande
     * Divide el engrama en sub-engramas más específicos si encuentra clusters naturales
     * 
     * @param engrama Engrama a analizar
     */
    private void aplicarClusteringInterno(Engrama engrama) {
        List<Neurona> neuronas = engrama.getNeuronas();
        
        // Agrupar por capa (heurística simple)
        Map<Integer, List<Neurona>> porCapa = agruparPorCapa(neuronas);
        
        // Si hay múltiples capas con suficientes neuronas, dividir
        List<List<Neurona>> clusters = new ArrayList<>();
        for (List<Neurona> neuronasEnCapa : porCapa.values()) {
            if (neuronasEnCapa.size() >= 3) {
                clusters.add(neuronasEnCapa);
            }
        }
        
        // Si encontramos clusters naturales (más de 1), dividir el engrama
        if (clusters.size() > 1) {
            eliminarEngrama(engrama.getId());
            for (List<Neurona> cluster : clusters) {
                String nuevoId = "cluster_" + contadorEngramas++;
                formarEngrama(nuevoId, cluster, System.currentTimeMillis());
                engramasFormados++;
            }
        }
    }
    
    /**
     * Agrupa neuronas por capa usando heurística de IDs
     * 
     * @param neuronas Lista de neuronas a agrupar
     * @return Mapa de capa -> lista de neuronas
     */
    private Map<Integer, List<Neurona>> agruparPorCapa(List<Neurona> neuronas) {
        Map<Integer, List<Neurona>> grupos = new HashMap<>();
        
        for (Neurona n : neuronas) {
            // Heurística: usar rango de IDs para determinar capa
            // Asumiendo que IDs se asignan secuencialmente por capa
            int capa = (int) (n.getId() / 100);
            grupos.computeIfAbsent(capa, k -> new ArrayList<>()).add(n);
        }
        
        return grupos;
    }
    
    /**
     * Clase interna para representar un par de engramas candidatos a fusión
     */
    private static class ParFusion {
        Engrama engrama1;
        Engrama engrama2;
        double similitud;
        
        ParFusion(Engrama e1, Engrama e2, double sim) {
            this.engrama1 = e1;
            this.engrama2 = e2;
            this.similitud = sim;
        }
    }

    
    /**
     * Obtiene estadísticas detalladas de engramas
     * 
     * @return Objeto con todas las métricas de engramas
     */
    public EstadisticasEngramas getEstadisticas() {
        int totalEngramas = engramas.size();
        
        // Calcular tamaño promedio, mínimo y máximo
        double tamañoPromedio = 0.0;
        int tamañoMin = Integer.MAX_VALUE;
        int tamañoMax = 0;
        
        if (totalEngramas > 0) {
            for (Engrama e : engramas.values()) {
                int tamaño = e.getNeuronas().size();
                tamañoPromedio += tamaño;
                tamañoMin = Math.min(tamañoMin, tamaño);
                tamañoMax = Math.max(tamañoMax, tamaño);
            }
            tamañoPromedio /= totalEngramas;
        } else {
            tamañoMin = 0;
        }
        
        return new EstadisticasEngramas(
            totalEngramas,
            engramasFormados,
            engramasFusionados,
            engramasPodados,
            tamañoPromedio,
            tamañoMin,
            tamañoMax,
            totalNeuronasRed
        );
    }
    
    /**
     * Clase interna para estadísticas de engramas
     */
    public static class EstadisticasEngramas {
        public final int totalActuales;
        public final int totalFormados;
        public final int totalFusionados;
        public final int totalPodados;
        public final double tamañoPromedio;
        public final int tamañoMin;
        public final int tamañoMax;
        public final int totalNeuronasRed;
        public final double porcentajePromedioNeuronas;
        
        public EstadisticasEngramas(int actuales, int formados, int fusionados, int podados,
                                   double promedio, int min, int max, int totalNeuronas) {
            this.totalActuales = actuales;
            this.totalFormados = formados;
            this.totalFusionados = fusionados;
            this.totalPodados = podados;
            this.tamañoPromedio = promedio;
            this.tamañoMin = min;
            this.tamañoMax = max;
            this.totalNeuronasRed = totalNeuronas;
            this.porcentajePromedioNeuronas = totalNeuronas > 0 ? 
                (promedio / totalNeuronas) * 100.0 : 0.0;
        }
        
        @Override
        public String toString() {
            return String.format(
                "Engramas: %d actuales (%d formados, %d fusionados, %d podados)\n" +
                "Tamaño: promedio=%.1f (%.1f%%), min=%d, max=%d",
                totalActuales, totalFormados, totalFusionados, totalPodados,
                tamañoPromedio, porcentajePromedioNeuronas, tamañoMin, tamañoMax
            );
        }
    }

    /**
     * Activa engramas parcialmente activos y facilita sus neuronas
     * Implementa el principio de Campillo (pg. 93): "el cerebro completa recuerdos parciales"
     * 
     * Proceso:
     * 1. Detectar engramas con activación parcial (>30% de neuronas activas)
     * 2. Facilitar las neuronas restantes del engrama (reducir umbral)
     * 3. Esto permite que el patrón se complete en la siguiente propagación
     * 
     * IMPORTANTE: Solo facilita, no activa directamente (para no interferir con procesamiento)
     * AJUSTADO: Umbral reducido a 30% para activación más agresiva
     * 
     * @param timestamp Timestamp actual para registrar activación
     */
    public void activarEngramasParciales(long timestamp) {
        if (!deteccionActiva) {
            return;
        }
        
        for (Engrama engrama : engramas.values()) {
            List<Neurona> neuronasEngrama = engrama.getNeuronas();
            
            if (neuronasEngrama.isEmpty()) {
                continue;
            }
            
            // Contar cuántas neuronas del engrama están activas
            int neuronasActivas = 0;
            for (Neurona n : neuronasEngrama) {
                if (n.estaActiva()) {
                    neuronasActivas++;
                }
            }
            
            // Calcular porcentaje de activación
            double porcentajeActivo = (double) neuronasActivas / neuronasEngrama.size();
            
            // AJUSTADO: Umbral reducido a 30% (antes 50%) para activación más agresiva
            // Esto permite que los engramas se activen con menos neuronas activas
            if (porcentajeActivo > 0.3 && porcentajeActivo < 1.0) {
                // Activar el engrama (registrar uso)
                engrama.activar(timestamp);
                
                // Facilitar las neuronas inactivas del engrama
                // Esto reduce su umbral de activación temporalmente
                double umbral = es.jastxz.nn.enums.PotencialMemoria.UMBRAL.getValor();
                
                for (Neurona n : neuronasEngrama) {
                    if (!n.estaActiva()) {
                        double potencialActual = n.getPotencial();
                        
                        // AJUSTADO: Umbral reducido a 50% (antes 70%) para facilitar más neuronas
                        if (potencialActual > umbral * 0.5) {
                            // Facilitar activación (reduce umbral temporalmente)
                            // Factor basado en qué tan activo está el engrama
                            n.facilitarActivacion(porcentajeActivo);
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Obtiene engramas que coinciden con el patrón de activación actual
     * Útil para análisis y debugging
     * 
     * @return Lista de IDs de engramas activos (>30% de neuronas activas)
     */
    public List<String> getEngramasActivos() {
        List<String> activos = new ArrayList<>();
        
        for (Map.Entry<String, Engrama> entry : engramas.entrySet()) {
            List<Neurona> neuronasEngrama = entry.getValue().getNeuronas();
            
            if (neuronasEngrama.isEmpty()) {
                continue;
            }
            
            int neuronasActivas = 0;
            for (Neurona n : neuronasEngrama) {
                if (n.estaActiva()) {
                    neuronasActivas++;
                }
            }
            
            double porcentajeActivo = (double) neuronasActivas / neuronasEngrama.size();
            
            if (porcentajeActivo > 0.3) {
                activos.add(entry.getKey());
            }
        }
        
        return activos;
    }
}
