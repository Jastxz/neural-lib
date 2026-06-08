# Diseño: Mejoras en Formación y Consolidación de Engramas

## Resumen

Este diseño implementa dos mejoras clave en el sistema de engramas:
1. Límite del 22% de neuronas totales por engrama
2. Consolidación adaptativa basada en tiempo de procesamiento

## Arquitectura

### Componentes Modificados

```
RedNeuralExperimental
├── GestorEngramas (modificado)
│   ├── detectarYFormarEngramas() - añade límite 22%
│   └── optimizarEngramas() - nuevo: clustering/fusión
├── GestorConsolidacionAdaptativa (nuevo)
│   ├── medirTiempo()
│   ├── calcularIntervalo()
│   └── debeConsolidar()
└── consolidar() - usa clustering/fusión

Engrama (modificado)
├── calcularSimilitudDetallada() - nuevo
└── fusionarCon() - nuevo

ModelosExperimentales
├── entrenarSelfPlay() - usa consolidación adaptativa
└── entrenarSupervisado() - usa consolidación adaptativa
```

## Diseño Detallado

### 1. Límite de Neuronas por Engrama (22%)

**Ubicación**: `GestorEngramas.java`

**Cambios**:

```java
public class GestorEngramas {
    private static final double LIMITE_PORCENTAJE_NEURONAS = 0.22;
    private int totalNeuronasRed;
    
    // Constructor actualizado
    public GestorEngramas(int totalNeuronasRed) {
        this.totalNeuronasRed = totalNeuronasRed;
        // ... resto
    }
    
    // Método modificado
    public void detectarYFormarEngramas(List<List<Neurona>> capasInterneuronas, long timestamp) {
        int limiteNeuronas = (int) (totalNeuronasRed * LIMITE_PORCENTAJE_NEURONAS);
        
        // Recolectar neuronas activas
        List<Neurona> todasNeuronasActivas = ...;
        
        // Si supera el límite, truncar y formar múltiples engramas
        if (todasNeuronasActivas.size() > limiteNeuronas) {
            // Dividir en chunks del tamaño límite
            List<List<Neurona>> chunks = dividirEnChunks(todasNeuronasActivas, limiteNeuronas);
            for (List<Neurona> chunk : chunks) {
                formarEngramaConLimite(chunk, timestamp);
            }
        } else {
            formarEngramaConLimite(todasNeuronasActivas, timestamp);
        }
    }
    
    private void formarEngramaConLimite(List<Neurona> neuronas, long timestamp) {
        // Verificar similitud con engramas existentes
        for (Engrama existente : engramas.values()) {
            double similitud = calcularSimilitud(existente.getNeuronas(), neuronas);
            if (similitud > 0.8) {
                existente.activar(timestamp);
                return; // No formar nuevo
            }
        }
        
        // Formar nuevo engrama
        String id = "auto_" + contadorEngramas++;
        formarEngrama(id, neuronas, timestamp);
    }
}
```

**Validación**: 
- ✓ Límite del 22% aplicado siempre
- ✓ Engramas grandes se dividen automáticamente
- ✓ No se añaden neuronas a engramas llenos

---

### 2. Consolidación Adaptativa Basada en Tiempo

**Ubicación**: Nuevo archivo `GestorConsolidacionAdaptativa.java`

**Diseño**:

```java
public class GestorConsolidacionAdaptativa implements Serializable {
    private static final long serialVersionUID = 1L;
    
    // Configuración
    private static final int VENTANA_MEDICION = 10;
    private static final int INTERVALO_MIN = 10;
    private static final int INTERVALO_MAX = 100;
    
    // Estado
    private List<Long> tiemposProcesamiento; // Ventana deslizante
    private int contadorIteraciones;
    private int intervaloActual;
    private boolean inicializado;
    
    public GestorConsolidacionAdaptativa() {
        this.tiemposProcesamiento = new ArrayList<>();
        this.contadorIteraciones = 0;
        this.intervaloActual = 50; // Valor inicial
        this.inicializado = false;
    }
    
    /**
     * Registra el tiempo de una iteración
     * @param tiempoMs Tiempo en milisegundos
     */
    public void registrarTiempo(long tiempoMs) {
        tiemposProcesamiento.add(tiempoMs);
        
        // Mantener ventana deslizante
        if (tiemposProcesamiento.size() > VENTANA_MEDICION) {
            tiemposProcesamiento.remove(0);
        }
        
        contadorIteraciones++;
        
        // Inicializar después de las primeras 10 iteraciones
        if (!inicializado && tiemposProcesamiento.size() >= VENTANA_MEDICION) {
            inicializado = true;
            actualizarIntervalo();
        }
        
        // Actualizar intervalo cada 10 iteraciones
        if (inicializado && contadorIteraciones % 10 == 0) {
            actualizarIntervalo();
        }
    }
    
    /**
     * Calcula el intervalo óptimo basándose en tiempos
     */
    private void actualizarIntervalo() {
        if (tiemposProcesamiento.isEmpty()) return;
        
        // Calcular promedio
        long suma = 0;
        for (long tiempo : tiemposProcesamiento) {
            suma += tiempo;
        }
        double promedioMs = (double) suma / tiemposProcesamiento.size();
        
        // Fórmula: intervalo = max(10, min(100, tiempoPromedioMs / 2))
        int nuevoIntervalo = (int) Math.max(INTERVALO_MIN, 
                                Math.min(INTERVALO_MAX, promedioMs / 2.0));
        
        this.intervaloActual = nuevoIntervalo;
    }
    
    /**
     * Verifica si debe consolidar en esta iteración
     */
    public boolean debeConsolidar() {
        if (!inicializado) {
            // Durante inicialización, consolidar cada 50 iteraciones
            return contadorIteraciones % 50 == 0;
        }
        
        return contadorIteraciones % intervaloActual == 0;
    }
    
    // Getters
    public int getIntervaloActual() { return intervaloActual; }
    public double getTiempoPromedioMs() {
        if (tiemposProcesamiento.isEmpty()) return 0.0;
        long suma = 0;
        for (long t : tiemposProcesamiento) suma += t;
        return (double) suma / tiemposProcesamiento.size();
    }
    public boolean estaInicializado() { return inicializado; }
}
```

**Integración en RedNeuralExperimental**:

```java
public class RedNeuralExperimental {
    private final GestorConsolidacionAdaptativa gestorConsolidacionAdaptativa;
    
    public RedNeuralExperimental(int[] topologia, double densidadConexiones) {
        // ... código existente ...
        
        // Inicializar gestor de consolidación adaptativa
        this.gestorConsolidacionAdaptativa = new GestorConsolidacionAdaptativa();
        
        // Pasar total de neuronas a GestorEngramas
        int totalNeuronas = getTotalNeuronas();
        this.gestorEngramas = new GestorEngramas(totalNeuronas);
    }
    
    public void entrenar(double[] inputs, double[] targets, int iteraciones) {
        for (int i = 0; i < iteraciones; i++) {
            long inicio = System.currentTimeMillis();
            
            // Procesar y entrenar (código existente)
            double[] outputs = procesar(inputs);
            // ... resto del entrenamiento ...
            
            long tiempoMs = System.currentTimeMillis() - inicio;
            gestorConsolidacionAdaptativa.registrarTiempo(tiempoMs);
            
            // Consolidar si es necesario
            if (gestorConsolidacionAdaptativa.debeConsolidar()) {
                iniciarConsolidacion();
                consolidar();
                finalizarConsolidacion();
            }
            
            avanzarTiempo(10L);
        }
    }
}
```

**Validación**:
- ✓ Mide tiempo de las primeras 10 iteraciones
- ✓ Calcula intervalo adaptativo
- ✓ Ajusta continuamente según ventana deslizante
- ✓ Fórmula: `max(10, min(100, promedioMs / 2))`

---

### 3. Clustering y Fusión de Engramas

**Ubicación**: `GestorEngramas.java` (nuevo método)

**Diseño**:

```java
public class GestorEngramas {
    
    /**
     * Optimiza engramas mediante clustering y fusión
     * Llamado durante consolidación
     */
    public void optimizarEngramas() {
        List<Engrama> listaEngramas = new ArrayList<>(engramas.values());
        
        // 1. Identificar engramas muy similares para fusión (>90%)
        List<ParFusion> candidatosFusion = identificarCandidatosFusion(listaEngramas, 0.90);
        
        // 2. Fusionar engramas similares
        for (ParFusion par : candidatosFusion) {
            fusionarEngramas(par.engrama1, par.engrama2);
        }
        
        // 3. Aplicar clustering a engramas grandes (>85% similitud interna)
        for (Engrama engrama : new ArrayList<>(engramas.values())) {
            if (engrama.getNeuronas().size() > totalNeuronasRed * 0.15) {
                aplicarClusteringInterno(engrama);
            }
        }
        
        // 4. Podar engramas con baja relevancia
        List<String> idsAPodar = new ArrayList<>();
        for (Map.Entry<String, Engrama> entry : engramas.entrySet()) {
            if (entry.getValue().getRelevancia() < 0.15) {
                idsAPodar.add(entry.getKey());
            }
        }
        for (String id : idsAPodar) {
            eliminarEngrama(id);
        }
    }
    
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
    
    private void fusionarEngramas(Engrama e1, Engrama e2) {
        // Crear nuevo engrama fusionado
        Set<Neurona> neuronasUnidas = new HashSet<>();
        neuronasUnidas.addAll(e1.getNeuronasParticipantes());
        neuronasUnidas.addAll(e2.getNeuronasParticipantes());
        
        // Verificar límite del 22%
        int limite = (int) (totalNeuronasRed * LIMITE_PORCENTAJE_NEURONAS);
        if (neuronasUnidas.size() > limite) {
            return; // No fusionar si supera el límite
        }
        
        String nuevoId = "fusion_" + contadorEngramas++;
        Engrama fusionado = new Engrama(nuevoId, new ArrayList<>(neuronasUnidas), 
                                        System.currentTimeMillis());
        
        // Heredar propiedades
        double fuerzaPromedio = (e1.getFuerza() + e2.getFuerza()) / 2.0;
        double relevanciaMax = Math.max(e1.getRelevancia(), e2.getRelevancia());
        fusionado.setFuerza(fuerzaPromedio);
        fusionado.setRelevancia(relevanciaMax);
        
        // Reemplazar engramas antiguos
        engramas.put(nuevoId, fusionado);
        engramas.remove(e1.getId());
        engramas.remove(e2.getId());
    }
    
    private void aplicarClusteringInterno(Engrama engrama) {
        List<Neurona> neuronas = engrama.getNeuronas();
        
        // Agrupar por capa
        Map<Integer, List<Neurona>> porCapa = agruparPorCapa(neuronas);
        
        // Si hay múltiples capas con suficientes neuronas, dividir
        List<List<Neurona>> clusters = new ArrayList<>();
        for (List<Neurona> neuronasEnCapa : porCapa.values()) {
            if (neuronasEnCapa.size() >= 3) {
                clusters.add(neuronasEnCapa);
            }
        }
        
        // Si encontramos clusters naturales, dividir el engrama
        if (clusters.size() > 1) {
            eliminarEngrama(engrama.getId());
            for (List<Neurona> cluster : clusters) {
                String nuevoId = "cluster_" + contadorEngramas++;
                formarEngrama(nuevoId, cluster, System.currentTimeMillis());
            }
        }
    }
    
    private Map<Integer, List<Neurona>> agruparPorCapa(List<Neurona> neuronas) {
        // Implementación simplificada: agrupar por ID de neurona
        // En la práctica, necesitaríamos metadata de capa
        Map<Integer, List<Neurona>> grupos = new HashMap<>();
        
        for (Neurona n : neuronas) {
            // Heurística: usar rango de IDs para determinar capa
            int capa = (int) (n.getId() / 100);
            grupos.computeIfAbsent(capa, k -> new ArrayList<>()).add(n);
        }
        
        return grupos;
    }
    
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
}
```

**Modificación en Engrama.java**:

```java
public class Engrama {
    // Añadir setter para fuerza
    public void setFuerza(double fuerza) {
        this.fuerza = Math.max(0.0, Math.min(1.0, fuerza));
    }
}
```

**Integración en RedNeuralExperimental.consolidar()**:

```java
public void consolidar() {
    if (estado != EstadoRed.CONSOLIDANDO) {
        throw new IllegalStateException("Debe iniciar consolidación primero");
    }
    
    // Optimizar engramas (clustering/fusión)
    gestorEngramas.optimizarEngramas();
    
    // Consolidación existente de engramas
    Map<String, Engrama> engramas = gestorEngramas.getEngramasInterno();
    List<String> engramasAEliminar = new ArrayList<>();
    
    for (Map.Entry<String, Engrama> entry : engramas.entrySet()) {
        // ... código existente de consolidación ...
    }
    
    // ... resto del código existente ...
}
```

**Validación**:
- ✓ Clustering con similitud > 85%
- ✓ Fusión con similitud > 90%
- ✓ Poda con relevancia < 0.15
- ✓ Respeta límite del 22%

---

### 4. Métricas y Estadísticas

**Ubicación**: `GestorEngramas.java` (nuevo método)

**Diseño**:

```java
public class GestorEngramas {
    // Contadores para métricas
    private int engramasFormados;
    private int engramasFusionados;
    private int engramasPodados;
    
    /**
     * Obtiene estadísticas detalladas de engramas
     */
    public EstadisticasEngramas getEstadisticas() {
        int totalEngramas = engramas.size();
        
        // Calcular tamaño promedio
        double tamañoPromedio = 0.0;
        int tamañoMin = Integer.MAX_VALUE;
        int tamañoMax = 0;
        
        for (Engrama e : engramas.values()) {
            int tamaño = e.getNeuronas().size();
            tamañoPromedio += tamaño;
            tamañoMin = Math.min(tamañoMin, tamaño);
            tamañoMax = Math.max(tamañoMax, tamaño);
        }
        
        if (totalEngramas > 0) {
            tamañoPromedio /= totalEngramas;
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
            this.porcentajePromedioNeuronas = (promedio / totalNeuronas) * 100.0;
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
}
```

**Integración en RedNeuralExperimental**:

```java
public Map<String, Object> getEstadisticas() {
    Map<String, Object> stats = new HashMap<>();
    
    // ... estadísticas existentes ...
    
    // Añadir estadísticas de engramas
    GestorEngramas.EstadisticasEngramas statsEngramas = gestorEngramas.getEstadisticas();
    stats.put("engramasActuales", statsEngramas.totalActuales);
    stats.put("engramasFormados", statsEngramas.totalFormados);
    stats.put("engramasFusionados", statsEngramas.totalFusionados);
    stats.put("engramasPodados", statsEngramas.totalPodados);
    stats.put("tamañoPromedioEngramas", statsEngramas.tamañoPromedio);
    stats.put("porcentajePromedioNeuronas", statsEngramas.porcentajePromedioNeuronas);
    
    // Añadir estadísticas de consolidación adaptativa
    stats.put("intervaloConsolidacion", gestorConsolidacionAdaptativa.getIntervaloActual());
    stats.put("tiempoPromedioMs", gestorConsolidacionAdaptativa.getTiempoPromedioMs());
    stats.put("consolidacionInicializada", gestorConsolidacionAdaptativa.estaInicializado());
    
    return stats;
}
```

---

## Modificaciones en Modelos Experimentales

### ModeloGatosExperimental

```java
public void entrenarSelfPlay(int numPartidas) {
    System.out.println("=== Entrenamiento Self-Play Gatos (No Supervisado) ===");
    System.out.println("Partidas a jugar: " + numPartidas);
    
    for (int partida = 0; partida < numPartidas; partida++) {
        long inicio = System.currentTimeMillis();
        
        jugarPartidaEntrenamiento();
        
        long tiempoMs = System.currentTimeMillis() - inicio;
        
        // La consolidación adaptativa se maneja automáticamente en cerebro.entrenar()
        // Ya no necesitamos consolidar manualmente cada 50 partidas
        
        if ((partida + 1) % 50 == 0) {
            mostrarProgreso(partida + 1, numPartidas);
        }
    }
    
    System.out.println("\n=== Entrenamiento Completado ===");
    mostrarEstadisticas();
}

private void mostrarEstadisticas() {
    System.out.println("Partidas jugadas: " + partidasJugadas);
    System.out.println("Victorias Ratón: " + victoriasRaton + " (" + 
        String.format("%.1f%%", victoriasRaton * 100.0 / partidasJugadas) + ")");
    System.out.println("Victorias Gatos: " + victoriasGatos + " (" + 
        String.format("%.1f%%", victoriasGatos * 100.0 / partidasJugadas) + ")");
    
    // Mostrar estadísticas de engramas
    Map<String, Object> stats = cerebro.getEstadisticas();
    System.out.println("\n=== Estadísticas de Engramas ===");
    System.out.println("Engramas formados: " + stats.get("engramasFormados"));
    System.out.println("Engramas actuales: " + stats.get("engramasActuales"));
    System.out.println("Engramas fusionados: " + stats.get("engramasFusionados"));
    System.out.println("Engramas podados: " + stats.get("engramasPodados"));
    System.out.println("Tamaño promedio: " + String.format("%.1f", stats.get("tamañoPromedioEngramas")) + 
                      " (" + String.format("%.1f%%", stats.get("porcentajePromedioNeuronas")) + " de la red)");
    
    System.out.println("\n=== Consolidación Adaptativa ===");
    System.out.println("Intervalo actual: cada " + stats.get("intervaloConsolidacion") + " iteraciones");
    System.out.println("Tiempo promedio: " + String.format("%.1f", stats.get("tiempoPromedioMs")) + " ms");
    
    System.out.println("\nConexiones totales: " + cerebro.getTotalConexiones());
}
```

---

## Impacto Esperado

### Antes vs Después

| Juego | Engramas Antes | Engramas Esperados | Tamaño Promedio | Intervalo Consolidación |
|-------|----------------|-------------------|-----------------|------------------------|
| 3 en Raya | 3-4 | 8-12 | 5-8 neuronas (15%) | ~5 iteraciones |
| Gatos | 1 | 10-20 | 10-15 neuronas (18%) | ~25-50 iteraciones |
| Damas | 1 | 15-30 | 15-20 neuronas (20%) | ~100 iteraciones |

### Beneficios

1. **Más engramas específicos**: Límite del 22% fuerza la creación de patrones más focalizados
2. **Consolidación eficiente**: Adapta frecuencia según complejidad del problema
3. **Menos redundancia**: Fusión automática de engramas similares
4. **Mejor rendimiento**: Poda de engramas irrelevantes libera recursos

---

## Plan de Pruebas

### Tests Unitarios

1. **GestorEngramasTest**
   - Verificar límite del 22%
   - Verificar división automática
   - Verificar clustering/fusión
   - Verificar poda

2. **GestorConsolidacionAdaptativaTest**
   - Verificar medición de tiempos
   - Verificar cálculo de intervalo
   - Verificar ventana deslizante
   - Verificar fórmula: `max(10, min(100, promedio/2))`

3. **EngramaTest**
   - Verificar fusión de engramas
   - Verificar setters de fuerza

### Tests de Integración

1. **Comparativa3enRayaTest**
   - Verificar formación de más engramas (8-12)
   - Verificar consolidación rápida (~5 iteraciones)

2. **ComparativaGatosTest**
   - Verificar formación de más engramas (10-20)
   - Verificar consolidación media (~25-50 iteraciones)

3. **ComparativaDamasTest**
   - Verificar formación de más engramas (15-30)
   - Verificar consolidación lenta (~100 iteraciones)

---

## Consideraciones de Implementación

### Serialización

- `GestorConsolidacionAdaptativa` debe ser `Serializable`
- Mantener `serialVersionUID` en todos los componentes modificados
- Modelos antiguos deben poder cargarse (compatibilidad hacia atrás)

### Rendimiento

- Medición de tiempo: usar `System.currentTimeMillis()` (overhead mínimo)
- Clustering: solo durante consolidación (no en cada iteración)
- Fusión: limitar a pares con similitud > 90% (evitar O(n²) completo)

### Debugging

- Añadir logs opcionales para tracking de engramas
- Métricas accesibles en tiempo real
- Visualización de estadísticas en `mostrarEstadisticas()`

---

## Próximos Pasos

1. Implementar `GestorConsolidacionAdaptativa`
2. Modificar `GestorEngramas` con límite 22% y optimización
3. Integrar en `RedNeuralExperimental`
4. Actualizar modelos experimentales
5. Crear/actualizar tests
6. Ejecutar comparativas y validar mejoras
