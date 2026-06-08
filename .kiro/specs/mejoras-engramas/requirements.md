# Mejoras en Formación y Consolidación de Engramas

## Contexto

Actualmente, la formación de engramas y la consolidación tienen limitaciones:
- Los engramas pueden crecer sin límite, incluyendo potencialmente todas las neuronas de la red
- La consolidación solo se activa por número de iteraciones/partidas, no considera el tiempo de procesamiento
- En juegos complejos (Gatos, Damas) se forman muy pocos engramas (solo 1)

## Objetivo

Mejorar la formación y consolidación de engramas para que sean más eficientes, específicos y adaptativos al contexto de procesamiento.

## User Stories

### 1. Límite de Neuronas por Engrama

**Como** sistema de memoria neuronal  
**Quiero** limitar el número de neuronas que puede contener un engrama  
**Para** evitar saturación de recursos y formar patrones más específicos

**Criterios de aceptación:**
- 1.1 Los engramas tienen un tamaño máximo del 22% del total de neuronas de la red
- 1.2 Si un patrón de activación supera el límite, NO se añaden más neuronas a ese engrama
- 1.3 Si se detecta un patrón que supera el límite, se forma un nuevo engrama con las neuronas restantes
- 1.4 El límite del 22% se aplica siempre, sin configuración
- 1.5 Durante la consolidación, se revisan engramas para clustering y fusión/poda

### 2. Consolidación Adaptativa Basada en Tiempo

**Como** sistema de aprendizaje  
**Quiero** consolidar más frecuentemente cuando el procesamiento es lento o intenso  
**Para** optimizar el uso de recursos y evitar pérdida de información

**Criterios de aceptación:**
- 2.1 Se mide el tiempo de procesamiento de las primeras 10 iteraciones/partidas
- 2.2 Basándose en el tiempo promedio, se calcula el intervalo óptimo de consolidación
- 2.3 El intervalo se ajusta continuamente según el tiempo de procesamiento actual
- 2.4 Fórmula: `intervalo = max(10, min(100, tiempoPromedioMs / 2))`
- 2.5 Se mantiene un historial de tiempos de las últimas 10 iteraciones (ventana deslizante)
- 2.6 La consolidación adaptativa está siempre activa (no configurable)

### 3. Clustering y Fusión de Engramas Durante Consolidación

**Como** sistema de consolidación  
**Quiero** analizar engramas similares y optimizarlos  
**Para** evitar redundancia y mejorar la eficiencia de memoria

**Criterios de aceptación:**
- 3.1 Durante consolidación, se aplica clustering a engramas con similitud > 85%
- 3.2 Engramas muy similares (>90% similitud) se fusionan en uno solo
- 3.3 Engramas con baja relevancia (<0.15) se podan
- 3.4 El clustering identifica sub-grupos naturales dentro de engramas grandes
- 3.5 Se mantiene la diversidad de patrones (no fusionar todos)

### 4. Métricas y Monitoreo

**Como** desarrollador  
**Quiero** monitorear la formación de engramas y la consolidación  
**Para** entender el comportamiento del sistema y ajustar parámetros

**Criterios de aceptación:**
- 4.1 Se registra el número de engramas formados por sesión de entrenamiento
- 4.2 Se registra el tamaño promedio de los engramas
- 4.3 Se registra la frecuencia de consolidación adaptativa
- 4.4 Se registra el número de engramas fusionados/podados por consolidación
- 4.5 Se puede obtener un reporte de estadísticas de engramas
- 4.6 Las métricas son accesibles durante y después del entrenamiento

## Restricciones Técnicas

- Mantener serialización funcional de modelos existentes
- No romper tests existentes (ajustar si es necesario)
- Rendimiento: la medición de tiempo debe ser mínima (<1% overhead)
- Las mejoras están siempre activas (no opcionales)

## Notas de Diseño

### Límite de Neuronas
- Límite fijo: `totalNeuronas * 0.22` (22% del total)
- Estrategia: rechazar neuronas adicionales, formar nuevo engrama si es necesario
- Durante consolidación: aplicar clustering para encontrar sub-grupos naturales

### Consolidación Adaptativa
- Medición inicial: promedio de las primeras 10 iteraciones
- Fórmula de intervalo: `max(10, min(100, tiempoPromedioMs / 2))`
- Ventana deslizante: últimas 10 iteraciones
- Ajuste continuo basado en tiempo real de procesamiento

### Clustering y Fusión
- Similitud para clustering: > 85%
- Similitud para fusión: > 90%
- Umbral de poda: relevancia < 0.15
- Algoritmo: similitud bidireccional (ya implementado en `GestorEngramas`)

### Impacto Esperado en Juegos
- **3 en raya**: ~5-10ms/iteración → consolidar cada ~5 iteraciones → más engramas
- **Gatos**: ~50-100ms/iteración → consolidar cada ~25-50 iteraciones → engramas balanceados
- **Damas**: ~200-500ms/iteración → consolidar cada ~100 iteraciones → engramas consolidados frecuentemente

## Dependencias

- `GestorEngramas`: Lógica de formación y división de engramas
- `RedNeuralExperimental`: Medición de tiempos y control de consolidación
- `Engrama`: Métodos para dividir engramas grandes
- Modelos experimentales: Ajustar lógica de consolidación

## Prioridad

**Alta** - Mejora significativa en la formación de engramas y eficiencia del sistema
