# Tasks: Mejoras en Formación y Consolidación de Engramas

## 1. Crear GestorConsolidacionAdaptativa

- [x] 1.1 Crear archivo `GestorConsolidacionAdaptativa.java` en `src/main/java/es/jastxz/nn/experimental/`
- [x] 1.2 Implementar medición de tiempos con ventana deslizante (10 iteraciones)
- [x] 1.3 Implementar cálculo de intervalo: `max(10, min(100, promedioMs / 2))`
- [x] 1.4 Implementar método `debeConsolidar()` con lógica adaptativa
- [x] 1.5 Añadir getters para métricas (intervalo, tiempo promedio, inicializado)
- [x] 1.6 Hacer la clase `Serializable` con `serialVersionUID`

## 2. Modificar GestorEngramas para Límite del 22%

- [x] 2.1 Añadir constante `LIMITE_PORCENTAJE_NEURONAS = 0.22`
- [x] 2.2 Añadir campo `totalNeuronasRed` y modificar constructor
- [x] 2.3 Modificar `detectarYFormarEngramas()` para aplicar límite del 22%
- [x] 2.4 Implementar método `dividirEnChunks()` para dividir listas grandes
- [x] 2.5 Implementar método `formarEngramaConLimite()` que verifica similitud antes de formar
- [x] 2.6 Añadir contadores: `engramasFormados`, `engramasFusionados`, `engramasPodados`

## 3. Implementar Clustering y Fusión en GestorEngramas

- [x] 3.1 Implementar método `optimizarEngramas()` (punto de entrada)
- [x] 3.2 Implementar `identificarCandidatosFusion()` con umbral 90%
- [x] 3.3 Implementar `fusionarEngramas()` respetando límite del 22%
- [x] 3.4 Implementar `aplicarClusteringInterno()` para engramas grandes (>15% de neuronas)
- [x] 3.5 Implementar `agruparPorCapa()` para clustering por capas
- [x] 3.6 Crear clase interna `ParFusion` para pares de engramas similares
- [x] 3.7 Actualizar contadores en cada operación (fusión, poda)

## 4. Añadir Métricas a GestorEngramas

- [x] 4.1 Crear clase interna `EstadisticasEngramas` con todos los campos
- [x] 4.2 Implementar método `getEstadisticas()` que calcula métricas
- [x] 4.3 Calcular tamaño promedio, mínimo y máximo de engramas
- [x] 4.4 Calcular porcentaje promedio de neuronas usadas
- [x] 4.5 Implementar `toString()` en `EstadisticasEngramas` para visualización

## 5. Modificar Engrama

- [x] 5.1 Añadir método `setFuerza(double fuerza)` con validación [0.0, 1.0]
- [x] 5.2 Verificar que todos los métodos necesarios existen (getId, getNeuronas, etc.)

## 6. Integrar en RedNeuralExperimental

- [x] 6.1 Añadir campo `gestorConsolidacionAdaptativa`
- [x] 6.2 Inicializar en constructor pasando `getTotalNeuronas()` a `GestorEngramas`
- [x] 6.3 Modificar método `entrenar()` para medir tiempos y consolidar adaptativamente
- [x] 6.4 Modificar método `consolidar()` para llamar a `gestorEngramas.optimizarEngramas()`
- [x] 6.5 Actualizar método `getEstadisticas()` con métricas de engramas y consolidación
- [x] 6.6 Asegurar que `gestorConsolidacionAdaptativa` es serializable

## 7. Actualizar ModeloGatosExperimental

- [ ] 7.1 Modificar `entrenarSelfPlay()` para eliminar consolidación manual cada 50 partidas
- [ ] 7.2 Actualizar `mostrarEstadisticas()` para mostrar métricas de engramas
- [ ] 7.3 Añadir sección de consolidación adaptativa en estadísticas
- [ ] 7.4 Verificar que `mostrarProgreso()` sigue funcionando correctamente

## 8. Actualizar Modelo3enRayaExperimental

- [ ] 8.1 Modificar `entrenarSelfPlay()` similar a ModeloGatosExperimental
- [ ] 8.2 Actualizar `mostrarEstadisticas()` con métricas de engramas
- [ ] 8.3 Verificar compatibilidad con entrenamiento supervisado

## 9. Actualizar ModeloDamasExperimental (si existe)

- [ ] 9.1 Aplicar mismos cambios que en ModeloGatosExperimental
- [ ] 9.2 Verificar que funciona con tableros grandes (8x8)

## 10. Tests Unitarios - GestorConsolidacionAdaptativa

- [x] 10.1 Crear `GestorConsolidacionAdaptativaTest.java`
- [x] 10.2 Test: medición de tiempos y ventana deslizante
- [x] 10.3 Test: cálculo de intervalo con fórmula correcta
- [x] 10.4 Test: inicialización después de 10 iteraciones
- [x] 10.5 Test: ajuste continuo cada 10 iteraciones
- [x] 10.6 Test: valores límite (min=10, max=100)

## 11. Tests Unitarios - GestorEngramas

- [x] 11.1 Actualizar tests existentes para pasar `totalNeuronas` al constructor
- [x] 11.2 Test: límite del 22% se aplica correctamente
- [x] 11.3 Test: división automática de patrones grandes
- [x] 11.4 Test: fusión de engramas similares (>90%)
- [x] 11.5 Test: clustering de engramas grandes (>15%)
- [x] 11.6 Test: poda de engramas con baja relevancia (<0.15)
- [x] 11.7 Test: estadísticas se calculan correctamente

## 12. Tests Unitarios - Engrama

- [x] 12.1 Test: `setFuerza()` valida rango [0.0, 1.0]
- [x] 12.2 Verificar que tests existentes siguen pasando

## 13. Tests de Integración - Comparativa3enRayaTest

- [x] 13.1 Ejecutar test y verificar que se forman más engramas (objetivo: 8-12)
- [x] 13.2 Verificar que consolidación es rápida (~5 iteraciones)
- [x] 13.3 Verificar que tamaño promedio de engramas es razonable (15%)
- [x] 13.4 Ajustar test si es necesario para nuevas métricas

## 14. Tests de Integración - ComparativaGatosTest

- [x] 14.1 Ejecutar test y verificar que se forman más engramas (objetivo: 10-20)
- [x] 14.2 Verificar que consolidación es media (~25-50 iteraciones)
- [x] 14.3 Verificar que tamaño promedio de engramas es razonable (18%)
- [x] 14.4 Ajustar test si es necesario

## 15. Tests de Integración - ComparativaDamasTest (si existe)

- [x] 15.1 Ejecutar test y verificar que se forman más engramas (objetivo: 15-30)
- [x] 15.2 Verificar que consolidación es lenta (~100 iteraciones)
- [x] 15.3 Verificar que tamaño promedio de engramas es razonable (20%)

## 16. Validación y Ajustes

- [x] 16.1 Ejecutar todos los tests y verificar que pasan
- [x] 16.2 Revisar logs y métricas de engramas en cada juego
- [x] 16.3 Ajustar umbrales si es necesario (similitud, clustering, etc.)
- [x] 16.4 Verificar que serialización funciona correctamente
- [x] 16.5 Verificar que modelos antiguos pueden cargarse

## 17. Documentación

- [x] 17.1 Actualizar comentarios en código con referencias biológicas
- [x] 17.2 Documentar nuevos parámetros y su justificación
- [x] 17.3 Crear ejemplos de uso de nuevas métricas
- [x] 17.4 Actualizar README si es necesario

## Notas de Implementación

- Prioridad: Tareas 1-6 son críticas (core functionality)
- Tareas 7-9 son actualizaciones de modelos (pueden hacerse en paralelo)
- Tareas 10-15 son validación (ejecutar después de implementación)
- Tarea 16 es ajuste fino basado en resultados
- Tarea 17 es documentación final

## Criterios de Éxito

- ✓ Límite del 22% se aplica en todos los engramas
- ✓ Consolidación adaptativa funciona según fórmula
- ✓ Se forman más engramas en todos los juegos
- ✓ Fusión y clustering reducen redundancia
- ✓ Métricas son precisas y útiles
- ✓ Tests pasan sin errores
- ✓ Serialización mantiene compatibilidad
