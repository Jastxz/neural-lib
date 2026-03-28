# Historial de Refactorizaciones

Este documento consolida todas las refactorizaciones importantes realizadas en el proyecto de red neuronal experimental.

---

## REFACTORIZACIÓN 1: Eliminación de Referencias Circulares
**Fecha:** 14 de febrero de 2026

### Problema
StackOverflowError al serializar modelos grandes (>10,000 conexiones) debido a referencias bidireccionales entre `Neurona` y `Conexion`.

### Solución Implementada
- Eliminadas listas de conexiones en `Neurona` (axones, dendritas)
- Añadido sistema de "vecinas" para navegación rápida
- Lógica centrada en iterar sobre conexiones (más eficiente O(n) vs O(n²))
- Referencias directas mantenidas en `Conexion`

### Cambios Principales

#### Neurona.java
```java
// ELIMINADO
private List<Conexion> axones;
private List<Conexion> dendritas;

// AÑADIDO
private List<Neurona> vecinas;  // Para navegación rápida
private transient double potencialAcumulado = 0.0;

public void recibirSeñal(double señal)
public boolean evaluarActivacion(long timestampActual)
```

#### PropagadorSeñal.java
```java
// Antes: iteraba sobre neuronas y buscaba conexiones
// Ahora: itera sobre conexiones directamente
public void propagarHaciaAdelante(List<Conexion> conexiones, 
                                  List<Neurona> todasNeuronas, long timestamp)
```

#### RedNeuralExperimental.java
```java
// ELIMINADO
private Map<Long, Neurona> neuronasById;
private Map<Long, List<Long>> conexionesSalientes;
private Map<Long, List<Long>> conexionesEntrantes;

// SIMPLIFICADO
private List<Conexion> conexiones;  // Solo lista de conexiones
```

### Beneficios
- Serialización funciona con modelos grandes (>49,000 conexiones)
- Rendimiento mejorado: O(n) vs O(n²)
- Código más simple y mantenible
- Más natural biológicamente (señales viajan por sinapsis)

### Archivos Modificados
- `Neurona.java`
- `Conexion.java`
- `EntrenadorHebiano.java`
- `PropagadorSeñal.java`
- `RedNeuralExperimental.java`
- `GestorPredicciones.java`

---

## REFACTORIZACIÓN 2: Eliminación de valorAlmacenado
**Fecha:** Febrero de 2026

### Problema Conceptual
El atributo `valorAlmacenado` en `Neurona` era biológicamente incorrecto:
- En el cerebro, el conocimiento está en las sinapsis (conexiones), no en las neuronas
- Las neuronas son procesadores que integran señales
- Causaba que todos los patrones convergieran al mismo valor

### Principio Biológico Correcto
```
Neurona: Procesador (integra señales → dispara o no)
Sinapsis: Memoria (almacena conocimiento en pesos)
Engrama: Conjunto de neuronas + conexiones específicas
```

### Cambios Implementados

#### 1. Pesos Iniciales Revertidos
```java
// Cambio de [0.1, 0.9] a [0.1, 0.4]
peso = random.nextDouble() * 0.3 + 0.1;  // [0.1, 0.4]
```

#### 2. PropagadorSeñal.java
```java
// ANTES (INCORRECTO)
public double[] getOutputs(List<Neurona> capaMotora) {
    outputs[i] = neurona.getValorAlmacenado();  // ❌
}

// DESPUÉS (CORRECTO)
public double[] getOutputs(List<Neurona> capaMotora, List<Conexion> conexiones) {
    // Calcular output desde pesos de conexiones activas
    double suma = 0.0;
    int contador = 0;
    
    for (Conexion c : conexiones) {
        if (c.getPostsinapticas().contains(neurona)) {
            Neurona pre = c.getPresinaptica();
            if (pre.estaActiva()) {
                suma += c.getPeso();  // ✓ Conocimiento en conexión
                contador++;
            }
        }
    }
    
    outputs[i] = contador > 0 ? suma / contador : 0.0;
}
```

#### 3. EntrenadorHebiano.java
```java
// ANTES (INCORRECTO)
neurona.setValorAlmacenado(nuevoValor);  // ❌

// DESPUÉS (CORRECTO)
// Ajustar pesos de conexiones que llegan a neuronas motoras
for (Conexion conexion : todasConexiones) {
    if (conexion.getPostsinapticas().contains(neuronaMotora)) {
        Neurona pre = conexion.getPresinaptica();
        if (pre.estaActiva()) {
            double ajuste = error * tasaAprendizaje;
            double nuevoPeso = conexion.getPeso() + ajuste;
            conexion.setPeso(Math.max(-1.0, Math.min(1.0, nuevoPeso)));
        }
    }
}
```

#### 4. Neurona.java
```java
// ELIMINADO
private double valorAlmacenado;
public double getValorAlmacenado() { ... }
public void setValorAlmacenado(double valor) { ... }

// MANTIENE
- Estado de activación (activa/reposo)
- Potencial de membrana (transitorio)
- Recursos asignados
- Timestamps
```

### Ejemplo Conceptual

#### Antes (Incorrecto)
```
Fibonacci → Activa neuronas [1,2,3] → Neurona motora.valorAlmacenado = 0.67
Aritmética → Activa neuronas [1,2,3] → Neurona motora.valorAlmacenado = 0.67
                                       ↑ Mismo valor ❌
```

#### Después (Correcto)
```
Fibonacci → Activa neuronas [1,2,3] → Conexiones [0.2, 0.3, 0.1] → Output: 0.20
Aritmética → Activa neuronas [4,5,6] → Conexiones [0.8, 0.9, 0.7] → Output: 0.80
                                       ↑ Diferentes conexiones ✓
```

### Beneficios
- Biológicamente correcto: conocimiento en sinapsis
- Permite especialización de patrones
- Diferentes engramas con diferentes conexiones
- Mejor diferenciación entre patrones

### Archivos Modificados
- `Neurona.java`
- `RedNeuralExperimental.java`
- `PropagadorSeñal.java`
- `EntrenadorHebiano.java`
- `GestorPredicciones.java`
- Tests actualizados

### Problema Identificado
Después de la refactorización, todas las neuronas intermedias se activaban para todos los patrones (30/30), causando que el promedio de pesos fuera el mismo.

**Solución:** Implementar inhibición lateral (ver siguiente refactorización).

---

## REFACTORIZACIÓN 3: Inhibición Lateral y Umbrales Variables
**Fecha:** 25 de febrero de 2026

Esta refactorización resolvió el problema de especialización identificado en la Refactorización 2.

### Cambios Implementados

#### 1. Conexiones Laterales Unidireccionales
- Cada conexión es unidireccional (pre → post)
- 90% inhibitorias (competición)
- 10% excitatorias (cooperación)
- Sin autoconexiones
- Densidad 37% en capas intermedias

#### 2. Umbrales de Activación Variables
Cada neurona tiene su propio umbral:
- Sensorial/Motora: 15-30% (más sensibles)
- Intermedias: 30-50% (más selectivas)

#### 3. Plasticidad Hebiana en Tiempo Real
La plasticidad ocurre DURANTE la propagación, no después:
- Refuerzo inmediato (LTP)
- Debilitamiento por desuso (LTD)

### Resultados
- Mejora del 48% en rendimiento (error 0.0759 vs 0.1470)
- Red experimental ahora supera a la clásica
- Inhibición lateral converge a estado estable (70-80% activas)

### Archivos Modificados
- `Neurona.java`: Umbral personalizado
- `RedNeuralExperimental.java`: Conexiones laterales, umbrales variables
- `PropagadorSeñal.java`: Plasticidad en tiempo real

**Documentación completa:** `MEJORAS_INHIBICION_LATERAL_Y_UMBRALES.md`

---

## COMPATIBILIDAD CON MODELOS GUARDADOS

### Refactorización 1 (Referencias Circulares)
✅ Compatible - Los modelos antiguos se pueden cargar

### Refactorización 2 (valorAlmacenado)
⚠️ Incompatible - Los modelos antiguos deben reentrenarse

**Opciones:**
1. Reentrenar modelos desde cero (recomendado)
2. Migración manual: cargar modelo antiguo, ignorar `valorAlmacenado`, guardar nuevo
3. Mantener versión antigua para modelos legacy

### Refactorización 3 (Inhibición Lateral)
✅ Compatible - Añade nuevas características sin romper API

---

## PRINCIPIOS BIOLÓGICOS IMPLEMENTADOS

### 1. Conocimiento en Sinapsis
Las conexiones (sinapsis) almacenan el conocimiento, no las neuronas. Las neuronas son procesadores.

### 2. Propagación Centrada en Conexiones
Las señales viajan por sinapsis, no por neuronas. Iterar sobre conexiones es más natural y eficiente.

### 3. Inhibición Lateral
Competición entre neuronas de la misma capa para especialización y sparse coding.

### 4. Diversidad Neuronal
Cada neurona tiene características únicas (umbral variable) que permiten especialización natural.

### 5. Plasticidad en Tiempo Real
Las sinapsis se modifican mientras se usan, no después. Refuerzo inmediato de patrones útiles.

---

## LECCIONES APRENDIDAS

### 1. Simplicidad es Mejor
Eliminar estructuras de datos complejas (Maps) simplificó el código y mejoró el rendimiento.

### 2. Seguir la Biología
Centrar la lógica en conexiones (sinapsis) es más natural y eficiente que centrarla en neuronas.

### 3. El Conocimiento Está en las Conexiones
Este principio biológico fundamental debe reflejarse en el código.

### 4. La Inhibición es Esencial
Sin inhibición lateral, no hay especialización. Las neuronas deben competir.

### 5. Iteración Incremental
Cada refactorización resolvió un problema específico, permitiendo validación paso a paso.

---

## ESTADO ACTUAL DEL PROYECTO

### Arquitectura
- Lógica centrada en conexiones (O(n))
- Sin referencias circulares
- Serialización funciona con modelos grandes
- Conocimiento almacenado en pesos sinápticos

### Mecanismos Biológicos
- ✅ Inhibición lateral
- ✅ Umbrales variables
- ✅ Plasticidad hebiana en tiempo real
- ✅ Coactivación y congelación
- ✅ Poda biológica
- ✅ Memoria de corto plazo

### Rendimiento
- Red experimental supera a clásica en 48%
- Serialización funciona con >49,000 conexiones
- Código más simple y mantenible

---

## ARCHIVOS DE DOCUMENTACIÓN

### Refactorizaciones
- `HISTORIAL_REFACTORIZACIONES.md` (este archivo)
- `REFACTORIZACION_MAPS.md` (detalle Refactorización 1)
- `PLAN_REFACTORIZACION_VALORALMACENADO.md` (plan Refactorización 2)
- `RESUMEN_REFACTORIZACION_VALORALMACENADO.md` (resumen Refactorización 2)

### Mejoras
- `HISTORIAL_MEJORAS_RED_EXPERIMENTAL.md` (todas las mejoras)
- `MEJORAS_INHIBICION_LATERAL_Y_UMBRALES.md` (mejoras recientes)

### Comparativas
- `RESUMEN_COMPARATIVA.md` (resumen ejecutivo)
- `COMPARATIVA_3ENRAYA.md` (3 en Raya)
- `COMPARATIVA_GATOS.md` (Gatos)
- `COMPARATIVA_DAMAS.md` (Damas)
- `COMPARATIVA_SUPERVISADO.md` (Backprop vs Hebbian)

---

**Última actualización:** 25 de febrero de 2026  
**Estado:** Todas las refactorizaciones completadas ✅  
**Tests pasando:** 126/126 ✅
