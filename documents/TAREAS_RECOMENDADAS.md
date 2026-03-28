# Tareas Recomendadas para Red Neuronal Experimental

Este documento lista tareas que deberían dársele especialmente bien a la red neuronal experimental debido a sus características biológicas: engramas, consolidación adaptativa, plasticidad hebiana y competición por recursos.

---

## 1. ✅ Reconocimiento de Patrones Secuenciales

**Estado:** IMPLEMENTADO - `src/test/java/es/jastxz/memo/SecuenciasTest.java`

**Por qué funciona bien:**
- Los engramas memorizan transiciones entre estados
- La consolidación refuerza patrones frecuentes
- La plasticidad hebiana asocia elementos consecutivos naturalmente

**Casos de prueba:**
- Fibonacci: 1,1,2,3,5,8,13,21...
- Potencias de 2: 1,2,4,8,16,32...
- Alternancia simple: A,B,A,B,A,B...
- Secuencias aritméticas: 2,4,6,8,10...
- Secuencias geométricas: 3,6,12,24,48...

**Métricas de éxito:**
- Precisión >80% en predicción del siguiente elemento
- Formación de engramas por cada tipo de secuencia
- Consolidación rápida (<20 iteraciones)

**Comparación con red clásica:**
- Red experimental: Memoriza patrones con engramas
- Red clásica: Necesita más ejemplos para generalizar

---

## 2. ⏳ Aprendizaje por Refuerzo con Memoria Episódica

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Engramas guardan episodios completos (estado → acción → recompensa)
- Consolidación refuerza estrategias exitosas
- Competición elimina estrategias fallidas

**Casos de prueba propuestos:**

### 2.1 Navegación en Laberinto Simple
```
Laberinto 5x5:
S . . . .
# # . # .
. . . # .
. # # # .
. . . . G

S = Start, G = Goal, # = Wall
```

**Objetivo:** Encontrar camino óptimo del inicio a la meta

**Ventajas de la red experimental:**
- Recuerda caminos exitosos como engramas
- No olvida rutas alternativas (múltiples engramas)
- Adaptación rápida si cambia el laberinto

**Implementación sugerida:**
```java
public class ModeloLaberintoExperimental {
    // Input: posición actual (x,y) + visión local (8 direcciones)
    // Output: dirección a tomar (arriba, abajo, izq, der)
    // Recompensa: +10 por llegar a meta, -1 por paso, -5 por pared
}
```

### 2.2 Juego de Cartas Simple (21/Blackjack simplificado)
**Objetivo:** Decidir pedir carta o plantarse

**Ventajas:**
- Engramas memorizan situaciones ganadoras
- Aprende estrategias sin necesitar millones de partidas

### 2.3 Problema del Bandido Multi-Brazo (Multi-Armed Bandit)
**Objetivo:** Maximizar recompensa eligiendo entre N opciones

**Ventajas:**
- Explora y explota naturalmente
- Engramas guardan qué opciones funcionan mejor

---

## 3. ⏳ Clasificación con Pocos Ejemplos (Few-Shot Learning)

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Engramas se forman con pocas exposiciones (principio biológico)
- No necesita miles de ejemplos como backpropagation
- Consolidación refuerza patrones importantes rápidamente

**Casos de prueba propuestos:**

### 3.1 Clasificación de Dígitos Manuscritos (MNIST Simplificado)
**Dataset:** 5-10 ejemplos por dígito (0-9)
**Objetivo:** Clasificar nuevos dígitos

**Comparación:**
- Red experimental: 50-100 ejemplos totales
- Red clásica: Necesita 1000+ ejemplos

**Implementación:**
```java
// Input: 28x28 píxeles = 784 inputs
// Hidden: 200 neuronas
// Output: 10 clases (0-9)
// Entrenar con solo 5 ejemplos por clase
```

### 3.2 Clasificación de Formas Geométricas
**Clases:** Círculo, Cuadrado, Triángulo, Estrella
**Dataset:** 3-5 ejemplos por forma

**Ventajas:**
- Problema más simple que MNIST
- Visualización clara de engramas por forma
- Rápido de entrenar

### 3.3 Detección de Anomalías
**Objetivo:** Identificar patrones anómalos con pocos ejemplos

**Casos:**
- Secuencias normales: [1,2,3,4,5]
- Anomalía: [1,2,9,4,5]
- Entrenar con 10 ejemplos normales, detectar anomalías

---

## 4. ⏳ Problemas Lógicos Compuestos

**Estado:** PENDIENTE (XOR ya implementado)

**Por qué funciona bien:**
- Plasticidad hebiana es natural para asociaciones lógicas
- Engramas memorizan tablas de verdad
- Ya funciona bien con AND, OR, XOR

**Casos de prueba propuestos:**

### 4.1 Operaciones Lógicas Compuestas
```java
// (A AND B) OR (C XOR D)
// (A OR B) AND NOT(C)
// (A XOR B) AND (C OR D)
```

**Objetivo:** Aprender operaciones de 3-4 variables

### 4.2 Circuitos Lógicos Simples
```
Half Adder: A + B = Sum, Carry
Full Adder: A + B + Cin = Sum, Cout
```

### 4.3 Tablas de Verdad de 3-4 Variables
**Objetivo:** Memorizar y reproducir tablas de verdad complejas

**Ventaja:** Los engramas memorizan cada fila de la tabla

---

## 5. ⏳ Detección de Cambios de Contexto

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Competición por recursos detecta cuando patrones antiguos ya no funcionan
- Consolidación permite mantener múltiples contextos
- No sufre "catastrophic forgetting"

**Casos de prueba propuestos:**

### 5.1 Cambio de Reglas en Juego
**Fase 1:** Jugar 3 en raya con reglas normales
**Fase 2:** Cambiar regla (ej: ganar con 4 en línea)
**Objetivo:** Adaptarse sin olvidar reglas originales

### 5.2 Detección de Anomalías en Secuencias
**Fase 1:** Secuencia normal [1,2,3,4,5,6...]
**Fase 2:** Introducir anomalía [1,2,9,4,5...]
**Objetivo:** Detectar cuándo cambia el patrón

### 5.3 Adaptación a Nuevos Patrones
**Fase 1:** Aprender Fibonacci
**Fase 2:** Aprender Potencias de 2
**Objetivo:** Mantener ambos patrones activos

**Ventaja:** Múltiples engramas permiten múltiples contextos

---

## 6. ⏳ Memoria Asociativa (Hopfield-like)

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Engramas están diseñados para completar patrones parciales
- Principio biológico: "el cerebro completa recuerdos parciales" (pg. 93 Campillo)
- Conexiones bidireccionales permiten recuperación

**Casos de prueba propuestos:**

### 6.1 Completar Patrones Visuales
```
Patrón completo:    Patrón parcial:    Objetivo:
X X X               X . X              X X X
X . X               X . .              X . X
X X X               . . .              X X X
```

### 6.2 Recuperación de Información por Asociación
**Entrenar:** Pares (nombre → teléfono)
**Test:** Dar nombre parcial → recuperar teléfono completo

### 6.3 Corrección de Errores en Patrones Ruidosos
**Entrenar:** Patrones limpios
**Test:** Dar patrón con ruido → recuperar patrón limpio

**Implementación:**
```java
// Usar conexiones bidireccionales
// Activar engrama con patrón parcial
// Dejar que la red complete el patrón
```

---

## 7. ⏳ Memoria de Trabajo (Working Memory)

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Ya implementada la memoria de corto plazo (neuronas mantienen activación)
- Engramas pueden mantener información temporalmente
- Reseteo explícito permite limpiar memoria

**Casos de prueba propuestos:**

### 7.1 Test de Span de Dígitos
**Objetivo:** Recordar N dígitos temporalmente
```
Mostrar: [3, 7, 2, 9, 1]
Esperar: 5 ciclos
Preguntar: ¿Cuál era el tercer número?
Respuesta: 2
```

### 7.2 Operaciones con Memoria Temporal
**Objetivo:** Realizar cálculos manteniendo resultados intermedios
```
Paso 1: 3 + 5 = 8 (guardar)
Paso 2: 2 × 4 = 8 (guardar)
Paso 3: Sumar resultados = 16
```

### 7.3 Seguimiento de Estado en Secuencias
**Objetivo:** Mantener contexto durante procesamiento secuencial

---

## 8. ⏳ Aprendizaje Incremental sin Olvido Catastrófico

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Múltiples engramas permiten múltiples tareas
- Consolidación protege conocimiento importante
- Competición elimina solo lo no usado

**Casos de prueba propuestos:**

### 8.1 Aprendizaje Secuencial de Operaciones Lógicas
```
Fase 1: Aprender XOR
Fase 2: Aprender AND
Fase 3: Aprender OR
Verificar: Todas siguen funcionando
```

### 8.2 Aprendizaje de Múltiples Juegos
```
Fase 1: Aprender 3 en Raya
Fase 2: Aprender Gatos
Verificar: Puede jugar ambos sin confundirse
```

### 8.3 Expansión de Vocabulario
```
Fase 1: Aprender 5 patrones
Fase 2: Aprender 5 patrones nuevos
Fase 3: Aprender 5 patrones más
Verificar: Recuerda los 15 patrones
```

---

## 9. ⏳ Generalización desde Ejemplos Ruidosos

**Estado:** PENDIENTE

**Por qué funciona bien:**
- Consolidación filtra ruido (refuerza patrones consistentes)
- Engramas capturan la esencia del patrón
- Competición elimina conexiones ruidosas

**Casos de prueba propuestos:**

### 9.1 Clasificación con Datos Ruidosos
**Entrenar:** Patrones con 10-20% de ruido
**Test:** Clasificar patrones limpios

### 9.2 Extracción de Señal desde Ruido
**Entrenar:** Señal + ruido aleatorio
**Test:** Recuperar señal limpia

---

## 10. ⏳ Predicción de Series Temporales

**Estado:** PENDIENTE (relacionado con Secuencias)

**Por qué funciona bien:**
- Engramas memorizan patrones temporales
- Modo predictivo reduce error de predicción
- Consolidación refuerza tendencias

**Casos de prueba propuestos:**

### 10.1 Predicción de Tendencias Simples
```
Serie: [1, 3, 5, 7, 9, ...]
Predecir: 11
```

### 10.2 Predicción con Estacionalidad
```
Serie: [10, 20, 15, 10, 20, 15, ...]
Predecir: 10 (patrón se repite)
```

### 10.3 Predicción de Valores Futuros
**Objetivo:** Predecir N pasos adelante

---

## Priorización Sugerida

### Fase 1: Fundamentos (Semana 1)
1. ✅ Secuencias (IMPLEMENTADO)
2. ⏳ Problemas Lógicos Compuestos
3. ⏳ Memoria de Trabajo

### Fase 2: Aprendizaje Avanzado (Semana 2)
4. ⏳ Few-Shot Learning (Formas Geométricas)
5. ⏳ Aprendizaje Incremental
6. ⏳ Memoria Asociativa

### Fase 3: Aplicaciones Complejas (Semana 3)
7. ⏳ Navegación en Laberinto
8. ⏳ Detección de Cambios de Contexto
9. ⏳ Generalización con Ruido
10. ⏳ Predicción de Series Temporales

---

## Métricas de Comparación

Para cada tarea, comparar:

| Métrica | Red Experimental | Red Clásica |
|---------|------------------|-------------|
| Ejemplos necesarios | ? | ? |
| Tiempo de entrenamiento | ? | ? |
| Precisión final | ? | ? |
| Engramas formados | ? | N/A |
| Intervalo consolidación | ? | N/A |
| Resistencia al olvido | ? | ? |

---

## Notas de Implementación

- Todos los tests en `src/test/java/es/jastxz/memo/`
- Cada test debe incluir comparación con red clásica
- Documentar engramas formados y consolidación
- Visualizar resultados cuando sea posible
- Medir tiempos de entrenamiento e inferencia

---

## Referencias Biológicas

- Engramas: pg. 84-85 Campillo
- Consolidación: pg. 87 Campillo
- Completar patrones: pg. 93 Campillo
- Plasticidad hebiana: pg. 46-47 Eagleman
- Competición: pg. 18-19, 229-230 Eagleman
