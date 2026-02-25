# Ejemplos de Uso: RedNeuralExperimental

Este documento proporciona ejemplos prácticos de cómo usar la `RedNeuralExperimental`.

---

## Ejemplo 1: Configuración Básica

```java
import es.jastxz.nn.RedNeuralExperimental;

// Crear red con topología simple: 2 inputs, 3 hidden, 2 outputs
RedNeuralExperimental red = new RedNeuralExperimental(
    new int[]{2, 3, 2},  // Topología
    0.8                   // 80% densidad de conexiones
);

// Procesar inputs
double[] input = {1.0, 0.0};
double[] output = red.procesar(input);

System.out.println("Output: " + Arrays.toString(output));
```

---

## Ejemplo 2: Entrenamiento con Plasticidad Hebiana

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);

// Datos de entrenamiento
double[] input = {1.0, 0.0};
double[] target = {1.0, 0.0};

// Entrenar 100 iteraciones
red.entrenar(input, target, 100);

// Verificar resultado
double[] output = red.procesar(input);
System.out.println("Después de entrenar: " + Arrays.toString(output));
```

---

## Ejemplo 3: Modo Predictivo

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);

// Activar modo predictivo
red.activarModoPredictivo(true);

// Procesar varios inputs
for (int i = 0; i < 10; i++) {
    double[] input = {Math.random(), Math.random()};
    double[] output = red.procesar(input);
    
    // Obtener errores de predicción
    double[] errores = red.getErroresPrediccion();
    System.out.println("Errores: " + Arrays.toString(errores));
}

// Los errores deberían disminuir con el tiempo
```

---

## Ejemplo 4: Competición por Recursos y Poda

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{3, 6, 3}, 0.7);

// Activar competición
red.activarCompeticionRecursos(true);

// Entrenar con un patrón específico
double[] input1 = {1.0, 0.0, 0.0};
double[] target1 = {1.0, 0.0, 0.0};

for (int i = 0; i < 50; i++) {
    red.entrenar(input1, target1, 5);
    
    // Competir por recursos cada 10 iteraciones
    if (i % 10 == 0) {
        red.competirPorRecursos();
        int podados = red.podarElementos();
        System.out.println("Elementos podados: " + podados);
    }
}

System.out.println("Conexiones finales: " + red.getTotalConexiones());
```

---

## Ejemplo 5: Formación y Uso de Engramas

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 5, 2}, 0.8);

// Activar detección automática de engramas
red.activarDeteccionEngramas(true);

// Procesar el mismo patrón varias veces
double[] patron = {1.0, 0.0};
for (int i = 0; i < 10; i++) {
    red.procesar(patron);
}

// Verificar engramas formados
Map<String, Engrama> engramas = red.getEngramas();
System.out.println("Engramas formados: " + engramas.size());

// Formar engrama manualmente
List<Neurona> neuronas = new ArrayList<>(
    red.getCapasInterneuronas().get(0).subList(0, 3)
);
red.formarEngrama("mi_engrama", neuronas);

// Activar engrama (facilita sus neuronas)
red.activarEngrama("mi_engrama");
```

---

## Ejemplo 6: Ciclo de Consolidación ("Sueño")

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);
red.activarDeteccionEngramas(true);

// Fase de aprendizaje (vigilia)
double[] input1 = {1.0, 0.0};
double[] input2 = {0.0, 1.0};

for (int i = 0; i < 20; i++) {
    red.procesar(input1);
    red.procesar(input2);
}

System.out.println("Engramas antes: " + red.getEngramas().size());

// Fase de consolidación (sueño)
red.iniciarConsolidacion();
red.consolidar();
red.finalizarConsolidacion();

System.out.println("Engramas después: " + red.getEngramas().size());

// Los engramas usados se fortalecen, los no usados se debilitan
```

---

## Ejemplo 7: Monitorización y Debugging

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 3, 2}, 0.8);

// Activar todos los sistemas
red.activarModoPredictivo(true);
red.activarCompeticionRecursos(true);
red.activarDeteccionEngramas(true);

// Entrenar
red.entrenar(new double[]{1.0, 0.0}, new double[]{1.0, 0.0}, 50);

// Obtener estadísticas
Map<String, Object> stats = red.getEstadisticas();
System.out.println("=== Estadísticas ===");
System.out.println("Total neuronas: " + stats.get("totalNeuronas"));
System.out.println("Total conexiones: " + stats.get("totalConexiones"));
System.out.println("Total engramas: " + stats.get("totalEngramas"));
System.out.println("Neuronas activas: " + stats.get("neuronasActivas"));
System.out.println("% Activación: " + stats.get("porcentajeActivacion"));

// Visualizar activaciones
System.out.println("\n" + red.visualizarActivaciones());

// Analizar engramas
System.out.println("\n" + red.analizarEngramas());

// Reporte de recursos
System.out.println("\n" + red.reporteRecursos());
```

---

## Ejemplo 8: Ciclo Completo de Aprendizaje

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 5, 2}, 0.8);

// Configurar sistemas
red.activarModoPredictivo(true);
red.activarCompeticionRecursos(true);
red.activarDeteccionEngramas(true);

// Datos de entrenamiento
double[][] inputs = {
    {1.0, 0.0},
    {0.0, 1.0},
    {1.0, 1.0},
    {0.0, 0.0}
};
double[][] targets = {
    {1.0, 0.0},
    {0.0, 1.0},
    {1.0, 1.0},
    {0.0, 0.0}
};

// Ciclo de entrenamiento con consolidación periódica
for (int epoca = 0; epoca < 10; epoca++) {
    System.out.println("\n=== Época " + (epoca + 1) + " ===");
    
    // Fase de vigilia: entrenar
    for (int i = 0; i < inputs.length; i++) {
        red.entrenar(inputs[i], targets[i], 10);
    }
    
    // Competir por recursos
    red.competirPorRecursos();
    int podados = red.podarElementos();
    System.out.println("Elementos podados: " + podados);
    
    // Fase de sueño: consolidar
    red.iniciarConsolidacion();
    red.consolidar();
    red.finalizarConsolidacion();
    
    // Estadísticas
    Map<String, Object> stats = red.getEstadisticas();
    System.out.println("Engramas: " + stats.get("totalEngramas"));
    System.out.println("Conexiones: " + stats.get("totalConexiones"));
}

// Evaluación final
System.out.println("\n=== Evaluación Final ===");
for (int i = 0; i < inputs.length; i++) {
    double[] output = red.procesar(inputs[i]);
    System.out.println("Input: " + Arrays.toString(inputs[i]) + 
                      " -> Output: " + Arrays.toString(output) +
                      " (Target: " + Arrays.toString(targets[i]) + ")");
}
```

---

## Ejemplo 9: Red con Múltiples Capas Intermedias

```java
// Red profunda: 3 inputs, 2 capas hidden (6 y 4 neuronas), 2 outputs
RedNeuralExperimental red = new RedNeuralExperimental(
    new int[]{3, 6, 4, 2},
    0.7  // Densidad más baja para red más grande
);

// La red maneja automáticamente la propagación entre todas las capas
double[] input = {1.0, 0.5, 0.0};
double[] output = red.procesar(input);

// Visualizar activaciones de todas las capas
System.out.println(red.visualizarActivaciones());
```

---

## Ejemplo 10: Reseteo de Memoria de Corto Plazo

```java
RedNeuralExperimental red = new RedNeuralExperimental(new int[]{2, 4, 2}, 0.8);

// Procesar secuencia de inputs (la red mantiene estado)
red.procesar(new double[]{1.0, 0.0});
red.procesar(new double[]{0.5, 0.5});
red.procesar(new double[]{0.0, 1.0});

// Las neuronas mantienen su activación como memoria de corto plazo

// Cambiar de tarea: resetear memoria de corto plazo
red.resetear();

// Ahora la red empieza "fresca" para una nueva tarea
red.procesar(new double[]{1.0, 1.0});

// NOTA: resetear() NO borra el conocimiento aprendido (pesos, engramas)
// Solo limpia el estado transitorio (activaciones, potenciales)
```

---

## Consejos de Uso

### Densidad de Conexiones
- **0.5-0.7**: Redes grandes (muchas neuronas)
- **0.7-0.9**: Redes medianas (uso general)
- **0.9-1.0**: Redes pequeñas (máxima conectividad)

### Modo Predictivo
- Útil para tareas secuenciales
- Reduce el "ruido" en la propagación
- Mejora la eficiencia del aprendizaje

### Competición por Recursos
- Activar después de entrenamiento inicial
- Ejecutar `competirPorRecursos()` periódicamente
- Podar elementos cada 10-20 iteraciones

### Consolidación
- Ejecutar después de sesiones de entrenamiento
- Simula el "sueño" del cerebro
- Fortalece conocimiento importante
- Elimina información irrelevante

### Engramas
- Detección automática para patrones repetidos
- Formación manual para memorias específicas
- Activación para "recordar" patrones

---

## Patrones Comunes

### Patrón 1: Entrenamiento Básico
```java
1. Crear red
2. Entrenar con datos
3. Evaluar resultados
```

### Patrón 2: Entrenamiento con Optimización
```java
1. Crear red
2. Activar competición
3. Entrenar
4. Competir y podar periódicamente
5. Evaluar
```

### Patrón 3: Aprendizaje Continuo
```java
1. Crear red
2. Activar todos los sistemas
3. Ciclo:
   - Entrenar (vigilia)
   - Competir y podar
   - Consolidar (sueño)
4. Evaluar
```

### Patrón 4: Memoria Explícita
```java
1. Crear red
2. Activar detección de engramas
3. Entrenar con patrones
4. Consolidar para fortalecer engramas
5. Activar engramas para recuperar memorias
```

---

## Debugging

### Problema: La red no aprende
```java
// Verificar estadísticas
Map<String, Object> stats = red.getEstadisticas();
System.out.println("Neuronas activas: " + stats.get("neuronasActivas"));

// Si muy pocas neuronas activas:
// - Aumentar densidad de conexiones
// - Verificar inputs (deben activar neuronas sensoriales)
// - Revisar topología (suficientes neuronas intermedias)
```

### Problema: Overfitting
```java
// Activar competición para podar conexiones débiles
red.activarCompeticionRecursos(true);
red.competirPorRecursos();
red.podarElementos();

// Consolidar periódicamente
red.iniciarConsolidacion();
red.consolidar();
red.finalizarConsolidacion();
```

### Problema: Memoria insuficiente
```java
// Verificar engramas
System.out.println(red.analizarEngramas());

// Activar detección si no está activa
red.activarDeteccionEngramas(true);

// Formar engramas manualmente para patrones importantes
```

---

## Recursos Adicionales

- **PLAN_RED_NEURONAL_EXPERIMENTAL.md**: Plan de implementación completo
- **RESUMEN_IMPLEMENTACION.md**: Resumen técnico
- **Apuntes_Neurologicos.md**: Fundamentos neurológicos
- **Tests**: Ver `src/test/java/es/jastxz/experimental/` para más ejemplos

---

¡Experimenta y diviértete explorando esta red neuronal biológicamente realista! 🧠✨
