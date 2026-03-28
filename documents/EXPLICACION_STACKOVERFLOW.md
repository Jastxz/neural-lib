# Explicación: StackOverflow en Serialización

## 🔴 Problema

Al intentar guardar modelos experimentales grandes (>49,000 conexiones), se produce un `StackOverflowError` durante la serialización con `ObjectOutputStream`.

```java
java.lang.StackOverflowError
    at java.base/java.io.ObjectOutputStream.writeObject0(...)
    at java.base/java.io.ObjectInputStream.readObject(...)
    at java.base/java.util.ArrayList.readObject(...)
    // ... se repite cientos de veces
```

## 🔍 Causa Raíz: Referencias Circulares

El problema está en la estructura de datos con **referencias bidireccionales**:

### Estructura Problemática

```
RedNeuralExperimental
├── List<Neurona> capaSensorial
│   └── Neurona
│       ├── List<Conexion> axones (salientes)
│       └── List<Conexion> dendritas (entrantes)
│
├── List<Conexion> conexiones
│   └── Conexion
│       ├── Neurona presináptica ──┐
│       └── List<Neurona> postsinápticas ──┐
│                                          │
└── (referencias circulares) ←────────────┘
```

### El Ciclo Infinito

1. **Serialización de RedNeuralExperimental**
   ```java
   ObjectOutputStream.writeObject(redNeuronal)
   ```

2. **Serializa capaSensorial** → Lista de Neuronas
   ```java
   writeObject(capaSensorial)
   ```

3. **Serializa cada Neurona** → Incluye sus axones y dendritas
   ```java
   writeObject(neurona)
   writeObject(neurona.axones)  // Lista de Conexiones
   ```

4. **Serializa cada Conexion** → Incluye neurona presináptica
   ```java
   writeObject(conexion)
   writeObject(conexion.presináptica)  // ¡Vuelve a Neurona!
   ```

5. **Vuelve al paso 3** → Ciclo infinito
   ```
   Neurona → Conexion → Neurona → Conexion → ...
   ```

### Visualización del Problema

```
Neurona A
  ├── axones: [Conexion1, Conexion2]
  │     └── Conexion1
  │           ├── presináptica: Neurona A  ← ¡Referencia circular!
  │           └── postsinápticas: [Neurona B]
  │                 └── Neurona B
  │                       └── dendritas: [Conexion1]  ← ¡Otra vez!
  │                             └── Conexion1 (ya serializada)
  │                                   └── presináptica: Neurona A
  │                                         └── axones: [Conexion1, ...]
  │                                               └── ... INFINITO
```

## 📊 Por Qué Ocurre con Redes Grandes

### Profundidad del Stack

Cada llamada recursiva consume espacio en el stack:

| Tamaño Red | Conexiones | Profundidad Stack | Resultado |
|------------|------------|-------------------|-----------|
| **3 en Raya** | ~1,000 | ~2,000 llamadas | ✅ OK |
| **Gatos** | ~10,000 | ~20,000 llamadas | ❌ StackOverflow |
| **Damas** | ~49,000 | ~98,000 llamadas | ❌ StackOverflow |

### Cálculo de Profundidad

Para cada conexión:
```
1 Conexion → 1 Neurona pre → N axones → N Conexiones → N Neuronas post → ...
```

Con 49,512 conexiones y densidad 0.85:
- Cada neurona tiene ~40 conexiones
- Profundidad = 49,512 × 2 (ida y vuelta) = ~99,024 llamadas
- Stack típico de JVM: ~10,000 frames
- **Resultado**: StackOverflow

## 💡 Análisis del Diseño Actual

### Redundancia Innecesaria

El diseño actual tiene **doble referencia**:

```java
// En Neurona.java
public class Neurona {
    private List<Conexion> axones;      // Conexiones salientes
    private List<Conexion> dendritas;   // Conexiones entrantes
}

// En Conexion.java
public class Conexion {
    private Neurona presináptica;
    private List<Neurona> postsinápticas;
}
```

**Problema**: Cada conexión se almacena 3 veces:
1. En `RedNeuralExperimental.conexiones`
2. En `Neurona.axones` (neurona presináptica)
3. En `Neurona.dendritas` (neuronas postsinápticas)

### ¿Por Qué Existe Esta Redundancia?

**Razón**: Optimización de búsqueda

```java
// Con referencias bidireccionales (actual):
// O(1) - Acceso directo
for (Conexion dendrita : neurona.getDendritas()) {
    if (dendrita.getPresinaptica().estaActiva()) {
        // Procesar...
    }
}

// Sin referencias en Neurona:
// O(n) - Buscar en todas las conexiones
for (Conexion c : todasLasConexiones) {
    if (c.getPostsinápticas().contains(neurona)) {
        // Procesar...
    }
}
```

**Trade-off**: Velocidad vs Memoria/Serialización

## 🛠️ Soluciones Posibles

### Opción A: Eliminar Referencias en Neurona (Más Simple)

**Propuesta del usuario**: Solo mantener conexiones en `RedNeuralExperimental`, sin listas en `Neurona`:

```java
public class Neurona {
    private final long id;
    private final TipoNeurona tipo;
    private double valorAlmacenado;
    private PotencialMemoria potencial;
    private boolean activa;
    
    // ❌ ELIMINAR:
    // private List<Conexion> axones;
    // private List<Conexion> dendritas;
}

public class Conexion {
    private Neurona presináptica;
    private List<Neurona> postsinápticas;
    private double peso;
}

public class RedNeuralExperimental {
    private List<Neurona> capaSensorial;
    private List<Conexion> conexiones;  // ✅ Única fuente de verdad
}
```

**Ventajas**:
- ✅ Elimina referencias circulares completamente
- ✅ Serialización funciona sin modificaciones
- ✅ Menos memoria (no duplicar referencias)
- ✅ Código más simple

**Desventajas**:
- ❌ Búsqueda más lenta: O(n) en lugar de O(1)
- ❌ Requiere iterar todas las conexiones para encontrar dendritas de una neurona

**Impacto en Rendimiento**:
```java
// Antes (O(1)):
for (Conexion dendrita : neurona.getDendritas()) { ... }

// Después (O(n)):
for (Conexion c : conexiones) {
    if (c.getPostsinápticas().contains(neurona)) { ... }
}
```

Para Damas con 49,512 conexiones y 449 neuronas:
- Cada evaluación de neurona: 49,512 iteraciones
- Total por procesamiento: 449 × 49,512 = ~22 millones de comparaciones
- **Impacto**: Procesamiento 100-1000x más lento

### Opción B: Usar Map de IDs (Propuesta del Usuario)

**Propuesta**: Usar `Map<Long, List<Long>>` para conectividad:

```java
public class RedNeuralExperimental {
    private Map<Long, Neurona> neuronasById;  // Índice rápido
    
    // Opción B1: Mapa de conexiones por neurona
    private Map<Long, List<ConexionDTO>> conexionesPorNeurona;
    
    // Opción B2: Mapa simple de adyacencia
    private Map<Long, List<Long>> grafoConexiones;  // id_pre -> [id_post1, id_post2, ...]
    private Map<Long, Double> pesos;  // "id_pre:id_post" -> peso
}

public class ConexionDTO {
    long idPre;
    List<Long> idsPost;
    double peso;
    TipoConexion tipo;
}
```

**Ventajas**:
- ✅ No hay referencias circulares
- ✅ Serialización trivial (solo primitivos y colecciones)
- ✅ Búsqueda eficiente: O(1) con índice
- ✅ Menos memoria que objetos completos

**Desventajas**:
- ❌ Más complejo de mantener (sincronizar mapas)
- ❌ Pierde encapsulación OOP
- ❌ Código menos legible

**Implementación**:
```java
// Buscar conexiones entrantes (dendritas)
List<ConexionDTO> dendritas = conexionesPorNeurona.get(neurona.getId());
for (ConexionDTO c : dendritas) {
    Neurona pre = neuronasById.get(c.idPre);
    if (pre.estaActiva()) {
        // Procesar...
    }
}
```

### Opción C: Referencias Transient + Reconstrucción

Mantener diseño actual pero marcar referencias como `transient`:

```java
public class Neurona {
    private final long id;
    
    // No se serializan (se reconstruyen al cargar)
    private transient List<Conexion> axones;
    private transient List<Conexion> dendritas;
}

public class Conexion {
    private long idPresináptica;  // Serializable
    private List<Long> idsPostsinápticas;  // Serializable
    
    private transient Neurona presináptica;  // No serializable
    private transient List<Neurona> postsinápticas;  // No serializable
}
```

**Ventajas**:
- ✅ Mantiene rendimiento O(1) en ejecución
- ✅ Serialización sin ciclos
- ✅ Mejor de ambos mundos

**Desventajas**:
- ❌ Requiere método de reconstrucción al cargar
- ❌ Más complejo de implementar

### 1. Romper Referencias Circulares (Opción C - Recomendado)

Usar IDs en lugar de referencias directas:

```java
public class Conexion implements Serializable {
    // En lugar de:
    // private Neurona presináptica;
    
    // Usar:
    private long idPresináptica;
    private List<Long> idsPostsinápticas;
    
    // Mantener referencias transitorias (no serializadas)
    private transient Neurona presináptica;
    private transient List<Neurona> postsinápticas;
}
```

**Ventajas**:
- Elimina ciclos infinitos
- Serialización lineal (O(n))
- Compatible con Java serialization

**Desventajas**:
- Requiere reconstruir referencias al cargar
- Más código de mantenimiento

### 2. Serialización Custom (writeObject/readObject)

Implementar métodos custom de serialización:

```java
public class RedNeuralExperimental implements Serializable {
    
    private void writeObject(ObjectOutputStream out) throws IOException {
        // Escribir campos primitivos
        out.defaultWriteObject();
        
        // Serializar neuronas sin conexiones
        out.writeInt(capaSensorial.size());
        for (Neurona n : capaSensorial) {
            out.writeLong(n.getId());
            out.writeDouble(n.getValorAlmacenado());
            // ... otros campos primitivos
        }
        
        // Serializar conexiones como IDs
        out.writeInt(conexiones.size());
        for (Conexion c : conexiones) {
            out.writeLong(c.getPresinaptica().getId());
            out.writeInt(c.getPostsinápticas().size());
            for (Neurona post : c.getPostsinápticas()) {
                out.writeLong(post.getId());
            }
            out.writeDouble(c.getPeso());
        }
    }
    
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        // Leer campos primitivos
        in.defaultReadObject();
        
        // Reconstruir neuronas
        Map<Long, Neurona> neuronasById = new HashMap<>();
        int numNeuronas = in.readInt();
        for (int i = 0; i < numNeuronas; i++) {
            long id = in.readLong();
            double valor = in.readDouble();
            Neurona n = new Neurona(id, ...);
            neuronasById.put(id, n);
            capaSensorial.add(n);
        }
        
        // Reconstruir conexiones
        int numConexiones = in.readInt();
        for (int i = 0; i < numConexiones; i++) {
            long idPre = in.readLong();
            int numPost = in.readInt();
            List<Long> idsPost = new ArrayList<>();
            for (int j = 0; j < numPost; j++) {
                idsPost.add(in.readLong());
            }
            double peso = in.readDouble();
            
            // Reconstruir conexión
            Neurona pre = neuronasById.get(idPre);
            List<Neurona> post = idsPost.stream()
                .map(neuronasById::get)
                .collect(Collectors.toList());
            
            Conexion c = new Conexion(pre, post, peso, ...);
            conexiones.add(c);
        }
    }
}
```

**Ventajas**:
- Control total sobre serialización
- Eficiente en espacio
- No requiere cambiar estructura de clases

**Desventajas**:
- Código complejo
- Propenso a errores
- Difícil de mantener

### 3. Usar Formato Alternativo (JSON/Protobuf)

Cambiar de Java serialization a formato basado en texto:

```java
// Usando Gson
public void guardarJSON(String filename) throws IOException {
    Gson gson = new GsonBuilder()
        .excludeFieldsWithModifiers(Modifier.TRANSIENT)
        .create();
    
    // Crear DTO sin referencias circulares
    RedNeuralDTO dto = toDTO();
    
    try (Writer writer = new FileWriter(filename)) {
        gson.toJson(dto, writer);
    }
}

private RedNeuralDTO toDTO() {
    RedNeuralDTO dto = new RedNeuralDTO();
    
    // Convertir neuronas a DTOs simples
    for (Neurona n : capaSensorial) {
        dto.neuronas.add(new NeuronaDTO(n.getId(), n.getValorAlmacenado(), ...));
    }
    
    // Convertir conexiones usando IDs
    for (Conexion c : conexiones) {
        dto.conexiones.add(new ConexionDTO(
            c.getPresinaptica().getId(),
            c.getPostsinápticas().stream().map(Neurona::getId).collect(Collectors.toList()),
            c.getPeso()
        ));
    }
    
    return dto;
}
```

**Ventajas**:
- Formato legible (JSON)
- Fácil de debuggear
- Compatible con otros lenguajes
- No hay problemas de referencias circulares

**Desventajas**:
- Archivos más grandes
- Más lento que serialización binaria
- Requiere librería externa (Gson, Jackson)

### 4. Aumentar Stack Size (Temporal)

Aumentar el tamaño del stack de la JVM:

```bash
java -Xss10m -jar mi-aplicacion.jar
```

**Ventajas**:
- Solución rápida
- No requiere cambios de código

**Desventajas**:
- Solo pospone el problema
- No escala a redes muy grandes
- Consume más memoria

## 📈 Comparación de Soluciones

| Solución | Complejidad | Rendimiento | Memoria | Serialización | Recomendado |
|----------|-------------|-------------|---------|---------------|-------------|
| **A: Sin refs en Neurona** | Baja | ❌ O(n) lento | ✅ Baja | ✅ Trivial | ❌ No (muy lento) |
| **B: Map de IDs** | Media | ✅ O(1) rápido | ✅ Media | ✅ Trivial | ⚠️ Posible |
| **C: Transient + rebuild** | Alta | ✅ O(1) rápido | ⚠️ Alta | ✅ Funciona | ✅ Sí |
| **D: JSON/Protobuf** | Baja | ✅ O(1) rápido | ⚠️ Alta | ✅ Funciona | ✅ Sí |
| **E: Aumentar stack** | Muy baja | ✅ O(1) rápido | ⚠️ Alta | ❌ Limitada | ❌ No |

## 🎯 Análisis Detallado

### Opción A: Eliminar Referencias en Neurona

**Código**:
```java
// Buscar dendritas de una neurona
public List<Conexion> getDendritas(Neurona neurona) {
    List<Conexion> dendritas = new ArrayList<>();
    for (Conexion c : conexiones) {
        if (c.getPostsinápticas().contains(neurona)) {
            dendritas.add(c);
        }
    }
    return dendritas;
}
```

**Benchmark estimado** (Damas: 449 neuronas, 49,512 conexiones):
- Antes: 1 procesamiento = 0.1 ms
- Después: 1 procesamiento = 100-500 ms
- **Impacto**: 1000-5000x más lento

**Veredicto**: ❌ No viable para redes grandes

### Opción B: Map de IDs (Propuesta del Usuario)

**Implementación Completa**:
```java
public class RedNeuralExperimental {
    // Índices para acceso rápido
    private Map<Long, Neurona> neuronasById;
    
    // Conectividad como grafo
    private Map<Long, List<ConexionInfo>> conexionesSalientes;  // axones
    private Map<Long, List<ConexionInfo>> conexionesEntrantes;  // dendritas
    
    private static class ConexionInfo {
        long idOtraNeurona;
        double peso;
        TipoConexion tipo;
    }
    
    // Método de búsqueda O(1)
    public List<ConexionInfo> getDendritas(long idNeurona) {
        return conexionesEntrantes.getOrDefault(idNeurona, Collections.emptyList());
    }
}
```

**Ventajas**:
- ✅ Serialización trivial (solo primitivos)
- ✅ Rendimiento O(1)
- ✅ Menos memoria que objetos Conexion completos
- ✅ Fácil de debuggear (ver IDs directamente)

**Desventajas**:
- ⚠️ Requiere refactorizar todo el código
- ⚠️ Pierde tipado fuerte (trabajas con IDs)
- ⚠️ Más propenso a errores (IDs inválidos)

**Veredicto**: ⚠️ Viable pero requiere refactorización grande

### Opción C: Transient + Reconstrucción

**Implementación**:
```java
public class Neurona implements Serializable {
    private final long id;
    private double valorAlmacenado;
    
    // No se serializan
    private transient List<Conexion> axones;
    private transient List<Conexion> dendritas;
    
    // Se reconstruyen al cargar
    public void reconstruirConexiones(List<Conexion> todasConexiones) {
        this.axones = new ArrayList<>();
        this.dendritas = new ArrayList<>();
        
        for (Conexion c : todasConexiones) {
            if (c.getIdPresináptica() == this.id) {
                this.axones.add(c);
            }
            if (c.getIdsPostsinápticas().contains(this.id)) {
                this.dendritas.add(c);
            }
        }
    }
}

public class Conexion implements Serializable {
    // Serializables
    private long idPresináptica;
    private List<Long> idsPostsinápticas;
    private double peso;
    
    // No serializables (se reconstruyen)
    private transient Neurona presináptica;
    private transient List<Neurona> postsinápticas;
    
    public void reconstruirReferencias(Map<Long, Neurona> neuronasById) {
        this.presináptica = neuronasById.get(idPresináptica);
        this.postsinápticas = idsPostsinápticas.stream()
            .map(neuronasById::get)
            .collect(Collectors.toList());
    }
}

public class RedNeuralExperimental implements Serializable {
    
    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        
        // Reconstruir índice de neuronas
        Map<Long, Neurona> neuronasById = new HashMap<>();
        for (Neurona n : capaSensorial) neuronasById.put(n.getId(), n);
        for (List<Neurona> capa : capasInterneuronas) {
            for (Neurona n : capa) neuronasById.put(n.getId(), n);
        }
        for (Neurona n : capaMotora) neuronasById.put(n.getId(), n);
        
        // Reconstruir referencias en conexiones
        for (Conexion c : conexiones) {
            c.reconstruirReferencias(neuronasById);
        }
        
        // Reconstruir listas en neuronas
        for (Neurona n : neuronasById.values()) {
            n.reconstruirConexiones(conexiones);
        }
    }
}
```

**Ventajas**:
- ✅ Mantiene diseño OOP actual
- ✅ Rendimiento O(1) en ejecución
- ✅ Serialización funciona
- ✅ Cambios mínimos en código existente

**Desventajas**:
- ⚠️ Reconstrucción O(n²) al cargar (pero solo una vez)
- ⚠️ Código de serialización más complejo

**Veredicto**: ✅ Mejor balance entre rendimiento y complejidad

### Opción D: JSON con DTOs

Ya explicada anteriormente. Sigue siendo válida y simple.

## 🎯 Recomendación Final Actualizada

### Para Solución Rápida: Opción C (Transient + Rebuild)

**Razones**:
1. Mantiene el diseño actual (menos cambios)
2. Rendimiento O(1) en ejecución
3. Serialización funciona sin StackOverflow
4. Reconstrucción solo ocurre al cargar (una vez)

### Para Solución a Largo Plazo: Opción B (Map de IDs)

**Razones**:
1. Diseño más limpio y escalable
2. Serialización trivial
3. Menos memoria
4. Más fácil de mantener a largo plazo

### NO Recomendado: Opción A (Sin Referencias)

**Razón**: Rendimiento inaceptable para redes grandes (1000x más lento)

## 📊 Tabla de Decisión

| Criterio | Opción A | Opción B | Opción C | Opción D |
|----------|----------|----------|----------|----------|
| **Cambios de código** | Medios | Grandes | Pequeños | Medios |
| **Rendimiento ejecución** | ❌ Muy lento | ✅ Rápido | ✅ Rápido | ✅ Rápido |
| **Rendimiento carga** | ✅ Rápido | ✅ Rápido | ⚠️ Lento | ⚠️ Lento |
| **Memoria** | ✅ Baja | ✅ Media | ❌ Alta | ❌ Alta |
| **Complejidad** | Baja | Media | Alta | Baja |
| **Mantenibilidad** | Media | Alta | Media | Alta |
| **Recomendado** | ❌ | ✅ Largo plazo | ✅ Corto plazo | ✅ Alternativa |

1. **Fácil de implementar**
2. **Debuggeable** (puedes ver el JSON)
3. **Escalable** (funciona con cualquier tamaño)
4. **Mantenible** (código simple)

### Implementación Recomendada

```java
// Añadir dependencia en pom.xml
<dependency>
    <groupId>com.google.code.gson</groupId>
    <artifactId>gson</artifactId>
    <version>2.10.1</version>
</dependency>
```

```java
// Crear DTOs simples
public class RedNeuralDTO {
    List<NeuronaDTO> neuronas;
    List<ConexionDTO> conexiones;
    int[] topologia;
    double densidad;
}

public class NeuronaDTO {
    long id;
    String tipo;
    double valorAlmacenado;
    double potencial;
}

public class ConexionDTO {
    long idPre;
    List<Long> idsPost;
    double peso;
    String tipo;
}
```

## 💬 Respuesta a Tu Pregunta

### "¿No podemos dejar en las conexiones la lista de neuronas y ya está?"

**Respuesta Corta**: Sí, pero con un costo de rendimiento significativo.

**Análisis**:

Tu propuesta eliminaría esto:
```java
// ❌ ELIMINAR de Neurona
private List<Conexion> axones;
private List<Conexion> dendritas;
```

Y solo mantendría:
```java
// ✅ MANTENER en RedNeuralExperimental
private List<Conexion> conexiones;

// ✅ MANTENER en Conexion
private Neurona presináptica;
private List<Neurona> postsinápticas;
```

**Problema**: Para evaluar una neurona necesitas sus dendritas:
```java
// Antes (O(1) - acceso directo):
for (Conexion dendrita : neurona.getDendritas()) {
    if (dendrita.getPresinaptica().estaActiva()) {
        sumaInputs += dendrita.getPeso();
    }
}

// Después (O(n) - buscar en todas):
for (Conexion c : todasLasConexiones) {
    if (c.getPostsinápticas().contains(neurona)) {
        if (c.getPresinaptica().estaActiva()) {
            sumaInputs += c.getPeso();
        }
    }
}
```

**Impacto en Damas**:
- 449 neuronas × 49,512 conexiones = 22 millones de comparaciones por procesamiento
- Tiempo estimado: 100-1000x más lento
- 20 minutos de entrenamiento → 33-333 horas

**Veredicto**: ❌ No viable para redes grandes

### "¿Podríamos usar Map<Long, List<Long>>?"

**Respuesta Corta**: ¡Sí! Esta es una excelente solución.

**Tu Propuesta**:
```java
public class RedNeuralExperimental {
    private Map<Long, Neurona> neuronasById;
    
    // Grafo de conectividad
    private Map<Long, List<Long>> conexionesSalientes;  // id_pre -> [id_post1, ...]
    private Map<Long, List<Long>> conexionesEntrantes;  // id_post -> [id_pre1, ...]
    
    // Pesos separados
    private Map<String, Double> pesos;  // "id_pre:id_post" -> peso
}
```

**Ventajas de tu propuesta**:
- ✅ Serialización trivial (solo primitivos)
- ✅ Rendimiento O(1) con índice
- ✅ Menos memoria
- ✅ No hay ciclos

**Implementación**:
```java
// Evaluar neurona con tu diseño
public boolean evaluar(long idNeurona, long timestamp) {
    Neurona neurona = neuronasById.get(idNeurona);
    List<Long> idsPresinápticas = conexionesEntrantes.get(idNeurona);
    
    double sumaInputs = 0.0;
    for (Long idPre : idsPresinápticas) {
        Neurona pre = neuronasById.get(idPre);
        if (pre.estaActiva()) {
            String key = idPre + ":" + idNeurona;
            double peso = pesos.get(key);
            sumaInputs += peso * pre.getPotencial();
        }
    }
    
    // ... resto de lógica
}
```

**Desventajas**:
- ⚠️ Requiere refactorizar mucho código
- ⚠️ Pierde tipado fuerte (trabajas con IDs)
- ⚠️ Más propenso a errores (IDs inválidos)

**Veredicto**: ✅ Excelente solución a largo plazo

### Mi Recomendación

**Para arreglar el problema ahora**: Opción C (Transient + Rebuild)
- Cambios mínimos
- Funciona inmediatamente
- Mantiene rendimiento

**Para rediseñar el sistema**: Tu propuesta (Map de IDs)
- Diseño más limpio
- Escalable
- Fácil de serializar

**Código de ejemplo para Opción C** (arreglo rápido):

```java
// En Neurona.java
public class Neurona implements Serializable {
    private final long id;
    private double valorAlmacenado;
    
    // Marcar como transient (no serializar)
    private transient List<Conexion> axones;
    private transient List<Conexion> dendritas;
    
    // Inicializar en constructor
    public Neurona(...) {
        // ...
        this.axones = new ArrayList<>();
        this.dendritas = new ArrayList<>();
    }
    
    // Reconstruir después de deserializar
    void reconstruirConexiones(List<Conexion> todasConexiones) {
        this.axones = new ArrayList<>();
        this.dendritas = new ArrayList<>();
        
        for (Conexion c : todasConexiones) {
            if (c.getPresinaptica() == this) {
                this.axones.add(c);
            }
            if (c.getPostsinápticas().contains(this)) {
                this.dendritas.add(c);
            }
        }
    }
}

// En RedNeuralExperimental.java
private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject();
    
    // Reconstruir listas transient en todas las neuronas
    List<Neurona> todasNeuronas = new ArrayList<>();
    todasNeuronas.addAll(capaSensorial);
    for (List<Neurona> capa : capasInterneuronas) {
        todasNeuronas.addAll(capa);
    }
    todasNeuronas.addAll(capaMotora);
    
    for (Neurona n : todasNeuronas) {
        n.reconstruirConexiones(conexiones);
    }
}
```

Este cambio:
- ✅ Arregla el StackOverflow inmediatamente
- ✅ Mantiene el rendimiento actual
- ✅ Requiere ~50 líneas de código
- ✅ No rompe código existente

## 📝 Conclusión

El StackOverflow ocurre por:
1. **Referencias bidireccionales** entre Neurona ↔ Conexion
2. **Serialización recursiva** que sigue las referencias infinitamente
3. **Redes grandes** que exceden la capacidad del stack

La solución más práctica es usar JSON con DTOs que rompen las referencias circulares mediante IDs.

---

**Archivos Afectados**:
- `src/main/java/es/jastxz/nn/RedNeuralExperimental.java`
- `src/main/java/es/jastxz/nn/Neurona.java`
- `src/main/java/es/jastxz/nn/Conexion.java`

**Modelos Afectados**:
- Gatos: 10,357 conexiones → StackOverflow
- Damas: 49,512 conexiones → StackOverflow

**Modelos Funcionales**:
- 3 en Raya: ~1,000 conexiones → OK
- OR/AND/XOR: <100 conexiones → OK
