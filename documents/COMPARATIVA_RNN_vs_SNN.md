# Comparativa Teórica: RNN vs SNN

## Modelo computacional fundamental

Una **RNN** (Red Neuronal Recurrente) opera con **valores continuos** y **tiempo discreto**.
Una **SNN** (Spiking Neural Network) opera con **eventos discretos** (spikes) y **tiempo continuo** (o pseudo-continuo por timesteps).

---

## Cálculo en una RNN (tipo Elman/LSTM)

En cada paso temporal `t`, cada neurona calcula:

```
hₜ = σ(W_h · hₜ₋₁ + W_x · xₜ + b)
yₜ = softmax(W_y · hₜ)
```

Donde:
- `hₜ` = estado oculto (vector continuo, se mantiene entre pasos)
- `xₜ` = entrada en el paso t
- `σ` = función de activación (tanh, ReLU...)
- El estado anterior `hₜ₋₁` retroalimenta → memoria

**Operaciones por paso:** multiplicación matricial + suma + activación no lineal.
Todo es diferenciable → backpropagation through time (BPTT).

---

## Cálculo en una SNN (modelo LIF)

En cada timestep `dt`, cada neurona integra:

```
τ · dV/dt = -(V - V_rest) + R · I(t)

Si V ≥ V_thresh → spike + V = V_reset (periodo refractario)
```

Donde:
- `V` = potencial de membrana (simula voltaje biológico)
- `I(t)` = corriente de entrada (suma de spikes ponderados por peso sináptico)
- `τ` = constante de tiempo de la membrana (decaimiento)
- No hay activación explícita: la no-linealidad **ES** el umbral de disparo

**Operaciones por paso:** integración diferencial + comparación con umbral + propagación de eventos.
No es diferenciable en el spike → no se puede usar BPTT directamente.

---

## Esquema de flujo comparativo

### RNN — Flujo de cálculo

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        RNN — FLUJO DE CÁLCULO                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Paso t                          Paso t+1                                 ║
║                                                                            ║
║   ┌───────┐                       ┌───────┐                                ║
║   │  xₜ   │  entrada              │ xₜ₊₁  │  entrada                      ║
║   └───┬───┘                       └───┬───┘                                ║
║       │                               │                                    ║
║       ▼                               ▼                                    ║
║   ┌───────────────┐  hₜ₋₁ ──►  ┌───────────────┐  hₜ ──►  ...            ║
║   │  W_x·xₜ      │             │  W_x·xₜ₊₁    │                          ║
║   │  + W_h·hₜ₋₁  │             │  + W_h·hₜ     │                          ║
║   │  + bias       │             │  + bias        │                          ║
║   │  ─────────    │             │  ─────────     │                          ║
║   │  σ(suma)      │             │  σ(suma)       │                          ║
║   └───────┬───────┘             └───────┬────────┘                         ║
║           │ hₜ (vector continuo)        │ hₜ₊₁                             ║
║           ▼                             ▼                                  ║
║   ┌──────────────┐              ┌──────────────┐                           ║
║   │ W_y · hₜ     │              │ W_y · hₜ₊₁   │                           ║
║   │ softmax      │              │ softmax       │                           ║
║   └──────┬───────┘              └──────┬────────┘                          ║
║          ▼                             ▼                                   ║
║        [yₜ]                          [yₜ₊₁]                                ║
║   (salida continua)             (salida continua)                          ║
║                                                                            ║
║   Valores que fluyen: VECTORES REALES en cada paso                         ║
║   Memoria: estado oculto hₜ (denso, todos los valores)                     ║
║   Entrenamiento: BPTT (gradiente fluye hacia atrás en el tiempo)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### SNN — Flujo de cálculo

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SNN — FLUJO DE CÁLCULO                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Timestep t                      Timestep t+1                             ║
║                                                                            ║
║   ┌───────┐                       ┌───────┐                                ║
║   │  xₜ   │  spike / no spike     │ xₜ₊₁  │  spike / no spike             ║
║   └───┬───┘  (0 ó 1)              └───┬───┘  (0 ó 1)                      ║
║       │                               │                                    ║
║       ▼                               ▼                                    ║
║   ┌──────────────────┐          ┌──────────────────┐                       ║
║   │ I(t) = Σ wⱼ·sⱼ  │          │ I(t+1) = Σ wⱼ·sⱼ│                       ║
║   │ (solo spikes      │          │ (solo spikes      │                      ║
║   │  activos aportan) │          │  activos aportan) │                      ║
║   └────────┬─────────┘          └────────┬─────────┘                       ║
║            ▼                             ▼                                  ║
║   ┌──────────────────┐          ┌──────────────────┐                       ║
║   │ V += decay       │          │ V += decay       │                       ║
║   │    + R·I(t)      │          │    + R·I(t+1)    │                       ║
║   │                  │          │                  │                        ║
║   │ V ──► decae      │  Vₜ ──► │ V ──► decae      │  Vₜ₊₁ ──►            ║
║   │       hacia       │          │       hacia       │                      ║
║   │       V_rest     │          │       V_rest     │                       ║
║   └────────┬─────────┘          └────────┬─────────┘                       ║
║            ▼                             ▼                                  ║
║   ┌──────────────────┐          ┌──────────────────┐                       ║
║   │ V ≥ umbral ?     │          │ V ≥ umbral ?     │                       ║
║   │                  │          │                  │                        ║
║   │  SÍ ──► SPIKE!  │          │  SÍ ──► SPIKE!  │                        ║
║   │  V = V_reset     │          │  V = V_reset     │                       ║
║   │                  │          │                  │                        ║
║   │  NO ──► silencio │          │  NO ──► silencio │                       ║
║   └────────┬─────────┘          └────────┬─────────┘                       ║
║            ▼                             ▼                                  ║
║        [0 ó 1]                       [0 ó 1]                               ║
║   (evento binario)              (evento binario)                           ║
║                                                                            ║
║   Valores que fluyen: EVENTOS BINARIOS (spike o nada)                      ║
║   Memoria: potencial de membrana V (analógico, con decaimiento)            ║
║   Entrenamiento: STDP (local, basado en timing de spikes)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Tabla comparativa de operaciones

```
┌─────────────────────┬──────────────────────────┬──────────────────────────┐
│                     │          RNN             │          SNN             │
├─────────────────────┼──────────────────────────┼──────────────────────────┤
│ Señal               │ Vector real continuo     │ Spike binario (0/1)      │
│ Tiempo              │ Discreto (pasos)         │ Pseudo-continuo (ms)     │
│ Op. principal       │ Multiplicación matricial │ Integración diferencial  │
│ No-linealidad       │ σ(x), tanh, ReLU         │ Umbral de disparo        │
│ Memoria             │ Estado oculto hₜ         │ Potencial de membrana V  │
│ Decaimiento         │ No (o gated en LSTM)     │ Sí (exponencial, τ)      │
│ Activación          │ Siempre (valor continuo) │ Sparse (solo al disparar)│
│ Coste por neurona   │ O(n²) por paso           │ O(k) solo si hay spike   │
│ Entrenamiento       │ BPTT (global, gradiente) │ STDP (local, temporal)   │
│ Diferenciable       │ Sí                       │ No (en el spike)         │
│ Eficiencia HW       │ GPU (paralelismo SIMD)   │ Neuromórfico (eventos)   │
│ Inspiración         │ Matemática/optimización  │ Biología/neurociencia    │
└─────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## La diferencia clave

La RNN procesa **todo** en cada paso: todas las neuronas calculan, todas producen salida. Es denso y predecible.

La SNN es **sparse** por naturaleza: solo las neuronas que alcanzan el umbral "hablan". En un timestep dado, quizá el 5-10% de las neuronas disparan. Esto es ineficiente en GPU (que quiere paralelismo masivo) pero extremadamente eficiente en **hardware neuromórfico** (como Intel Loihi o IBM TrueNorth), donde cada spike es un evento que consume energía solo cuando ocurre.

---

## Implicaciones prácticas

En la comparativa del 3 en raya de este proyecto, la red clásica gana porque backprop es un algoritmo de optimización matemáticamente superior para minimizar error supervisado. La SNN con STDP está aprendiendo de forma más "biológica" pero menos dirigida — como la diferencia entre estudiar con un profesor particular (backprop) vs aprender por exposición repetida (STDP).

Para acercar el rendimiento de una SNN al de una RNN habría que:

1. **Usar un algoritmo de entrenamiento supervisado adaptado a SNNs** (como surrogate gradient o e-prop) en vez de STDP puro
2. **Darle al AG más presupuesto** — más población, más generaciones, más muestras de entrenamiento
3. **Afinar la codificación temporal** para que la información se preserve mejor en la conversión valor→spike→valor
