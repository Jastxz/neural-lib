---
inclusion: always
---

# Architecture — neural-lib

## Stack
- **Framework:** Spring Boot (demo/test only — library is framework-agnostic)
- **Language:** Java
- **Build:** Maven (pom.xml)
- **Package:** es.jastxz

## Module Boundaries (from graphify: 3074 nodes, 8171 edges, 136 communities)
- God nodes: `getValor()` (69 edges), `RedNeuralExperimental` (54), `ConfiguracionRedBuilder` (44), `RedNeuralSpiking` (44)
- `es.jastxz.nn/` — Neural network core
  - `spiking/` — Spiking neural networks (NeuronaSpiking, RedNeuralSpiking, STDP learning)
  - `experimental/` — Experimental neural architectures (RedNeuralExperimental)
  - `nube/` — Cloud/random-cloud-based neural networks
  - `genetico/` — Genetic algorithm optimization
  - `benchmark/` — Performance benchmarking framework (ConfiguracionBenchmark, NivelComplejidad)
  - `enums/` — Shared enumerations
- `es.jastxz.math/` — Mathematical utilities (SmallMatrix, linear algebra)
- `es.jastxz.models/` — Shared model classes
- `es.jastxz.util/` — General utilities
- `es.jastxz.services/` — Service layer (for Spring Boot demo)

## Dependency Rules
- Library core (`nn/`, `math/`, `models/`) MUST NOT depend on Spring
- `services/` depends on Spring — it is the demo/integration layer only
- `getValor()` is the universal value accessor — all neural types expose it
- `ConfiguracionRedBuilder` is the builder pattern for network configuration
- `math/` is a pure foundation — depended on by all `nn/` subpackages
- Spiking module is self-contained — STDP learning rules are internal
- Benchmark module tests all other modules — depends on everything

## Design Principles
- Builder pattern for network configuration (`ConfiguracionRedBuilder`)
- Multiple neural paradigms: feedforward, spiking, experimental, cloud-based
- Genetic algorithms for hyperparameter optimization
- Benchmark framework for comparative evaluation across architectures
