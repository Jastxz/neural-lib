---
inclusion: fileMatch
fileMatchPattern: "**/*Test.java,**/*Tests.java"
---

# Testing — neural-lib

## Framework
- JUnit 5

## Key Test Classes
- `NeuronaSpikingTest` — Spiking neuron behavior verification
- `GestorSTDPTest` — STDP learning rule correctness
- Benchmark tests via `ConfiguracionBenchmark`

## Run
- `mvn test`

## Conventions
- Test classes in `src/test/java/` mirroring source structure
- Each neural module has dedicated test classes
- Benchmark tests for performance regression detection
- Test naming: `should_ExpectedBehavior_When_Condition`
- Numerical tolerance: use `assertEquals(expected, actual, delta)` for floating point
