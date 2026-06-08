---
inclusion: always
---

# Coding Standards — neural-lib

## Java
- Standard Java conventions: PascalCase classes, camelCase methods
- Library code (`nn/`, `math/`, `models/`) must NOT depend on Spring
- Spring only in `services/` package for demo/integration
- Document public API with Javadoc — especially `getValor()` contracts
- Prefer composition over inheritance where possible
- Immutable data objects where practical
- Builder pattern for complex configuration (see `ConfiguracionRedBuilder`)

## Package Conventions
- `es.jastxz.nn.*` — Neural network implementations
- `es.jastxz.math` — Pure math/matrix operations
- `es.jastxz.models` — Shared domain models
- `es.jastxz.util` — Stateless utility functions
- `es.jastxz.services` — Spring-bound service layer (demo only)

## Build
- `mvn clean install` to build and install locally
- `mvn test` to run tests (JUnit 5)
- Version managed in pom.xml — update on breaking changes

## Performance
- Avoid autoboxing in numerical code — use primitive arrays
- Minimize object allocations in matrix operations (`SmallMatrix`)
- `getValor()` is called extensively — keep it allocation-free
- Profile with JMH for critical paths via benchmark module

## Testing
- Test classes in `src/test/java/`
- Spiking tests: `NeuronaSpikingTest`, `GestorSTDPTest`
- Benchmark configs for comparative performance testing
