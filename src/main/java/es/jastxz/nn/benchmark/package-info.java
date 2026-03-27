/**
 * Paquete para la Suite de Benchmarks de la Red Neuronal Spiking (SNN).
 *
 * <p>Este paquete contiene el marco de pruebas paramétrico que evalúa sistemáticamente
 * el rendimiento de {@code RedNeuralSpiking} variando topologías de red y niveles
 * de complejidad de problemas (puertas lógicas → 3 en Raya → Gatos → Damas).</p>
 *
 * <p>Componentes principales:</p>
 * <ul>
 *   <li>{@code NivelComplejidad} - Enum con niveles de complejidad por espacio de estados</li>
 *   <li>{@code ConfiguracionBenchmark} - Configuración de topología + problema para un benchmark</li>
 *   <li>{@code ResultadoBenchmark} - Resultado con métricas de una ejecución de benchmark</li>
 *   <li>{@code GeneradorTopologias} - Genera combinaciones paramétricas de topologías</li>
 *   <li>{@code RecolectorMetricas} - Ejecuta benchmarks y recopila métricas</li>
 *   <li>{@code DetectorLimites} - Detecta límites operativos de la red</li>
 *   <li>{@code GeneradorInforme} - Genera informe comparativo por consola</li>
 *   <li>{@code GeneradorGraficas} - Genera gráficas PNG e índice HTML con XChart</li>
 * </ul>
 *
 * @since 1.1
 */
package es.jastxz.nn.benchmark;
