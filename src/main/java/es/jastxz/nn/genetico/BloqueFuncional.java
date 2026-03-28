package es.jastxz.nn.genetico;

/**
 * Agrupación de genes dependientes dentro del Cromosoma que deben
 * mantenerse juntos durante el cruce para preservar coherencia funcional.
 */
public enum BloqueFuncional {
    TOPOLOGIA,      // capasOcultas, neuronasPorCapa[]
    LIF,            // umbralDisparo, potencialReposo, constanteDecaimiento, duracionRefractario
    STDP,           // amplitudLTP, amplitudLTD, tauLTP, tauLTD
    CODIFICACION,   // frecuenciaMaxima, modoCodificacion, ventanaDecodificacion
    REGULACION,     // homeostasisActiva, tasaDisparoObjetivo, tasaAjusteHomeostasis,
                    // inhibicionLateralActiva, radioInhibicion, fuerzaInhibicion
    COMPETICION     // wtaActivo, wtaCapaSalida, wtaCapasOcultas, radioWTA,
                    // fuerzaWTA, umbralActivacionWTA
}
