package org.javig.util;

/**
 * Helper class to store simulation results
 */
public class ResultadoSimulacion {
    final String nombreModelo;
    final int victoriasModelo;
    final int victoriasMinimax;

    public ResultadoSimulacion(String nombreModelo, int victoriasModelo, int victoriasMinimax) {
        this.nombreModelo = nombreModelo;
        this.victoriasModelo = victoriasModelo;
        this.victoriasMinimax = victoriasMinimax;
    }

    public String getNombreModelo() {
        return nombreModelo;
    }

    public int getVictoriasModelo() {
        return victoriasModelo;
    }

    public int getVictoriasMinimax() {
        return victoriasMinimax;
    }

    public double getPorcentajeVictoriasModelo() {
        return (double) victoriasModelo / (victoriasModelo + victoriasMinimax);
    }

    public double getPorcentajeVictoriasMinimax() {
        return (double) victoriasMinimax / (victoriasModelo + victoriasMinimax);
    }
}