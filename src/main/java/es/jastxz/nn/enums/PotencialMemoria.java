package es.jastxz.nn.enums;

public enum PotencialMemoria {
    REPOSO(-70.0),
    UMBRAL(-55.0),
    PICO(40.0);

    private final double valor;

    PotencialMemoria(double valor) {
        this.valor = valor;
    }

    public double getValor() {
        return valor;
    }
}
