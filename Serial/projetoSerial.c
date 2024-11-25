# include <stdio.h>
# include <omp.h>
#include <math.h>

# define N 2000  // Tamanho da grade
# define T 500 // Número de iterações
# define D 0.1  // Coeficiente de difusão
# define DELTA_T 0.01
# define DELTA_X 1.0

int NThreads = 1;

void diff_eq(double C[N][N], double C_new[N][N]) {
    for (int t = 0; t < T; t++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }
        // Atualizar matriz para a próxima iteração
        double difmedio = 0.;
        for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    difmedio += fabs(C_new[i][j] - C[i][j]);
                    C[i][j] = C_new[i][j];
            }
        }
        if ((t%100) == 0)
            printf("iteracao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main() {
    printf("Concentração final no centro | ");
    printf("Tempo de processamento | ");
    printf("Numero de Threads\n");
    for(int i = 0; i < 10; i++){
        double C[N][N] = {0};      // Concentração inicial
        double C_new[N][N] = {0};  // Concentração para a próxima iteração
        double startTime, endTime;

        // Inicializar uma concentração alta no centro
        C[N/2][N/2] = 1.0;

        // Executar a equação de difusão
        startTime = omp_get_wtime();
        diff_eq(C, C_new);
        endTime = omp_get_wtime();

        // Exibir resultado para verificação
        printf("%f | ", C[N/2-2][N/2-2]);
        printf("%f | ", endTime-startTime);
        printf("%d\n", NThreads);
    }
    return 0;
}