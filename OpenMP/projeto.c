#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 2000  // Tamanho da grade
#define T 500   // Número de iterações
#define D 0.1   // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

int NThreads = 2;

void diff_eq(double **C, double **C_new) {
    for (int t = 0; t < T; t++) {
        double difmedio = 0.0;

        // Cálculo da nova concentração (escrevendo em C_new)
        #pragma omp parallel for num_threads(NThreads) collapse(2)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        // Troca os ponteiros, em vez de copiar os valores
        double **temp = C;
        C = C_new;
        C_new = temp;

        // Cálculo da diferença média para verificação de convergência
        difmedio = 0.0;
        #pragma omp parallel for num_threads(NThreads) collapse(2) reduction(+:difmedio)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
            }
        }

        if ((t % 100) == 0)
            printf("Iteração %d - Diferença Média = %g\n", t, difmedio / ((N-2) * (N-2)));
    }
}

int main() {
    
    for (NThreads = 2; NThreads <= 16; NThreads *= 2) {  
        // Alocação das matrizes
        double **C = (double **)malloc(N * sizeof(double *));
        double **C_new = (double **)malloc(N * sizeof(double *));
        if (C == NULL || C_new == NULL) {
            fprintf(stderr, "Erro na alocação de memória\n");
            return 1;
        }
        
        for (int i = 0; i < N; i++) {
            C[i] = (double *)malloc(N * sizeof(double));
            C_new[i] = (double *)malloc(N * sizeof(double));
            if (C[i] == NULL || C_new[i] == NULL) {
                fprintf(stderr, "Erro na alocação de memória\n");
                return 1;
            }
        }
        
        // Inicializa as matrizes com zero
        #pragma omp parallel for num_threads(NThreads) collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0.0;
                C_new[i][j] = 0.0;
            }
        }
        
        // Inicializa a concentração no centro
        C[N/2][N/2] = 1.0;
        
        // Executa a equação de difusão
        double startTime = omp_get_wtime();
        diff_eq(C, C_new);
        double endTime = omp_get_wtime();
        
        // Exibe resultado
        printf("Concentração final no centro | ");
        printf("Tempo de processamento | ");
        printf("Número de Threads\n");
        printf("%f | ", C[N/2][N/2]);
        printf("%f | ", endTime - startTime);
        printf("%d\n", NThreads);

        // Libera memória
        for (int i = 0; i < N; i++) {
            free(C[i]);
            free(C_new[i]);
        }
        free(C);
        free(C_new);
    }

    return 0;
}