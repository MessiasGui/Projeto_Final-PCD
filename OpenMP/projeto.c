#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

# define N 2000 // Tamanho da grade
# define T 500 // Número de iterações
# define D 0.1  // Coeficiente de difusão
# define DELTA_T 0.01
# define DELTA_X 1.0

int NThreads = 2;

void diff_eq(double **C, double **C_new) {
    for (int t = 0; t < T; t++) {
      double difmedio = 0.0;
      #pragma omp parallel num_threads(NThreads)
      {
        #pragma omp for collapse (2)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }
        
        #pragma omp for collapse (2) reduction(+:difmedio)
        // Atualizar matriz para a próxima iteração
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }

        if ((t % 100) == 0)
            printf("iteracao %d - diferenca=%g\n", t, difmedio / ((N-2) * (N-2)));
      }
    }
}

int main() {
  printf("Concentração final no centro | ");
    printf("Tempo de processamento | ");
    printf("Numero de Threads\n");
    
        // Concentração inicial
        double **C = (double **)malloc(N * sizeof(double *));
        if (C == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    for(int i = 0; i <= 10; i++){
        for (int i = 0; i < N; i++) {
            C[i] = (double *)malloc(N * sizeof(double));
            if (C[i] == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return 1;
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
            C[i][j] = 0.;
            }
        }
        
        // Concentração para a próxima iteração
        double **C_new = (double **)malloc(N * sizeof(double *));
        if (C_new == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        for (int i = 0; i < N; i++) {
            C_new[i] = (double *)malloc(N * sizeof(double));
            if (C_new[i] == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return 1;
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C_new[i][j] = 0.;
            }
        }
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
        if(i == 10 && NThreads < 16){
          i = 0;
          NThreads = NThreads * 2;
        }
    }
    return 0;
}