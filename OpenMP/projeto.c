# include <stdio.h>
# include <omp.h>
# define N 500 // Tamanho da grade
# define T 1000 // Número de iterações
# define D 0.1  // Coeficiente de difusão
# define DELTA_T 0.01
# define DELTA_X 1.0

int NThreads = 2;

void diff_eq(double C[N][N], double C_new[N][N]) {
    for (int t = 0; t < T; t++) {
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
        #pragma omp for collapse (2)
        // Atualizar matriz para a próxima iteração
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C[i][j] = C_new[i][j];
            }
        }
      }
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
    if(i == 9 && NThreads < 16){
      i = 0;
      NThreads = NThreads * 2;
    }
  }

  return 0;
}