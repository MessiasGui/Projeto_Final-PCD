#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>
#include <math.h>

#define N 2000   // Tamanho da grade
#define T 500    // Número de iterações
#define D 0.1    // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double *C, double *C_new, int local_rows, int rank, int size) {
    MPI_Request send_reqs[2], recv_reqs[2];

    for (int t = 0; t < T; t++) {
        double local_difmedio = 0.0, global_difmedio = 0.0;

        // Comunicação MPI: Troca das linhas de borda com vizinhos
        if (rank > 0)
            MPI_Irecv(&C[0 * N], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_reqs[0]);
        if (rank < size - 1)
            MPI_Irecv(&C[(local_rows - 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_reqs[1]);

        if (rank > 0)
            MPI_Isend(&C[1 * N], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_reqs[0]);
        if (rank < size - 1)
            MPI_Isend(&C[(local_rows - 2) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_reqs[1]);

        // Computação da difusão para o núcleo interno (exceto as bordas)
        for (int i = 2; i < local_rows - 2; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i * N + j] = C[i * N + j] + D * DELTA_T * (
                    (C[(i + 1) * N + j] + C[(i - 1) * N + j] +
                     C[i * N + j + 1] + C[i * N + j - 1] - 4 * C[i * N + j]) /
                    (DELTA_X * DELTA_X)
                );
                local_difmedio += fabs(C_new[i * N + j] - C[i * N + j]); // Calcula a diferença média local
            }
        }

        // Aguarda recebimento das bordas antes de atualizar as bordas
        if (rank > 0) MPI_Wait(&recv_reqs[0], MPI_STATUS_IGNORE);
        if (rank < size - 1) MPI_Wait(&recv_reqs[1], MPI_STATUS_IGNORE);

        // Atualização das linhas de borda
        for (int j = 1; j < N - 1; j++) {
            if (rank > 0) {
                C_new[1 * N + j] = C[1 * N + j] + D * DELTA_T * (
                    (C[2 * N + j] + C[0 * N + j] + C[1 * N + j + 1] + C[1 * N + j - 1] - 4 * C[1 * N + j]) /
                    (DELTA_X * DELTA_X)
                );
                local_difmedio += fabs(C_new[1 * N + j] - C[1 * N + j]);
            }
            if (rank < size - 1) {
                C_new[(local_rows - 2) * N + j] = C[(local_rows - 2) * N + j] + D * DELTA_T * (
                    (C[(local_rows - 1) * N + j] + C[(local_rows - 3) * N + j] +
                     C[(local_rows - 2) * N + j + 1] + C[(local_rows - 2) * N + j - 1] - 4 * C[(local_rows - 2) * N + j]) /
                    (DELTA_X * DELTA_X)
                );
                local_difmedio += fabs(C_new[(local_rows - 2) * N + j] - C[(local_rows - 2) * N + j]);
            }
        }

        // Aguarda envio das bordas antes de avançar
        if (rank > 0) MPI_Wait(&send_reqs[0], MPI_STATUS_IGNORE);
        if (rank < size - 1) MPI_Wait(&send_reqs[1], MPI_STATUS_IGNORE);

        // Redução para calcular a diferença média total
        MPI_Reduce(&local_difmedio, &global_difmedio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        global_difmedio /= ((N - 2) * (N - 2)); // Normaliza a diferença média

        // Troca as matrizes
        double *temp = C;
        C = C_new;
        C_new = temp;

        // Impressão nas iterações específicas (0, 100, 200, 300, 400)
        if ((t == 0 || t == 100 || t == 200 || t == 300 || t == 400) && rank == 0) {
            printf("Iteração %d - Diferença Média: %g\n", t, global_difmedio);
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = (N / size) + 2;  // Cada processo recebe um pedaço da matriz
    double *C = (double *)calloc(local_rows * N, sizeof(double));
    double *C_new = (double *)calloc(local_rows * N, sizeof(double));

    // Apenas um processo inicializa a concentração no centro
    if (rank == size / 2) {
        int local_center = (N / size) / 2 + 1;
        C[local_center * N + N / 2] = 1.0;
    }

    // Inicia medição de tempo
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double start_time = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    // Executa a equação de difusão
    diff_eq(C, C_new, local_rows, rank, size);

    
    // Processo 0 imprime os resultados finais
    double local_concentration = C[(local_rows / 2) * N + N / 2];
    double final_concentration = 0.0;
    MPI_Reduce(&local_concentration, &final_concentration, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // Finaliza medição de tempo
    gettimeofday(&tv, NULL);
    double end_time = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    if (rank == 0) {
        printf("Concentração final no centro | Tempo de processamento | Número de Processos\n");
        printf("%f | %fs | %d\n", final_concentration, end_time - start_time, size);
    }

    free(C);
    free(C_new);
    MPI_Finalize();
    return 0;
}