%%writefile MPIcomOpenMP.c

%%writefile projeto.c

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <sys/time.h>

#define N 1000   // Tamanho da grade
#define T 1000  // Número de iterações
#define D 0.1   // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double *C, double *C_new, int start, int end, int rank, int size, int local_rows) {
    MPI_Request send_reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Request recv_reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    for (int t = 0; t < T; t++) {
        if (rank > 0)
            MPI_Irecv(&C[0 * N], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_reqs[0]);
        if (rank < size - 1)
            MPI_Irecv(&C[(local_rows - 1) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_reqs[1]);
        if (rank > 0)
            MPI_Isend(&C[1 * N], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_reqs[0]);
        if (rank < size - 1)
            MPI_Isend(&C[(local_rows - 2) * N], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_reqs[1]);

        #pragma omp parallel for collapse(2)
        for (int i = 2; i < local_rows - 2; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i * N + j] = C[i * N + j] + D * DELTA_T * (
                    (C[(i + 1) * N + j] + C[(i - 1) * N + j] + C[i * N + j + 1] + C[i * N + j - 1] - 4 * C[i * N + j]) / (DELTA_X * DELTA_X)
                );
            }
        }
        MPI_Waitall(2, recv_reqs, MPI_STATUSES_IGNORE);

        #pragma omp parallel for
        for (int j = 1; j < N - 1; j++) {
            if (rank > 0) {
                C_new[1 * N + j] = C[1 * N + j] + D * DELTA_T * (
                    (C[2 * N + j] + C[0 * N + j] + C[1 * N + j + 1] + C[1 * N + j - 1] - 4 * C[1 * N + j]) / (DELTA_X * DELTA_X)
                );
            }
            if (rank < size - 1) {
                C_new[(local_rows - 2) * N + j] = C[(local_rows - 2) * N + j] + D * DELTA_T * (
                    (C[(local_rows - 1) * N + j] + C[(local_rows - 3) * N + j] + C[(local_rows - 2) * N + j + 1] + C[(local_rows - 2) * N + j - 1] - 4 * C[(local_rows - 2) * N + j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        MPI_Waitall(2, send_reqs, MPI_STATUSES_IGNORE);

        double *temp = C;
        C = C_new;
        C_new = temp;
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_rows = (N / size) + 2;
    double *C = (double *)calloc(local_rows * N, sizeof(double));
    double *C_new = (double *)calloc(local_rows * N, sizeof(double));

    int start = (N / size) * rank;

    if (start <= N / 2 && (start + local_rows) > N / 2) {
        C[(N/2 - start + 1) * N + N/2] = 1.0;
    }

    double start_time, end_time;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    start_time = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    diff_eq(C, C_new, start, start + local_rows - 2, rank, size, local_rows);

    gettimeofday(&tv, NULL);
    end_time = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    if (rank == 0) {
        printf("Tempo de cálculo da difusão: %f segundos\n", end_time - start_time);
        printf("Concentração final no centro: %f\n", C[(N/2 - start + 1) * N + N/2]);
    }

    free(C);
    free(C_new);
    MPI_Finalize();
    return 0;
}
