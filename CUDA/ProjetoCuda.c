#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 2000 // Tamanho da grade
#define T 500  // Numero de iteracoes no tempo
#define D 0.1  // Coeficiente de difusao
#define DELTA_T 0.01
#define DELTA_X 1.0

__global__ void diff_eq_kernel(double *d_in, double *d_out) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < N - 1 && j < N - 1) {
        int idx = i * N + j;
        int idx_up = (i - 1) * N + j;
        int idx_down = (i + 1) * N + j;
        int idx_left = i * N + (j - 1);
        int idx_right = i * N + (j + 1);

        d_out[idx] = d_in[idx] + D * DELTA_T *
                     ((d_in[idx_up] + d_in[idx_down] + d_in[idx_left] + d_in[idx_right] - 4.0 * d_in[idx]) / (DELTA_X * DELTA_X));
    }
}

void diff_eq(double *h_in, double *h_out) {
    double *d_in, *d_out;

    size_t size = N * N * sizeof(double);
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    for (int t = 0; t < T; t++) {
        diff_eq_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out);
        cudaDeviceSynchronize();

        // Alternar os ponteiros para a próxima iteração
        double *temp = d_in;
        d_in = d_out;
        d_out = temp;

        if ((t % 100) == 0 || t == T - 1) {
            cudaMemcpy(h_in, d_out, size, cudaMemcpyDeviceToHost);
            double difmedio = 0.0;
            for (int i = 1; i < N - 1; i++) {
                for (int j = 1; j < N - 1; j++) {
                    int idx = i * N + j;
                    difmedio += fabs(h_in[idx] - h_out[idx]);
                }
            }
            printf("Iteracao %d - diferenca=%g\n", t, difmedio / ((N - 2) * (N - 2)));
        }
    }

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    size_t size = N * N * sizeof(double);

    double *h_in = (double *)malloc(size);
    double *h_out = (double *)malloc(size);

    if (!h_in || !h_out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            h_in[idx] = 0.0;
            h_out[idx] = 0.0;
        }
    }

    // Inicializar uma concentracao alta no centro
    h_in[(N / 2) * N + (N / 2)] = 1.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Executar as iteracoes no tempo para a equacao de difusao
    diff_eq(h_in, h_out);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Tempo gasto: %f s\n", elapsedTime/1000);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_in);
    free(h_out);

    return 0;
}