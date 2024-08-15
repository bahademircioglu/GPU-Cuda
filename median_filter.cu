#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 512  // Input'un size'ı
#define FILTER_SIZE 3
#define BLOCK_SIZE 16

// 3x3 medyan filtresinin uygulanması için CUDA işlemleri
__global__ void medianFilterKernel(int *input, int *output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= width) return;

    int filter[FILTER_SIZE * FILTER_SIZE];
    int filterIndex = 0;

    // Komşuları toplama
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int nx = x + i;
            int ny = y + j;

            // Sınır kontrolü
            if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
                filter[filterIndex++] = input[nx + ny * width];
            } else {
                filter[filterIndex++] = 0;
            }
        }
    }

    // Medyanı bulma için filtre dizisinin sıralanması
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
        for (int j = i + 1; j < FILTER_SIZE * FILTER_SIZE; j++) {
            if (filter[i] > filter[j]) {
                int temp = filter[i];
                filter[i] = filter[j];
                filter[j] = temp;
            }
        }
    }

    // Medyan outputu
    output[x + y * width] = filter[FILTER_SIZE * FILTER_SIZE / 2];
}

// Hostta 3x3 medyan filtre uygulama fonksiyonu
void medianFilterHost(int *input, int *output, int width) {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            int filter[FILTER_SIZE * FILTER_SIZE];
            int filterIndex = 0;

            // Komşuları toplama
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int nx = x + i;
                    int ny = y + j;

                    // Sınır kontrolü
                    if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
                        filter[filterIndex++] = input[nx + ny * width];
                    } else {
                        filter[filterIndex++] = 0;
                    }
                }
            }

            // Medyanı bulma için filtre dizisi sıralanması
            for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) {
                for (int j = i + 1; j < FILTER_SIZE * FILTER_SIZE; j++) {
                    if (filter[i] > filter[j]) {
                        int temp = filter[i];
                        filter[i] = filter[j];
                        filter[j] = temp;
                    }
                }
            }

            // Medyan outputu
            output[x + y * width] = filter[FILTER_SIZE * FILTER_SIZE / 2];
        }
    }
}

// Arrayleri kıyaslama
bool compareArrays(int *a, int *b, int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// Execution zamanlarının kıyaslanması
float measureTime(void (*function)(int*, int*, int), int *input, int *output, int width) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    function(input, output, width);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    int *input, *output_gpu, *output_cpu;
    int *d_input, *d_output;
    int size = N * N * sizeof(int);

    // Host arrayleri için hafıza ayrımı
    input = (int *)malloc(size);
    output_gpu = (int *)malloc(size);
    output_cpu = (int *)malloc(size);

    // Device arrayleri için hafıza ayrımı
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Random değerlerle input hazırlama
    for (int i = 0; i < N * N; i++) {
        input[i] = rand() % 256;  // 0-255 arası random değerler
    }

    // Device'a input verilerini aktarma
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Device için execution zamanı hesaplama
    float device_time = measureTime([](int *in, int *out, int w) {
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        medianFilterKernel<<<dimGrid, dimBlock>>>(in, out, w);
        cudaDeviceSynchronize();  // Senkrozine ile bitişten emin olma
    }, d_input, d_output, N);
    printf("Device execution time: %f ms\n", device_time);

    // Sonuçları Host'a aktarma
    cudaMemcpy(output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    // Host için execution zamanını hesaplama
    float host_time = measureTime(medianFilterHost, input, output_cpu, N);
    printf("Host execution time: %f ms\n", host_time);

    // Medyan filtre çalıştırma (host)
    medianFilterHost(input, output_cpu, N);

    // Sonuçları kıyaslama
    if (compareArrays(output_gpu, output_cpu, N * N)) {
        printf("Results are identical!\n");
    } else {
        printf("Results are different!\n");
    }

    // Hafızayı serbest bırakma
    free(input);
    free(output_gpu);
    free(output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
