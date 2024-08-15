#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// vektor ekleme
__global__ void vecAdd(int *A, int *B, int *C, int *D, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        D[i] = A[i] + B[i] + C[i];
    }
}

// rastgele degerlerle vektor olusturma 
void randomInit(int *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % 100; // 0-99 arasi rastgele sayilar
    }
}

// Hostta vektor ekleme, host = cpu islemi
void vecAddHost(int *A, int *B, int *C, int *D, int N) {
    for (int i = 0; i < N; i++) {
        D[i] = A[i] + B[i] + C[i];
    }
}

//hizlari ekrana yazdirma
void printSpeedComparison(float prevTime, float currentTime, int prevN, int currentN, const char* processor) {
    float speedChange = currentTime / prevTime;
    printf("Speed comparison of elements by %s (%d - %d elements): %f\n", processor, prevN, currentN, speedChange);
}

int main(void) {
    int sizes[] = {1 << 10, 1 << 12, 1 << 14, 1 << 16};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    float prevGpuTime = 0.0, prevCpuTime = 0.0;
    int prevN = 0;
	
	for (int i = 0; i < numSizes; i++) {
        int N = sizes[i];
        size_t size = N * sizeof(int);

        int *h_A, *h_B, *h_C, *h_D, *h_D_host;
        int *d_A, *d_B, *d_C, *d_D;

        // Bellek ayirimi
        h_A = (int *)malloc(size);
        h_B = (int *)malloc(size);
        h_C = (int *)malloc(size);
        h_D = (int *)malloc(size);
        h_D_host = (int *)malloc(size);

        // Hostta vektorleri olusturma
        srand(time(NULL));
        randomInit(h_A, N);
        randomInit(h_B, N);
        randomInit(h_C, N);

        // Cuda bellek ayirimi
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        cudaMalloc(&d_D, size);

        // hosttan gpuya verilerin aktarimi
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

        // gpu timer baslangici
        float gpuTime;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // gpu vektor eklenimi
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);

        // timer sonu
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, start, stop);
        printf("Time for vector addition on GPU with %d elements: %f ms\n", N, gpuTime);

        // hosta sonucun aktarimi
        cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

        // gpu memorylerin serbest birakimi
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);

        // host icin timer baslangici
        clock_t start_host = clock();

        // hostta vektor eklenmesi
        vecAddHost(h_A, h_B, h_C, h_D_host, N);

        // timer sonu 
        clock_t stop_host = clock();
        double cpuTime = ((double)(stop_host - start_host)) / CLOCKS_PER_SEC * 1000;
        printf("Time for vector addition on CPU with %d elements: %f ms\n", N, cpuTime);

        // hiz kiyaslamasi
        double speedup = cpuTime / gpuTime;
        printf("Speed-up (CPU time / GPU time): %f\n", speedup);
		
		// hiz kiyaslamasini yazdirma
        if (i > 0) {
            printSpeedComparison(prevGpuTime, gpuTime, prevN, N, "GPU");
            printSpeedComparison(prevCpuTime, cpuTime, prevN, N, "CPU");
        }

        prevGpuTime = gpuTime;
        prevCpuTime = cpuTime;
        prevN = N;

        // memory serbest birakimi
						   
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_D);
        free(h_D_host);
    }

    return 0;
}
