#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_BINS 256
#define MAX_VAL 255
#define MIN_VAL 0

// Cuda Error Kontrolü
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Zaman hesaplaması için anlık zamanı alma
double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Shared memory kullanarak GPU ile histogram hesaplama
__global__ void histogramKernel(int *input, int *bins, int size) {
    __shared__ int temp[NUM_BINS];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    // Shared memory oluşturma
    if (threadIdx.x < NUM_BINS) {
        temp[threadIdx.x] = 0;
    }
    __syncthreads();

    // Histogram Hesaplama
    while (i < size) {
        atomicAdd(&temp[input[i]], 1);
        i += offset;
    }
    __syncthreads();

    // Global memory'e yazmak
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&(bins[threadIdx.x]), temp[threadIdx.x]);
    }
}

// CPU'ya histogram hesaplatma
void histogramCPU(int *input, int *bins, int size) {
    for (int i = 0; i < NUM_BINS; ++i) {
        bins[i] = 0;
    }
    for (int i = 0; i < size; ++i) {
        bins[input[i]]++;
    }
}

// Histogramların ilk 10 elemanını ekrana yazdırmak
void printHistogram(int *bins, const char* processor) {
    printf("First 10 elements of the %s histogram:\n", processor);
    for (int i = 0; i < 10; ++i) {
        printf("%d ", bins[i]);
    }
    printf("\n");
}

// Farklı büyüklüklerde histogram oluşturma ve oluşma zamanlarını hesaplama
void runHistogram(int size) {
    int *input, *bins, *dev_input, *dev_bins;
    double start, stop;

    // Allocate memory
    input = (int *)malloc(size * sizeof(int));
    bins = (int *)malloc(NUM_BINS * sizeof(int));
    cudaMalloc((void **)&dev_input, size * sizeof(int));
    cudaMalloc((void **)&dev_bins, NUM_BINS * sizeof(int));

    // input oluşturma
    for (int i = 0; i < size; ++i) {
        input[i] = rand() % (MAX_VAL + 1);
    }

    // oluşturulan inputun aktarılması
    cudaMemcpy(dev_input, input, size * sizeof(int), cudaMemcpyHostToDevice);

    // GPU da histogram oluşturma
    start = seconds();
    histogramKernel<<<(size + 255)/256, 256>>>(dev_input, dev_bins, size);
    cudaDeviceSynchronize();
    stop = seconds();
    printf("Size: %d, GPU Time: %.6f seconds\n", size, stop - start);

    // Sonuçların GPU'dan çağrılması
    cudaMemcpy(bins, dev_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    printHistogram(bins, "GPU");

    // CPU'ya histogram oluşturtma
    start = seconds();
    histogramCPU(input, bins, size);
    stop = seconds();
    printf("Size: %d, CPU Time: %.6f seconds\n", size, stop - start);
    printHistogram(bins, "CPU");

    // ayrılan memorylerin vs temizlenerek lock'tan çıkartılması
    free(input);
    free(bins);
    cudaFree(dev_input);
    cudaFree(dev_bins);
}

int main() {
    // Farklı input değerleri kullanacağız
    int sizes[] = {262144, 524288, 1048576, 2097152}; // Büyüklükler: 256kb, 512kb, 1mb ve 2mb
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        runHistogram(sizes[i]);
    }

    return 0;
}
