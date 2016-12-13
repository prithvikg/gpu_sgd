#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

#define SAMPLES 50
#define DIMENSIONALITY 10
#define THREADS 5
#define ITERATIONS 50
#define MINI_BATCH 10

#define STEP_SIZE 0.05

//read a csv of input data X, a matrix and store it
//implement function to calculate the gradient of of w wrt one datapoint
//implement update step
//implement kernel

int loadCSV(char *fileName, float *matrix, int rows, int cols);

__global__
void sgd_kernel(float* X, float* y, float *w,
                unsigned long long int num_samples,
                unsigned long long int dimensionality,
                unsigned long long int H,
                int num_iterations);

__device__
void take_gradient_step(float *X, float *w, float *y,
                        unsigned long long int threadId,
                        unsigned long long int t,
                        unsigned long long int dimensionality,
                        unsigned long long int num_samples );

int main()
{
    unsigned long long int num_samples = SAMPLES;
    unsigned long long int dimensionality = DIMENSIONALITY;
    int num_threads = THREADS;
    int num_iterations = ITERATIONS;

    unsigned long long int rows, cols;
    rows = num_samples;
    cols = dimensionality;

    // Seed RNG
    srand(time(NULL));

    // Allocate the arrays
    float *mat = (float*)malloc(rows * cols * sizeof(float));
    float *maty = (float*)malloc(rows * sizeof(float));
    
    int result = loadCSV((char*)"./xmatrix.csv", mat, rows, cols);
    int result2 = loadCSV((char*)"./yvector.csv", maty, 1, num_samples);

    if (result < 0 || result2 < 0)
    {
        printf("Unable to load file\n");
        return result;
    }

    int nThread = num_threads;
    unsigned long long int H = num_samples / nThread; //This is the number of samples each thread is going to handle
    if (num_samples % nThread != 0)
        nThread++;

    printf("H is %llu\n", H);

    float* X;
    float* w;
    float* y;
    float *host_w = (float*)malloc(nThread * dimensionality * sizeof(float));

    for (unsigned long long int i = 0; i < nThread; i++) {
        for (unsigned long long int j = 0; j < dimensionality; j++) {
            host_w[index(i, j, dimensionality)] = 0;
        }
    }

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&X, rows * cols * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc of X failed!");
        return -1;
    }

    cudaStatus = cudaMalloc(&w, nThread * dimensionality * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc of w failed!");
        return -1;
    }

    cudaStatus = cudaMalloc(&y, rows * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc of y failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(X, mat, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy of X to device failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(
        w, host_w, nThread * dimensionality * sizeof(float), cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy of w to device failed!");
        return -1;
    }

    cudaStatus = cudaMemcpy(y, maty, rows * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy of w to device failed!");
        return -1;
    }

    int blockWidth = 128;
    int totalBlocks = nThread / blockWidth;
    if (nThread % blockWidth != 0)
        totalBlocks++;

    printf("totalBLock is %d, blockWidth is %d\n", totalBlocks, blockWidth);
    sgd_kernel << <totalBlocks, blockWidth >> > (X, y, w, rows, cols, H, num_iterations);

    cudaStatus = cudaMemcpy(
        host_w, w, nThread * dimensionality * sizeof(float), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy of w to host failed!");
        return -1;
    }

    printf("printing gpu result\n");

    for (unsigned long long int j = 0; j < dimensionality; j++)
    {
        for (int i = 0; i < nThread; i++)
        {
            printf("% f", host_w[index(i, j, dimensionality)]);
        }
        printf("\n");
    }
    printf("\n");

    //need to accumulate all the individual w vectors from each thread
    float *final_w = (float*)malloc(dimensionality * sizeof(float));

    for (int j = 0; j < dimensionality; j++)
    {
        float sum = 0;
        for (int i = 0; i < nThread; i++)
        {
            sum += host_w[index(i, j, dimensionality)];
        }
        final_w[j] = sum / nThread;
    }


    for (unsigned long long int j = 0; j < dimensionality; j++)
    {
        printf("%.6f ", final_w[j]);
    }
    printf("\n");

    free(mat);
    free(maty);
    free(host_w);
    free(final_w);

    return 0 ;

}

__global__
void sgd_kernel(float* X, float * y, float *w,
                unsigned long long int num_samples,
                unsigned long long int dimensionality,
                unsigned long long int H,
                int num_iterations)
{
    unsigned long long int s = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int startIndex = s * H;
    unsigned long long int endIndex = startIndex + H;

    /*if (s < 5){
        printf("%llu, %llu, %llu, %llu\n", num_samples, dimensionality, H, num_iterations);
        printf("thread id, %llu H is %llu, si, %llu ei %llu\n", s, H, startIndex, endIndex);
    }*/

    if (endIndex > num_samples)
        endIndex = num_samples;

    unsigned long long int guage = endIndex - startIndex;

    if (startIndex >= num_samples)
        return;

    //printf("thread id %llu, guage %llu\n", s, guage);

    for (int epoch = 0; epoch < num_iterations; epoch++)
    {
        for (int times = 0; times < guage; times++)
        {
            unsigned long long int t = startIndex + (times);
            //printf("thread id %llu, t %llu\n", s, t);
            //take a step with gradient on the t^{th} data point in X
            take_gradient_step(X, w, y, s, t, dimensionality, num_samples);
        }
    }
}

__device__
void take_gradient_step(float *X, float *w, float *y, 
                        unsigned long long int threadId,
                        unsigned long long int t,
                        unsigned long long int dimensionality,
                        unsigned long long int num_samples)
{
    float xwsum = 0;
    float term1[DIMENSIONALITY], term2[DIMENSIONALITY];
    float diff1, diff2;
    float temp = 0;
    int randomNode = (threadId == THREADS - 1) ? 0 : (threadId + 1);
    int delta;

    // Calculate X.W
    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        xwsum += X[index(t, i, dimensionality)] * w[index(threadId, i, dimensionality)];
    }
    xwsum -= y[t];
    
    // Build term1 and term 2
    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        term1[i] = w[index(threadId, i, dimensionality)] - 
                    STEP_SIZE * (xwsum * X[index(t, i, dimensionality)]);
    }

    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        term2[i] = w[index(randomNode, i, dimensionality)];
    }

    // Compute differences
    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        diff1 += term1[i] - term2[i];
    }

    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        diff2 += w[index(randomNode, i, dimensionality)] - w[index(threadId, i, dimensionality)];
    }

    // Compute delta
    diff1 = diff1 * diff1;
    diff2 = diff2 * diff2;
    delta = (diff1 < diff2) ? 1 : 0;

    // Compute [w^i - 1/2(w^i - w^j)] * delta
    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        temp += w[index(threadId, i, dimensionality)] - 
                w[index(threadId, randomNode, dimensionality)];
    }
    temp *= delta * 0.5;

    // Update weights
    for (unsigned long long int i = 0; i < dimensionality; i++)
    {
        w[index(threadId, i, dimensionality)] -= 
            STEP_SIZE * (temp + (xwsum * X[index(t, i, dimensionality)]));
    }
}

int loadCSV(char *fileName, float *matrix, int rows, int cols)
{
    char buffer[cols * 50];
    char *record, *line;
    int i = 0, j = 0;

    FILE *fstream = fopen(fileName, "r");
    if (fstream == NULL)
    {
        printf("\n file opening failed ");
        return -1 ;
    }
    while ((line = fgets(buffer, sizeof(buffer), fstream)) != NULL)
    {
        j = 0;
        record = strtok(line, ",");
        while (record != NULL)
        {
            matrix[index(i, j, cols)] = atof(record) ;
            j++;
            record = strtok(NULL, ",");
        }
        ++i ;
    }
    return 0;
}
