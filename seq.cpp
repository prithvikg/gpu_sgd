#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <thread>
#include <vector>

/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

#define SAMPLES 50
#define DIMENSIONALITY 10
#define THREADS 5
#define ITERATIONS 50

int loadCSV(char *fileName, float *matrix, int rows, int cols);

void sgd_kernel(float* X, float * y, float *w,
                unsigned long long int num_samples,
                unsigned long long int dimensionality,
                unsigned long long int H,
                int num_iterations,
                int i, int j, int col_size);

void take_gradient_step(float *X, float *w, float *y,
                        unsigned long long int threadId,
                        unsigned long long int t,
                        unsigned long long int dimensionality,
                        unsigned long long int num_samples );

using namespace std;

int main()
{
    unsigned long long int num_samples = SAMPLES;
    unsigned long long int dimensionality = DIMENSIONALITY;
    int num_threads = THREADS;
    int num_iterations = ITERATIONS;

    unsigned long long int rows, cols;
    rows = num_samples;
    cols = dimensionality;

    //allocate the array
    // float *mat = (float*)malloc(rows * cols * sizeof(float));
    // float *maty = (float*)malloc(rows * sizeof(float));

    int nThread = num_threads;
    unsigned long long int H = num_samples / nThread; //This is the number of samples each thread is going to handle
    if (num_samples % nThread != 0)
        nThread++;

    printf("H is %llu\n", H);

    float* X;
    float* w;
    float* y;

    X = (float*) calloc(rows * cols, sizeof(float));
    w = (float*) calloc(nThread * dimensionality, sizeof(float));
    y = (float*) calloc(rows, sizeof(float));

    int result = loadCSV((char*)"./xmatrix.csv", X, rows, cols);
    int result2 = loadCSV((char*)"./yvector.csv", y, 1, num_samples);

    if (result < 0 || result2 < 0)
    {
        printf("Unable to load file\n");
        return result;
    }

    int blockWidth = 128;
    int totalBlocks = nThread / blockWidth;
    if (nThread % blockWidth != 0)
        totalBlocks++;

    printf("totalBLock is %d, blockWidth is %d\n", totalBlocks, blockWidth);

    vector<thread*> pthreads;

    for (int i = 0; i < totalBlocks; i++) {

        for (int j = 0; j < blockWidth; j++) {

            thread* pthread = new thread(
                sgd_kernel, 
                X, y, w, rows, cols, H, num_iterations, i, j, blockWidth);

            pthreads.push_back(pthread);
        }
    }

    // Wait for all to complete
    for (int i = 0; i < THREADS; i++) {
        pthreads[i]->join();
    }

    printf("printing gpu result\n");

    for (unsigned long long int j = 0; j < dimensionality; j++)
    {
        for (int i = 0; i < nThread; i++)
        {
            printf("% f", w[index(i, j, dimensionality)]);
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
            sum += w[index(i, j, dimensionality)];
        }
        final_w[j] = sum / nThread;
    }


    for (unsigned long long int j = 0; j < dimensionality; j++)
    {
        printf("%.6f ", final_w[j]);
    }
    printf("\n");

    // free(mat);
    // free(maty);
    free(X);
    free(y);
    free(w);
    free(final_w);

    return 0 ;

}

void sgd_kernel(float* X, float * y, float *w,
                unsigned long long int num_samples,
                unsigned long long int dimensionality,
                unsigned long long int H,
                int num_iterations,
                int i, int j, int col_size)
{
    unsigned long long int s = i * col_size + j;
    unsigned long long int startIndex = s * H;
    unsigned long long int endIndex = startIndex + H;

    // printf("In %d, %d = %d\n", i, j, s);

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

void take_gradient_step(float *X, float *w, float *y, 
                        unsigned long long int threadId,
                        unsigned long long int t,
                        unsigned long long int dimensionality,
                        unsigned long long int num_samples)
{
    float xwsum = 0;
    float stepSize = 0.05;
    for (unsigned long long int j = 0; j < dimensionality; j++)
    {
        printf("thread id %llu, f %f, s %f\n", 
            threadId, X[index(t, j, dimensionality)], w[index(threadId, j, dimensionality)]);

        xwsum += X[index(t, j, dimensionality)] * w[index(threadId, j, dimensionality)];
    }
    xwsum -= y[t];
    
    // printf("thread id %llu, t %llu, xwsum %f\n", threadId, t, xwsum);

    for (unsigned long long int j = 0; j < dimensionality; j++)
    {
        w[index(threadId, j, dimensionality)] -= 
            stepSize * (xwsum * X[index(t, j, dimensionality)]);
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
