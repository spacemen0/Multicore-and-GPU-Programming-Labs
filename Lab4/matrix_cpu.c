#include <stdio.h>
#include <stdlib.h>
#include <milli.h>

void add_matrix(float *a, float *b, float *c, int N)
{
    int index;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            index = i + j * N;
            c[index] = a[index] + b[index];
        }
}

int main()
{
    const int N = 1024;

    // Allocate memory for matrices
    float *a = (float *)malloc(N * N * sizeof(float));
    float *b = (float *)malloc(N * N * sizeof(float));
    float *c = (float *)malloc(N * N * sizeof(float));

    // Check if memory allocation is successful
    if (a == NULL || b == NULL || c == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            a[i + j * N] = 10 + i;
            b[i + j * N] = (float)j / N;
        }

    double start = GetSeconds();
    add_matrix(a, b, c, N);
    double end = GetSeconds();

    printf("Execution Time: %0.6lf seconds\n", (end - start));

    // Uncomment the following code if you want to print the result matrix
    
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%0.2f ", c[i + j * N]);
        }
        printf("\n");
    }
    

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}
