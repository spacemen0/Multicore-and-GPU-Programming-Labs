#include <stdio.h>

void exchange(unsigned int *a, unsigned int *b)
{
    unsigned int temp = *a;
    *a = *b;
    *b = temp;
}

void printArray(unsigned int *data, int N)
{
    printf("Array: ");
    for (int i = 0; i < N; i++)
    {
        printf("%u ", data[i]);
    }
    printf("\n");
}

void bitonic_cpu(unsigned int *data, int N)
{
    unsigned int i, j, k;

    printf("CPU sorting.\n");

    for (k = 2; k <= N; k = 2 * k) // Outer loop, double size for each step
    {
        printf("\nOuter loop: k = %u\n", k);

        for (j = k >> 1; j > 0; j = j >> 1) // Inner loop, half size for each step
        {
            printf("  Inner loop: j = %u\n", j);

            for (i = 0; i < N; i++) // Loop over data
            {
                int ixj = i ^ j; // Calculate indexing!
                if ((ixj) > i)
                {
                    printf("    i=%u, ixj=%u\n", i, ixj);

                    if ((i & k) == 0 && data[i] > data[ixj])
                    {
                        exchange(&data[i], &data[ixj]);
                        printf("      Swap: data[%u] = %u, data[%u] = %u\n", i, data[i], ixj, data[ixj]);
                        printArray(data, N);
                    }

                    if ((i & k) != 0 && data[i] < data[ixj])
                    {
                        exchange(&data[i], &data[ixj]);
                        printf("      Swap: data[%u] = %u, data[%u] = %u\n", i, data[i], ixj, data[ixj]);
                        printArray(data, N);
                    }
                }
            }
        }
    }
}

int main()
{
    unsigned int data[] = {4, 2, 9, 1, 5, 6, 8, 7, 3, 6, 7, 8, 9, 0, 1, 2};
    int N = sizeof(data) / sizeof(data[0]);

    printf("Original array:\n");
    printArray(data, N);

    bitonic_cpu(data, N);

    printf("\nSorted array:\n");
    printArray(data, N);

    return 0;
}
