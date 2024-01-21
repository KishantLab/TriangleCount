#include <stdio.h>
#include <stdlib.h>

#define ONE_GB 1073741824ULL  // Use unsigned long long int for GB

int main() {
    unsigned long long int array_size;

    printf("Enter the size of the array: ");
    scanf("%llu", &array_size);  // Read input as unsigned long long int

    // Dynamically allocate memory for the array
    unsigned long long int *my_array = (unsigned long long int *)malloc(array_size * sizeof(unsigned long long int));

    // Check for allocation failure
    if (my_array == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Calculate and print array size in GB (using unsigned long long int)
    unsigned long long int array_size_bytes = sizeof(my_array[0]) * array_size;
    double array_size_gb = (double)array_size_bytes / ONE_GB;
    printf("Total array size in memory: %.2f GB\n", array_size_gb);

    // Free the allocated memory
    free(my_array);

    return 0;
}

