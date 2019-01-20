#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
using namespace std;

#include "gObject.h"// memory allocate
void* util_malloc(int size)
{
    void* ptr = malloc(size);

    if (ptr == NULL) // if failed, end the program
    {
        printf("Memory allocation error!\n");
        exit(EXIT_FAILURE);
    }

    // success, return
    return ptr;
}