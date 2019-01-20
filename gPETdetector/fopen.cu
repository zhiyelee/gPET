#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
using namespace std;

#include "gObject.h"// package fopen 
FILE* util_fopen(char name[100], char* access)
{
    FILE* fp = fopen(name, access);

    if (fp == NULL) // if open failed, terminate
    {
        printf("Error opening file %s!\n", name);
        exit(EXIT_FAILURE);
    }

    // if success, return;
    return  fp;
}
