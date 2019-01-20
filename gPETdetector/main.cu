// includes, system

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <omp.h>
#include<cstring>
using namespace std;

// includes, project
#include "cutil_math.h"
#include <cuda.h>
#include <cublas.h>
#include <curand_kernel.h>

// includes
#include "cuPrintf.cu"
#include "gPETInternal.h"
#include "gObject.h"
#include "main.h"
#include "iniDevice.cu"
#include "iniPanel.cu"
#include "iniPhysics.cu"
#include "iniSource.cu"
#include "gPET.cu"
#include "finalize.cu"
//
#include "libphoton.cu"
#include "libpenmath.cu"

#include "buildObjectArray.cu"
#include "fopen.cu"
#include "memoryAllocate.cu"
void itoa(int n, char s[]);
void reverse(char s[]);
int strlen(char s[]);
/****************************************************
        main program
****************************************************/
int main( )
{
    cout << endl << "****************************************" << endl;
    cout << "Computation parameters..." << endl;
    cout << "****************************************" << endl ;
    // clock setting
    clock_t start_time, end_time1, end_time2, end_time3;
    float time_diff1, time_diff2, time_diff3;

    start_time = clock();
    char fname0[100], fname2[100], fname3[100];

// geometry file
    readinput(fname0);

// output file
    readinput(fname3);

// source PSF file
    readinput(fname2);

// photon history
    int NSource;
    readinput(&NSource);

// PSF repeat times
    int NRepeat;
    readinput(&NRepeat);

// device #
    int deviceNo;
    readinput(&deviceNo);

// initialize device
    iniDevice(deviceNo);
    printf("\n");

// load the object structure
    struct object_t* panelArray;
    struct object_v* panelMaterial;
    int total_Panels=0;
    read_file_ro(&panelArray,&panelMaterial,&total_Panels,fname0);

// load PSF file
    Source source;
    source = ReadSource(fname2, NSource, NRepeat);

    end_time1 = clock();
    time_diff1 = ((float)end_time1 - (float)start_time)/CLOCKS_PER_SEC;

// initialize panel geometry
    iniPanel(panelArray,panelMaterial,total_Panels);
// initialize physics
    iniPhysics(panelMaterial);

    end_time2 = clock();
    time_diff2 = ((float)end_time2 - (float)end_time1)/CLOCKS_PER_SEC;

// transport photon and record
    runCalculation(source, fname3);

// finalize
    fina();

    end_time3 = clock();
    time_diff3 = ((float)end_time3 - (float)start_time)/CLOCKS_PER_SEC;
    printf("\n\n****************************************\n");
    printf("total time: %f s.\n\n",time_diff3);
    printf("initialize time: %f s.\n\n",time_diff2);
    printf("loading parameters: %f s.\n\n",time_diff1);
    printf("****************************************\n\n\n");//*/
}
