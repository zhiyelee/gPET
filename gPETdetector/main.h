#ifndef __MAIN_H__
#define __MAIN_H__

#include <vector>
using namespace std;

#include "gObject.h"
// the lengths of the geometry in each dimension

#define unxVoxel 506
#define unyVoxel 506
#define unzVoxel 506
#define offSetx -12.65
#define offSety -12.65
#define offSetz -12.62
// two kinds of voxel size
#define voxelSize 0.05f
#define voxelSize2 2.0f



// package the memory 'malloc' function
void* util_malloc(int size);

// package the file 'fopen' function
FILE* util_fopen(char name[100], char* access);

// initialize an object 
struct object_t InitializeObject();

// read data from file to build up the object array
void read_file(struct object_t** objectArray, struct object_v** objectMaterial, int* total_Panels, char fname[100]);

// read data from rotational geometry file to build up the object array
void read_file_ro(struct object_t** objectArray, struct object_v** objectMaterial, int* total_Panels, char fname[100]);

// read data from file to build up the object size information
struct object_v loadObjectVoxel(char fname[100]);

// judge if a point with coordinates 'coords[3]' is in object 'p'
void getSign(int *sign, float coords[3], object_t p);

// score density into an array 'densityScore'
void score(float** densityScore, float voxelSizes[3], int nVoxel[3], float shifts[3], struct object_t* objectArray, int size, int total_Objects);

// write density to a file, in binary
void filize(float* densityScore, int size);
#endif