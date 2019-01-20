#ifndef __DIGITIZER_H__
#define __DIGITIZER_H__
#define NCRYSTAL 100*1024
#include <math.h>
#include <algorithm>
#include <string.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__device__ int pannelN=1, moduleN=100,crystalN=900;

typedef struct Event
{
    int parn,pann,modn,cryn,siten;//siten: the index in current depth
    float t, E, x,y,z;
} Event;
struct compare_parn
{
    __host__ __device__ bool operator()(Event a, Event b)
    {
        return a.parn < b.parn;
    }
};
struct compare_siten
{
    __host__ __device__ bool operator()(Event a, Event b)
    {
        return a.siten < b.siten;
    }
};
struct compare_t
{
    __host__ __device__ bool operator()(Event a, Event b)
    {
        return a.t < b.t;
    }
};

typedef struct Coincidence
{
    Event a;
    Event b;
} Coincidence;

#endif