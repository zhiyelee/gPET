#ifndef __GOBJECT_H__
#define __GOBJECT_H__

#include <vector>
using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

#define M 100+1 // length of the largest name in character
// typedef unsigned int size_t;
// define the object structure
typedef struct object_t
{ 
// panel index
	int panel;
// panel dimension
	float lengthx, lengthy, lengthz;	
// module size
	float MODx, MODy, MODz;
// module space size
	float Mspacex, Mspacey, Mspacez;
// LSO size
	float LSOx, LSOy, LSOz;
// space size
	float spacex, spacey, spacez;
// offset (top surface center, local coordinate origin) of each module
	float offsetx, offsety, offsetz;
// module local direction
	float directionx, directiony, directionz;
// unit vector along x direction of each module
	float UniXx, UniXy, UniXz;
// unit vector along y direction of each module
	float UniYx, UniYy, UniYz;
// unit vector along z direction of each module
	float UniZx, UniZy, UniZz;
} OBJECT; // name of the structure

typedef struct object_v
{ // material index
	int material;
// material density
	float density;
}OBJECT_V;

typedef struct
{   
  vector<float4> xbuffer, vxbuffer; // xbuffer: x,y,z,T;
                                                // vxbuffer: vx,vy,vz,E;
  int NSource;
}Source;

typedef struct
{     
  vector<float4> xevent, vxevent; // xbuffer: x,y,z,T;
                                                // vxbuffer: module#,crystal#,photon#,E;
}PETevent;



Source ReadSource(char fname[100],int NSource);
void getnam2(int iounit,char physics[80], int *n);
#endif
