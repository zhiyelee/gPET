#ifndef __GBRACHY_H__
#define __GBRACHY_H__

#include <vector>

typedef struct
{   
   vector<vector<float4> > xbuffer, vxbuffer;	// xbuffer: x,y,z,w;
                                             // vxbuffer: vx,vy,vz,E;
   int NBuffer; // number of energy bin
	int NParticle;

}Particle;

typedef struct
{   
   int   *mat;
   float *dens;
   int   Unxvox, Unyvox, Unzvox;
   float dx, dy, dz;
   float Offsetx, Offsety, Offsetz;
   float Sizex, Sizey, Sizez;

}Patient;

typedef struct
{   
   
   float *doseAve, *doseStd;
   float  totalParticleWeight;

}PatientDose;

typedef struct
{   
   float *XSource;
   float *YSource;
   float *ZSource;
   float *TotalWeight;
   float *C10Weight;
   float *C11Weight;
   float *N13Weight; 
   float *O15Weight;
   int numSource;
}BeamData;


Patient loadPatient(char fname[100],char fname2[100]);
Particle SampleParticle();
Particle ReadParticle(char fname[100],int NParticle,float user_esrcmax, float user_esrcmin, Patient patient);
BeamData loadBeamData(char fname[100]);

#endif


