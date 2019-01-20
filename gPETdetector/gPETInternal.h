#ifndef __GPETINTERNAL_H__
#define __GPETINTERNAL_H__

#include <vector>
#include "main.h"
#include "digitizer.h"

void __global__ energywindow(int* counts, Event* events, int total, float thresholder, float upholder);
void quicksort(Event* events, int start, int stop, int sorttype);
void quicksort_d(Event* events_d, int start, int stop, int sorttype);
void orderevents(int* counts,Event* events_d);
void __global__ coinsorter(int* counts,Event* events,float interval, int secdiffer, int total);
void __global__  deadtime(int* counts,Event* events, int total, float interval, int deadtype);
void __global__ addnoise(int* counts, Event* events_d, float f, float Emean, float res, float interval);
int __device__ addevents(int*ntotal_d, Event* totalevents_d, int* counts_d, Event* events_d);
int __device__ adder(int* counts_d, Event* events_d, Event event);
int __device__ readout(int* counts_d, Event* events_d,int depth, int policy);
int outevents(int* num_d, Event* totalevents_d, const char *outputfilename);
//==========================================================
//      GPU configurations
//==========================================================

#define NTHREAD_PER_BLOCK_GPET 256//32
#define NTHREAD_PER_BLOCK_ELECTRON 64  
//      number of threads per block

#define NBLOCKX 32768
//      the leading dimension of the 2d thread grid

//==========================================================
//      global variables
//==========================================================

//	common variable group /source photon/
__device__ __constant__ int ifEnergyBlur_PET;

float4 *vx_phap_gPET;
float4 *x_phap_gPET;


// common variable group for panel geometry
__device__ __constant__ int dev_totalPanels;
float *dens_panel;
int *mat_panel;
int *panelID;
float *lengthx_panel, *lengthy_panel, *lengthz_panel;
float *MODx_panel, *MODy_panel, *MODz_panel;
float *Mspacex_panel, *Mspacey_panel, *Mspacez_panel;
float *LSOx_panel, *LSOy_panel, *LSOz_panel;
float *spacex_panel, *spacey_panel, *spacez_panel;
float *offsetx_panel, *offsety_panel, *offsetz_panel;
float *directionx_panel, *directiony_panel, *directionz_panel;
float *UniXx_panel, *UniXy_panel, *UniXz_panel;
float *UniYx_panel, *UniYy_panel, *UniYz_panel;
float *UniZx_panel, *UniZy_panel, *UniZz_panel;

//==========================================================
//      global variables for physics
//==========================================================
#define NPART 500000 
//	common variable group /rseed/
int iseed1_h[NPART];
__device__ int iseed1[NPART];
__device__ curandState cuseed[NPART];

// cutoff energy
__device__ __constant__ float eabsph;
float eabsph_h;

// material 
#define MAXMAT 4
__device__ __constant__ int nmat;
int nmat_h;

// total cross section
#define NLAPH 2048
__device__ __constant__ float idleph;
__device__ __constant__ float elaph0;
float idleph_h;
float elaph_h[NLAPH],lamph_h[NLAPH*MAXMAT],lampha_h[NLAPH*MAXMAT],lamphb_h[NLAPH*MAXMAT],
	lamphc_h[NLAPH*MAXMAT],lamphd_h[NLAPH*MAXMAT];
//	lampha~d_h is necessary, they are used on CPU side for initialize Woodkock method
cudaArray *lamph;
texture<float,1,cudaReadModeElementType> lamph_tex;

// compton scattering 
#define NCMPT 2048

__device__ __constant__ float idlecp;
__device__ __constant__ float ecmpt0;
float idlecp_h;
float ecmpt_h[NCMPT],compt_h[NCMPT*MAXMAT];
cudaArray *compt;
texture<float,1,cudaReadModeElementType> compt_tex;

#define NCPCM 1001
#define NECM 51
float idcpcm_h, idecm_h;
__device__ __constant__ float idcpcm,idecm;
float mucmpt_h[NCPCM*NECM*MAXMAT];
cudaArray* sArray;
texture<float, 3, cudaReadModeElementType> s_tex;


//	common variable group /dpmpte/
#define NPHTE 2048

__device__ __constant__ float idlepe;
__device__ __constant__ float ephte0;
float idlepe_h;
float ephte_h[NPHTE],phote_h[NPHTE*MAXMAT];
cudaArray *phote;
texture<float,1,cudaReadModeElementType> phote_tex;
  

//      common variable group /dpmray/
#define NRAYL 2048

__device__ __constant__ float idlerl;
__device__ __constant__ float erayl0;
float idlerl_h;
float erayl_h[NRAYL],rayle_h[NRAYL*MAXMAT];
cudaArray *rayle;	//	cross section data
texture<float,1,cudaReadModeElementType> rayle_tex;

#define NCPRL 1001
#define NERL 51
float idcprl_h, iderl_h;
__device__ __constant__ float idcprl,iderl;
float murayl_h[NCPRL*NERL*MAXMAT];
cudaArray* fArray;
texture<float, 3, cudaReadModeElementType> f_tex;



//	common variable group /dpmwck/
#define NWCK 2048
__device__ __constant__ float idlewk;
__device__ __constant__ float wcke0;
float idlewk_h, wcke0_h;

float woock_h[NWCK];
cudaArray *woock;
texture<float,1,cudaReadModeElementType> woock_tex;



//==========================================================
//      global variables for particle transport
//==========================================================
// common variable for loading source PSF

int nactive_h, ptype_h, nparload_h;
__device__ int nactive, ptype;

__device__ float4 vx_gPET[NPART];
__device__ float4 x_gPET[NPART];

float4 xbuffer[NPART];
float4 vxbuffer[NPART];

float totalWeight_gPET = 0.0f;
long long totalSimPar = 0;

int bufferHeadId;


#define NSSTACK 15*NPART
#define NSSTACKSHARED (15*NTHREAD_PER_BLOCK_GPET)
__device__ int nsstk;
__device__ float sf[NSSTACK];
__device__ int sid[NSSTACK];


//==========================================================
//      physical and mathematical constants
//==========================================================
#define PI 3.1415926535897932384626433f
#define TWOPI 6.2831853071795864769252867f

#define MC2 510.9991e3
#define IMC2 1.95695060911e-6
#define TWOMC2 1021.9982e3
#define MC2SQ2 361.330928790e3

#define KCMAX 0.4999999999

#define RE2 7.940791479481e-26
#define NAVO 6.022137e23
#define CFAC 153537.4865001838
#define ZERO 1.0e-20
#define SZERO 1.0e-4
#define USCALE 4.656612873077392578125e-10
#define INF 1.0e20
#define REV 5.1099906e5

#define EPS 1.0e-8
#define ZSRC -90.0F

#define C1 15.0e-27
#define C2 0.990
#define IC2 1.010101010101010

//==========================================================
//      declare penmath functions
//==========================================================
void spline(float *X, float *Y, float *A, float *B, float *C, float *D, float S1, float SN, int N);
//	cubic spline function
void inirngG();
//	generate rand seed for all threads
__global__ void setupcuseed();
//      setup random seeds

//==========================================================
//	declare iniDevice functions
//==========================================================
void iniDevice(int deviceNo);
void printDevProp(int device);

//==========================================================
//	declare inipanel functions
//==========================================================
void iniPanel(struct object_t* objectArray, int totalOb);
void iniSource(Source source);

//==========================================================
//	declare iniPhysics functions
//==========================================================
void iniPhysics(struct object_t* objectArray);

void getnam(FILE *fp,int iounit,char physics[80], int *n);
void getna2(FILE *iounit,char physics[80], int *n);
void readinput(char* ptr);
void readinput(int* ptr);
void readinput(float* ptr);
// read material info
void rmater(char fname[50], float *eminph, float *emax);
//	read photon total inverse mean free path data info
void rlamph(char fname[50]);
//	read compton inverse mean free path data info
void rcompt(char fname[50]);
//	read compton scattering function data info
void rcmpsf(char fname[50]);
//	read photoelectric inverse mean free path data info
void rphote(char fname[50]);
//	read rayleigh inverse mean free path data info
void rrayle(char fname[50]);
//	read Rayleigh scattering form factor info
void rrayff(char fname[50]);
//	initialize woodcock
void iniwck(float eminph,float emax, struct object_v* objectMaterial);
//	interpolation of phi
float itphip(int matid, float e);

//==========================================================
//      declare/define other functions
//==========================================================
__host__ __device__ int ind2To1(int i, int j, int nx, int ny)
//      convert a voxel real indices into a single index
{
//      different ordering
        return i*ny+j;
//      return j*nx+i;
}

//==========================================================
//      declare particle transport functions
//==========================================================
void runCalculation(Source source, char fname[100]);
void simulateParticles(Source source, char fname[100]);
bool loadFromPSfile(Source source, int first, int last);



//==========================================================
//      declare photon transport functions
//==========================================================
void clrStk();
// photon transport
extern "C"
__global__ void photon(Event* events, int* counts,const int nactive, const int bufferID, float* dens, int *mat, int *panelID, float *lenx, float *leny, float *lenz,
	                   float *MODx, float *MODy, float *MODz, float *Msx, float *Msy, float *Msz, 
	                   float *LSOx, float *LSOy, float *LSOz, float *sx, float *sy, float *sz, 
	                   float *ox, float *oy, float *oz, float *dx, float *dy, float *dz, float *UXx, float *UXy, float *UXz, 
	                   float *UYx, float *UYy, float *UYz,float *UZx, float *UZy, float *UZz);

// search LSo index and material type
__device__ void LSOsearch(float4 xtemp2,float leny_S,float lenz_S,
	float MODy_S,float MODz_S,float Msy_S,float Msz_S,float LSOy_S,float LSOz_S,
	float sy_S,float sz_S,float dy_S, float dz_S, int *m_id, int *M_id, int *L_id);
// 1D guassian sampling
__device__ float box_muller1(float m, curandState *localState_pt);
// rayleigh scattering
__device__ void rylsam(float energytemp, int matid, curandState *localState_pt, float *costhe);
// compton scattering
//__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe, int matid);
__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe);
__device__ void rotate(float *u, float *v, float *w, float costh, float phi);
//==========================================================
//      declare final functions
//==========================================================
void outputData(const char *srcname, const int size, const char *outputfilename, const char *mode);
void outputData(void *src, const int size, const char *outputfilename, const char *mode);
void fina();

/*
void printDevProp(int device);
//      print out device properties
void runCalculation(Particle particle, BeamData beamdata, int NRepeat); //use phase space file 
//      run entire DPM calculation
void runCalculation(long long NParticle); // use source model
PatientDose getDose();

//==========================================================
//	declare initialization/finalization functions
//==========================================================

void init(struct object_t* objectArray, struct object_v objectVoxel, BeamData sourceinfo, int ifDoseToWater_h, int deviceNo, int totalOb);
//	initialize DPM
void initObjectArray(struct object_t* objectArray, int totalOb);
void initObjectVoxel(struct object_v objectVoxel);
void initBodyindex(struct object_t* objectArray, struct object_v objectVoxel);
int getID(float coords[3], struct object_t* objectArray);
void getSign(int *sign, float coords[3], object_t p);
void getnam(FILE *fp,int iounit,char physics[80], int *n);
void getna2(FILE *iounit, char physics[80], int *n);
//	get names, used in initialization
void rmater(char fname[50], float *eminph, float *emax);
//	read material info
void rlamph(char fname[50]);
//	read info
void rcompt(char fname[50]);
//	read info
void rcmpsf(char fname[50]);
//	read info
void rphote(char fname[50]);
//	read info
void rrayle(char fname[50]);
//	read info
void rrayff(char fname[50]);
//	read info
void rvoxg(char fname[100]);
//	vox infomation
void rconfig(char fname[100]);
//      read config info

void rmear(char fname[40]); 
// read info


void outputData(void *src, const int size, const char *outputfilename, const char *mode);
void outputData(const char *srcname, const int size, const char *outputfilename, const char *mode);
//      output data to file
void fina();
void fina2();
//	finalize dpm code

//==========================================================
// function related to source model
//===========================================================
void init_icdf_EDist(char fname[50], char fname2[50]);
void init_icdf_ZDist(char fname[50]);
void init_icdf_PhiDist(char fname[50]);

void source(int num, int iEbin);
__global__ void setSource(int num, int iEbin, int nZbin);

void initSourceInfo(BeamData sourcedata);
//__global__ void Relocate_Rotate_PhapParticle(int num, int idSource, int bufferheadid, float4 *x_phap_gBrachy,float4 *vx_phap_gBrachy );
__global__ void Relocate_Rotate_PhapParticle(int num, int idSource, int bufferheadid, int NRepeat, int idRepeat, float4 *x_phap_gBrachy,float4 *vx_phap_gBrachy );



//==========================================================
//      declare photon functions
//==========================================================

__global__ void photon(float *escore, const int nactive, int *material, float* density, int* n_bounds, int* n_offsprings, float* parameters);
//	transport a photon
__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe, int matid);
//__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe);
//	sample cross section
__device__ void rylsam(float energytemp, int matid, curandState *localState_pt, float *costhe);
//	sample Rayleigh cross section 
__device__ float itphip_G(int matid, float e);
//      interpolation of phi on GPU
__device__ float lamwck(float e);
//	mfp of photon
__device__ float icptip(int matid, float e);
//	inverse compton mfp
__device__ float irylip(int matid, float e);
//	inverse rayleigh mfp

//==========================================================
//      declare gCTD functions
//==========================================================
void convertDose();
void simulateParticles(Particle particle,BeamData beamdata,int NRepeat);
void simulateParticles(long long NParticle, int iEbin);

bool loadFromPSfile(Particle particle);
//      generate source particles
__device__ void scoreDose(float edep, int4 voxid, curandState *localState_pt,float *escore);
__device__ void scoreDoseToWater(float edep, float energy, int4 voxid, curandState *localState_pt,float *escore,int tempMaterial);
//	perform addition with float number
void loadFromStack(int itype, int nstk);
//      load particles from given stack

__global__ void getEStkPcl();
//      move particles from stack to simulation space
void clrStat();
//      clean the dose counter for statistics


void finStat();
//	finalize statistics
__global__ void calStat(float *escore,float *fEscore, float *fEscor2);
//      calculate final statis
__global__ void convertDoseKernal(float *dscore, float *escore,float totalWeight_gBrachy);
__global__ void calculateDensityKernal(float *dscore, float* density, int* n_bounds, int* n_offsprings, float* parameters, float* delta, float* delta2);




//==========================================================
//      declare geometry functions
//==========================================================
__device__ void rotate(float *u, float *v, float *w, float costh, float phi);
//	rotate a vector
__host__ __device__ int getabs(int xvox, int yvox, int zvox, int nx, int ny, int nz);
//	convert a 3-d voxel indices to one index
__device__ int4 getAbsVox(float4 xtemp);

//==========================================================
//      declare/define other functions
//==========================================================
__host__ __device__ int ind2To1(int i, int j, int nx, int ny)
//      convert a voxel real indices into a single index
{
//      different ordering
        return i*ny+j;
//      return j*nx+i;
}

//==========================================================
//      declare/define functions for electron transport
//==========================================================

void restep(char *fname);
void rerstpw(char *fname);
void rescpw(char *fname);
void reqsurf(char *fname);
void rebw(char *fname);
void inisub();
void clrStk();

__global__ void electr(float *escore, int nactive, int *tempMaterial, float *tempDensity);
__device__ void esamsca(float e, float *mu, curandState *localState_pt);
__device__ float exbwip(float ie);
__device__ float exq2Dip(float u,float ie);
__device__ void esubabs(float4* xtemp, float4 *vtemp, int4 *voxtemp, curandState *localState_pt, float *escore);
__device__ float estepip(float e);
__device__ float erstpip(int matid, float e);
__device__ float escpwip(int matid, float e);
__device__ void flight(float4 *xtemp, float4 *vxtemp, int4 *voxtemp, int* eventid, float *fuelel, float *fuelxt, float *escore, float *smax, int *indexvox, int *dvox, int *tempMaterial, float *tempDensity);
__device__ void chvox(int4 *voxtemp, int indexvox, int dvox);
__device__ float inters(float4 *vtemp, float4* xtemp, int4 *voxtemp, int *indexvox, int *dvox);
__device__ float comele(float energytemp, float efrac, float costhe);
__device__ void putElectron(float4 *vxtemp, float4 *xtemp,float de, curandState *localState_pt);
__device__ int getID(float4 *coords, int *n_boundsShared, float *parametersShared, int *n_offspringsShared);
__device__ void getSign(int *sign, float4 *coords, int i, int *n_boundsShared, float *parametersShared);

*/
#endif
