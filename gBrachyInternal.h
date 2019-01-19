#ifndef __GBRACHYINTERNAL_H__
#define __GBRACHYINTERNAL_H__

#include <vector>
//==========================================================
//      GPU configurations
//==========================================================

#define NTHREAD_PER_BLOCK_GBRACHY 256//32
#define NTHREAD_PER_BLOCK_ELECTRON 64  
//      number of threads per block

#define NBLOCKX 32768
//      the leading dimension of the 2d thread grid


//==========================================================
//      global variables
//==========================================================
// variable for source model
int ifUseModel;
__device__ __constant__ int ifDoseToWater_brachy;
int matid_water_h;
__device__ __constant__ int matid_water;

int nicdf_ZDist, nEbin, nicdf_EDist, nicdf_PhiDist, nZbin, nEk;

float *pEbin;

cudaArray *icdf_ZDist;
texture<float,1,cudaReadModeElementType> icdf_ZDist_tex;

cudaArray* icdf_EDist;
texture<float, 2, cudaReadModeElementType> icdf_EDist_tex;

cudaArray* icdf_PhiDist;
texture<float, 2, cudaReadModeElementType> icdf_PhiDist_tex;


__device__ __constant__ float idpZ;
__device__ __constant__ float idpE;
__device__ __constant__ float idpPhi;
__device__ __constant__ float idZbin;
__device__ __constant__ float threE;
__device__ __constant__ float Zbeg;
__device__ __constant__ float Ztip;
__device__ __constant__ float sourceR;

__device__ __constant__ int NSource;

#define MAXNSOURCE 350000
__device__ float X0_source[MAXNSOURCE];
__device__ float Y0_source[MAXNSOURCE];
__device__ float Z0_source[MAXNSOURCE];
__device__ float TotalWeight_source[MAXNSOURCE];
__device__ float C10Weight_source[MAXNSOURCE];
__device__ float C11Weight_source[MAXNSOURCE];
__device__ float N13Weight_source[MAXNSOURCE];
__device__ float O15Weight_source[MAXNSOURCE];

//	common variable group /dpmpart/
#define NPART 262144 //131072//16384 //262144
__device__ float4 vx_gBrachy[NPART];
__device__ float4 x_gBrachy[NPART];

__device__ float4 vx0_gBrachy[NPART];
__device__ float4 x0_gBrachy[NPART];

float4 *vx_phap_gBrachy;
float4 *x_phap_gBrachy;

int nactive_h, ptype_h, nparload_h;
__device__ int nactive, ptype;

float4 xbufferRepeat[NPART];
float4 vxbufferRepeat[NPART];

//#define MAX_NXYZ 128*128*128
//__device__ float fEscore[MAX_NXYZ], fEscor2[MAX_NXYZ];

float *escore;
float *fEscore, *fEscor2;

const char *outputAveName = "resultAve.dat";
const char *outputStdName = "resultStd.dat";
float totalWeight_gBrachy = 0.0f;
long long totalSimPar = 0;

//	common variable group /dpmsim/ only used in CPU
#define NBATCH 1
signed long long NXYZ;
int NXZ;

vector<int> bufferHeadId;
int bufferHeadId2;

//	common variable group /dpmsrc/
__device__ __constant__ float eabsph;
float eabsph_h;
__device__ __constant__ float xsrc,ysrc,zsrc;
__device__ __constant__ float esrcmin,esrcmax;

//	common variable group /dpmmat/
#define MAXMAT 9
__device__ __constant__ int nmat;
int nmat_h;


//	common variable group /dpmvox/
cudaExtent volumeSize;
cudaMemcpy3DParms copyParams = {0};

cudaArray *mat;
texture<int,3,cudaReadModeElementType> mat_tex;

cudaArray *dens;
texture<float,3,cudaReadModeElementType> dens_tex;

__device__ __constant__ float dx_gBrachy,dy_gBrachy,dz_gBrachy;

__device__ __constant__ int Unxvox,Unyvox,Unzvox;




//	common variable group /dpmrsp/
cudaArray *dersp;
texture<float,1,cudaReadModeElementType> dersp_tex;
__device__ __constant__ float idersp;
__device__ __constant__ float edersp0;
float idersp_h, edersp0_h;


//	common variable group /dpmspc/
#define NSPEC 128

float despec_h;
float espec_h[NSPEC], pspec_h[NSPEC];
float psum_h;
int nspecdata_h;

//	common variable group /dpmlph/ 
#define NLAPH 2048

__device__ __constant__ float idleph;
__device__ __constant__ float elaph0;
float idleph_h;
float elaph_h[NLAPH],lamph_h[NLAPH*MAXMAT],lampha_h[NLAPH*MAXMAT],lamphb_h[NLAPH*MAXMAT],
	lamphc_h[NLAPH*MAXMAT],lamphd_h[NLAPH*MAXMAT];
//	lampha~d_h is necessary, they are used on CPU side for initialize Woodkock method
cudaArray *lamph;
texture<float,1,cudaReadModeElementType> lamph_tex;

//	common variable group /dpmcmp/
#define NCMPT 2048

__device__ __constant__ float idlecp;
__device__ __constant__ float ecmpt0;
float idlecp_h;
float ecmpt_h[NCMPT],compt_h[NCMPT*MAXMAT];
cudaArray *compt;
texture<float,1,cudaReadModeElementType> compt_tex;

#define NCPCM 101
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

#define NCPRL 101
#define NERL 51
float idcprl_h, iderl_h;
__device__ __constant__ float idcprl,iderl;
float murayl_h[NCPRL*NERL*MAXMAT];
cudaArray* fArray;
texture<float, 3, cudaReadModeElementType> f_tex;

#define NMEAR 1024

__device__ __constant__ float idmear;
__device__ __constant__ float emear0;
float idmear_h;
float emear_h[NMEAR],mear_h[NMEAR*MAXMAT];
cudaArray *mear;
texture<float,1,cudaReadModeElementType> mear_tex;


//	common variable group /cgeom3/
__device__ __constant__ float idx_gBrachy,idy_gBrachy,idz_gBrachy;
float idx_gBrachy_h,idy_gBrachy_h,idz_gBrachy_h;

__device__ __constant__ float Offsetx_gBrachy,Offsety_gBrachy,Offsetz_gBrachy;


//	common variable group /dpmwck/
#define NWCK 2048
__device__ __constant__ float idlewk;
__device__ __constant__ float wcke0;
float idlewk_h, wcke0_h;

float woock_h[NWCK];
cudaArray *woock;
texture<float,1,cudaReadModeElementType> woock_tex;


//	common variable group /rseed/
int iseed1_h[NPART];
__device__ int iseed1[NPART];
__device__ curandState cuseed[NPART];

//	common variable for electron transport
__device__ __constant__ float eabs;
float eabs_h;	//electron absorption energy

__device__ __constant__ float subden,subfac,substp;
float subden_h,subfac_h,substp_h;	//subthreshold transport

#define NSSTACK 8*2*NPART
#define NSSTACKSHARED (8*NTHREAD_PER_BLOCK_GBRACHY)
__device__ int nsstk;
__device__ float sf[NSSTACK];


#define NESTACK (4*NPART)
#define NESTACKSHARED (3*NTHREAD_PER_BLOCK_ELECTRON )

__device__ int nestk;
__device__ float4 esx[NESTACK];
__device__ float4 esvx[NESTACK];
__device__ bool esifp[NESTACK];


#define NSCSR 512	//scattering strength
__device__ __constant__ float idless;
__device__ __constant__ float escsr0;
float idless_h;
float escsr_h[NSCSR],scssp_h[NSCSR];
cudaArray *scssp;
texture<float,1,cudaReadModeElementType> scssp_tex;

#define NST 1024	//restricted stopping power
__device__ __constant__ float idlest;	
__device__ __constant__ float est0;
float idlest_h;
float est_h[NST],stsp_h[NST*MAXMAT];
cudaArray *stsp;
texture<float,1,cudaReadModeElementType> stsp_tex;

#define NSCP 2048	//1st TMP
__device__ __constant__ float idlesc;
__device__ __constant__ float escp0;
float idlesc_h;
float escp_h[NSCP],scpsp_h[NSCP*MAXMAT];
cudaArray *scpsp;
texture<float,1,cudaReadModeElementType> scpsp_tex;

#define NUQ 128		//q surface
#define NEQ 256
__device__ __constant__ float le0q,idleq,iduq;
float le0q_h,idleq_h,iduq_h;
float q_h[NUQ*NEQ];
cudaArray* qArray;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
texture<float, 2, cudaReadModeElementType> q_tex;

#define NBW 512		//screening parameters
__device__ __constant__ float idlebw;
__device__ __constant__ float ebw0;
float idlebw_h;
float ebw_h[NBW],bwsp_h[NBW];
cudaArray *bwsp;
texture<float,1,cudaReadModeElementType> bwsp_tex;

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
//      declare main functions
//==========================================================
void printDevProp(int device);
//      print out device properties
void runCalculation(Particle particle, BeamData beamdata, int NRepeat); //use phase space file 
//      run entire DPM calculation
void runCalculation(long long NParticle); // use source model
PatientDose getDose();

//==========================================================
//	declare initialization/finalization functions
//==========================================================

void init(Patient patient, BeamData sourcedata, int ifDoseToWater_h, int deviceNo);
//	initialize DPM
void initpatient(Patient patient);
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
void iniwck(float eminph,float emax, Patient patient);
//	find info for photon
float itphip(int matid, float e);
//	interpolation of phi
void rmear(char fname[40]); 
// read info


void outputData(void *src, const int size,const char *outputfilename, const char *mode);
void outputData(const char *srcname, const int size, const char *outputfilename, const char *mode);
//      output data to file
void fina();
//	finalize dpm code

//==========================================================
// function related to source model
//===========================================================
void init_icdf_EDist(char fname[50], char fname2[50]);
void init_icdf_ZDist(char fname[50]);
void init_icdf_PhiDist(char fname[50]);

void source(int num);
__global__ void setSource(int num);
__device__ int binarySearch(float num);
__device__ float box_muller1(float m, float s, curandState *localState_pt);
__device__ float getDistance(float4 coords, float4 direcs);

void initSourceInfo(BeamData sourceinfo);
//__global__ void Relocate_Rotate_PhapParticle(int num, int idSource, int bufferheadid, float4 *x_phap_gBrachy,float4 *vx_phap_gBrachy );
__global__ void Relocate_Rotate_PhapParticle(int num, int idSource, int bufferheadid, int NRepeat, int idRepeat, float4 *x_phap_gBrachy,float4 *vx_phap_gBrachy );



//==========================================================
//      declare photon functions
//==========================================================

__global__ void photon(float *escore, const int nactive);
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
void simulateParticles(long long NParticle);

bool loadFromPSfile(Particle particle);
//      generate source particles
__device__ void scoreDose(float edep, int4 voxid, curandState *localState_pt,float *escore);
__device__ void scoreDoseToWater(float edep, float energy, int4 voxid, curandState *localState_pt,float *escore);
__device__ void inline atomicFloatAdd(float *address, float val);
//	perform addition with float number
void loadFromStack(int itype, int nstk);
//      load particles from given stack

__global__ void getEStkPcl();
//      move particles from stack to simulation space
void clrStat();
//      clean the dose counter for statistics


/* void finStat();
//	finalize statistics
__global__ void calStat(float *escore,float *fEscore, float *fEscor2); */
//      calculate final statis
__global__ void convertDoseKernal(float *escore,float totalWeight_gBrachy);

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

__global__ void electr(float *escore, int nactive);
__device__ void esamsca(float e, float *mu, curandState *localState_pt);
__device__ float exbwip(float ie);
__device__ float exq2Dip(float u,float ie);
__device__ void esubabs(float4* xtemp, float4 *vtemp, int4 *voxtemp, curandState *localState_pt, float *escore);
__device__ float estepip(float e);
__device__ float erstpip(int matid, float e);
__device__ float escpwip(int matid, float e);
__device__ void flight(float4 *xtemp, float4 *vxtemp, int4 *voxtemp, int* eventid, float *fuelel, float *fuelxt, float *escore, float *smax, int *indexvox, int *dvox);
__device__ void chvox(int4 *voxtemp, int indexvox, int dvox);
__device__ float inters(float4 *vtemp, float4* xtemp, int4 *voxtemp, int *indexvox, int *dvox);
__device__ float comele(float energytemp, float efrac, float costhe);
__device__ void putElectron(float4 *vxtemp, float4 *xtemp,float de, curandState *localState_pt);

#endif
