// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <time.h>
using namespace std;

// includes, project
#include <cutil_math.h>
#include <cuda.h>
#include <cublas.h>
#include <curand_kernel.h>

// includes
#include "cuPrintf.cu"
#include "gBrachy.h"
#include "gBrachyInternal.h"
#include "initialize.cu"
#include "finalize.cu"
#include "gBrachy.cu"
#include "libphoton.cu"
#include "libpenmath.cu"
#include "libgeom.cu"
#include "libelectron.cu"



/****************************************************
        main program
****************************************************/
int main( )
{
    cout << endl << "****************************************" << endl;
    cout << "Computation parameters..." << endl;
    cout << "****************************************" << endl ;

	clock_t start_time, end_time;
    float time_diff;    
	start_time = clock();
	printf("start time %ld\n",start_time);		
	char buffer[200];
	char fname0[100], fname1[100],fname2[100], fname3[100];
	
    readinput(fname0);
    readinput(fname1);
    
    Patient patient;
    patient=loadPatient(fname0,fname1); // It is users' responsibility to load their own patient density and material
		
    readinput(fname2);
	
    readinput(fname3);	
		
	BeamData beamdata;
	//char fname2[100]="patientcase/lungslab/sourceSpect.dat";
	beamdata = loadBeamData(fname2); 
		
	int ifDoseToWater_h;
    readinput(&ifDoseToWater_h);
			
	readinput(&ifUseModel);
	
    int NParticle;
    readinput(&NParticle);	

	int NRepeat;
    readinput(&NRepeat);
	    
	int deviceNo;
    readinput(&deviceNo);
		
    init(patient, beamdata, ifDoseToWater_h,deviceNo); 
		
	cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1);
		
	if(ifUseModel)
	{
		//long long NParticle = 1e6;
		runCalculation(NParticle);
	}
	else
	{//this is set up for future use
		Particle particle;
//      particle = SampleParticle(); //It is users' responsibility to load sampled particles for dose calculation
        float user_esrcmax = 1400000.0f;
	    float user_esrcmin = 9000.0f;
	
	    int NParticle = 42031272;
        
        char fname1[100]="varian_psf_2e7.bin";
		  
        
        particle = ReadParticle(fname1,NParticle,user_esrcmax,user_esrcmin,patient); 
		  
		float loadtime = clock();
		float time_diff2 = ((float)loadtime - (float)start_time)/CLOCKS_PER_SEC;
		printf("loading time: %f ms.\n\n",time_diff2);   
               
        runCalculation(particle, beamdata, NRepeat);
    }
		
    PatientDose patientDose;
    patientDose = getDose(); 
           
        
    fina();
//      finalize the entire compuation

    end_time = clock();
	printf("start time %ld\n",start_time);
	printf("end time %ld\n",end_time);
	cudaEventRecord(event2);
    cudaEventSynchronize(event2);
    float dt_ms = 0;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    cout << "Total Computation time: " << dt_ms << " ms" << endl;
		
		
    time_diff = ((float)end_time - (float)start_time)/1000.0;
	float time_diff1 = ((float)end_time - (float)start_time)/CLOCKS_PER_SEC;
    printf("\n\n****************************************\n");
    printf("Total time: %f ms.\n\n",time_diff);   
	printf("Total time: %f s.\n\n",time_diff1);
    printf("****************************************\n\n\n");
		
		
        

    FILE *fp;
    fp = fopen(fname3, "w");
    fwrite(patientDose.doseAve, NXYZ*sizeof(float), 1 , fp );
    fclose(fp); 
        
      //  cout<<"Totalweight="<<patientDose.totalParticleWeight<<endl;
        
    delete[] patient.mat;
    delete[] patient.dens; 
    delete[] patientDose.doseAve;
    //delete[] patientDose.doseStd;
    delete[] beamdata.XSource;
	delete[] beamdata.YSource;
	delete[] beamdata.ZSource;
	delete[] beamdata.TotalWeight;
	delete[] beamdata.C10Weight;
	delete[] beamdata.C11Weight;
	delete[] beamdata.N13Weight; 
    delete[] beamdata.O15Weight;
		
    printf("Have a nice day!\n");
    cudaThreadExit();
    return 0;  
}   
