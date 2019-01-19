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
		int len;
			
		gets(buffer);
        printf("%s\n", buffer);
        getnam2(5, fname0, &len);
		printf("%s\n",fname0);
        getnam2(5, fname1, &len);
        printf("%s\n",fname1);
        Patient patient;
	    patient=loadPatient(fname0,fname1); // It is users' responsibility to load their own patient density and material
		
        gets(buffer);
        printf("%s\n", buffer);
        getnam2(5, fname2, &len);
		printf("%s\n",fname2);
		
		gets(buffer);
        printf("%s\n", buffer);
        getnam2(5, fname3, &len);
		printf("%s\n",fname3);
		
		BeamData beamdata;
		//char fname2[100]="patientcase/lungslab/sourceSpect.dat";
		beamdata = loadBeamData(fname2); 
		
		int ifDoseToWater_h;
		
		gets(buffer);
        printf("%s\n", buffer);
        scanf("%d\n", &ifDoseToWater_h);
        printf("%d\n", ifDoseToWater_h);
		
			
	    gets(buffer);
        printf("%s\n", buffer);
		scanf("%d\n", &ifUseModel);
		printf("%d\n", ifUseModel);
		

		int NRepeat;
		gets(buffer);
        printf("%s\n", buffer);
		scanf("%d\n", &NRepeat);
		printf("%d\n", NRepeat);
	    
		int deviceNo;
		gets(buffer);
        printf("%s\n", buffer);
		scanf("%d\n", &deviceNo);
		printf("%d\n", deviceNo);
		
        init(patient, beamdata, ifDoseToWater_h,deviceNo); 
		
		cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1);
		
		if(ifUseModel)
		{
		   long long NParticle = 1e6;
		   runCalculation(NParticle);
		}
		else
		{
		  Particle particle;
//        particle = SampleParticle(); //It is users' responsibility to load sampled particles for dose calculation
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
        
     
}

Patient loadPatient(char fname[100], char fname1[100])
/*******************************************************************
c*    Reads voxel geometry from an input file                      *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c*    Comments:                                                    *
c*      -> rmater must be called first to set nmat.                *
c******************************************************************/
{
        char buffer[100];
        Patient patient;

        printf("\n");
        printf("\n");
        printf("loading patient phantom ... \n");
       	
        FILE *fp = fopen(fname,"r");
       
        fscanf(fp, "%d %d %d\n",  &patient.Unxvox, &patient.Unyvox, &patient.Unzvox );
        printf("CT dimension: %d %d %d\n", patient.Unxvox, patient.Unyvox, patient.Unzvox);
          
		fscanf(fp, "%f %f %f\n",  &patient.Offsetx, &patient.Offsety, &patient.Offsetz );
        printf("CT offset: %f %f %f\n", patient.Offsetx, patient.Offsety, patient.Offsetz);
		
		fscanf(fp, "%f %f %f\n",  &patient.Sizex, &patient.Sizey, &patient.Sizez );
        printf("CT Size: %f %f %f\n", patient.Sizex, patient.Sizey, patient.Sizez);
		
        patient.dx = patient.Sizex/patient.Unxvox; 
		patient.dy = patient.Sizey/patient.Unyvox; 
		patient.dz = patient.Sizez/patient.Unzvox; 
		
        patient.mat = new int[patient.Unxvox*patient.Unyvox*patient.Unzvox];
        patient.dens = new float[patient.Unxvox*patient.Unyvox*patient.Unzvox];

        //      read density
        printf("Reading CT densities...\n");
         for(int k = 0; k<patient.Unzvox; k++)
        {
                for(int j=0; j < patient.Unyvox; j++)
                {
                        for (int i=0; i < patient.Unxvox; i++)
                        {
                                int absvoxtemp = getabs(i,j,k,patient.Unxvox,patient.Unyvox,patient.Unzvox);
                                fscanf(fp, "%f ",&patient.dens[absvoxtemp]);
                        }
                        fscanf(fp,"\n");
                }
                fscanf(fp,"\n");
        }

//      
        fclose(fp);  

        FILE *fp1 = fopen(fname1,"r");
        
        printf("Reading CT materials...\n");
        for(int k = 0; k<patient.Unzvox; k++)
        {
                for(int j=0; j < patient.Unyvox; j++)
                {
                        for (int i=0; i < patient.Unxvox; i++)
                        {
                                int absvoxtemp = getabs(i,j,k,patient.Unxvox,patient.Unyvox,patient.Unzvox);
                                fscanf(fp1, "%d ",&patient.mat[absvoxtemp]);
                               // printf("%d ", patient.mat[absvoxtemp]);				
                        }
                        fscanf(fp1,"\n");
						//printf("\n");
                }
                fscanf(fp1,"\n");
        }

        fclose(fp1);

        
        
 	  fp = fopen("dens1.dat", "w");
        fwrite(patient.dens, patient.Unxvox*patient.Unyvox*patient.Unzvox*sizeof(float), 1 , fp );
        fclose(fp);
	fp = fopen("mat1.dat", "w");
        fwrite(patient.mat, patient.Unxvox*patient.Unyvox*patient.Unzvox*sizeof(int), 1 , fp );
        fclose(fp);   

     return patient;
   }


/* Particle SampleParticle()
{
	float user_esrcmax = 150000.0f;
	float user_esrcmin = 0.0f;
	int NParticle = 100000;
	
//	float4 *user_vx, *user_x;
//	
//	cudaMalloc( (void **) &user_vx, sizeof(float4)*NParticle) ;
//    cudaMemset(user_vx, 0, sizeof(float4)*NParticle) ;
//    
//    cudaMalloc( (void **) &user_x, sizeof(float4)*NParticle) ;
//    cudaMemset(user_x, 0, sizeof(float4)*NParticle) ;
    

//    int NTHREAD_PER_BLOCK_src = 256;
//    
//    int nblocks = 1 + (NParticle - 1)/NTHREAD_PER_BLOCK_src ;
//    
//    setSourceDirandPos<<<nblocks, NTHREAD_PER_BLOCK_src>>>(NParticle,user_vx,user_x);
//	//setSourceEne<<<nblocks, NTHREAD_PER_BLOCK_src>>>(NParticle,user_vx,user_esrcmax,user_esrcmin);
//    cudaThreadSynchronize();
        
    float4 *user_vx_h, *user_x_h;
    user_vx_h = new float4[NParticle];  
    user_x_h = new float4[NParticle];  
//    
//    cudaMemcpy(user_vx_h, user_vx, sizeof(float4)*NParticle, cudaMemcpyDeviceToHost);
//    cudaMemcpy(user_x_h, user_x, sizeof(float4)*NParticle, cudaMemcpyDeviceToHost);
    float temp1, temp2, temp3, temp4,length;
    
    for(int i=0; i<NParticle;i++)
    {
    	temp1=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
    	temp2=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
    	temp3=((float)rand()/(float)RAND_MAX)*2.0f-1.0f;
    	temp4=((float)rand()/(float)RAND_MAX)*(user_esrcmax-user_esrcmin)+user_esrcmin;
    	
    	length = sqrtf(temp1*temp1+temp2*temp2+temp3*temp3);
    	user_vx_h[i].x = temp1/length;
    	user_vx_h[i].y = temp2/length;
    	user_vx_h[i].z = temp3/length;
    	user_vx_h[i].w = temp4;
    	
    	user_x_h[i].x = 12.8;
    	user_x_h[i].y = 12.8;
    	user_x_h[i].z = 12.8;
    	user_x_h[i].w = 1;
    }
    
    FILE *fp;
    
    fp = fopen("vx.dat", "w");
        fwrite(user_vx_h, sizeof(float)*NParticle*4, 1 , fp );
        fclose(fp);
	fp = fopen("x.dat", "w");
        fwrite(user_x_h, sizeof(float)*NParticle*4, 1 , fp );
        fclose(fp);
        
    
    Particle particle;
    particle.NBuffer = 10;
    
    float esrc_dbuffer = (user_esrcmax - user_esrcmin)/particle.NBuffer;
    
    for(int i = 0; i<particle.NBuffer; i++)
	{
		vector<float4> temp1;
		particle.xbuffer.push_back(temp1);
		vector<float4> temp2;
		particle.vxbuffer.push_back(temp2);
	}
	
    int bufferid;
    
    for(int i = 0; i<NParticle; i++)
    {
    	bufferid = int((user_vx_h[i].w - user_esrcmin)/esrc_dbuffer);
						
		bufferid = (bufferid > particle.NBuffer)? particle.NBuffer-1 : bufferid;	//truncate
		bufferid = (bufferid < 0)? 0 : bufferid;
			
		particle.xbuffer[bufferid].push_back(user_x_h[i]);
		particle.vxbuffer[bufferid].push_back(user_vx_h[i]); 	
    }
    
    delete[] user_vx_h;
    delete[] user_x_h;
    
//    cudaFree(user_vx);
//    cudaFree(user_x);
   
	return particle;
}*/


Particle ReadParticle(char fname[100],int NParticle,float user_esrcmax, float user_esrcmin, Patient patient)
{
	Particle particle;
    particle.NBuffer = 6;
	
    
    int Num_per_buffer[particle.NBuffer];
	int aNum_per_buffer[particle.NBuffer];
    
    float esrc_dbuffer = (user_esrcmax - user_esrcmin)/(particle.NBuffer+4);
    
    printf("energy span for buffer: %f \n", esrc_dbuffer);
    
    for(int i = 0; i<particle.NBuffer; i++)
	{
		vector<float4> temp1;
		particle.xbuffer.push_back(temp1);
		vector<float4> temp2;
		particle.vxbuffer.push_back(temp2);
		
		Num_per_buffer[i]=0;
	}
	
    int bufferid;
    
    FILE *fp;
    
    fp = fopen(fname,"rb");
    
    float kEenergy, posX, posY, posZ, dirX, dirY, dirZ;
    
    //float weight = 1.0f;
	float weight = 0.01f;
	
     for(int i = 0; i<NParticle; i++)
    {
    	fread(&kEenergy, sizeof(float), 1, fp);
    	fread(&posX, sizeof(float), 1, fp);
    	fread(&posY, sizeof(float), 1, fp);
    	fread(&posZ, sizeof(float), 1, fp);
    	fread(&dirX, sizeof(float), 1, fp);
    	fread(&dirY, sizeof(float), 1, fp);
    	fread(&dirZ, sizeof(float), 1, fp);
    	
    	kEenergy *= 1.0e6; // covert the energy from MeV to eV 
    	
    	if(kEenergy <= user_esrcmin) continue;
    		
    	bufferid = int((kEenergy - user_esrcmin)/esrc_dbuffer);
						
		bufferid = (bufferid >= particle.NBuffer)? particle.NBuffer-1 : bufferid;	//truncate
		bufferid = (bufferid < 0)? 0 : bufferid;
			
		posX = (posX/10.0f);
		posY = (posY/10.0f);
		posZ = (posZ/10.0f)-10.0f;
			
		particle.xbuffer[bufferid].push_back(make_float4(
						posX, posY, posZ, weight));
		particle.vxbuffer[bufferid].push_back(make_float4(
						dirX, dirY, dirZ,kEenergy )); 	
						
	   if(i<10){
	   	printf("particle %d: %f %f %f %f %f %f %f %d\n",i,kEenergy,posX,posY,posZ,dirX,dirY,dirZ,bufferid );
	   }
	   
	   Num_per_buffer[bufferid]++; 
	   
    }
	float4 *tempx = new float4[NParticle];
    float4 *tempvx = new float4[NParticle];
	
    cudaMalloc( (void **) &x_phap_gBrachy, sizeof(float4)*NParticle);
	cudaMalloc( (void **) &vx_phap_gBrachy, sizeof(float4)*NParticle);
	
	//cudaMemcpy(x_phap_gBrachy, &(particle.xbuffer[0][0]), sizeof(float4)*Num_per_buffer[0], cudaMemcpyHostToDevice);
	printf("particle numer for %d buffer: %d \n",0,Num_per_buffer[0] );
	
     memcpy(&(tempx[0]), &(particle.xbuffer[0][0]), sizeof(float4)*Num_per_buffer[0]);
	 memcpy(&(tempvx[0]), &(particle.vxbuffer[0][0]), sizeof(float4)*Num_per_buffer[0]);
	 
	 aNum_per_buffer[0]=Num_per_buffer[0];
	 
	for(int i = 1; i<particle.NBuffer; i++){
      printf("particle numer for %d buffer: %d \n",i,Num_per_buffer[i] ); 
	  
	  aNum_per_buffer[i] = aNum_per_buffer[i-1]+Num_per_buffer[i];
	   printf("particle numer for %d buffer: %d \n",i,aNum_per_buffer[i] ); 
	   
	  memcpy(&(tempx[aNum_per_buffer[i-1]]), &(particle.xbuffer[i][0]), sizeof(float4)*Num_per_buffer[i]);
	  memcpy(&(tempvx[aNum_per_buffer[i-1]]), &(particle.vxbuffer[i][0]), sizeof(float4)*Num_per_buffer[i]);
	  
    }
    cudaMemcpy(x_phap_gBrachy, tempx, sizeof(float4)*NParticle, cudaMemcpyHostToDevice); 
	cudaMemcpy(vx_phap_gBrachy, tempvx, sizeof(float4)*NParticle, cudaMemcpyHostToDevice); 
    fclose(fp);
	
	particle.NParticle = aNum_per_buffer[particle.NBuffer-1];
	
	cout<<"Total Particle = "<<particle.NParticle <<endl;
    free(tempx);
	free(tempvx);
	
	
    
	return particle;
} 

 BeamData loadBeamData(char fname[100])
{
  BeamData mybeam;
  
  FILE *fp = fopen(fname,"r"); 
  
  char buffer[200];
  fgets(buffer,200,fp);
  printf("%s\n", buffer);
  fscanf(fp,"%d\n", &mybeam.numSource);  
  printf("numSource = %d\n", mybeam.numSource);
  
  mybeam.XSource = new float[mybeam.numSource];
  mybeam.YSource = new float[mybeam.numSource];
  mybeam.ZSource = new float[mybeam.numSource];
  mybeam.TotalWeight = new float[mybeam.numSource];
  mybeam.C10Weight = new float[mybeam.numSource];
  mybeam.C11Weight = new float[mybeam.numSource];
  mybeam.N13Weight = new float[mybeam.numSource];  
  mybeam.O15Weight = new float[mybeam.numSource]; 
  
  for (int i=0; i < mybeam.numSource; i++)
  {

    fscanf(fp, "%f",&mybeam.TotalWeight[i]);
    fscanf(fp, "%f",&mybeam.C10Weight[i]);
    fscanf(fp, "%f",&mybeam.C11Weight[i]);
    fscanf(fp, "%f",&mybeam.N13Weight[i]);
    fscanf(fp, "%f",&mybeam.O15Weight[i]);
	fscanf(fp, "%f",&mybeam.ZSource[i]);
    fscanf(fp, "%f",&mybeam.YSource[i]); 
    fscanf(fp, "%f",&mybeam.XSource[i]); 	
	fscanf(fp,"\n");
    if (i<=100)
	printf("%f %f %f %f %f %f %f %f\n", mybeam.TotalWeight[i],mybeam.C10Weight[i],mybeam.C11Weight[i],mybeam.N13Weight[i],mybeam.O15Weight[i],mybeam.XSource[i],mybeam.YSource[i],mybeam.ZSource[i]);
  }
   
    fclose(fp);
	
	return mybeam;
  
}   
