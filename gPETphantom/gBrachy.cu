#ifndef __GBRACHY_CU__
#define __GBRACHY_CU__
void runCalculation(long long NParticle)
{


    time_t start_time, end_time;
    float time_diff;

    start_time = clock();
    long  NParticle_iEbin;
    NParticle_iEbin=16384;
    int maxBin=int(NParticle/NParticle_iEbin);
    cout<<"maxBin "<<maxBin<<endl;
    long  cumuPar=0;

    for(int iEbin = 0; iEbin<maxBin+1; iEbin++)
    {

        if(NParticle-cumuPar>NParticle_iEbin)
        {
            cout<<"simulate particle number:"<<NParticle_iEbin<<endl;
            simulateParticles(NParticle_iEbin);
            cumuPar+=NParticle_iEbin;
        }
        else
        {
            cout<<"simulate particle number:"<<NParticle-cumuPar<<endl;
            simulateParticles(NParticle-cumuPar);
            cumuPar+=NParticle-cumuPar;
        }

    }

    /*for(int iEbin = 0; iEbin<100; iEbin++){

      NParticle_iEbin = NParticle * 0.01;
      cout<<"simulate particle number:"<<NParticle_iEbin<<endl;
      simulateParticles(NParticle_iEbin);
    }*/

    /* float *escore_temp = new float[NXYZ*NBATCH];
            cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;

            FILE *fp;
            fp = fopen("escore1.dat", "w");
            fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
            fclose(fp); */
    convertDose();
    /* cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;


    fp = fopen("escore2.dat", "w");
    fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
    fclose(fp); */

    // finStat();

    /* cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;


            fp = fopen("escore3.dat", "w");
            fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
            fclose(fp); */

    cudaThreadSynchronize();
    end_time = clock();
    time_diff = ((float)end_time - (float)start_time)/1000.0;
    printf("\n\n****************************************\n");
    printf("Simulation time: %f ms.\n\n",time_diff);
    printf("****************************************\n\n\n");
}

void runCalculation(Particle particle, BeamData beamdata, int NRepeat)
{
    time_t start_time, end_time;
    float time_diff;

    start_time = clock();
    bufferHeadId2 = 0;
    /*  for(int i = 0; i<particle.NBuffer; i++)
     {
      bufferHeadId.push_back(0);
     } */
    simulateParticles(particle,beamdata, NRepeat);

    /* float *escore_temp = new float[NXYZ*NBATCH];
            cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;

            FILE *fp;
            fp = fopen("escore1.dat", "w");
            fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
            fclose(fp);
             */
    convertDose();
    // cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;
    /* fp = fopen("escore2.dat", "w");
    fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
    fclose(fp); */

    //  finStat();

    //cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;
    /*   fp = fopen("escore3.dat", "w");
      fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
      fclose(fp); */

    cudaThreadSynchronize();

    end_time = clock();
    time_diff = ((float)end_time - (float)start_time)/1000.0;
    printf("\n\n****************************************\n");
    printf("Simulation time: %f ms.\n\n",time_diff);
    printf("****************************************\n\n\n");
}


/* void finStat()
//      finalize statistics
{
	dim3 nblocks;

        nblocks.x = NBLOCKX;
        nblocks.y = ((1 + (NXYZ - 1)/NTHREAD_PER_BLOCK_GBRACHY) - 1) / NBLOCKX + 1;

        calStat<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(escore, fEscore, fEscor2);
        cudaThreadSynchronize();
}

__global__ void calStat(float *escore, float *fEscore, float *fEscor2)
//      calculate final statis
{
	const int tid = (blockIdx.y*NBLOCKX + blockIdx.x)*blockDim.x + threadIdx.x;
//      obtain current id on thread

        if (tid < Unzvox*Unyvox*Unxvox)
        {
                float sumq = 0;
                float sumq2 = 0;
		for(int i = 0; i<NBATCH;i++)
		{
			float tmp = escore[tid + i*Unzvox*Unyvox*Unxvox];
			sumq += tmp;
			sumq2 += tmp*tmp;
		}

		sumq /= NBATCH;
		fEscore[tid] = sumq*NBATCH;
		fEscor2[tid] = __fsqrt_rn((sumq2/NBATCH - sumq*sumq)*NBATCH);
		escore[tid+Unzvox*Unyvox*Unxvox] = fEscore[tid];
        }
} */

void convertDose()
//      convert energy to dose
{
    dim3 nblocks;
    nblocks.x = NBLOCKX;
    nblocks.y = ((1 + (NXYZ - 1)/NTHREAD_PER_BLOCK_GBRACHY) - 1) / NBLOCKX + 1;

    convertDoseKernal<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(escore,totalWeight_gBrachy);
    cudaThreadSynchronize();
}

__global__ void convertDoseKernal(float *escore, float totalWeight_gBrachy)
//      convert energy to dose in each batch counter
{
    const int tid = (blockIdx.y*NBLOCKX + blockIdx.x)*blockDim.x + threadIdx.x;
//      obtain current id on thread

    if (tid < Unzvox*Unyvox*Unxvox)
    {
        float voxvol = dx_gBrachy*dy_gBrachy*dz_gBrachy;
        int xvox = tid%Unxvox;
        int yvox = (tid/Unxvox)%Unyvox;
        int zvox = tid/Unxvox/Unyvox;
        float dens = tex3D(dens_tex,xvox, yvox, zvox);
        //float dens = 1.0f;

        /* if(tid ==100 )
         printf("xvox=%d, yvox=%d, zvox=%d,den = %f\n", xvox, yvox, zvox,dens); 	 */
        float tempvalue;

        for(int i = 0; i < NBATCH; i++)
        {
            tempvalue = escore[tid + i*Unzvox*Unyvox*Unxvox]/(totalWeight_gBrachy*voxvol*dens*1.0e6F);
            if(tempvalue!=tempvalue)
            {
                tempvalue = 0.0f;
            }
            escore[tid + i*Unzvox*Unyvox*Unxvox] = tempvalue;

        }
    }
}


__device__ inline void atomicFloatAdd(float *address, float val)
{
    int tmp0 = *address;
    int i_val = __float_as_int(val + __int_as_float(tmp0));
    int tmp1;
    while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0 )
    {
        tmp0 = tmp1;
        i_val = __float_as_int(val + __int_as_float(tmp1));
    }
}


__device__ void scoreDose(float edep, int4 voxid, curandState *localState_pt,float *escore)
/*******************************************************************
c*    Deposites energy in the corresponding counters               *
c*                                                                 *
c*    Input:                                                       *
c*      edep -> energy being deposited (eV)                        *
c*    Comments:                                                    *
c******************************************************************/
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x;

    int ind = voxid.x + voxid.y*Unxvox + voxid.z*Unyvox * Unxvox;

    /* if(NBATCH>1)
    {
     int ind2 = NBATCH*curand_uniform(localState_pt);
     if(ind2 == NBATCH)
     {
       ind2 = NBATCH - 1;
     }
     ind = ind + ind2*Unzvox*Unyvox*Unxvox;
    } */

    if(edep!=edep)
    {
        printf("voxid.x = %d, voxid.y = %d , voxid.z = %d, voxid.w = %d, ind = %d, edep = %f\n", voxid.x, voxid.y,voxid.z,voxid.w,ind,edep);
    }
    atomicFloatAdd(&escore[ind],edep);


}

__device__ float mearip(int matid, float e)
/*******************************************************************
c*    Photon total inverse mean free path --3spline interpolation  *
c*    this is the GPU version of itphip                            *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      Total inverse mean free path in cm^2/g                     *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
    float i = idleph*(e-elaph0) + 0.5;
    return tex1D(lamph_tex,matid * NLAPH + i);
}


__device__ void scoreDoseToWater(float edep, float energy, int4 voxid, curandState *localState_pt,float *escore)
/*******************************************************************
c*    Deposites energy in the corresponding counters               *
c*                                                                 *
c*    Input:                                                       *
c*      edep -> energy being deposited (eV)                        *
c*    Comments:                                                    *
c******************************************************************/
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x;

    int ind = voxid.x + voxid.y*Unxvox + voxid.z*Unyvox * Unxvox;

    int matid = tex3D(mat_tex, voxid.x, voxid.y, voxid.z);

    // float coef = itphip_G(matid_water, energy)/itphip_G(matid, energy);
    float i = idmear*(energy-emear0) + 0.5;
    float coef = tex1D(mear_tex, matid * NMEAR + i);

    if(edep!=edep)
    {
        printf("voxid.x = %d, voxid.y = %d , voxid.z = %d, voxid.w = %d, ind = %d, edep = %f\n", voxid.x, voxid.y,voxid.z,voxid.w,ind,edep);
    }
    atomicFloatAdd(&escore[ind],edep*coef);


}

bool loadFromPSfile(Particle particle)
//	load particles from PS file, return TRUE if no particles available
{

    int nblocks;

    for(int ibuffer = 0; ibuffer < particle.NBuffer; ibuffer++)
        //for(int ibuffer = 4; ibuffer < 10; ibuffer++)
    {
        int np = particle.xbuffer[ibuffer].size();
        if(np > 0 && bufferHeadId[ibuffer] < np-1)
//	found a buffer nonempty
        {
//	compute range
            int first = bufferHeadId[ibuffer];
            int last = first + NPART -1;
            last = (last > np-1)? np-1 : last;
            int n = last - first + 1;

//	load to GPU

            /* for(int k=0; k<NRepeat; k++) {
               memcpy(&(xbufferRepeat[k*n]),&(particle.xbuffer[ibuffer][first]),sizeof(float4)*n);
               memcpy(&(vxbufferRepeat[k*n]),&(particle.vxbuffer[ibuffer][first]),sizeof(float4)*n);
            }


            if(cudaMemcpyToSymbol(x_gBrachy, &(xbufferRepeat[0]),
            	sizeof(float4)*n*NRepeat, 0, cudaMemcpyHostToDevice)!= cudaSuccess)

            	 cout << "error in setting x_gBrachy" << endl;

            if(cudaMemcpyToSymbol(vx_gBrachy, &(vxbufferRepeat[0]),
                                sizeof(float4)*n*NRepeat, 0, cudaMemcpyHostToDevice)!= cudaSuccess)
            	cout << "error in setting vx_gBrachy" << endl;


               nblocks = 1 + (n*NRepeat - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
               Relocate_Rotate_PhapParticle<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(n*NRepeat); */

            memcpy(&(xbufferRepeat),&(particle.xbuffer[ibuffer][first]),sizeof(float4)*n);
            memcpy(&(vxbufferRepeat),&(particle.vxbuffer[ibuffer][first]),sizeof(float4)*n);

            if(cudaMemcpyToSymbol(x0_gBrachy, &(xbufferRepeat[0]),
                                  sizeof(float4)*n, 0, cudaMemcpyHostToDevice)!= cudaSuccess)

                cout << "error in setting x_gBrachy" << endl;

            if(cudaMemcpyToSymbol(vx0_gBrachy, &(vxbufferRepeat[0]),
                                  sizeof(float4)*n, 0, cudaMemcpyHostToDevice)!= cudaSuccess)
                cout << "error in setting vx_gBrachy" << endl;



//	advance the buffer head index
            bufferHeadId[ibuffer] = last + 1;

//	manipulate particles before launching
//             int temp;
//			 temp = n*NRepeat;
//
//             if( cudaMemcpyToSymbol(nactive, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
//             {
//                cout << "error in setting nactive" << endl;
//
//                exit(1);
//             }
//
//             temp = 0;
//             if( cudaMemcpyToSymbol(ptype, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
//             cout << "error in setting ptype" << endl;

            nparload_h = n;


            /*float *tempaddress;
             if(cudaGetSymbolAddress((void**) &tempaddress,x_gBrachy) != cudaSuccess)
            cout	<< "error in getting symbol address while computing weights" << endl;
            totalWeight_gBrachy += cublasSasum(n*NRepeat, tempaddress+3, 4); */
            /* totalWeight_gBrachy += n*NRepeat;

            cout << "Simulating " << n*NRepeat << " particles from bin " << ibuffer << endl;
            cout << "totalWeight="<<totalWeight_gBrachy << endl; */

            cout << "Loading " << n << " particles from bin " << ibuffer << endl;
            return false;
        }
    }
    return true;

//	return true if no particles loaded, meaning simulated enough
//	if(totalWeight > 1000000.0)
//		return true;
//	else
//	{
//		source(NPART);
//		cout << "simulated " << totalWeight << " particles." << endl;
//		return false;
//	}
}




//      run simulations from phase space file
void simulateParticles(Particle particle,BeamData beamdata,int NRepeat)
{
//      if enough number of particles are simulated
    bool ifenough_load = false;
    bool ifenough = false;
    int idSource = 0;
    int idRepeat = 0;

    clrStk();

//	loop until all stacks are empty and no particles from ps file
    for(;;)
    {
//      obtain stack status
        int nestk_h = 0;

        if( cudaMemcpyFromSymbol(&nestk_h, nestk, sizeof(int), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "error in getting nestk" << endl;
        cout << "nestk =" << nestk_h << endl;

//      break if all particles simulated
        if( nestk_h == 0 && ifenough == true)
        {
            break;
            // cout << "1" << endl;
        }

        if(ifenough == true && nestk_h > 0)
        {

            loadFromStack(-1, nestk_h);
            //  cout << "2" << endl;
        }
        else
        {
            if(nestk_h >= NPART)
            {
                loadFromStack(-1, nestk_h);
                //     cout << "3" << endl;
            }
            else
            {
//      generate source particles from ps file
                int first, last;
                if(idSource == 0 && idRepeat == 0)
                {

                    first = bufferHeadId2;
                    last = first + NPART -1;
                    if(last > particle.NParticle-1)
                    {
                        last = particle.NParticle-1;
                        ifenough_load = true;
                    }

                    nparload_h = last - first + 1;
                    bufferHeadId2 = last;
                }

                //if(idRepeat == 0)
                //{

                cout << "Rotating " << nparload_h << " particles for Source" << idSource<< "from particle"<< first<< "to particle"<<last<<endl;
                int nblocks = 1 + (nparload_h- 1)/NTHREAD_PER_BLOCK_GBRACHY ;
                //Relocate_Rotate_PhapParticle<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(nparload_h,idSource,first,NRepeat,idRepeat,x_phap_gBrachy,vx_phap_gBrachy);
                cudaThreadSynchronize();

                //}

                cout << "Simulating " << nparload_h << " particles for repeat time" << idRepeat<<endl;

                idRepeat += 1;
                if(idRepeat == NRepeat)
                {
                    idRepeat = 0;
                    idSource +=1;
                    if(idSource == beamdata.numSource)
                        idSource = 0;
                }
                if(ifenough_load == true && idSource == 0 && idRepeat ==0 )
                    ifenough = true;

                nactive_h = nparload_h;
                ptype_h = 0;

                totalSimPar += nactive_h;

                float *tempaddress;

                if(cudaGetSymbolAddress((void**) &tempaddress,x_gBrachy) != cudaSuccess)
                    cout	<< "error in getting symbol address while computing weights" << endl;
                totalWeight_gBrachy += cublasSasum(nactive_h, tempaddress+3, 4);

                cout << "totalWeight="<<totalWeight_gBrachy << endl;
                cout << "totalSimPar="<< totalSimPar<< endl;
                //    cout << "4" << endl;
            }
        }

//                int nactive_h = 0;
//                if( cudaMemcpyFromSymbol(&nactive_h, nactive, sizeof(int),
//			0, cudaMemcpyDeviceToHost) != cudaSuccess)
//                        cout << "error in getting nactive" << endl;
        cout << " Number of active particles: " << nactive_h << endl;


//                int ptype_h = 0;
//                if( cudaMemcpyFromSymbol(&ptype_h, ptype, sizeof(int),
//			0, cudaMemcpyDeviceToHost) != cudaSuccess)
//                        cout << "error in getting ptype" << endl;
        cout << "Particle type: " << ptype_h << endl;

//      simulate a batch particles
        if (ptype_h == 0 && nactive_h>0)
        {
            int nblocks = 1 + (nactive_h - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
            photon<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(escore, nactive_h);
            cudaThreadSynchronize();

//                        int temp = 0;
//                        cudaMemcpyToSymbol(nactive, &temp, sizeof(int),
//				0, cudaMemcpyHostToDevice);
            nactive_h = 0;
        }
        else if (ptype_h != 0 && nactive_h>0)
        {
            int nblocks = 1 + (nactive_h - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
            // electr<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(escore, nactive_h);
            cudaThreadSynchronize();

//                        int temp = 0;
//                        cudaMemcpyToSymbol(nactive, &temp, sizeof(int),
//				0, cudaMemcpyHostToDevice);
            nactive_h = 0;
            cout << " electron transport" <<endl;
        }
    }

    /*  float *escore_temp = new float[NXYZ*NBATCH];
                 cudaMemcpy(escore_temp, escore, NXYZ*sizeof(float)*NBATCH, cudaMemcpyDeviceToHost) ;

                 FILE *fp;
                 fp = fopen("escore.dat", "w");
                 fwrite(escore_temp, NXYZ*NBATCH*sizeof(float), 1 , fp );
                 fclose(fp);

                 delete[] escore_temp; */
}

//      run simulations from source model
void simulateParticles(long long NParticle)
{
//  if enough number of particles are simulated
    bool ifenough = false;

    //      number of particle simulated
    long long nsimu = 0;

    clrStk();

//	loop until all stacks are empty and no particles from ps file
    while(1)
    {
        int nsstk_h = 0;
        if( cudaMemcpyFromSymbol(&nsstk_h, nsstk, sizeof(int), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "error in getting nsstk" << endl;
        cout<< "particle number    " << nsstk_h << endl;
        /*if(nsstk_h >= NSSTACK)
        {*/
        void *tempData;
        cudaMalloc( (void **) &tempData, sizeof(float)*nsstk_h);
        if( cudaMemcpyFromSymbol(tempData, sf, sizeof(float)*nsstk_h, 0,
                                 cudaMemcpyDeviceToDevice) != cudaSuccess)
            cout << "error in getting sf " << endl;

        outputData(tempData,sizeof(float)*(nsstk_h), "phaseSpace.dat", "ab");
        cudaFree(tempData);
        int temp = 0;
        if( cudaMemcpyToSymbol(nsstk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice)
                != cudaSuccess)
            cout << "error in setting nsstk" << endl;
        //}
//      obtain stack status
        int nestk_h = 0;

        if( cudaMemcpyFromSymbol(&nestk_h, nestk, sizeof(int),
                                 0, cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "error in getting nestk 1" << endl;
        //   cout << "nestk =" << nestk_h << endl;

//      break if all particles simulated
        if( nestk_h == 0 && ifenough == true)
        {
            break;
            //   cout << "1" << endl;
        }
        if(ifenough == true && nestk_h > 0)
        {

            loadFromStack(-1, nestk_h);
            //   cout << "2" << endl;
        }
        else
        {
            if(nestk_h >= NPART)
            {
                loadFromStack(-1, nestk_h);
                //     cout << "3" << endl;
            }
            else
            {
//      generate source particles from ps file
                if(NParticle - nsimu >= NPART)
                {
                    source(NPART);
                    nsimu += NPART;
                    //     cout << "4" << endl;
                }
                else
                {
                    source(NParticle-nsimu);
                    nsimu = NParticle;
                    //  cout << "5" << endl;
                }
                if(nsimu >= NParticle)
                    ifenough = true;
            }
        }

        cout << " Number of active particles: " << nactive_h << endl;
        // cout << "Particle type: " << ptype_h << endl;

//      simulate a batch particles
        if (ptype_h == 0 && nactive_h>0)
        {
            // cout << " photon transport" <<endl;
            int nblocks = 1 + (nactive_h - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
            photon<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(escore, nactive_h);
            cudaThreadSynchronize();
            nactive_h = 0;
        }
        else if (ptype_h != 0 && nactive_h>0)
        {
            //  cout << " electron transport" <<endl;
            int nblocks = 1 + (nactive_h - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
            //                     electr<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(escore, nactive_h);
            cudaThreadSynchronize();
            nactive_h = 0;
        }
    }

    int nsstk_h = 0;
    if( cudaMemcpyFromSymbol(&nsstk_h, nsstk, sizeof(int), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "error in getting nsstk" << endl;
    //cout<< "particle number    " << nsstk_h << endl;

    int size=sizeof(float)*(nsstk_h);
    void *tempData;
    cudaMalloc( (void **) &tempData, size);

    if( cudaMemcpyFromSymbol(tempData, sf, size, 0,
                             cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "error in getting sf " << endl;

    outputData(tempData,size, "phaseSpace.dat", "ab");
    cudaFree(tempData);
    int temp = 0;
    if( cudaMemcpyToSymbol(nsstk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice)
            != cudaSuccess)
        cout << "error in setting nsstk" << endl;
}

void loadFromStack(int itype, int nstk)
//    Retrieves particles from the secondary stack
{
    int nblocks;

    if(itype == -1)
    {
        nblocks = 1 + (NPART - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
        getEStkPcl<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>();
        cudaThreadSynchronize();

        int temp = 0;
        temp = nstk - NPART;
        temp = (temp<0)? 0 : temp;

        if( cudaMemcpyToSymbol(nestk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
            cout << "error in setting nestk" << endl;


        temp = nstk - temp;
        nactive_h = temp;
        ptype_h = -1;

        /* if( cudaMemcpyToSymbol(nactive, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
                        cout << "error in setting nactive" << endl;
        temp = -1;
        if( cudaMemcpyToSymbol(ptype, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
                        cout << "error in setting ptype" << endl; */
    }

}


__global__ void getEStkPcl()
//      move electron from stack to simulation space
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
//      obtain current id on thread

    if( tid < NPART )
    {
        int istart = nestk - NPART;
        istart = (istart < 0)? 0 : istart;
        int iend = nestk-1;
        int ind = tid + istart;

        if(ind <= iend)
        {
            vx_gBrachy[tid] = esvx[ind];
            x_gBrachy[tid] = esx[ind];
        }
    }
}

/****************************************************
        run the gCTD simulation
****************************************************/
void clrStk()
//      clear both particle stacks
{
    int temp = 0;
    if( cudaMemcpyToSymbol(nestk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
        cout << "error in setting nestk" << endl;
    /*  if( cudaMemcpyToSymbol(npstk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice) != cudaSuccess)
             cout << "error in setting npstk" << endl; */
}

void source(int num)
/*******************************************************************
c*    Creates new particle state                           *
c*                                                                 *
c*    Output:                                                      *
c*      ptype -> -1 if it is an electron, 0 when it is a photon    *
c*      energy -> kinetic energy                                   *
c*      {vx,vy,vz} -> direction of flight                          *
c*      {x,y,z} -> position                                        *
c*      vox -> voxel#                                              *
c*    Comments:                                                    *
c*      -> It is this routine's responsability to make sure that   *
c*         all dynamic variables are assigned valid values; in     *
c*         particular, kinetic energy must be in the interval      *
c*         (Emin,Emax) defined by the material data generated with *
c*         predpm.                                                 *
c******************************************************************/
{
    int nblocks;
    nblocks = 1 + (num - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
    setSource<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>(num);

    cudaThreadSynchronize();

    nactive_h = 2.0f*num;
    ptype_h = 0;

    float *tempaddress;
    if(cudaGetSymbolAddress((void**) &tempaddress,x_gBrachy) != cudaSuccess)
        cout	<< "error in getting symbol address while computing weights" << endl;
    totalWeight_gBrachy += cublasSasum(num, tempaddress+3, 4);
    totalSimPar += nactive_h;
    cout << "totalWeight="<<totalWeight_gBrachy << endl;
    cout << "totalSimPar="<< totalSimPar<< endl;
}


__global__ void setSource(int num)
//      set source direction, num means total_weight
{
    const int tid = blockIdx.x*blockDim.x + threadIdx.x;
    const float ene=511.0f;
    const float C11R=0.9979f; // no need, use total contant decay rate
    const float ThalfC10=19.290f; // half life in second
    const float ThalfC11=1220.04f;
    const float ThalfN13=598.2f; // half life in second
    const float ThalfO15=122.24f;
//  obtain current id on thread
//  which voxel, which isotope, which decay chanel, which direction
    if (tid < num)
    {
        // set which voxel
        curandState localState = cuseed[tid];
        if (tid==0)
        {

            printf("localState=cuseed[tid]= %d\n", localState);
        }
        float num = curand_uniform(&localState);
        if (tid==0)
        {
            printf("cuseed[tid]=num= %f\n", num);
        }
        int ind;
        ind = binarySearch(num);

        // set which isotope and decay time
        float pro = curand_uniform(&localState);
        float t1=C10Weight_source[ind];
        float t2=C11Weight_source[ind];
        float t3=N13Weight_source[ind];
        float t4=O15Weight_source[ind];
        float i_sum=1.0f/(t1+t2+t3+t4);
        if (tid==0)
        {

            // printf("x_par = %f,y_par = %f, z_par = %f\n", x_par,y_par,z_par);
            // printf("vx_par = %f,vy_par = %f, vz_par = %f,energy = %f\n", vx_par,vy_par,vz_par,energy);
            // printf(" idSource= %d,R11_source = %f, R12_source=%f,R21_source=%f,R22_source=%f,dirXY_source=%f\n",idSource, R11_source[idSource],R12_source[idSource],R21_source[idSource],R22_source[idSource],dirXY_source[idSource]);
            printf("weighting: ind=%d, t1 = %f,t2 = %f, t3 = %f, t4 = %f\n", ind,t1,t2,t3,t4);
            printf("position: x = %f,y = %f, z = %f\n", X0_source[ind],Y0_source[ind],Z0_source[ind]);
            printf("delta: x = %f,y = %f, z = %f\n", dx_gBrachy,dy_gBrachy,dz_gBrachy);
        }


        float proDecay = curand_uniform(&localState);
        float Tdecay=0.0f;
        Tdecay=-__logf(proDecay)/__logf(2.0f)*ThalfC10; // only simulate C10
        // four isotopes
        /*if(pro<=t1*i_sum) // to simulate C10 particle
        {
        	Tdecay=-__logf(proDecay)/__logf(2.0f)*ThalfC10;

        }
        else if (pro<=(t1+t2)*i_sum) // to simulate C11 decay
        {
        	Tdecay=-__logf(proDecay)/__logf(2.0f)*ThalfC11;
        }
        else if (pro<=(t1+t2+t3)*i_sum) // to simulate C11 decay
        {
        	Tdecay=-__logf(proDecay)/__logf(2.0f)*ThalfN13;
        }
        else
        	Tdecay=-__logf(proDecay)/__logf(2.0f)*ThalfO15;*/

//      set location
        float prox = curand_uniform(&localState);
        float proy = curand_uniform(&localState);
        float proz = curand_uniform(&localState);
        float ta=0.0f;
        float tb=0.0f;

        // suppose PEN is at the iso-center
        x_gBrachy[2*(tid-1)+1] = make_float4(0.0f,0.0f,0.0f,Tdecay); // cm
        x_gBrachy[2*tid] = make_float4(0.0f,0.0f,0.0f,Tdecay);

        // according to real readin files

        /*x_gBrachy[2*(tid-1)+1] = make_float4((X0_source[ind]+prox)*dx_gBrachy+ta,(Y0_source[ind]+proy)*dy_gBrachy+ta,(Z0_source[ind]+proz)*dz_gBrachy+tb,Tdecay);
        x_gBrachy[2*tid] = make_float4((X0_source[ind]+prox)*dx_gBrachy+ta,(Y0_source[ind]+proy)*dy_gBrachy+ta,(Z0_source[ind]+proz)*dz_gBrachy+tb,Tdecay);
        */
//      set direction for the first photon
        float theta=curand_uniform(&localState)*TWOPI;
        float u= 2*curand_uniform(&localState)-1;
        float4 vtem;
        vtem.x = __fsqrt_rn(1-u*u)*cosf(theta);
        vtem.y = __fsqrt_rn(1-u*u)*sinf(theta);
        vtem.z = u;

        vx_gBrachy[2*(tid-1)+1] = make_float4(vtem.x,vtem.y,vtem.z,ene*1000.0f);
//		set direction for the second photon
        float mean=0.0f;
        float sigma=0.00f;
        theta=box_muller1(mean, sigma, &localState);  // polar angle
        float phi=curand_uniform(&localState)*TWOPI;  //azimusal angle
        float costhe=__cosf(theta);
        vtem.x=-vtem.x;
        vtem.y=-vtem.y;
        vtem.z=-vtem.z;
        //rotate(&vtem.x,&vtem.y,&vtem.z,costhe,phi);
        vx_gBrachy[2*tid] = make_float4(vtem.x,vtem.y,vtem.z,ene*1000.0f);
        /*if(tid ==0)
        {

            // printf("x_par = %f,y_par = %f, z_par = %f\n", x_par,y_par,z_par);
            // printf("vx_par = %f,vy_par = %f, vz_par = %f,energy = %f\n", vx_par,vy_par,vz_par,energy);
            // printf(" idSource= %d,R11_source = %f, R12_source=%f,R21_source=%f,R22_source=%f,dirXY_source=%f\n",idSource, R11_source[idSource],R12_source[idSource],R21_source[idSource],R22_source[idSource],dirXY_source[idSource]);
            printf("first photon: x = %f,y = %f, z = %f, time = %f\n", x_gBrachy[2*(tid-1)+1].x,x_gBrachy[2*(tid-1)+1].y,x_gBrachy[2*(tid-1)+1].z,x_gBrachy[2*(tid-1)+1].w);
            printf("second photon: x = %f,y = %f, z = %f, time = %f\n", x_gBrachy[2*tid].x,x_gBrachy[2*tid].y,x_gBrachy[2*tid].z,x_gBrachy[2*tid].w);
            printf("first photon: vx = %f,vy = %f, vz = %f, energy=%f\n", vx_gBrachy[2*(tid-1)+1].x,vx_gBrachy[2*(tid-1)+1].y,vx_gBrachy[2*(tid-1)+1].z,vx_gBrachy[2*(tid-1)+1].w);
            printf("second photon: vx = %f,vy = %f, vz = %f, energy= %f\n", vx_gBrachy[2*tid].x,vx_gBrachy[2*tid].y,vx_gBrachy[2*tid].z,vx_gBrachy[2*tid].w);
        }*/
        //		if(tid ==1 )
        // printf("tid=1: curand_uniform(&localState)=%f,localState=%f\n", curand_uniform(&localState), localState);
        cuseed[tid] = localState;
    }
}
__device__ int binarySearch(float num)
{
//bisection method to locate the source that index "num"
    int left = 1;
    int right = NSource;
    int flag = 0;
    int ind=0;
    int mid=100;
    while (left <= right)
    {

        mid = ceilf((left + right) / 2);

        if (TotalWeight_source[mid]== num)
        {
            ind = mid;
            flag = 1;
            break;
        }
        else if (TotalWeight_source[mid]> num)
            right = mid - 1;
        else
            left = mid + 1;
    }

    if (flag == 0)
    {
        if(TotalWeight_source[mid]> num)
            ind = mid;
        else
            ind=mid+1;
    }
    return ind;
}

__device__ float box_muller1(float m, float s, curandState *localState_pt)	/* normal random variate generator */
{   /* mean m, standard deviation s, energy call */
    float x1, x2, w, y1;
    while(1)
    {
        x1 = 2.0 * curand_uniform(localState_pt) - 1.0;
        x2 = 2.0 * curand_uniform(localState_pt) - 1.0;
        w = x1 * x1 + x2 * x2;
        if (w<1.0f)
            break;
    }
    w = sqrtf( (-2.0 * __logf( w ) ) / w );
    y1 = x1 * w;

    return (m + y1 * s);
}
__device__ float getDistance(float4 coords, float4 direcs)
/*******************************************************************
c*   get the distance to the recording plane  		   *
c*                                                                 *
c*                                                                 *
c*    Input:                                                       *
c*      coords -> particle position and weight                     *
c*      i -> the ith object in the object file                     *
c*    Output:                                                      *
c*      distance-> nearest distance to current body boundaries     *
c******************************************************************/
{
    float pa[10]= {0.0f};

    // allocate a block of memory and initialize it with infi, each boundary has as much as two roots;
    float root[2];
    float infi=1.0e4f;
    float wup=1.0e-5f;
    int count=0;
    for(int i=0; i<2; i++)
    {
        root[i]=infi;
    }

    //	cout << coords[0] << endl;
    pa[0]=1.0f;
    pa[1]=1.0f;
    pa[2]=0.0f;
    pa[3]=0.0f;
    pa[4]=0.0f;
    pa[5]=0.0f;
    pa[6]=-10.1f;//0.0f;
    pa[7]=-10.1f;//0.0f;
    pa[8]=0.0f;
    pa[9]=-48.995f;//-102.01f;


    // parameters for quandric surface function a*s*s + b*s + c=0.
    float a = pa[0] * direcs.x * direcs.x + pa[1] * direcs.y * direcs.y + pa[2] * direcs.z * direcs.z +
              pa[3] * direcs.x * direcs.y + pa[4] * direcs.y * direcs.z + pa[5] * direcs.z * direcs.x ;
    float b = 2.0f*pa[0] * direcs.x * coords.x + 2.0f*pa[1] * direcs.y * coords.y + 2.0f*pa[2] * direcs.z * coords.z +
              pa[3] * (direcs.x * coords.y + direcs.y * coords.x) + pa[4] * (direcs.y * coords.z + direcs.z*coords.y) +
              pa[5] * (direcs.z * coords.x + direcs.x*coords.z) + pa[6]*direcs.x + pa[7]*direcs.y + pa[8]*direcs.z;
    float c = pa[0] * coords.x * coords.x + pa[1] * coords.y * coords.y + pa[2] * coords.z * coords.z +
              pa[3] * coords.x * coords.y + pa[4] * coords.y * coords.z + pa[5] * coords.z * coords.x +
              pa[6] * coords.x + pa[7] * coords.y + pa[8] * coords.z + pa[9];
    if (fabs(a)<wup)
    {
        if (fabs(b)>wup && (c/b)<-wup)
        {
            root[1]=-c/b;

        }
        else
            root[1]=infi;
    }
    else
    {
        float delta = b*b - 4.0f*a*c;
        float wupsilon=1e-7f;
        float t1=-b/(2.0f*a)-sqrtf(delta)/(2.0f*fabs(a));
        float t2=-b/(2.0f*a)+sqrtf(delta)/(2.0f*fabs(a));

        root[0]=t1;
        root[1]=t2;
    }
    return root[1];
}

#endif
