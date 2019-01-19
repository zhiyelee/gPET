#ifndef __INITIALIZE_CU__
#define __INITIALIZE_CU__


void init(Patient patient, BeamData sourceinfo, int ifDoseToWater_h, int deviceNo)
/*******************************************************************
c*    Initializes the gCTD system                                  *
c******************************************************************/
{
//      mark the start time    
        time_t start_time, end_time;
	    float time_diff;
	    
	    start_time = clock();

        cudaSetDevice(deviceNo);
		
		cudaDeviceReset();
		
        printDevProp(deviceNo);
        
//      initialize cuda printf
	    cudaPrintfInit();

        printf(" \n");
        printf("init: Reading simulation setup from STDIN;\n");
        printf("      information from this stream follows:\n\n");
       
        initpatient(patient);
		initSourceInfo(sourceinfo);
		cudaMemcpyToSymbol(ifDoseToWater_brachy, &ifDoseToWater_h, sizeof(int), 0, cudaMemcpyHostToDevice);
        
		FILE *fp = fopen("./patientcase/PET.in","r");		
		char buffer[200];

//	max number of histories (removed)
//	    gets(buffer);
//        printf("%s\n", buffer);
//        scanf("%Ld\n", &maxhis_h);
//        printf("%Ld\n", maxhis_h);


//	absorption energy
	    fgets(buffer,200,fp);
        printf("%s\n", buffer);
        fscanf(fp,"%f\n", &eabsph_h);
        printf("%e\n", eabsph_h);
        cudaMemcpyToSymbol(eabsph, &eabsph_h, sizeof(float), 0, cudaMemcpyHostToDevice);

//	initialize number of histories simulated
	   
	
//      in GPU, initialize rand seed with rand numbers
        inirngG();
        printf("\n");

	   char prefix[40], fname[50], fname2[50];
        int len;

//      Read the prefix with the name of the data files, truncate it:
        fgets(buffer,200,fp);
        printf("%s\n", buffer);
        getnam(fp,5, prefix, &len);
        printf("%s\n", prefix);

//      Read material data
        strcpy(fname, prefix);
        strcat(fname,".matter");
	   float emax, eminph;
        rmater(fname, &eminph, &emax);

        printf("\n");
        printf("\n");
        if(eabsph_h <eminph)
        {
                printf("init:error: Eabs out of range.\n");
                exit(1);
        }

//	laod total cross section
        strcpy(fname, prefix);
        strcat(fname,".lamph");
        rlamph(fname);

//	load compton cross section
        strcpy(fname, prefix);
        strcat(fname,".compt");
        rcompt(fname);
		strcpy(fname, prefix);
        strcat(fname,".cmpsf");
        rcmpsf(fname);

//	load photoelectric cross section
        strcpy(fname, prefix);
        strcat(fname,".phote");
        rphote(fname);

//      load rayleigh cross section and form factors
        strcpy(fname, prefix);
        strcat(fname,".rayle");
        rrayle(fname);
		strcpy(fname, prefix);
        strcat(fname,".rayff");
        rrayff(fname);


//      iniwck must be called after reading esrc & eabsph:
        iniwck(eminph, emax, patient);
		
		if(ifDoseToWater_h)
	{
	   strcpy(fname, prefix);
       strcat(fname,".mear");
	   rmear(fname);
	}	

	
	
	   // inisub();

	    

//      initialize the dose counter and dose output to be 0.
        clrStat();
        
//      print out
        printf("\n");
        printf("\n");
        printf("Initialize: Done.\n");
        
               
        end_time = clock();
        time_diff = ((float)end_time - (float)start_time)/1000.0;
        printf("\n\n****************************************\n");
        printf("Initializing time: %f ms.\n\n",time_diff);
        printf("****************************************\n\n\n");
        
    
}

void rdrsp(char fname[100])
//      load detector response curve
{
	char buffer[100];

	cout << endl;
    printf("rdrsp: Reading %s\n", fname);
    FILE *fp = fopen(fname,"r");
	
	fgets(buffer,100,fp);
    printf("%s", buffer);
    int ndata;
	fscanf(fp,"%d \n",&ndata);
	printf("        %d\n", ndata);

	fgets(buffer,100,fp);
	float *etemp = (float*)malloc(sizeof(float)*ndata);
	float *dersp_h = (float*)malloc(sizeof(float)*ndata);;
	for(int i = 0; i < ndata; i++)
	{
		fscanf(fp,"%f %f\n",&etemp[i],&dersp_h[i]);
//	convert to eV, rsp file is in keV
		etemp[i] *= 1000.0f;	
		dersp_h[i] *= 1000.0f;
		cout << etemp[i] << " " << dersp_h[i] << endl;
	}
    fclose(fp);

	idersp_h = (etemp[ndata-1] - etemp[0])/(ndata-1);
	idersp_h = 1.0f/idersp_h;
	cout << idersp_h << endl;
	edersp0_h = etemp[0];
	cout << edersp0_h << endl;

//	copy to GPU
	cudaMemcpyToSymbol(idersp, &idersp_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
	cudaMemcpyToSymbol(edersp0, &edersp0_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

	cudaMallocArray(&dersp, &dersp_tex.channelDesc, ndata, 1);
    cudaMemcpyToArray(dersp, 0, 0, dersp_h, sizeof(float)*ndata, cudaMemcpyHostToDevice);
    dersp_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(dersp_tex, dersp);
	
	free(etemp);
	free(dersp_h);
}









float itphip(int matid, float e)
/*******************************************************************
c*    Photon total inverse mean free path --3spline interpolation  *
c*                                                                 *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      Total inverse mean free path in cm^2/g                     *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
        int i;

        i = int(idleph_h*(e-elaph_h[0]));

        return  lampha_h[ind2To1(matid,i,MAXMAT,NLAPH)]
                        + e*(lamphb_h[ind2To1(matid,i,MAXMAT,NLAPH)]
                        + e*(lamphc_h[ind2To1(matid,i,MAXMAT,NLAPH)]
                        + e*lamphd_h[ind2To1(matid,i,MAXMAT,NLAPH)] ));
}




void iniwck(float eminph,float emax, Patient patient)
/*******************************************************************
c*    Finds information used to transport photons with the Woodcock*
c*    technique                                                    *
c*                                                                 *
c*    Input:                                                       *
c*      eminph -> minimum photon energy in data files (eV)         *
c*      emax -> maximum photon energy in data files (eV)           *
c*    Output                                                       *
c*      bytes -> space allocated for arrays                        *
c*    Comments:                                                    *
c*      -> common /dpmsrc/ must be loaded previously               *
c*      -> rlamph() must be called previously                      *
c*      -> rvoxg() must be called first                            *
c*      -> emax reduced to avoid reaching the end of interpol table*
c******************************************************************/
{
	float maxden[MAXMAT],de,e,ymax,ycanbe;
        const float eps = 1.0e-10F;

        printf("\n");
        printf("\n");
        printf("iniwck: Started.\n");
//      Find the largest density for each present material:
        for(int i = 0; i < MAXMAT; i++)
        {
                maxden[i] = 0.0F;
        }
        for(int vox = 0; vox < NXYZ; vox++)
        {
                if (patient.dens[vox] > maxden[patient.mat[vox]])
                        maxden[patient.mat[vox]] = patient.dens[vox];
        }

		
//      Prepare data:
        wcke0_h = eminph;
        de = (emax*(1.0F - eps ) - wcke0_h ) / NWCK;
        idlewk_h = 1.0F/de;

        for(int i = 0; i < NWCK; i++)
        {
                e = wcke0_h + de*i;
                ymax = 0.0;
                for(int j = 0; j < nmat_h; j++)
                {
                        ycanbe = itphip(j,e)*maxden[j];

                        if (ycanbe > ymax) ymax = ycanbe;
                }
                woock_h[i] = 1.0F/ymax;
        }
        FILE *fp = fopen("woock.dat", "w");
        fwrite(woock_h, NWCK*sizeof(float), 1 , fp );
        fclose(fp);

	   cudaMemcpyToSymbol(idlewk, &idlewk_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
       cudaMemcpyToSymbol(wcke0, &wcke0_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

        cudaMallocArray(&woock, &woock_tex.channelDesc, NWCK, 1);
        cudaMemcpyToArray(woock, 0, 0, woock_h, sizeof(float)*NWCK, cudaMemcpyHostToDevice);
        woock_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(woock_tex, woock);

}



//void rvoxg(char fname[100])
void initpatient(Patient patient)
/*******************************************************************
c*    Reads voxel geometry from an input file                      *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c*    Comments:                                                    *
c*      -> rmater must be called first to set nmat.                *
c******************************************************************/
{
	    NXYZ = patient.Unxvox*patient.Unyvox*patient.Unzvox;
      
        printf("CT dimension: %d %d %d\n", patient.Unxvox, patient.Unyvox, patient.Unzvox);
   
        printf("CT resolution: %f %f %f\n", patient.dx, patient.dy, patient.dz);
        
//      if(NXYZ>MAX_NXYZ)
//		{
//        	printf("CT dimension error: Too many voxels! Increase nxyz.\n");
//          exit(1);
//      }
        
        cudaMemcpyToSymbol(Unxvox, &patient.Unxvox, sizeof(int), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(Unyvox, &patient.Unyvox, sizeof(int), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(Unzvox, &patient.Unzvox, sizeof(int), 0, cudaMemcpyHostToDevice) ;

       
        cudaMemcpyToSymbol(dx_gBrachy, &patient.dx, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(dy_gBrachy, &patient.dy, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(dz_gBrachy, &patient.dz, sizeof(float), 0, cudaMemcpyHostToDevice) ;

	    idx_gBrachy_h = 1.0F/patient.dx;
        cudaMemcpyToSymbol(idx_gBrachy, &idx_gBrachy_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        idy_gBrachy_h = 1.0F/patient.dy;
        cudaMemcpyToSymbol(idy_gBrachy, &idy_gBrachy_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        idz_gBrachy_h = 1.0F/patient.dz;
        cudaMemcpyToSymbol(idz_gBrachy, &idz_gBrachy_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
		
		cudaMemcpyToSymbol(Offsetx_gBrachy, &patient.Offsetx, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(Offsety_gBrachy, &patient.Offsety, sizeof(float), 0, cudaMemcpyHostToDevice) ;
		cudaMemcpyToSymbol(Offsetz_gBrachy, &patient.Offsetz, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        
        
	    volumeSize = make_cudaExtent(patient.Unxvox, patient.Unyvox, patient.Unzvox);
        cudaMalloc3DArray(&mat, &mat_tex.channelDesc, volumeSize) ;
        cudaMalloc3DArray(&dens, &dens_tex.channelDesc, volumeSize);
        
//      create a 3d array on device

	copyParams.srcPtr   = make_cudaPitchedPtr((void*)patient.mat, volumeSize.width*sizeof(int), volumeSize.width, volumeSize.height);
        copyParams.dstArray = mat;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams) ;
//      copy data from host to device
        mat_tex.normalized = false;
        mat_tex.filterMode = cudaFilterModePoint;
        cudaBindTextureToArray(mat_tex, mat, mat_tex.channelDesc);
//      bind to texture memory

        copyParams.srcPtr   = make_cudaPitchedPtr((void*)patient.dens, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
        copyParams.dstArray = dens;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams) ;
//      copy data from host to device
        dens_tex.normalized = false;
        dens_tex.filterMode = cudaFilterModePoint;
        cudaBindTextureToArray(dens_tex, dens, dens_tex.channelDesc);
//      bind to texture memory

     
        
//// for debug
//        FILE *fp;
//	fp = fopen("dens.dat", "w");
//        fwrite(patient.dens, NXYZ*sizeof(float), 1 , fp );
//        fclose(fp);
//	fp = fopen("mat.dat", "w");
//        fwrite(patient.mat, NXYZ*sizeof(int), 1 , fp );
//        fclose(fp);

        printf("finish initializing patient texture\n");
}

void initSourceInfo(BeamData sourceinfo)
{
     if(sourceinfo.numSource>MAXNSOURCE)
	 {
	   printf("source number error: Too many sources! Increase MAXNSOURCE.\n");
                exit(1);
	 }
    
	 cudaMemcpyToSymbol(NSource, &sourceinfo.numSource, sizeof(int), 0, cudaMemcpyHostToDevice) ;
	 cudaMemcpyToSymbol(X0_source, sourceinfo.XSource, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
	 cudaMemcpyToSymbol(Y0_source, sourceinfo.YSource, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
	 cudaMemcpyToSymbol(Z0_source, sourceinfo.ZSource, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
	 cudaMemcpyToSymbol(TotalWeight_source, sourceinfo.TotalWeight, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
	 cudaMemcpyToSymbol(C10Weight_source, sourceinfo.C10Weight, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
	 cudaMemcpyToSymbol(C11Weight_source, sourceinfo.C11Weight, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
     cudaMemcpyToSymbol(N13Weight_source, sourceinfo.N13Weight, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
     cudaMemcpyToSymbol(O15Weight_source, sourceinfo.O15Weight, sizeof(float)*sourceinfo.numSource, 0,cudaMemcpyHostToDevice) ;
}


void rspec(char fname[40])
//	load spectrum data
{
	char buffer[100];
//        int ndata;

	cout << endl;
        printf("rspec: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
	
	fgets(buffer,100,fp);
        printf("%s", buffer);
	fscanf(fp,"%d \n",&nspecdata_h);
	printf("        %d\n", nspecdata_h);
	if(nspecdata_h > NSPEC)
	{
		printf("rspec:error: spectrum data too large:\n");
                printf("%d %d\n", nspecdata_h, NSPEC);
                exit(1);
	}

	fgets(buffer,100,fp);
	psum_h = 0;
	for(int i = 0; i < nspecdata_h; i++)
	{
		fscanf(fp,"%f %f\n",&espec_h[i],&pspec_h[i]);
//	convert to eV, spec file is in keV
		espec_h[i] *= 1000.0f;
		psum_h += pspec_h[i];
//		cout << espec_h[i] << " " << pspec_h[i] << endl;
	}
        fclose(fp);

	despec_h = (espec_h[nspecdata_h-1]-espec_h[0])/(nspecdata_h-1);



}






void rphote(char fname[40])
/*******************************************************************
c*    Reads photoelectric inverse mean free path data from file and*
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];
        int ndata;

        printf("rphote: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NPHTE)
                {
                        printf("rphote:error: Array dim do not match:\n");
                        printf("%d %d\n", ndata,NPHTE);
                        exit(1);
                }
                fgets(buffer,100,fp);
//      Preparing interpolation
                for(int i = 0; i < NPHTE; i++)
                {
                        fscanf(fp,"%f %f\n",&ephte_h[i],&phote_h[ind2To1(j,i,MAXMAT,NPHTE)]);
//                      if(j == nmat-1)
//                              printf("%e %e\n",ephte[i],phote[i]);
                }
                fgets(buffer,100,fp);
        }
        fclose(fp);

        idlepe_h = (NPHTE-1)/(ephte_h[NPHTE-1]-ephte_h[0]);
        cudaMemcpyToSymbol(idlepe, &idlepe_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(ephte0, &ephte_h[0], sizeof(float), 0, cudaMemcpyHostToDevice);

	cudaMallocArray(&phote, &phote_tex.channelDesc, NPHTE*MAXMAT, 1);
        cudaMemcpyToArray(phote, 0, 0, phote_h, sizeof(float)*NPHTE*MAXMAT, cudaMemcpyHostToDevice);
        phote_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(phote_tex, phote);

}



void rcmpsf(char fname[40])
/*******************************************************************
c*    Reads Compton scattering function data from file and         *
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];        

        printf("rcmpsf: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
//	read sf data
                fgets(buffer,100,fp);
                float temp;
		int ndata;
                fscanf(fp,"%d %f %f %f\n",&ndata,&temp,&temp,&temp);                
                fgets(buffer,100,fp);
                for(int i = 0; i < ndata; i++)
                {
                        fscanf(fp,"%f %f %f\n",&temp, &temp, &temp);
                }             

//	read s surface
		fgets(buffer,100,fp);				
		int ncp, ne;
		float dcp, de;
		fscanf(fp,"%d %f %f %f %d %f %f %f\n", &ncp, &temp, &temp, &dcp, &ne, &temp, &temp, &de);
		if (ncp != NCPCM)
                {
                        printf("rcmpsf:error: NCP dim do not match:\n");
                        printf("%d %d\n", ncp,NCPCM);
                        exit(1);
                }
		if (ne != NECM)
                {
                        printf("rcmpsf:error: NE dim do not match:\n");
                        printf("%d %d\n", ne,NECM);
                        exit(1);
                }
		idcpcm_h = 1.0f/dcp;
		idecm_h = 1.0f/de;
		for(int icp=0; icp <ncp; icp++)
			fscanf(fp,"%f ",&temp);
		fscanf(fp,"\n");
		for(int ie=0; ie <ne; ie++)
			fscanf(fp,"%f ",&temp);
		fscanf(fp,"\n");
		for(int icp=0; icp <ncp; icp++)
		{
			for(int ie = 0; ie<ne; ie++)
			{
				fscanf(fp,"%f ",&mucmpt_h[j*NCPCM*NECM+icp*NECM+ie]);
//				if(mucmpt_h[j*NCPCM*NECM+icp*NECM+ie] > 1.0f || mucmpt_h[j*NCPCM*NECM+icp*NECM+ie]<-1.0f)
//					cout << "error in data" << mucmpt_h[j*NCPCM*NECM+icp*NECM+ie] << endl;
			}
			fscanf(fp,"\n");					
		}
		fscanf(fp,"\n");
        }
        fclose(fp);

//	load to GPU
	cudaMemcpyToSymbol(idcpcm, &idcpcm_h, sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(idecm, &idecm_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
		
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	const cudaExtent volumeSize = make_cudaExtent(NECM, NCPCM, MAXMAT);
		
	cudaMalloc3DArray(&sArray, &channelDesc, volumeSize) ;
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr((void*)mucmpt_h, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
        copyParams.dstArray = sArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams) ;
	
	s_tex.normalized = false;
        s_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(s_tex, sArray, channelDesc);
}





void rrayff(char fname[40])
/*******************************************************************
c*    Reads Rayleigh scattering form factor data from file and     *
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];        

        printf("rrayff: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
//	read ff data
                fgets(buffer,100,fp);
                float temp;
		int ndata;
                fscanf(fp,"%d %f %f %f\n",&ndata,&temp,&temp,&temp);                
                fgets(buffer,100,fp);
                for(int i = 0; i < ndata; i++)
                {
                        fscanf(fp,"%f %f  %f\n",&temp, &temp, &temp);
                }             

//	read f surface
		fgets(buffer,100,fp);				
		int ncp, ne;
		float dcp, de;
		fscanf(fp,"%d %f %f %f %d %f %f %f\n", &ncp, &temp, &temp, &dcp, &ne, &temp, &temp, &de);
		if (ncp != NCPRL)
                {
                        printf("rrayff:error: NCP dim do not match:\n");
                        printf("%d %d\n", ncp,NCPRL);
                        exit(1);
                }
		if (ne != NERL)
                {
                        printf("rrayff:error: NE dim do not match:\n");
                        printf("%d %d\n", ne,NERL);
                        exit(1);
                }
		idcprl_h = 1.0f/dcp;
		iderl_h = 1.0f/de;
		for(int icp=0; icp <ncp; icp++)
			fscanf(fp,"%f ",&temp);
		fscanf(fp,"\n");
		for(int ie=0; ie <ne; ie++)
			fscanf(fp,"%f ",&temp);
		fscanf(fp,"\n");
		for(int icp=0; icp <ncp; icp++)
		{
			for(int ie = 0; ie<ne; ie++)
			{
				fscanf(fp,"%f ",&murayl_h[j*NCPRL*NERL+icp*NERL+ie]);
//                              	if(murayl_h[j*NCPRL*NERL+icp*NERL+ie] > 1.0f || murayl_h[j*NCPRL*NERL+icp*NERL+ie]<-1.0f)
//                                      cout << "error in data" << murayl_h[j*NCPRL*NERL+icp*NERL+ie] << endl;
			}
			fscanf(fp,"\n");					
		}
		fscanf(fp,"\n");
//		cout << murayl_h[j*NCPRL*NERL+(NCPRL-2)*NERL+1] << endl;
        }
        fclose(fp);

//	load to GPU
	cudaMemcpyToSymbol(idcprl, &idcprl_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(iderl, &iderl_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
		
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	const cudaExtent volumeSize = make_cudaExtent(NERL, NCPRL, MAXMAT);
		
	cudaMalloc3DArray(&fArray, &channelDesc, volumeSize) ;
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr((void*)murayl_h, volumeSize.width*sizeof(float), volumeSize.width, volumeSize.height);
        copyParams.dstArray = fArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
	
	f_tex.normalized = false;
        f_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(f_tex, fArray, channelDesc);
}




void rrayle(char fname[40])
/*******************************************************************
c*    Reads rayleigh inverse mean free path data from file and     *
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];
        int ndata;

        printf("rrayle: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NRAYL)
                {
                        printf("rrayle:error: Array dim do not match:\n");
                        printf("%d %d\n", ndata,NRAYL);
                        exit(1);
                }
                fgets(buffer,100,fp);
//      Preparing interpolation
                for(int i = 0; i < NRAYL; i++)
                {
                        fscanf(fp,"%f %f\n",&erayl_h[i],&rayle_h[ind2To1(j,i,MAXMAT,NRAYL)]);
                }
                fgets(buffer,100,fp);
        }
        fclose(fp);

        idlerl_h = (NRAYL-1)/(erayl_h[NRAYL-1]-erayl_h[0]);
        cudaMemcpyToSymbol(idlerl, &idlerl_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(erayl0, &erayl_h[0], sizeof(float), 0, cudaMemcpyHostToDevice) ;

        cudaMallocArray(&rayle, &rayle_tex.channelDesc, NRAYL*MAXMAT, 1);
        cudaMemcpyToArray(rayle, 0, 0, rayle_h, sizeof(float)*NRAYL*MAXMAT, cudaMemcpyHostToDevice);
        rayle_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(rayle_tex, rayle);

}







void rcompt(char fname[40])
/*******************************************************************
c*    Reads Compton inverse mean free path data from file and sets *
c*    up interpolation matrices                                    *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];
        int ndata;

        printf("rcompt: Reading %s\n", fname);
        FILE *fp = fopen(fname, "r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NCMPT)
                {
                        printf("rcompt:error: Array dim do not match:\n");
                        printf("%d %d \n", ndata,NCMPT);
                        exit(1);
                }
                fgets(buffer,100,fp);
//      Preparing interpolation:
                for(int i = 0; i <NCMPT; i++)
                {
                        fscanf(fp,"%f %f\n",&ecmpt_h[i],&compt_h[ind2To1(j,i,MAXMAT,NCMPT)]);
//                      if(j == nmat-1)
//                              printf("%e %e\n",ecmpt[i],compt[i]);
                }
                fgets(buffer,100,fp);

        }
        fclose(fp);

        idlecp_h = (NCMPT-1)/(ecmpt_h[NCMPT-1]-ecmpt_h[0]);
        cudaMemcpyToSymbol(idlecp, &idlecp_h, sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(ecmpt0, &ecmpt_h[0], sizeof(float), 0, cudaMemcpyHostToDevice) ;

	cudaMallocArray(&compt, &compt_tex.channelDesc, NCMPT*MAXMAT, 1);
        cudaMemcpyToArray(compt, 0, 0, compt_h, sizeof(float)*NCMPT*MAXMAT, cudaMemcpyHostToDevice);
        compt_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(compt_tex, compt);
}








void rlamph(char fname[40])
/*******************************************************************
c*    Reads photon total inverse mean free path data from file and *
c*    sets up interpolation matrices                               *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];
        int ndata;
        float dummya[NLAPH],dummyb[NLAPH],dummyc[NLAPH],dummyd[NLAPH];

        printf("rlamph: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NLAPH)
                {
                        printf("rlamph:error: Array dim do not match:\n");
                        printf("%d %d\n", ndata,NLAPH);
                        exit(1);
                }
                fgets(buffer,100,fp);
//      Preparing interpolation:
                for(int i = 0;i < NLAPH; i++)
                {
                        fscanf(fp,"%f %f\n",&elaph_h[i],&lamph_h[ind2To1(j,i,MAXMAT,NLAPH)]);
                      if(i<100)
                             printf("%e %e\n",elaph_h[i],lamph_h[ind2To1(j,i,MAXMAT,NLAPH)]);
                }
                fgets(buffer,100,fp);
                spline(elaph_h, &lamph_h[ind2To1(j,0,MAXMAT,NLAPH)],dummya,dummyb,dummyc,dummyd,0.0F,0.0F,NLAPH);
//      Loading dummy arrays into multimaterial sp matrices:
                for(int i = 0; i < NLAPH; i++)
                {
                        lampha_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummya[i];
                        lamphb_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummyb[i];
                        lamphc_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummyc[i];
                        lamphd_h[ind2To1(j,i,MAXMAT,NLAPH)] = dummyd[i];
                }
        }
        fclose(fp);

        idleph_h = (NLAPH-1)/(elaph_h[NLAPH-1]-elaph_h[0]);
        cudaMemcpyToSymbol(idleph, &idleph_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
        cudaMemcpyToSymbol(elaph0, &elaph_h[0], sizeof(float), 0, cudaMemcpyHostToDevice);

        cudaMallocArray(&lamph, &lamph_tex.channelDesc, NLAPH*MAXMAT, 1);
        cudaMemcpyToArray(lamph, 0, 0, lamph_h, sizeof(float)*NLAPH*MAXMAT, cudaMemcpyHostToDevice);
        lamph_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(lamph_tex, lamph);

}









void rmater(char fname[40], float *eminph, float *emax)
/*******************************************************************
c*    Reads material data from file                                *
c*                                                                 *
c*    Output:                                                      *
c*      fname -> input file name                                   *
c*      [Emin,Eminph,Emax] -> interval where data will be gen (eV) *
c*      refz -> total atomic no of the reference material          *
c*      refz2 -> atomic no^2 of the reference material             *
c*      refmas -> atomic weight of the reference material          *
c*      refden -> density of the reference material (g/cm^3)       *
c******************************************************************/
{
        char buffer[100];
        int len;
        float shigh,slow,ecross, temp,wcc,wcb;
        char mname[40];
        
		char watername[] = "Water\n";
		
        printf(" \n");
        printf(" \n");
        printf("rmater: Reading %s\n", fname);
        printf("        information from this file follows:\n");

        FILE *fp = fopen(fname,"r");
        fgets(buffer, 100, fp);
        fgets(buffer, 100, fp);
        fgets(buffer, 100, fp);
        printf("%s\n",buffer);

        fgets(buffer, 100, fp);
        printf("%s\n",buffer);
        fscanf(fp,"%f %f %f\n",eminph, &temp, emax);
        printf("%e %e %e\n",*eminph,temp, *emax);

        fgets(buffer, 100, fp);
        printf("%s\n",buffer);
        fscanf(fp,"%f %f\n",&wcc, &wcb);
        printf("%e %e\n",wcc,wcb);

        fgets(buffer, 100, fp);
        printf("%s\n",buffer);
        fscanf(fp,"%f %f %f\n",&shigh,&slow,&ecross);
        printf("%e %e %e\n",shigh,slow,ecross);

        fgets(buffer, 100, fp);
        printf("%s\n",buffer);
        fscanf(fp,"%d\n", &nmat_h);
        printf("%d\n",nmat_h);
        cudaMemcpyToSymbol(nmat, &nmat_h, sizeof(int), 0, cudaMemcpyHostToDevice) ;
        if (nmat_h > MAXMAT)
        {
                printf("rmater:error: Too many materials.\n");
                exit(1);
        }

        for(int i = 0; i < nmat_h; i++)
        {
//      Read name of material, remove trailing blanks:
		float matden;
		int nelem;
                getna2(fp, mname, &len);
                fgets(buffer, 100, fp);
                printf("MATERIAL: %s\n",mname);
                fscanf(fp,"%f\n", &matden);
                printf("%e\n", matden);
                fgets(buffer, 100, fp);
                printf("%s\n",buffer);
                fscanf(fp,"%d\n",&nelem);
                printf("%d\n", nelem);
                for(int j = 0; j < nelem; j++)
                {
                        fgets(buffer, 100, fp);
                        printf("%s\n",buffer);
                }
                fgets(buffer, 100, fp);
                printf("%s\n",buffer);
                float atnotemp,atno2temp;
                fscanf(fp,"%f %f %f\n",&atnotemp, &atno2temp, &temp);
                printf("%e %e\n", atnotemp,atno2temp);
                fgets(buffer, 100, fp);
                printf("%s\n",buffer);
		float mass;
                fscanf(fp,"%f\n", &mass);
                printf("%e\n", mass);
                fgets(buffer, 100, fp);
                printf("%s\n",buffer);
		float zmass,z2mass;
                fscanf(fp,"%f %f\n", &zmass,&z2mass);
                printf("%e %e\n\n\n", zmass,z2mass);
				
				if(strcmp(mname,watername)==0)
				{
				  matid_water_h= i; 
				  printf("material id for water is %d\n\n\n", matid_water_h);
				}
				
				matid_water_h = 0;
        }
        fclose(fp);

       
		
		cudaMemcpyToSymbol(matid_water, &matid_water_h, sizeof(int), 0, cudaMemcpyHostToDevice);
		
        printf(" \n");
        printf("rmater: Done.\n");
}



void getna2(FILE *iounit, char physics[40], int *n)
{
        char phyname[80];

        if(iounit == NULL)
        {
                scanf("%80s\n",phyname);
        }
        else
        {
                fgets(phyname, 80, iounit);
        }

        int istart;
        for(int i = 0; i < 80; i++)
        {
                if(phyname[i] == ':')
                {
                        istart = i + 2;
                        break;
                }
        }
//      find first ':'

        for(int i = istart; i < 80; i++)
        {
                int j = i - istart;
                if(phyname[i] != '\0')
                        physics[j] = phyname[i];
                else
                {
                        *n = i-istart;
                        physics[j] = '\0';
                        return;
                }
        }
}





void getnam(FILE *fp,int iounit,char physics[80], int *n)
//	get a file name from std
{
        char phyname[80];

        if(iounit == 5)
        {
                fscanf(fp,"%80s\n",phyname);
        }
        else
        {
                printf("ERROR, io from other unit has not been programed...\n");
                exit(1);
        }
        strcpy(physics,phyname);
        *n = (int)strlen(phyname);

        return;
}

void getnam2(int iounit,char physics[80], int *n)
{
    	char phyname[80];
 
    	if(iounit == 5)
	{
		scanf("%80s\n",phyname);                         
	}
    	else
	{
		printf("ERROR, io from other unit has not been programed...\n");
		exit(1);
	} 
	strcpy(physics,phyname);
	*n = (int)strlen(phyname);

	return;
}

void clrStat()
//	clean all dose counters for statistics
{
//	    float *tempe;
//        cudaMalloc( (void **) &tempe, sizeof(float)*NXYZ) ;
//        cudaMemset(tempe, 0, sizeof(float)*NXYZ) ;
//        if( cudaMemcpyToSymbol(fEscore, tempe, sizeof(tempe), 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
//                cout << "error in setting fEscore" << endl;
//	    if( cudaMemcpyToSymbol(fEscor2, tempe, sizeof(tempe), 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
//                cout << "error in setting fEscor2" << endl;
//        cudaFree(tempe) ;
        
//        float *tempe2;
//	    cudaMalloc( (void **) &tempe2, sizeof(float)*NXYZ*NBATCH);
//        cudaMemset(tempe2, 0, sizeof(float)*NXYZ*NBATCH) ;
//        if( cudaMemcpyToSymbol(escore, tempe2, sizeof(tempe2), 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
//                cout << "error in setting escore" << endl;
//        cudaFree(tempe2) ;
        
        cudaMalloc( (void **) &escore, sizeof(float)*NXYZ*NBATCH);
        cudaMemset(escore, 0, sizeof(float)*NXYZ*NBATCH) ;
		
        
        /* cudaMalloc( (void **) &fEscore, sizeof(float)*NXYZ);
        cudaMemset(fEscore, 0, sizeof(float)*NXYZ) ;
        
        cudaMalloc( (void **) &fEscor2, sizeof(float)*NXYZ);
        cudaMemset(fEscor2, 0,  sizeof(float)*NXYZ) ; */
		
		
}

void printDevProp(int device)
//      print out device properties
{
        int devCount;
        cudaDeviceProp devProp;
//      device properties

        cudaGetDeviceCount(&devCount);
	cout << "Number of device:              " << devCount << endl;
	cout << "Using device #:                " << device << endl;
        cudaGetDeviceProperties(&devProp, device);
	
	printf("Major revision number:         %d\n",  devProp.major);
    	printf("Minor revision number:         %d\n",  devProp.minor);
    	printf("Name:                          %s\n",  devProp.name);
    	printf("Total global memory:           %7.2f MB\n",  
		devProp.totalGlobalMem/1024.0/1024.0);
    	printf("Total shared memory per block: %5.2f kB\n",  
		devProp.sharedMemPerBlock/1024.0);
    	printf("Total registers per block:     %u\n",  devProp.regsPerBlock);
    	printf("Warp size:                     %d\n",  devProp.warpSize);
    	printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    	printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    	
	printf("Maximum dimension of block:    %d*%d*%d\n", 			
		devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
	printf("Maximum dimension of grid:     %d*%d*%d\n", 
		devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
    	printf("Clock rate:                    %4.2f GHz\n",  devProp.clockRate/1000000.0);
    	printf("Total constant memory:         %5.2f kB\n",  devProp.totalConstMem/1024.0);
    	printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    	printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    	printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    	printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
//      obtain computing resource


}

void restep(char *fname)
//	read scattering strength data
{
	char buffer[100];
        int ndata;
        float shigh,slow,ecross;

        printf("rstep: Reading %s\n",fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer, 100, fp);
        fgets(buffer, 100, fp);
        fgets(buffer, 100, fp);
        printf("rstep: \n   %s\n",buffer);
        fscanf(fp,"%f %f %f\n",&shigh,&slow,&ecross);
        printf("%e %e %e\n", shigh,slow,ecross);
        fgets(buffer, 100, fp);
        float temp;
        fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
        if (ndata != NSCSR)
        {
                printf("rstep:error: Array dim do not match:\n");
                printf("%d %d", ndata,NSCSR);
                exit(1);
        }
        fgets(buffer, 100, fp);

//      Prepare interpolation:
        for(unsigned int i = 0; i < NSCSR; i++)
        {
                fscanf(fp,"%f %f\n",&escsr_h[i],&scssp_h[i]);
//              printf("%e %e\n",escsr[i],scssp[i]);
        }
        fclose(fp);

        idless_h = (NSCSR-1)/(escsr_h[NSCSR-1]-escsr_h[0]);
        cudaMemcpyToSymbol(idless, &idless_h, sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(escsr0, &escsr_h[0], sizeof(float), 0, cudaMemcpyHostToDevice);

        cudaMallocArray(&scssp, &scssp_tex.channelDesc, NSCSR, 1);
        cudaMemcpyToArray(scssp, 0, 0, scssp_h, sizeof(float)*NSCSR, cudaMemcpyHostToDevice);
        scssp_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(scssp_tex, scssp);
}

void rerstpw(char *fname)
//	read stopping power restricted data
{
	char buffer[100];
        int ndata;

        printf("rerstpw: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < MAXMAT; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NST)
                {
                        printf("rerstpw:error: Array dim do not match:\n");
                        printf("%d %d\n", ndata,NST);
                        exit(1);
                }
                fgets(buffer,100,fp);
                for(int i = 0; i < NST; i++)
                {
                        fscanf(fp,"%f %f\n",&est_h[i],&stsp_h[ind2To1(j,i,MAXMAT,NST)]);
//                      if(j == nmat-1)
//                              printf("%e %e\n",est[i],stsp[i]);
                }
                fgets(buffer,100,fp);
        }
        fclose(fp);

        idlest_h = (NST-1)/(est_h[NST-1]-est_h[0]);
	cudaMemcpyToSymbol(idlest, &idlest_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(est0, &est_h[0], sizeof(float), 0,
                cudaMemcpyHostToDevice);

        cudaMallocArray(&stsp, &stsp_tex.channelDesc, NST*MAXMAT, 1);
        cudaMemcpyToArray(stsp, 0, 0, stsp_h, sizeof(float)*NST*MAXMAT, cudaMemcpyHostToDevice);
        stsp_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(stsp_tex, stsp);
}

void rescpw(char *fname)
/*******************************************************************
c*    Reads 1st TMFP from file and sets up interpolation matrices  *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c*    Output:                                                      *
c*      bytes -> memory filled up by interpolation arrays          *
c******************************************************************/
{
        char buffer[256];
        int ndata;

        printf("rscpw: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);

        for(int j = 0; j < MAXMAT; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NSCP)
                {
                        printf("rscpw:error: Array dim do not match:\n");
                        printf("%d %d\n", ndata,NSCP);
                        exit(1);
                }
                fgets(buffer,100,fp);
                //      Prepare interpolation:
                for(int i = 0; i < NSCP; i++)
                {
                        fscanf(fp,"%e %e\n",&escp_h[i],&scpsp_h[ind2To1(j,i,MAXMAT,NSCP)]);
                        //                      printf("%e %e\n",escp[i],scpsp[i]);
                }
                fgets(buffer,100,fp);
        }
        fclose(fp);

        idlesc_h = (NSCP-1)/(escp_h[NSCP-1]-escp_h[0]);
        cudaMemcpyToSymbol(idlesc, &idlesc_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(escp0, &escp_h[0], sizeof(float), 0,
                cudaMemcpyHostToDevice);

        cudaMallocArray(&scpsp, &scpsp_tex.channelDesc, NSCP*MAXMAT, 1);
	cudaMemcpyToArray(scpsp, 0, 0, scpsp_h, sizeof(float)*NSCP*MAXMAT,
                cudaMemcpyHostToDevice);
        scpsp_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(scpsp_tex, scpsp);
}

void reqsurf(char *fname)
/*******************************************************************
c*    Reads q surface data data from file and sets up interpolation*
c*    matrices                                                     *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c*    Output:                                                      *
c*      bytes -> memory filled up by interpolation arrays          *
c******************************************************************/
{
        char buffer[256];
        int nu,ne;
        float e0,e1,u,qmax,effic,qval;

        printf("rqsurf: Reading %s\n", fname);
        FILE *fp = fopen(fname,"r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);

        float temp;
        fscanf(fp,"%d %d %f %f %f %f %f\n",&nu,&ne,&temp,&temp,&temp,&temp,&temp);
        if ((nu != NUQ) || (ne !=NEQ))
        {
                printf("rqsurf:error: Array dim do not match:\n");
                printf("%d %d %d %d", nu,NUQ,ne,NEQ);
                exit(1);
        }
        fgets(buffer,100,fp);
        fscanf(fp,"%f %f\n",&qmax,&temp);
        //      printf("%e %e\n",qmax,temp);
        fgets(buffer,100,fp);
        effic = 1.0F/qmax;
        printf("rqsurf: q rejection efficiency(%):\n");
        printf("%e\n", 100.0F*effic);
        for(int i = 0; i < NEQ; i++)
        {
                fscanf(fp,"%f\n", &e1);
                //              printf("%e\n",e1);
                if (i == 0)
                {
                        e0 = e1;
                }
                for(int j = 0; j < NUQ; j++)
                {
                        fscanf(fp,"%e %e\n", &u,&qval);
                        //      Incorporating the efficiency in the q function:
                        q_h[ind2To1(j,i,NUQ,NEQ)] = qval*effic;
                }
        }
        fgets(buffer,100,fp);

        fclose(fp);

        le0q_h = -1.0F/e0;
        cudaMemcpyToSymbol(le0q, &le0q_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);

        idleq_h = (NEQ-1)/(-1.0F/e1-le0q_h);

        cudaMemcpyToSymbol(idleq, &idleq_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);

        iduq_h = NUQ-1.0F;
        cudaMemcpyToSymbol(iduq, &iduq_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);

        cudaMallocArray(&qArray, &channelDesc, NEQ, NUQ);
        cudaMemcpyToArray(qArray, 0,0, q_h, sizeof(float)*NUQ*NEQ,
                cudaMemcpyHostToDevice);
        q_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(q_tex, qArray, channelDesc);
        //      bind to texture memory

}

void rebw(char *fname)
/*******************************************************************
c*    Reads screening parameter data from file and sets up         *
c*    interpolation matrices                                       *
c*                                                                 *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c*    Output:                                                      *
c*      bytes -> memory filled up by interpolation arrays          *
c******************************************************************/
{
        char buffer[256];
        int ndata;

        printf("rbw: Reading %s\n", fname);
        FILE *fp = fopen(fname, "r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        float temp;
        fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
        fgets(buffer,100,fp);
        if (ndata != NBW)
        {
                printf("rbw:error: Array dim do not match:\n");
                printf("%d %d\n", ndata,NBW);
                exit(1);
        }
        //      Prepare interpolation:
        for(int i = 0; i < NBW; i++)
        {
                fscanf(fp,"%e %e\n",&ebw_h[i],&bwsp_h[i]);
                //              printf("%e %e\n",ebw[i],bwsp[i]);
                ebw_h[i] = -1.0F/ebw_h[i];
        }
        fgets(buffer,100,fp);
        fclose(fp);

        idlebw_h = (NBW-1)/(ebw_h[NBW-1]-ebw_h[0]);
        cudaMemcpyToSymbol(idlebw, &idlebw_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(ebw0, &ebw_h[0], sizeof(float), 0,
                cudaMemcpyHostToDevice);

        cudaMallocArray(&bwsp, &bwsp_tex.channelDesc, NBW, 1);
        cudaMemcpyToArray(bwsp, 0, 0, bwsp_h, sizeof(float)*NBW, cudaMemcpyHostToDevice);
        bwsp_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(bwsp_tex, bwsp);
}

void inisub()
/*******************************************************************
c*    Initializes data used when transporting electrons below      *
c*    Eabs and reads StopPower from file to set up interpolation   *
c*    matrices                                                     *
c*                                                                 *
c*    Input:                                                       *
c*      refden -> reference material density (g/cm^3)              *
c*    Comments:                                                    *
c*      -> must be called after reading common /dpmsrc/            *
c******************************************************************/
{
        //      Density and StopPower for SubEabs transport are set quite arbitrarily:
        subden_h = 1.0/10.0F;
        substp_h = 2.0e6F;
        subfac_h = subden_h/eabs_h;

        cudaMemcpyToSymbol(subden, &subden_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(substp, &substp_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(subfac, &subfac_h, sizeof(float), 0,
                cudaMemcpyHostToDevice);

        printf("inisub: Minimum density for SubEabs motion (g/cm^3):\n");
        printf("%e\n",subden_h);
        printf("inisub: StopPower for SubEabs motion (eV*cm^2/g):\n");
        printf("%e\n", substp_h);
        printf("inisub: Factor for SubEabs motion:\n");
        printf("%e\n",subfac_h);
        printf("\n\n\n");
}


void init_icdf_ZDist(char fname[100])
{
       float *icdf_ZDist_h = new float[nicdf_ZDist];
  
       FILE *fp = fopen(fname,"r");
       fread(icdf_ZDist_h, sizeof(float),nicdf_ZDist, fp);
       fclose(fp);
  
        cudaMallocArray(&icdf_ZDist, &icdf_ZDist_tex.channelDesc, nicdf_ZDist, 1);
        cudaMemcpyToArray(icdf_ZDist, 0, 0, icdf_ZDist_h, sizeof(float)*nicdf_ZDist, cudaMemcpyHostToDevice);
		
        icdf_ZDist_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(icdf_ZDist_tex, icdf_ZDist); 
		
		float idpZ_h = (nicdf_ZDist-1)/1.0f;
		if(cudaMemcpyToSymbol(idpZ, &idpZ_h, sizeof(float), 0, cudaMemcpyHostToDevice)!= cudaSuccess)
		 cout << "error in setting idpZ" << endl;
     
	    delete[] icdf_ZDist_h;
}

void init_icdf_EDist(char fname1[100],char fname2[100])
{
       float *icdf_EDist_h = new float[nicdf_EDist*nEbin];
  
       FILE *fp = fopen(fname1,"r");
       fread(icdf_EDist_h, sizeof(float),nicdf_EDist*nEbin,fp);
       fclose(fp);
  
        cudaMallocArray(&icdf_EDist, &channelDesc, nicdf_EDist, nEbin);
        cudaMemcpyToArray(icdf_EDist, 0, 0, icdf_EDist_h, sizeof(float)*nicdf_EDist*nEbin, cudaMemcpyHostToDevice);
		
        icdf_EDist_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(icdf_EDist_tex, icdf_EDist);
		
		pEbin = new float[nEbin];
		
		fp = fopen(fname2,"r");
		fread(pEbin, sizeof(float),nEbin,fp);
        fclose(fp);
		
		float idpE_h = (nicdf_EDist-1)/1.0f;
		if(cudaMemcpyToSymbol(idpE, &idpE_h, sizeof(float), 0,cudaMemcpyHostToDevice)!= cudaSuccess)
        cout << "error in setting idpE" << endl;
		
	    delete[] icdf_EDist_h;
}

void init_icdf_PhiDist(char fname[100])
{
       float *icdf_PhiDist_h = new float[nicdf_PhiDist*nZbin*nEk];
	  
  
       FILE *fp = fopen(fname,"r");
       fread(icdf_PhiDist_h, sizeof(float),nicdf_PhiDist*nZbin*nEk,fp);
       fclose(fp);
	   
	   cudaMallocArray(&icdf_PhiDist, &channelDesc, nicdf_PhiDist, nZbin*nEk);
       cudaMemcpyToArray(icdf_PhiDist, 0, 0, icdf_PhiDist_h, sizeof(float)*nicdf_PhiDist*nZbin*nEk, cudaMemcpyHostToDevice);
		
        icdf_PhiDist_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(icdf_PhiDist_tex, icdf_PhiDist);
  
		
		float idpPhi_h = (nicdf_PhiDist-1)/1.0f;
		if(cudaMemcpyToSymbol(idpPhi, &idpPhi_h, sizeof(float), 0,cudaMemcpyHostToDevice)!= cudaSuccess)
        cout << "error in setting idpPhi" << endl;
		
		delete[] icdf_PhiDist_h;
}

void rmear(char fname[40])
/*******************************************************************
c*    Reads mass energy absorption coefficient ratio               *
c*    mu_water/mu_material                                         *
c*    Input:                                                       *
c*      fname -> input file name                                   *
c******************************************************************/
{
        char buffer[100];
        int ndata;

        printf("rmear: Reading %s\n", fname);
        FILE *fp = fopen(fname, "r");
        fgets(buffer,100,fp);
        fgets(buffer,100,fp);
        for(int j = 0; j < nmat_h; j++)
        {
                fgets(buffer,100,fp);
                float temp;
                fscanf(fp,"%d %f %f %f %f\n",&ndata,&temp,&temp,&temp,&temp);
                if (ndata != NMEAR)
                {
                        printf("rmear:error: Array dim do not match:\n");
                        printf("%d %d \n", ndata,NMEAR);
                        exit(1);
                }
                fgets(buffer,100,fp);
//      Preparing interpolation:
                for(int i = 0; i <NMEAR; i++)
                {
                        fscanf(fp,"%f %f\n",&emear_h[i],&mear_h[ind2To1(j,i,MAXMAT,NMEAR)]);
                      if(i == 0)
                              printf("%e %e\n",emear_h[i],mear_h[j*NMEAR+i]);
                }
                fgets(buffer,100,fp);

        }
        fclose(fp);

        idmear_h = (NMEAR-1)/(emear_h[NMEAR-1]-emear_h[0]);
        cudaMemcpyToSymbol(idmear, &idmear_h, sizeof(float), 0, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(emear0, &emear_h[0], sizeof(float), 0, cudaMemcpyHostToDevice) ;

	cudaMallocArray(&mear, &mear_tex.channelDesc, NMEAR*MAXMAT, 1);
        cudaMemcpyToArray(mear, 0, 0, mear_h, sizeof(float)*NMEAR*MAXMAT, cudaMemcpyHostToDevice);
        mear_tex.filterMode = cudaFilterModeLinear;
        cudaBindTextureToArray(mear_tex, mear);
}

#endif
