#ifndef __INIPHYSICS_CU__
#define __INIPHYSICS_CU__


void iniPhysics(struct object_v* objectMaterial)
/*******************************************************************
c*    Initializes the physics subsystem                                  *
c******************************************************************/
{
	printf(" \n");
    printf("init: physics model;\n");
    printf("      information from this stream follows:\n");
     
	FILE *fp = fopen("./patientcase/PETConfiguration/physics.in","r");		
	char buffer[200];


//	absorption energy
	fgets(buffer,200,fp);
    printf("%s\n", buffer);
    fscanf(fp,"%f\n", &eabsph_h);
    printf("%e\n", eabsph_h);
    cudaMemcpyToSymbol(eabsph, &eabsph_h, sizeof(float), 0, cudaMemcpyHostToDevice);
	   
	
//  in GPU, initialize rand seed with rand numbers
    inirngG();
    printf("\n");

	char prefix[40], fname[50], fname2[50];
    int len;

//  Read the prefix with the name of the data files, truncate it:
    fgets(buffer,200,fp);
    printf("%s\n", buffer);
    getnam(fp,5, prefix, &len);
    printf("%s\n", prefix);

//  Read material data
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

//	load total cross section
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

//  load rayleigh cross section and form factors
    strcpy(fname, prefix);
    strcat(fname,".rayle");
    rrayle(fname);
	strcpy(fname, prefix);
    strcat(fname,".rayff");
    rrayff(fname);

//  iniwck must be called after reading esrc & eabsph:
    iniwck(eminph, emax, objectMaterial);		
	
	fclose(fp);	

	printf("finish init: physics model;\n\n");            
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
        printf("ERROR, io from other unit has not been programmed...\n");
        exit(1);
    }
    strcpy(physics,phyname);
    *n = (int)strlen(phyname);

    return;
}

void getna2(FILE *iounit,char physics[80], int *n)
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
				
    }
    fclose(fp);
		
    printf(" \n");
    printf("rmater: Done.\n");
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
            if(i<100&j == nmat_h-1)
            printf("%d, %e %e\n",nmat_h, elaph_h[i],lamph_h[ind2To1(j,i,MAXMAT,NLAPH)]);
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
            printf("rcompt:error: Array did do not match:\n");
            printf("%d %d \n", ndata,NCMPT);
            exit(1);
        }
        fgets(buffer,100,fp);
//      Preparing interpolation:
        for(int i = 0; i <NCMPT; i++)
        {
            fscanf(fp,"%f %f\n",&ecmpt_h[i],&compt_h[ind2To1(j,i,MAXMAT,NCMPT)]);
//          if(j == nmat-1)
//          printf("%e %e\n",ecmpt[i],compt[i]);
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
//				cout << "error in data" << mucmpt_h[j*NCPCM*NECM+icp*NECM+ie] << endl;
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
//           if(j == nmat-1)
//          printf("%e %e\n",ephte[i],phote[i]);
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
//  if(murayl_h[j*NCPRL*NERL+icp*NERL+ie] > 1.0f || murayl_h[j*NCPRL*NERL+icp*NERL+ie]<-1.0f)
//  cout << "error in data" << murayl_h[j*NCPRL*NERL+icp*NERL+ie] << endl;
			}
			fscanf(fp,"\n");					
		}
		fscanf(fp,"\n");
//	cout << murayl_h[j*NCPRL*NERL+(NCPRL-2)*NERL+1] << endl;
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



void iniwck(float eminph,float emax, struct object_v* objectMaterial)
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
//  Find the largest density for each present material:
    for(int i = 0; i < MAXMAT; i++)
    {
        maxden[i] = 0.0F;
    }
    for(int i=0; i<2; i++)
    {
        if (objectMaterial[i].density > maxden[objectMaterial[i].material])
            maxden[objectMaterial[i].material] = objectMaterial[i].density;
    }

		
//  Prepare data:
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
           // if (i<10)
            //    printf("j=%d,ycanbe =%f,maxden=%f\n",j,ycanbe,maxden[j]);
            if (ycanbe > ymax) 
				ymax = ycanbe;
        }
        woock_h[i] = 1.0F/ymax;
    }

	cudaMemcpyToSymbol(idlewk, &idlewk_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;
    cudaMemcpyToSymbol(wcke0, &wcke0_h, sizeof(float), 0, cudaMemcpyHostToDevice) ;

    cudaMallocArray(&woock, &woock_tex.channelDesc, NWCK, 1);
    cudaMemcpyToArray(woock, 0, 0, woock_h, sizeof(float)*NWCK, cudaMemcpyHostToDevice);
    woock_tex.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(woock_tex, woock);

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

#endif
