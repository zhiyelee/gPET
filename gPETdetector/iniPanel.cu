#ifndef __INIPANEL_CU__
#define __INIPANEL_CU__


void iniPanel(struct object_t* objectArray, struct object_v* objectMaterial,int totalOb)
/*******************************************************************
c*    Initializes the module system                                  *
c******************************************************************/
{

	printf(" \n");
    printf("init: Panel geometry;\n");
    printf("      information from this stream follows:\n");

//  copy arrays from host to device
	int *ma=new int[2];
	float *den=new float[2];
	int *p_id=new int[totalOb];

	float *lx_m=new float[totalOb];
	float *ly_m=new float[totalOb];
	float *lz_m=new float[totalOb];

	float *Mx_m=new float[totalOb];
	float *My_m=new float[totalOb];
	float *Mz_m=new float[totalOb];

	float *Msx_m=new float[totalOb];
	float *Msy_m=new float[totalOb];
	float *Msz_m=new float[totalOb];

	float *Lx_m=new float[totalOb];
	float *Ly_m=new float[totalOb];
	float *Lz_m=new float[totalOb];

	float *sx_m=new float[totalOb];
	float *sy_m=new float[totalOb];
	float *sz_m=new float[totalOb];

	float *ox_m=new float[totalOb];
	float *oy_m=new float[totalOb];
	float *oz_m=new float[totalOb];

	float *dx_m=new float[totalOb];
	float *dy_m=new float[totalOb];
	float *dz_m=new float[totalOb];

	float *UXx_m=new float[totalOb];
	float *UXy_m=new float[totalOb];
	float *UXz_m=new float[totalOb];

	float *UYx_m=new float[totalOb];
	float *UYy_m=new float[totalOb];
	float *UYz_m=new float[totalOb];

	float *UZx_m=new float[totalOb];
	float *UZy_m=new float[totalOb];
	float *UZz_m=new float[totalOb];

	for (int i=0;i<2;i++)
	{
		ma[i]=objectMaterial[i].material;
		den[i]=objectMaterial[i].density;
	}

	for (int i=0;i<totalOb;i++)
	{
		p_id[i]=objectArray[i].panel;

		lx_m[i]=objectArray[i].lengthx;
		ly_m[i]=objectArray[i].lengthy;
		lz_m[i]=objectArray[i].lengthz;

		Mx_m[i]=objectArray[i].MODx;
		My_m[i]=objectArray[i].MODy;
		Mz_m[i]=objectArray[i].MODz;

		Msx_m[i]=objectArray[i].Mspacex;
		Msy_m[i]=objectArray[i].Mspacey;
		Msz_m[i]=objectArray[i].Mspacez;

		Lx_m[i]=objectArray[i].LSOx;
		Ly_m[i]=objectArray[i].LSOy;
		Lz_m[i]=objectArray[i].LSOz;

		sx_m[i]=objectArray[i].spacex;
		sy_m[i]=objectArray[i].spacey;
		sz_m[i]=objectArray[i].spacez;

		ox_m[i]=objectArray[i].offsetx;
		oy_m[i]=objectArray[i].offsety;
		oz_m[i]=objectArray[i].offsetz;

		dx_m[i]=objectArray[i].directionx;
		dy_m[i]=objectArray[i].directiony;
		dz_m[i]=objectArray[i].directionz;

		UXx_m[i]=objectArray[i].UniXx;
		UXy_m[i]=objectArray[i].UniXy;
		UXz_m[i]=objectArray[i].UniXz;

		UYx_m[i]=objectArray[i].UniYx;
		UYy_m[i]=objectArray[i].UniYy;
		UYz_m[i]=objectArray[i].UniYz;

		UZx_m[i]=objectArray[i].UniZx;
		UZy_m[i]=objectArray[i].UniZy;
		UZz_m[i]=objectArray[i].UniZz; 
		
	}
	for (int i=0;i<totalOb;i++)
	{
		printf("totalPanel on CPU is: %d, UZx_m=%f, UZy_m=%f,UZz_m=%f\n",totalOb, UZx_m[i],UZy_m[i],UZz_m[i]);
	}
	
	
	cudaMemcpyToSymbol(dev_totalPanels, &totalOb, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&mat_panel, 2 * sizeof(int)); 
	cudaMalloc((void**)&dens_panel, 2 * sizeof(float)); 
	cudaMalloc((void**)&panelID, totalOb * sizeof(int)); 

	cudaMalloc((void**)&lengthx_panel, totalOb * sizeof(float)); 
	cudaMalloc((void**)&lengthy_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&lengthz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&MODx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&MODy_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&MODz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&Mspacex_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&Mspacey_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&Mspacez_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&LSOx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&LSOy_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&LSOz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&spacex_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&spacey_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&spacez_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&offsetx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&offsety_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&offsetz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&directionx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&directiony_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&directionz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&UniXx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&UniXy_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&UniXz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&UniYx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&UniYy_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&UniYz_panel, totalOb * sizeof(float));

	cudaMalloc((void**)&UniZx_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&UniZy_panel, totalOb * sizeof(float));
	cudaMalloc((void**)&UniZz_panel, totalOb * sizeof(float));

    cudaMemcpy(mat_panel, ma, 2*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dens_panel, den, 2*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(panelID, p_id, totalOb*sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(lengthx_panel, lx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lengthy_panel, ly_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(lengthz_panel, lz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(MODx_panel, Mx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(MODy_panel, My_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(MODz_panel, Mz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(Mspacex_panel, Msx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Mspacey_panel, Msy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Mspacez_panel, Msz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(LSOx_panel, Lx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(LSOy_panel, Ly_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(LSOz_panel, Lz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(spacex_panel, sx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(spacey_panel, sy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(spacez_panel, sz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(offsetx_panel, ox_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(offsety_panel, oy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(offsetz_panel, oz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(directionx_panel, dx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(directiony_panel, dy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(directionz_panel, dz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(UniXx_panel, UXx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(UniXy_panel, UXy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(UniXz_panel, UXz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(UniYx_panel, UYx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(UniYy_panel, UYy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(UniYz_panel, UYz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(UniZx_panel, UZx_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(UniZy_panel, UZy_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(UniZz_panel, UZz_m, totalOb*sizeof(float), cudaMemcpyHostToDevice);

	delete[] ma;
	delete[] den;
	delete[] p_id;

	delete[] lx_m;
	delete[] ly_m;
	delete[] lz_m;

	delete[] Mx_m;
	delete[] My_m;
	delete[] Mz_m;

	delete[] Msx_m;
	delete[] Msy_m;
	delete[] Msz_m;

	delete[] Lx_m;
	delete[] Ly_m;
	delete[] Lz_m;

	delete[] sx_m;
	delete[] sy_m;
	delete[] sz_m;

	delete[] ox_m;
	delete[] oy_m;
	delete[] oz_m;

	delete[] dx_m;
	delete[] dy_m;
	delete[] dz_m;

	delete[] UXx_m;
	delete[] UXy_m;
	delete[] UXz_m;

	delete[] UYx_m;
	delete[] UYy_m;
	delete[] UYz_m;

	delete[] UZx_m;
	delete[] UZy_m;
	delete[] UZz_m;
	
    printf("finish init: Module geometry;\n\n");
}
#endif