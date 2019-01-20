#ifndef __LIBPHOTON_H__
#define __LIBPHOTON_H__

__device__ float lamwck(float e)
/*******************************************************************
c*    Mean free path prepared to play the Woodcock trick           *
c*                                                                 *
c*    Input:                                                       *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      Minimum mean free path in cm                               *
c*    Comments:                                                    *
c*      -> iniwck() must be called before first call               *
c******************************************************************/
{
    float i = idlewk*(e-wcke0) + 0.5f;
    return tex1D(woock_tex, i);
}

__device__ float itphip_G(int matid, float e)
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

__device__ float irylip(int matid, float e)
/*******************************************************************
c*    Inverse Rayleigh mean free path --3spline interpolation      *
c*                                                                 *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      Inverse total mean free path in cm^2/g                     *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
    float i = idlerl*(e - erayl0) + 0.5;
    return tex1D(rayle_tex,matid * NRAYL + i);
}

__device__ float icptip(int matid, float e)
/*******************************************************************
c*    Inverse Compton mean free path --3spline interpolation       *
c*                                                                 *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      Inverse total mean free path in cm^2/g                     *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
    float i = idlecp*(e - ecmpt0) + 0.5;
    return tex1D(compt_tex,matid * NCMPT + i);
}



__global__ void photon(float *escore, const int nactive)
/*******************************************************************
c*    Transports a photon until it either escapes from the         *
c*    universe or its energy drops below EabsPhoton                *
c*                                                                 *
c*    Input:                                                       *
c*      photon initial state                                       *
c*    Output:                                                      *
c*      deposits energy in counters and updates secondary stack    *
c******************************************************************/
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid=threadIdx.x;
//      obtain current id on thread
    const float spe=29979245800.0f;
    const float ispe=1.0f/spe;
    __shared__ int nsstktemp;
    __shared__ float sftemp[NSSTACKSHARED];
    if(tid==0)
    {
        nsstktemp = 0;
    }
    if( id < nactive)
    {
        float4 xtemp = x_gBrachy[id];
        float4 vxtemp = vx_gBrachy[id];
        curandState localState = cuseed[id];
//      Loop until it either escapes or is absorbed:
        while(1)
        {
//      Get lambda from the minimum lambda at the current energy:
            float lammin = lamwck(vxtemp.w);
            float s = -lammin*__logf(curand_uniform(&localState));
            xtemp.x += s*vxtemp.x;
            xtemp.y += s*vxtemp.y;
            xtemp.z += s*vxtemp.z;
            xtemp.w+=abs(s)*ispe;
            int4 absvoxtemp = getAbsVox(xtemp);

            if(id ==0 )
                printf("x=%f,y=%f,z=%f,e=%f\n",xtemp.x,xtemp.y,xtemp.z,vxtemp.w);

            if (absvoxtemp.w == -1)//means the particle is outside the phantom
            {
                float r=getDistance(xtemp,vxtemp);
                float xs= xtemp.x+r*vxtemp.x;//get the nearest position that the particle goes out of the phantom
                float ys= xtemp.y+r*vxtemp.y;
                float zs= xtemp.z+r*vxtemp.z;
                float ws=xtemp.w+abs(r)*ispe;
                if (nsstktemp> NSSTACKSHARED) break;
                if (abs(vxtemp.z-1.0f)<=1e-7||abs(vxtemp.z+1.0f)<=1e-7||abs(zs)>395) break;
                int ind = atomicAdd(&nsstktemp,8);
                sftemp[ind] = xs; // to cm
                sftemp[ind+1] = ys;
                sftemp[ind+2] = zs;
                sftemp[ind+3] = ws;//s;
                sftemp[ind+4] = vxtemp.x;
                sftemp[ind+5] = vxtemp.y;
                sftemp[ind+6] = vxtemp.z;
                sftemp[ind+7] = vxtemp.w;
                if(id==0)
                    printf("x=%f,y=%f,z=%f,e=%f,lammin=%f,s=%f\n",xtemp.x,xtemp.y,xtemp.z,vxtemp.w,lammin,s);
                break;
            }

//	get density
            float voxden = tex3D(dens_tex, int((xtemp.x+Offsetx_gBrachy)*idx_gBrachy), int((xtemp.y+Offsety_gBrachy)*idy_gBrachy), int((xtemp.z+Offsetz_gBrachy)*idz_gBrachy));
//	get mat id
            int voxmatid = tex3D(mat_tex, int((xtemp.x+Offsetx_gBrachy)*idx_gBrachy), int((xtemp.y+Offsety_gBrachy)*idy_gBrachy), int((xtemp.z+Offsetz_gBrachy)*idz_gBrachy));

            if (voxden !=1.00f)//boundry conditions, only valid for water 
            {
                float r=getDistance(xtemp,vxtemp);
                float xs= xtemp.x+r*vxtemp.x;
                float ys= xtemp.y+r*vxtemp.y;
                float zs= xtemp.z+r*vxtemp.z;
                float ws=xtemp.w+abs(r)*ispe;
                if (nsstktemp> NSSTACKSHARED) break;
                if (abs(vxtemp.z-1.0f)<=1e-7||abs(vxtemp.z+1.0f)<=1e-7||abs(zs)>395) break;
                int ind = atomicAdd(&nsstktemp,8);
                sftemp[ind] = xs;
                sftemp[ind+1] = ys;
                sftemp[ind+2] = zs;
                sftemp[ind+3] = ws;//s;
                sftemp[ind+4] = vxtemp.x;
                sftemp[ind+5] = vxtemp.y;
                sftemp[ind+6] = vxtemp.z;
                sftemp[ind+7] = vxtemp.w;
                if(id==0)

                    printf("x=%f,y=%f,z=%f,e=%f,lammin=%f,s=%f\n",xtemp.x,xtemp.y,xtemp.z,vxtemp.w,lammin,s);
                break;
            }

//      Apply Woodcock trick:
            float lamden = lammin*voxden;
            float prob = 1.0-lamden*itphip_G(voxmatid, vxtemp.w);
            float randno = curand_uniform(&localState);
//      No real event; continue jumping:
            if (randno < prob) continue;

//      Compton:
            prob += lamden*icptip(voxmatid, vxtemp.w);
            if (randno < prob)
            {
                float efrac, costhe;
                comsam(vxtemp.w, &localState, &efrac, &costhe, voxmatid);
//				comsam(vxtemp.w, &localState, &efrac, &costhe);
                float de = vxtemp.w * (1.0f-efrac);
                float phi = TWOPI*curand_uniform(&localState);


                vxtemp.w -= de;
                if (vxtemp.w < eabsph)
                {

                    break;
                }

                rotate(&vxtemp.x,&vxtemp.y,&vxtemp.z,costhe,phi);
                // if(id ==0 )
                //printf("costhe = %f, vxtemp.w=%f\n", costhe,vxtemp.w);
                continue;
            }
//	Rayleigh:
            prob += lamden*irylip(voxmatid, vxtemp.w);
            if (randno < prob)
            {
                float costhe;
                rylsam(vxtemp.w, voxmatid, &localState, &costhe);
                float phi = TWOPI*curand_uniform(&localState);

                rotate(&vxtemp.x,&vxtemp.y,&vxtemp.z,costhe,phi);
                continue;
            }
//  Photoelectric:
            break;
        }

        cuseed[id] = localState;

    }
    __syncthreads();
    __shared__ int istart;
    if(tid == 0)
    {
        istart = atomicAdd(&nsstk, nsstktemp);
//		if(istart2 +nsstktemp > NSSTACK)

//              if(nestktemp>20)
//                      cuPrintf("istart = %d, iend = %d, nstktmp = %d\n", istart, istart + nestktemp, nestktemp);
    }
    __syncthreads();

    for(int i = 0; i < 1+(nsstktemp)/blockDim.x; i++)
    {
        if(nsstktemp == 0)
            break;

        int ind = istart + i*blockDim.x + tid;

        if(ind < istart + nsstktemp && ind<NSSTACK)
        {

            sf[ind] = sftemp[i*blockDim.x + tid];
        }
    }
    __syncthreads();

}

__device__ void rylsam(float energytemp, int matid, curandState *localState_pt, float *costhe)
/*******************************************************************
c*    Samples a Rayleigh event following its DCS                   *
c*                                                                 *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*    Output:                                                      *
c*      costhe -> cos(theta) of the 2nd photon                     *
c*    Comments:                                                    *
c*      -> inirng() must be called before 1st call                 *
c******************************************************************/
{
    float indcp = curand_uniform(localState_pt)*idcprl;
    float inde = energytemp*iderl;
    float temp = tex3D(f_tex,inde+0.5f, indcp+0.5f, matid+0.5f);
    if(temp > 1.0f) temp = 1.0f;
    if(temp < -1.0f) temp = -1.0f;
    *costhe = temp;
}

__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe, int matid)
//this is the revised KN model
/*******************************************************************
c*    Samples a Compton event following Klein-Nishina DCS          *
c*                                                                 *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*    Output:                                                      *
c*      efrac -> fraction of initial energy kept by 2nd photon     *
c*      costhe -> cos(theta) of the 2nd photon                     *
c*    Comments:                                                    *
c*      -> inirng() must be called before 1st call                 *
c******************************************************************/
{
    float indcp = curand_uniform(localState_pt)*idcpcm;
    float inde = energytemp*idecm;
    float temp = tex3D(s_tex,inde+0.5f, indcp+0.5f, matid+0.5f);
    if(temp > 1.0f) temp = 1.0f;
    if(temp < -1.0f) temp = -1.0f;
    *costhe = temp;

    *efrac = 1.0f/(1.0f + energytemp*IMC2*(1.0f-temp));
}

//__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe)
//this is the standard KN model which treat electron as free and no dopler effect
/*******************************************************************
c*    Samples a Compton event following Klein-Nishina DCS          *
c*                                                                 *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*    Output:                                                      *
c*      efrac -> fraction of initial energy kept by 2nd photon     *
c*      costhe -> cos(theta) of the 2nd photon                     *
c*    Comments:                                                    *
c*      -> inirng() must be called before 1st call                 *
c******************************************************************/
/*{
	float e0,twoe,kmin2,loge,mess;

    	e0 = energytemp*IMC2;
    	twoe = 2.0*e0;
    	kmin2 = 1.0/((1.0+twoe)*(1.0+twoe));
    	loge = __logf(1.0+twoe);

	for(;;)
	{
        	if (curand_uniform(localState_pt)*(loge+twoe*(1.0+e0)*kmin2) < loge)
		{
			*efrac = expf(-curand_uniform(localState_pt)*loge);
		}
        	else
		{
			*efrac = sqrtf(kmin2+curand_uniform(localState_pt)*(1.0-kmin2));
		}
        	mess = e0*e0*(*efrac)*(1.0+(*efrac)*(*efrac));

		if (curand_uniform(localState_pt)*mess <= mess-(1.0-*efrac)*((1.0+twoe)*(*efrac)-1.0))break;
	}

    	*costhe = 1.0-(1.0-*efrac)/((*efrac)*e0);
}*/

__device__ float comele(float energytemp, float efrac, float costhe)
/*******************************************************************
c*    Compton angular deviation of the secondary electron          *
c*    Input:                                                       *
c*      energy -> photon energy in eV                              *
c*      efrac -> fraction of initial energy kept by 2nd photon     *
c*      costhe-> photon scattering angle                           *
c*    Output:                                                      *
c*      -> cos(theta) of the 2nd electron                          *
c******************************************************************/
{
    float e0;

//	opt-ComptonE-ON -> switch following lines for last three
//    e0=energy**2+(energy*efrac*(energy*efrac-2.d0*energy*costhe))
//     if(e0.gt.1.0d-12) then
//       comele=(energy-energy*efrac*costhe)/dsqrt(e0)
//     else
//       comele=1.d0
//     endif

    e0 = energytemp*IMC2;
    return (1.0+e0)*sqrtf((1.0-efrac)/(e0*(2.0+e0*(1.0-efrac))));
}

__device__ void putElectron(float4 *vxtemp, float4 *xtemp,
                            float de, curandState *localState_pt)
//	put an ionization electron in the stack
{
//	if stack is full
    if (nestk == NESTACKSHARED)
        return;

//	put location information
    int ind = atomicAdd(&nestk, 1);
    esx[ind] = *xtemp;

//	put velocity information
    esvx[ind] = *vxtemp;

    esvx[ind].w = de;
}

#endif
