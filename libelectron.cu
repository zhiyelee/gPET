#ifndef __LIBELECTRON_H__
#define __LIBELECTRON_H__


__global__ void electr(float *escore, int nactive)
/*******************************************************************
c*    Transports an electron until it either escapes from the      *
c*    universe or its energy drops below Eabs                      *
c*                                                                 *
c*    Input:                                                       *
c*      electron initial state                                     *
c*    Output:                                                      *
c*      deposits energy in counters and updates secondary stack    *
c******************************************************************/
{
        const int id = blockIdx.x*blockDim.x + threadIdx.x;
//      obtain current id on thread

        if( id < nactive) 
        {
        	float smax;
	        int indexvox, dvox;
	
		float4 xtemp = x_gBrachy[id];
        float4 vxtemp = vx_gBrachy[id];
		int4 voxtemp = getAbsVox(xtemp);;
		
		curandState localState = cuseed[id];			

//      Loading fuel before take off:
		float ebefor = vxtemp.w;
		float fuelel = estepip(vxtemp.w);
		float fuelxt = fuelel*curand_uniform(&localState);
		fuelel -= fuelxt;

//      Set switches to steer the flight:
		int eventid = -1;
        int modeel = 1;

//      Repeat for every flight stop:
                for(; ;)
                {
                        flight(&xtemp, &vxtemp, &voxtemp, &eventid, &fuelel, 
				&fuelxt, escore,&smax, &indexvox, &dvox);

                        if (eventid == 2)
                        {
//      Elastic stop for refueling:
                                if (modeel == 0)
                                {
//      End of second scattering substep, no real interaction:					
                                        ebefor = vxtemp.w;
					fuelel = estepip(vxtemp.w);
			                fuelxt = fuelel*curand_uniform(&localState);
			                fuelel -= fuelxt;
                                        modeel = 1;
                                        eventid = 20;
                                }
                                else
                                {
//      Elastic scattering event:
//      opt-qSingleMat ON: activate next line and deact next+1:
					float costhe = 1.0f;
                                        esamsca(ebefor, &costhe, &localState);
//      call xsamsca(mat(absvox),ebefor,costhe)

                                        rotate(&vxtemp.x,&vxtemp.y,&vxtemp.z,costhe,
						TWOPI*curand_uniform(&localState) );
 
                                        fuelel = fuelxt;
                                        modeel = 0;
                                }
                        }
                        else
                        {
//      Other event values indicate that history ended:
                                break;
                        }
                }
		
		cuseed[id] = localState;
        }

}











__device__ void esamsca(float e, float *mu, curandState *localState_pt)
/*******************************************************************
c*    Samples cos(theta) according to the G&S distribution.        *
c*    Uses interpolated data for bw and the q surface and this     *
c*    latter quantity to perform a rejection procedure.            *
c*                                                                 *
c*    Input:                                                       *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      mu -> polar angle -cos(theta)-                             *
c******************************************************************/
{
        float bw,onebw,u,ie;

        ie = -1.0/e;
        bw = exbwip(ie);

        onebw = 1.0 + bw;
        for(;;)
        {
                u = curand_uniform(localState_pt);
                *mu = (onebw - u*(onebw + bw))/(onebw - u);
                if (curand_uniform(localState_pt) <= exq2Dip(u,ie)) break;
        }
}





__device__ float exbwip(float ie)
/*******************************************************************
c*    3spline interpolation for bw as a function energy            *
c*                                                                 *
c*    Input:                                                       *
c*      ie -> -1/energy in eV^-1  --kinetic energy--               *
c*    Output:                                                      *
c*      bw, broad screening parameter that gets flattest q surf    *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c*    Comments:                                                    *
c*      -> Identical to bwip() but uses only the reference mat.    *
c******************************************************************/
{
	float i = idlebw * (ie- ebw0) + 0.5;
	return tex1D(bwsp_tex,i);
}





__device__ float exq2Dip(float u,float ie)
/*******************************************************************
c*    Linearly interpolated q(u;energy) surface                    *
c*                                                                 *
c*    Input:                                                       *
c*      u -> angular variable                                      *
c*      ie -> -1/energy in eV^-1  --kinetic energy--               *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c*    Comments:                                                    *
c*      -> Identical to q2Dip() but uses only the reference mat.   *
c******************************************************************/
{
        float ru = u * iduq;
        float rle = idleq * (ie-le0q);
		return tex2D(q_tex,  rle+0.5, ru+0.5);
}










__device__ void esubabs(float4* xtemp, float4 *vtemp, int4 *voxtemp, 
	curandState *localState_pt, float *escore)
/*******************************************************************
c*    Transport of e- below the nominal Eabs until absorption      *
c*                                                                 *
c*    Input:                                                       *
c*      {x,y,z} -> take off location                               *
c*      {vx,vy,vz} -> direction of flight                          *
c*      energy -> initial kinetic energy                           *
c*    Output:                                                      *
c*      escore -> energy deposited by CSDA                         *
c*    Comments:                                                    *
c*      -> the particle flies following a straight path.           *
c*      -> uses a constant StopPow extracted from reference mat.   *
c******************************************************************/
{
	float temp = vtemp->w;
	
        for(;;)
        {
		 float voxdentemp = tex3D(dens_tex, voxtemp->x, voxtemp->y, voxtemp->z);
		  
//                float voxdentemp = tex1Dfetch(dens_tex, voxtemp->w);
//      Determine if further transport is needed:

                if (voxdentemp > temp*subfac)
                {
			/* if(ifDoseToWater)
                        {
                        	scoreDoseToWater(temp, *voxtemp,
                                        localState_pt, vtemp->w);
                        }
                        else */
                        //{
								
    						scoreDose(temp*xtemp->w, *voxtemp, localState_pt,escore);
                       // }
                        return;
                }
		int indexvox,dvox;
                float s = inters(vtemp, xtemp, voxtemp, &indexvox, &dvox);
                float de = substp * voxdentemp * s;
                temp -= de;

		/* if(ifDoseToWater)
                {
                        scoreDoseToWater(de, *voxtemp,
                                localState_pt, vtemp->w);
                }
                else */
               // {
			          
						scoreDose(de*xtemp->w, *voxtemp, localState_pt,escore);
               // }
                       xtemp->x += s* vtemp->x;
		               xtemp->y += s* vtemp->y;
		               xtemp->z += s* vtemp->z;
                chvox(voxtemp, indexvox, dvox);
                              
                if (voxtemp->w == -1)
                {
                        return;
                }
                
        }
}






__device__ float estepip(float e)
/*******************************************************************
c*    3spline interpolation for scattering strength as a function  *
c*    of kinetic energy; this quantity is related to the step      *
c*    length                                                       *
c*                                                                 *
c*    Input:                                                       *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      K = scattering strength = integ{ds/lambda1(s)}             *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
	float i = idless*(e-escsr0) + 0.5;
	return tex1D(scssp_tex, i);
}







__device__ float erstpip(int matid, float e)
/*******************************************************************
c*    Restricted stopping power --linear interpolation             *
c*                                                                 *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> energy in eV  --kinetic energy--                      *
c*    Output:                                                      *
c*      StopPow in eV*cm^2/g                                       *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
	float i = idlest*(e-est0) + 0.5;
	return tex1D(stsp_tex,matid * NST + i);
}





__device__ float escpwip(int matid, float e)
/*******************************************************************
c*    Inverse 1st transport MFP --linear interpolation             *
c*                                                                 *
c*    Input:                                                       *
c*      matid -> material id#                                      *
c*      e -> kinetic energy in eV                                  *
c*    Output:                                                      *
c*      -> lambda_1^{-1} in cm^2/g                                 *
c*    Comments:                                                    *
c*      -> init() must be called before first call                 *
c******************************************************************/
{
	float i = idlesc*(e-escp0) + 0.5;
	return tex1D(scpsp_tex, matid*NSCP + i);
}







__device__ void flight(float4 *xtemp, float4 *vxtemp, int4 *voxtemp, int* eventid, 
	float *fuelel, float *fuelxt, float *escore, float *smax, int *indexvox, int *dvox)
/*******************************************************************
c*    Transports the particle following a rectiliniar trajectory,  *
c*    taking care of interface crossings and keeping track of      *
c*    energy losses in the corresponding counters (using CSDA)     *
c*                                                                 *
c*    Input:                                                       *
c*      e- initial state                                           *
c*      Fuel variables affecting the flight                        *
c*      event -> event that stopped flight last time; in addition  *
c*                to the output codes,                             *
c*                -1 new particle                                  *
c*                20 end of elastic step                           *
c*    Output:                                                      *
c*      e- final state                                             *
c*      Remaining fuel vars                                        *
c*      event -> kind of event that causes flight to stop:         *
c*                1 run out of energy; absorbed                    *
c*                2 run out of elastic fuel                        *
c*                3 run out of Moller fuel                         *
c*                4 run out of bremsstrahlung fuel                 *
c*               99 escaped from universe                          *
c*    Comments:                                                    *
c*      -> this routine does NOT transport correctly e- below Eabs *
c*      -> internally, 'event' can also have the value 0 to        *
c*         indicate that the particle hit a voxel boundary; the    *
c*         flight is resumed without a stop.                       *
c*      -> the 'fuel vars' are contained in /dpmjmp/.              *
c******************************************************************/
{
//	float smax;
//	int indexvox, dvox;
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	
//      Init according to the dynamic vars changed:
        if (*eventid  == 2 || *eventid == -1)
        {
		*smax = inters(vxtemp, xtemp, voxtemp, indexvox, dvox);
        }
        *eventid = 0;

//      Loop until it runs out of fuel:
        for(;;)
        {
                float s = *smax;

//      Calculate fuel burn rate in the current voxel:
		int matid = tex3D(mat_tex, voxtemp->x, voxtemp->y, voxtemp->z);
		float voxden = tex3D(dens_tex, voxtemp->x, voxtemp->y, voxtemp->z);
		
		 /* if(id ==0 )
	     printf("voxtemp->x=%d, voxtemp->y=%d, voxtemp->z=%d,voxden = %f, voxmatid = %d\n", voxtemp->x, voxtemp->y, voxtemp->z,voxden, matid); 	 		
		
		if(matid == 1 || matid ==2)
		 printf("voxtemp->x=%d, voxtemp->y=%d, voxtemp->z=%d,voxden = %f, voxmatid = %d\n", voxtemp->x, voxtemp->y, voxtemp->z,voxden, matid); 	 		
		  */
                float dedx = erstpip(matid, vxtemp->w) * voxden;

                float newe = vxtemp->w - 0.5*dedx*s;
                if(eabs > newe) newe = eabs;

                float burnel = escpwip(matid, newe) * voxden;

//      Burn elastic fuel:
                float infuel = *fuelel;
                *fuelel -= s*burnel;

                if (*fuelel < 0.0f)
                {
//      Refine calculation of scattering 1st MFP:
                        float news = infuel/(escpwip(matid, vxtemp->w) * voxden);
                        newe = vxtemp->w - 0.5*dedx*news;
                        if(eabs > newe) newe = eabs;

                        news = infuel/(escpwip(matid, newe) * voxden);
                        if (news > s) news = s;
                        float sback = s - news;
                        s = news;
                        *fuelel = 0.0;
                        *eventid = 2;
                }
//      Accounting for continous energy loss:
                newe = vxtemp->w - 0.5*dedx*s;
                if(eabs > newe) newe = eabs;
                float de = s * erstpip(matid, newe)*voxden;
//	to make sure at most deposit *ene 
		if(de > vxtemp->w) 
		{
			de = vxtemp->w;
		}
                vxtemp->w -= de;
		/* if(ifDoseToWater)
                {
                        scoreDoseToWater(de, *voxtemp,
                                &cuseed[blockIdx.x*blockDim.x + threadIdx.x], vxtemp->w);
                }
                else */
                //{
                        scoreDose(de*xtemp->w, *voxtemp, &cuseed[blockIdx.x*blockDim.x + threadIdx.x],escore);
			
                //}

                if (vxtemp->w < eabs)
                {
//      Determine if subEabs transport is needed:
                        if (voxden > subden)
                        {
                                /* if(ifDoseToWater)
		                {
                		        scoreDoseToWater(vxtemp->w, *voxtemp,
                                		&cuseed[blockIdx.x*blockDim.x + threadIdx.x], 
						vxtemp->w);
                		}
                		else */
                		//{    
				
						scoreDose(vxtemp->w*xtemp->w, *voxtemp, 
						&cuseed[blockIdx.x*blockDim.x + threadIdx.x], escore);
						
						
                		//}
                        }
                        else
                        {
				esubabs(xtemp, vxtemp, voxtemp, 
					&cuseed[blockIdx.x*blockDim.x + threadIdx.x], escore);
                        }
                        *eventid = 1;
                        return;
                }

//      Move the electron:
                xtemp->x += s* vxtemp->x;
		        xtemp->y += s* vxtemp->y;
		        xtemp->z += s* vxtemp->z;
                *smax -= s;

//      Check whether an interaction has not ocurred:
                if (*eventid == 0)
                {
                        chvox(voxtemp, *indexvox, *dvox);
                         
		            
                        if (voxtemp->w == -1)
                        {
                                *eventid = 99;
                                return;
                        }
                        *smax = inters(vxtemp, xtemp, voxtemp, indexvox, dvox);

                        continue;
                }
                break;
        }
//      Otherwise, run out of fuel so return:

}











#endif
