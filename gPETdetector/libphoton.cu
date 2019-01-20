#ifndef __LIBPHOTON_H__
#define __LIBPHOTON_H__
#include "digitizer.h"
#include "cuda_fp16.h"
/*void __device__ multiples()*/
void __global__ coinsorter(int* counts,Event* events,float interval,int secdiffer, int total)
{
	//this is for the coincidence sorter part in digitizer
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    int flag=0,num=0;
    while(id<total-1)
    {
        flag=0;
        flag=((abs(events[id+1].pann-events[id].pann)==secdiffer) && (events[id+1].t-events[id].t<interval));
        if(id==0) {
            if(!flag) events[0].t=10000;
            num++;
            id+=blockDim.x*gridDim.x;
            continue;
        }
        if(id==total-2) {
            if(!flag) events[total-1].t=10000;
            num++;
        }
        flag+=((abs(events[id].pann-events[id-1].pann)==secdiffer) && (events[id].t-events[id-1].t<interval));
        if(!flag) {
            events[id].t=10000;
            num++;
        }
        id+=blockDim.x*gridDim.x;
    }
    atomicSub(counts,num);
}
void __global__ energywindow(int* counts, Event* events,int total, float thresholder, float upholder)
{
	//this is for the energy thresholder part in digitizer
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    int num=0;
    while(id<total)
    {
        if(events[id].E<thresholder || events[id].E>upholder)
        {
            events[id].t=10000;
            num++;
        }
        id+=blockDim.x*gridDim.x;
    }
    if(num) atomicSub(counts,num);
}
void __global__ deadtime(int* counts,Event* events,int total, float interval, int deadtype)
{
	//this is the deadtime part in digitizer
   //deadtype 0 for paralysable, 1 for non
    int id=blockIdx.x*blockDim.x + threadIdx.x;
    int start,current,i,k;
    float tdead;
    while(id<total)
    {
        start=id;
        if(start==0||events[start].siten!=events[start-1].siten||events[start].t>(events[start-1].t+interval))//find the start index
        {
            current=start;
            i=current+1;
            k=0;
            tdead=events[start].t;
            while(i<total)
            {
                while(events[i].siten==events[current].siten && events[i].t<(tdead+interval))
                {
                    //events[current].E+=events[i].E;
                    if(!deadtype) {
                        tdead=events[i].t;    //paralyzable accounts for pile-up effect
                        events[current].t=events[i].t;
                    }
                    events[i].t=10000;
                    i++;
                    k++;
                    if(i==total) break;
                }
                if(i==total) break;
                if(events[i].siten!=events[i-1].siten||events[i].t>(events[i-1].t+interval))
                    break;
                current=i;
                tdead=events[current].t;
                i++;
            }
            //if(!(id%1000)) printf("id is %d k is %d\n", id, k);
            atomicSub(counts,k);
        }
        id+=blockDim.x*gridDim.x;
    }
}
void __global__ addnoise(int* counts, Event* events_d, float lambda, float Emean, float sigma, float interval)
{
	//this is the noise part for digitizer
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Event events[6];
    float t=id*interval;//0;
    curandState localstate = cuseed[id];
    int i=0, ind=0;
    while(t<(id+1)*interval)
    {
        t+=-__logf(curand_uniform(&localstate))*lambda;
        if(t<(id+1)*interval)
        {
            events[i].t=t;//+id*interval;
            events[i].E=Emean+sigma*curand_normal(&localstate);//2.355;
            events[i].x=curand_uniform(&localstate);//need to be implemented to be matched to global coordinates
            events[i].y=curand_uniform(&localstate);
            events[i].z=curand_uniform(&localstate);
            events[i].parn=-1;
            events[i].pann=floor(pannelN*curand_uniform(&localstate));
            events[i].modn=floor(moduleN*curand_uniform(&localstate));
            events[i].cryn=floor(crystalN*curand_uniform(&localstate));
            events[i].siten=events[i].pann*moduleN*crystalN+events[i].modn*crystalN+events[i].cryn;
            i=(i+1)%6;
            if(!i)
            {
                ind=atomicAdd(counts,6);
                for(int j=0; j<6; j++)
                    events_d[ind+j]=events[j];
            }
        }
    }
    cuseed[id]=localstate;
    ind=atomicAdd(counts,i);
    for(int j=0; j<i; j++)
        events_d[ind+j]=events[j];
}

int __device__ adder(int* counts_d, Event* events_d, Event event)
{
	//this is the adder part in digitizer
    for(int i=0; i < counts_d[0]; i++)
    {
        if(event.siten == events_d[i].siten)
        {
            events_d[i].x = (events_d[i].x*events_d[i].E + event.x*event.E)/(events_d[i].E + event.E);
            events_d[i].y = (events_d[i].y*events_d[i].E + event.y*event.E)/(events_d[i].E + event.E);
            events_d[i].z = (events_d[i].z*events_d[i].E + event.z*event.E)/(events_d[i].E + event.E);
            events_d[i].E = (events_d[i].E + event.E);
            return 1;
        }
    }
    events_d[counts_d[0]]=event;
    counts_d[0]++;
    return 1;
}
int __device__ readout(int* counts_d, Event* events_d,int depth, int policy)
{
	//this is for the readout part in digitizer
    //depth means the readout level. 0,1,2,3 represents world,panel,module,cry
    //policy 0,1 for winnertakeall and energy centroid
    if(policy==1) depth = 2;
    if(depth==3) return 1;
    //the readout part
    switch(depth)
    {
    case 0:
    {
        for(int i=0; i<counts_d[0]; i++)
            events_d[i].siten=0;
        break;
    }
    case 1:
    {
        for(int i=0; i<counts_d[0]; i++)
            events_d[i].siten=events_d[i].pann;
        break;
    }
    case 2:
    {
        for(int i=0; i<counts_d[0]; i++)
            events_d[i].siten=events_d[i].pann*moduleN+events_d[i].modn;
        break;
    }
    }
    int ind=0;
    for(int i=0; i<counts_d[0]; i++)
    {
        Event event0 = events_d[i];
        if(event0.t>9998) continue;
        for(int j=i+1; j<counts_d[0]; j++)
        {
            Event event = events_d[j];
            if((event.parn==event0.parn)&&(event.siten == event0.siten))
            {
                if(policy==1)
                {
                    events_d[ind].x = (event0.x*event0.E + event.x*event.E)/(event0.E + event.E);
                    events_d[ind].y = (event0.y*event0.E + event.y*event.E)/(event0.E + event.E);
                    events_d[ind].z = (event0.z*event0.E + event.z*event.E)/(event0.E + event.E);
                    events_d[ind].E = (event0.E + event.E);
                    events_d[j].t=10000;
                    continue;
                }
                events_d[ind]=(event0.E>event.E)?event0:event;
                events_d[j].t=10000;
            }
        }
        ind++;
    }
    counts_d[0]=ind;
    return 1;
}
int __device__ blur(curandState localstate, int* counts, Event* events, int Eblurpolicy=0, float Eref=511.0f, float Rref=0.26f, float slope=0.0f, float Spaceblur=0.0f)
{
	//this is the energy blurring part in digitizer
    for(int i=0; i<counts[0]; i++)
    {
        if(Eblurpolicy==0) events[i].E+=curand_normal(&localstate)*sqrt(Eref*events[i].E)*Rref/2.355;
        if(Eblurpolicy==1)
        {
            float R=Rref+slope*(events[i].E-Eref);
            events[i].E+=curand_normal(&localstate)*R*events[i].E/2.355;
        }
//other distribution of energy blurring need to be implemented
        if(Spaceblur>0)
        {   events[i].x+=Spaceblur*curand_normal(&localstate);
            events[i].y+=Spaceblur*curand_normal(&localstate);
            events[i].z+=Spaceblur*curand_normal(&localstate);
        }
    }
    return 1;
}

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
    float i = idleph*(e-elaph0) + 0.5f;
    return tex1D(lamph_tex, matid * NLAPH + i);
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
    return tex1D(rayle_tex, matid * NRAYL + i);
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
    return tex1D(compt_tex, matid * NCMPT + i);
}


extern "C"
__global__ void photon(Event* events_d, int* counts_d, const int nactive, const int bufferID, float* dens, int *mat, int *panelID, float *lenx, float *leny, float *lenz,
                       float *MODx, float *MODy, float *MODz, float *Msx, float *Msy, float *Msz, float *LSOx, float *LSOy, float *LSOz, float *sx, float *sy, float *sz,
                       float *ox, float *oy, float *oz, float *dx, float *dy, float *dz, float *UXx, float *UXy, float *UXz,
                       float *UYx, float *UYy, float *UYz,float *UZx, float *UZy, float *UZz)
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
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int tid=threadIdx.x;

    Event events[5];
    int counts[2]= {0,5};
    Event event;

    float tempDen=0.0f;
    int tempMat=0;

    __shared__ int nsstktemp;
    __shared__ float sftemp[NSSTACKSHARED];
    __shared__ int sidtemp[NSSTACKSHARED];
    extern __shared__ float s[];
    float *dens_S = s;
    int *mat_S = (int*)&dens_S[2];
    int *panelID_S = (int*)&mat_S[2];
    float *lenx_S = (float*)&panelID_S[dev_totalPanels];
    float *leny_S = (float*)&lenx_S[dev_totalPanels];
    float *lenz_S = (float*)&leny_S[dev_totalPanels];
    float *MODx_S = (float*)&lenz_S[dev_totalPanels];
    float *MODy_S = (float*)&MODx_S[dev_totalPanels];
    float *MODz_S = (float*)&MODy_S[dev_totalPanels];
    float *Msx_S = (float*)&MODz_S[dev_totalPanels];
    float *Msy_S = (float*)&Msx_S[dev_totalPanels];
    float *Msz_S = (float*)&Msy_S[dev_totalPanels];
    float *LSOx_S = (float*)&Msz_S[dev_totalPanels];
    float *LSOy_S = (float*)&LSOx_S[dev_totalPanels];
    float *LSOz_S = (float*)&LSOy_S[dev_totalPanels];
    float *sx_S = (float*)&LSOz_S[dev_totalPanels];
    float *sy_S = (float*)&sx_S[dev_totalPanels];
    float *sz_S = (float*)&sy_S[dev_totalPanels];
    float *ox_S = (float*)&sz_S[dev_totalPanels];
    float *oy_S = (float*)&ox_S[dev_totalPanels];
    float *oz_S = (float*)&oy_S[dev_totalPanels];
    float *dx_S = (float*)&oz_S[dev_totalPanels];
    float *dy_S = (float*)&dx_S[dev_totalPanels];
    float *dz_S = (float*)&dy_S[dev_totalPanels];
    float *UXx_S = (float*)&dz_S[dev_totalPanels];
    float *UXy_S = (float*)&UXx_S[dev_totalPanels];
    float *UXz_S = (float*)&UXy_S[dev_totalPanels];
    float *UYx_S = (float*)&UXz_S[dev_totalPanels];
    float *UYy_S = (float*)&UYx_S[dev_totalPanels];
    float *UYz_S = (float*)&UYy_S[dev_totalPanels];
    float *UZx_S = (float*)&UYz_S[dev_totalPanels];
    float *UZy_S = (float*)&UZx_S[dev_totalPanels];
    float *UZz_S = (float*)&UZy_S[dev_totalPanels];

    if(tid==0)
    {
        nsstktemp = 0;
    }

    if(tid==0)
    {
        for (int i=0; i<2; i++)
        {
            mat_S[i]=mat[i];
            dens_S[i]=dens[i];
        }
        for(int i=0; i<dev_totalPanels; i++)
        {
            panelID_S[i]=panelID[i];
            lenx_S[i]=lenx[i];
            leny_S[i]=leny[i];
            lenz_S[i]=lenz[i];
            MODx_S[i]=MODx[i];
            MODy_S[i]=MODy[i];
            MODz_S[i]=MODz[i];
            Msx_S[i]=Msx[i];
            Msy_S[i]=Msy[i];
            Msz_S[i]=Msz[i];
            LSOx_S[i]=LSOx[i];
            LSOy_S[i]=LSOy[i];
            LSOz_S[i]=LSOz[i];
            sx_S[i]=sx[i];
            sy_S[i]=sy[i];
            sz_S[i]=sz[i];
            ox_S[i]=ox[i];
            oy_S[i]=oy[i];
            oz_S[i]=oz[i];
            dx_S[i]=dx[i];
            dy_S[i]=dy[i];
            dz_S[i]=dz[i];
            UXx_S[i]=UXx[i];
            UXy_S[i]=UXy[i];
            UXz_S[i]=UXz[i];
            UYx_S[i]=UYx[i];
            UYy_S[i]=UYy[i];
            UYz_S[i]=UYz[i];
            UZx_S[i]=UZx[i];
            UZy_S[i]=UZy[i];
            UZz_S[i]=UZz[i];
        }
    }
    __syncthreads();


//  obtain current id on thread
    if( id < nactive)
    {
        curandState localState = cuseed[id];
        float4 xtemp = x_gPET[id];
        float4 vxtemp = vx_gPET[id];

        // change global coordinates to local coordinates
        int paID=-1;
        float4 xtemp2;
        float4 vxtemp2;
        //get the panelid crystal id that the particle enters???
        for (int i=0; i<dev_totalPanels; i++)
        {

            float tempx=(xtemp.x-ox_S[i])*UXx_S[i]+(xtemp.y-oy_S[i])*UXy_S[i]+(xtemp.z-oz_S[i])*UXz_S[i];
            float tempy=(xtemp.x-ox_S[i])*UYx_S[i]+(xtemp.y-oy_S[i])*UYy_S[i]+(xtemp.z-oz_S[i])*UYz_S[i];
            float tempz=(xtemp.x-ox_S[i])*UZx_S[i]+(xtemp.y-oy_S[i])*UZy_S[i]+(xtemp.z-oz_S[i])*UZz_S[i];

            float tempvx=vxtemp.x*UXx[i]+vxtemp.y*UXy[i]+vxtemp.z*UXz[i];
            float tempvy=vxtemp.x*UYx[i]+vxtemp.y*UYy[i]+vxtemp.z*UYz[i];
            float tempvz=vxtemp.x*UZx[i]+vxtemp.y*UZy[i]+vxtemp.z*UZz[i];

            float tempx2=0.0f;
            if(tempvx*dx_S[i]>=0)
            {

                float tempy2=tempy-tempx/tempvx*tempvy;
                float tempz2=tempz-tempx/tempvx*tempvz;

                if(abs(tempy2)<leny_S[i]/2 & abs(tempz2)<lenz_S[i]/2)
                {
                    xtemp2.x=tempx2;
                    xtemp2.y=tempy2;
                    xtemp2.z=tempz2;
                    xtemp2.w=xtemp.w;

                    vxtemp2.x=tempvx;
                    vxtemp2.y=tempvy;
                    vxtemp2.z=tempvz;
                    vxtemp2.w=vxtemp.w;
                    paID=panelID_S[i];
                    break;
                }
            }
            float tempy2=tempy-tempx/tempvx*tempvy;
            float tempz2=tempz-tempx/tempvx*tempvz;
            xtemp2.x=tempx2;
            xtemp2.y=tempy2;
            xtemp2.z=tempz2;
            xtemp2.w=xtemp.w;

            vxtemp2.x=tempvx;
            vxtemp2.y=tempvy;
            vxtemp2.z=tempvz;
            vxtemp2.w=vxtemp.w;
            paID=-1;
        }

//      Loop until it either escapes or is absorbed:
        for(;;)
        {
            if (paID==-1)
                break;
            //      Get lambda from the minimum lambda at the current energy:
            float lammin = lamwck(vxtemp2.w);
            float  s = -lammin*__logf(curand_uniform(&localState));

            //		Get the coordinates of the photon after passing a free length
            xtemp2.x += s*vxtemp2.x;
            xtemp2.y += s*vxtemp2.y;
            xtemp2.z += s*vxtemp2.z;

            // if out of panel
            if (paID==-1|abs(xtemp2.y)>leny_S[paID]/2| abs(xtemp2.z)>lenz_S[paID]/2 |(xtemp2.x*dx_S[paID])<0 |(xtemp2.x*dx_S[paID])>lenx_S[paID])
            {
                break;
            }

            int m_id=-1; //material id
            int M_id=-1; // module id
            int L_id=-1; // LSO id

            LSOsearch(xtemp2,leny_S[paID],lenz_S[paID],MODy_S[paID],MODz_S[paID],Msy_S[paID],Msz_S[paID], LSOy_S[paID],LSOz_S[paID],sy_S[paID],sz_S[paID],dy_S[paID], dz_S[paID], &m_id, &M_id, &L_id);

            tempDen = dens_S[m_id];
            tempMat = mat_S[m_id];


            //  Apply Woodcock trick:
            float lamden = lammin*tempDen;
            float prob = 1.0-lamden*itphip_G(tempMat, vxtemp2.w);
            float randno = curand_uniform(&localState);
            //  Compton:
            //  No real event; continue jumping:
            if (randno < prob)
                continue;


            prob += lamden*icptip(tempMat, vxtemp2.w);

            if (randno < prob)
            {
                float efrac, costhe;
                comsam(vxtemp2.w, &localState, &efrac, &costhe);//, tempMat);
                float de = vxtemp2.w * (1.0f-efrac);
                float phi = TWOPI*curand_uniform(&localState);

                //record events
                if (nsstktemp!= NSSTACKSHARED)
                {
                    int ind = atomicAdd(&nsstktemp,5);
                    sidtemp[ind] = id+bufferID-nactive+1;
                    sidtemp[ind+1] = paID;
                    sidtemp[ind+2] = M_id;
                    sidtemp[ind+3] = L_id;
                    sidtemp[ind+4] = 1;

                    event.parn=id+bufferID-nactive+1;
                    event.cryn=L_id-1;
                    event.modn= M_id-1;
                    event.pann=paID;
                    event.siten=event.pann*moduleN*crystalN+event.modn*crystalN+event.cryn;//maybe should use event.pann-1 or event.cryn-1 to make sure siten start from 0

                    sftemp[ind] = de;//s;
                    sftemp[ind+1] = xtemp2.w;//s;
                    sftemp[ind+2] = xtemp2.x;
                    sftemp[ind+3] = xtemp2.y;
                    sftemp[ind+4] = xtemp2.z;
                    event.E = de;
                    event.t = xtemp2.w;
                    event.x = xtemp2.x;
                    event.y = xtemp2.y;
                    event.z = xtemp2.z;
                    adder(counts,events,event);
                }

                vxtemp2.w -= de;
                if (vxtemp2.w < eabsph)
                {

                    if (nsstktemp!= NSSTACKSHARED)
                    {
                        int ind = atomicAdd(&nsstktemp,5);
                        sidtemp[ind] = id+bufferID-nactive+1;
                        sidtemp[ind+1] = paID;
                        sidtemp[ind+2] = M_id;
                        sidtemp[ind+3] = L_id;
                        sidtemp[ind+4] = 2;

                        event.parn=id+bufferID-nactive+1;
                        event.cryn=L_id-1;
                        event.modn= M_id-1;
                        event.pann=paID;
                        event.siten=event.pann*moduleN*crystalN+event.modn*crystalN+event.cryn;


                        sftemp[ind] = vxtemp2.w;//s;
                        sftemp[ind+1] = xtemp2.w;//s;
                        sftemp[ind+2] = xtemp2.x;
                        sftemp[ind+3] = xtemp2.y;
                        sftemp[ind+4] = xtemp2.z;

                        event.E = vxtemp2.w;
                        event.t = xtemp2.w;
                        event.x =xtemp2.x;
                        event.y = xtemp2.y;
                        event.z = xtemp2.z;
                        adder(counts,events,event);
                    }
                    break;
                }

                rotate(&vxtemp2.x,&vxtemp2.y,&vxtemp2.z,costhe,phi);
                continue;
            }

//	Rayleigh:
            prob += lamden*irylip(tempMat, vxtemp2.w);
            if (randno < prob)
            {
                float costhe;
                rylsam(vxtemp2.w, tempMat, &localState, &costhe);
                float phi = TWOPI*curand_uniform(&localState);

                if (nsstktemp!= NSSTACKSHARED)
                {
                    int ind = atomicAdd(&nsstktemp,5);
                    sidtemp[ind] = id+bufferID-nactive+1;
                    sidtemp[ind+1] = paID;
                    sidtemp[ind+2] = M_id;
                    sidtemp[ind+3] = L_id;
                    sidtemp[ind+4] = 3;
                    sftemp[ind] = 0.0f;//s;
                    sftemp[ind+1] = xtemp2.w;//s;
                    sftemp[ind+2] = xtemp2.x;
                    sftemp[ind+3] = xtemp2.y;
                    sftemp[ind+4] = xtemp2.z;
                }
                rotate(&vxtemp2.x,&vxtemp2.y,&vxtemp2.z,costhe,phi);
                continue;
            }
//  Photoelectric:
            if (nsstktemp!= NSSTACKSHARED)
            {
                int ind = atomicAdd(&nsstktemp,5);
                sidtemp[ind] = id+bufferID-nactive+1;
                sidtemp[ind+1] = paID;
                sidtemp[ind+2] = M_id;
                sidtemp[ind+3] = L_id;
                sidtemp[ind+4] = 4;

                sftemp[ind] = vxtemp2.w;//s;
                sftemp[ind+1] = xtemp2.w;//s;
                sftemp[ind+2] = xtemp2.x;
                sftemp[ind+3] = xtemp2.y;
                sftemp[ind+4] = xtemp2.z;

                event.parn=id+bufferID-nactive+1;
                event.cryn=L_id-1;
                event.modn= M_id-1;
                event.pann=paID;
                event.siten=event.pann*moduleN*crystalN+event.modn*crystalN+event.cryn;
                event.E = vxtemp2.w;
                event.t = xtemp2.w;
                event.x = xtemp2.x;
                event.y = xtemp2.y;
                event.z = xtemp2.z;
                adder(counts, events, event);//this is for digitizer
            }
            break;
        }
        //readout(counts,events,3,0);
        blur(localState,counts,events,0, 511000.0f, 0.19f, 0.0f, 0.0f);
        if(counts[0])
        {
            int ind=atomicAdd(counts_d,counts[0]);
            //if(!(id%60000)) printf("%d %d\n",counts[0],ind);
            for(int i=0; i<counts[0]; i++)
                events_d[ind+i]=events[i];
        }
        cuseed[id] = localState;
        //id+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    __shared__ int istart;
    if(threadIdx.x==0)
    {
        //printf("nsstktemp1 = %d\n",nsstktemp);
        istart = atomicAdd(&nsstk, nsstktemp);
        //printf("istart = %d\n",istart);
    }
    id=(blockIdx.x*blockDim.x + threadIdx.x);
    //if(id==0) printf("total events=%d\ncurrent total hits=%d\n", counts_d[0],nsstk);
    __syncthreads();

    for(int i = 0; i < 1+(nsstktemp)/blockDim.x; i++)
    {
        if(nsstktemp == 0)
            break;

        int ind = istart + i*blockDim.x + tid;


        if(ind < istart + nsstktemp && ind<NSSTACK)
        {
            sf[ind] = sftemp[i*blockDim.x + tid];//this is for hits events
            sid[ind] = sidtemp[i*blockDim.x + tid];
        }
    }
    __syncthreads();

}

// crystal index and material type
__device__ void LSOsearch(float4 xtemp2,float leny_S,float lenz_S,float MODy_S,float MODz_S,float Msy_S,float Msz_S,float LSOy_S,float LSOz_S,float sy_S,float sz_S,float dy_S, float dz_S, int *m_id, int *M_id, int *L_id)
{

    float delt=1e-8f;
    float delt2=1e-3f;
    float deltay=leny_S/2*dy_S+xtemp2.y;
    float deltaz=lenz_S/2*dz_S+xtemp2.z;

    int M_id_y=0;
    int M_id_z=0;

    if(deltay/(MODy_S+Msy_S)-float(floor(deltay/(MODy_S+Msy_S)))>delt)
    {
        M_id_y=ceilf(deltay/(MODy_S+Msy_S));   // module id along x
    }
    else
        M_id_y=floor(deltay/(MODy_S+Msy_S))>0?floor(deltay/(MODy_S+Msy_S)):1;
    if(deltaz/(MODz_S+Msz_S)-float(floor(deltaz/(MODz_S+Msz_S)))>delt)
    {
        M_id_z=ceilf(deltaz/(MODz_S+Msz_S));   // module id along y
    }
    else
        M_id_z=floor(deltaz/(MODz_S+Msz_S))>0?floor(deltaz/(MODz_S+Msz_S)):1;

    int Md_y=ceilf(leny_S/(MODy_S+Msy_S));     // total module along x
    int Md_z=ceilf(lenz_S/(MODz_S+Msz_S));     // total module along y

    *M_id=(M_id_z-1)*Md_z+M_id_y;

    int L_id_y=0;
    int L_id_z=0;


    if((deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S)-float(floor((deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S)))>delt2)
    {
        L_id_y=ceilf((deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S)); // module id along y
    }
    else
        L_id_y=floor((deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S))>0?floor((deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S)):1;

    if((deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S)-float(floor((deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S)))>delt2)
    {
        L_id_z=ceilf((deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S));  // module id along x
    }
    else
        L_id_z=floor((deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S))>0?floor((deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S)):1;

    int d_z=ceilf((MODz_S+Msz_S)/(LSOz_S+sz_S));

    *L_id=(L_id_z-1)*d_z+L_id_y;

    float L_y=(deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S)-float(floor((deltay-(M_id_y-1)*(MODy_S+Msy_S))/(LSOy_S+sy_S)));
    float L_z=(deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S)-float(floor((deltaz-(M_id_z-1)*(MODz_S+Msz_S))/(LSOz_S+sz_S)));
    float r_y=LSOy_S/(LSOy_S+sy_S);
    float r_z=LSOz_S/(LSOz_S+sz_S);

    if(L_y>=0 & L_y<=r_y& L_z>=0 &L_z<=r_z)
    {
        *m_id=0;
    }
    else
        *m_id=1;
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

/*__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe, int matid)
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
	float indcp = curand_uniform(localState_pt)*idcpcm;
	float inde = energytemp*idecm;
	float temp = tex3D(s_tex,inde+0.5f, indcp+0.5f, matid+0.5f);
	if(temp > 1.0f) temp = 1.0f;
    if(temp < -1.0f) temp = -1.0f;
    *costhe = temp;

	*efrac = 1.0f/(1.0f + energytemp*IMC2*(1.0f - temp));
}*/
__device__ void comsam(float energytemp, curandState *localState_pt, float *efrac, float *costhe)
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
    float e0,kmin2,loge,onecost,reject;

    e0 = energytemp/MC2;
    kmin2 = 1.0/((1.0+2.0*e0)*(1.0+2.0*e0));
    loge = __logf(1.0+2.0*e0);

    do
    {
        if (curand_uniform(localState_pt)*(loge+(1.0-kmin2)/2)< loge)
        {
            *efrac = expf(-curand_uniform(localState_pt)*loge);
        }
        else
        {
            *efrac = sqrtf(kmin2+curand_uniform(localState_pt)*(1.0-kmin2));
        }
        onecost = (1.0-(*efrac))/((*efrac)*e0);
        reject = 1.0-(*efrac)*onecost*(2.0-onecost)/(1.0+(*efrac)*(*efrac));
    } while(reject<curand_uniform(localState_pt));

    *costhe = 1.0-onecost;
}

__device__ void rotate(float *u, float *v, float *w, float costh, float phi)
/*******************************************************************
c*    Rotates a vector; the rotation is specified by giving        *
c*    the polar and azimuthal angles in the "self-frame", as       *
c*    determined by the vector to be rotated.                      *
c*                                                                 *
c*    Input:                                                       *
c*      (u,v,w) -> input vector (=d) in the lab. frame             *
c*      costh -> cos(theta), angle between d before and after turn *
c*      phi -> azimuthal angle (rad) turned by d in its self-frame *
c*    Output:                                                      *
c*      (u,v,w) -> rotated vector components in the lab. frame     *
c*    Comments:                                                    *
c*      -> (u,v,w) should have norm=1 on input; if not, it is      *
c*         renormalized on output, provided norm>0.                *
c*      -> The algorithm is based on considering the turned vector *
c*         d' expressed in the self-frame S',                      *
c*           d' = (sin(th)cos(ph), sin(th)sin(ph), cos(th))        *
c*         and then apply a change of frame from S' to the lab     *
c*         frame. S' is defined as having its z' axis coincident   *
c*         with d, its y' axis perpendicular to z and z' and its   *
c*         x' axis equal to y'*z'. The matrix of the change is then*
c*                   / uv/rho    -v/rho    u \                     *
c*          S ->lab: | vw/rho     u/rho    v |  , rho=(u^2+v^2)^0.5*
c*                   \ -rho       0        w /                     *
c*      -> When rho=0 (w=1 or -1) z and z' are parallel and the y' *
c*         axis cannot be defined in this way. Instead y' is set to*
c*         y and therefore either x'=x (if w=1) or x'=-x (w=-1)    *
c******************************************************************/
{
    float rho2,sinphi,cosphi,sthrho,urho,vrho,sinth,norm;

    rho2 = (*u)*(*u)+(*v)*(*v);
    norm = rho2 + (*w)*(*w);
//      Check normalization:
    if (fabs(norm-1.0) > SZERO)
    {
//      Renormalize:
        norm = 1.0/__fsqrt_rn(norm);
        *u = (*u)*norm;
        *v = (*v)*norm;
        *w = (*w)*norm;
    }

    sinphi = __sinf(phi);
    cosphi = __cosf(phi);
//      Case z' not= z:

    float temp = costh*costh;
    if (rho2 > ZERO)
    {
        if(temp < 1.0f)
            sthrho = __fsqrt_rn((1.00-temp)/rho2);
        else
            sthrho = 0.0f;

        urho =  (*u)*sthrho;
        vrho =  (*v)*sthrho;
        *u = (*u)*costh - vrho*sinphi + (*w)*urho*cosphi;
        *v = (*v)*costh + urho*sinphi + (*w)*vrho*cosphi;
        *w = (*w)*costh - rho2*sthrho*cosphi;
    }
    else
//      2 especial cases when z'=z or z'=-z:
    {
        if(temp < 1.0f)
            sinth = __fsqrt_rn(1.00-temp);
        else
            sinth = 0.0f;

        *v = sinth*sinphi;
        if (*w > 0.0)
        {
            *u = sinth*cosphi;
            *w = costh;
        }
        else
        {
            *u = -sinth*cosphi;
            *w = -costh;
        }
    }
}
#endif
