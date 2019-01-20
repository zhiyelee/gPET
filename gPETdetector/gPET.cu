#ifndef __GPET_CU__
#define __GPET_CU__
#include "digitizer.h"

void runCalculation(Source source, char fname[100])
{
    //  setup timer events
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    cudaEventRecord(event1);


    cout << endl << endl;
    cout << "================================" << endl;
    cout << "Performing gPET simulations...." << endl;
    bufferHeadId = 0;

    simulateParticles(source, fname);

    cudaThreadSynchronize();

    //      output timer
    cudaEventRecord(event2);
    cudaEventSynchronize(event2);
    float dt_ms = 0;
    cudaEventElapsedTime(&dt_ms, event1, event2);
    cout << "simulation time: " << dt_ms << " ms" << endl;
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

//      run simulations from phase space file
void simulateParticles(Source source, char fname[100])
{
    clrStk();
//  if enough number of particles are simulated
    bool ifenough_load = false;
    bool ifenough = false;
    int idRepeat = 0;

    int totalPanels_h = 0;
    if( cudaMemcpyFromSymbol(&totalPanels_h, dev_totalPanels, sizeof(int), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "error in getting total panels" << endl;
    char fname0[100], fname1[100];
    strcpy(fname0,fname);
    strcat(fname0,".dat");
    strcpy(fname1,fname);
    strcat(fname1,"ID.dat");

//    clrStk();
//  loop until no particles from ps file
//  count the loop times
    Event* events_d;
    cudaMalloc(&events_d,sizeof(Event)*(NPART*3));
    cout<<source.NSource<<"  total particle number\n";
    int temptemp[2]= {0,NPART*3};
    int* counts_d;
    cudaMalloc(&counts_d,sizeof(int)*2);


    int n=0;
    while(1)
    {
        int nsstk_h = 0;
        if( cudaMemcpyFromSymbol(&nsstk_h, nsstk, sizeof(int), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
            cout << "first error in getting nsstk" << endl;
        void *tempData;
        cudaMalloc( (void **) &tempData, sizeof(int)*nsstk_h);
        if( cudaMemcpyFromSymbol(tempData, sid, sizeof(int)*nsstk_h, 0,cudaMemcpyDeviceToDevice) != cudaSuccess)
            cout << "first error in getting sid " << endl;
        outputData(tempData,sizeof(float)*(nsstk_h), fname1, "ab");
        cudaFree(tempData);

        void *tempData2;
        cudaMalloc( (void **) &tempData2, sizeof(float)*nsstk_h);
        if( cudaMemcpyFromSymbol(tempData2, sf, sizeof(float)*nsstk_h, 0,cudaMemcpyDeviceToDevice) != cudaSuccess)
            cout << "first error in getting sf " << endl;
        outputData(tempData2,sizeof(float)*(nsstk_h), fname0, "ab");
        cudaFree(tempData2);

        int temp = 0;
        if( cudaMemcpyToSymbol(nsstk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice)!= cudaSuccess)
            cout << "first error in setting nsstk" << endl;

//  break if all particles simulated
        if( ifenough == true)
        {
            break;
            //  cout << "1" << endl;
        }
        else
        {
//  generate source particles from ps file
            int first, last;

            first = bufferHeadId;
            last = first + NPART -1;
            if(last >= source.NSource-1)
            {
                last = source.NSource-1;
                ifenough= true;
                //  cout << "success 2" << endl;
            }
            loadFromPSfile(source,first, last);
            nparload_h = last - first + 1;
            bufferHeadId = last;

            //cout << "Simulating " << nparload_h << " particles from PSF file " <<endl;

            nactive_h = nparload_h;
            ptype_h = 0;
            totalSimPar += nactive_h;
            float *tempaddress;

            if(cudaGetSymbolAddress((void**) &tempaddress,x_gPET) != cudaSuccess)
                cout<< "error in getting symbol address while computing weights" << endl;
            totalWeight_gPET += cublasSasum(nactive_h, tempaddress+3, 4);
        }

        //  cout << " Number of active particles: " << nactive_h << endl;
        //  cout << "Particle type: " << ptype_h << endl;

//      simulate a batch particles
        if (ptype_h == 0 && nactive_h>0)
        {
            int nblocks = 1 + (nactive_h - 1)/NTHREAD_PER_BLOCK_GPET ;

// the size of the external shared memory is calculated as follows, but the specific number is hard coding!!! Be careful when changing the input parameters!!!
            size_t nShared = (totalPanels_h+2)*sizeof(int)+(30*totalPanels_h+2)*sizeof(float);

            cudaMemcpy(counts_d,temptemp,sizeof(int)*2,cudaMemcpyHostToDevice);
            //printf(" shared memory is totalPanels_h=%d\n", totalPanels_h);
            photon<<<nblocks, NTHREAD_PER_BLOCK_GPET,nShared>>>(events_d,counts_d,nactive_h, bufferHeadId,dens_panel, mat_panel, panelID, lengthx_panel, lengthy_panel, lengthz_panel,
                    MODx_panel, MODy_panel, MODz_panel, Mspacex_panel, Mspacey_panel, Mspacez_panel,
                    LSOx_panel, LSOy_panel, LSOz_panel, spacex_panel, spacey_panel, spacez_panel,
                    offsetx_panel, offsety_panel, offsetz_panel, directionx_panel, directiony_panel, directionz_panel,
                    UniXx_panel, UniXy_panel, UniXz_panel, UniYx_panel, UniYy_panel, UniYz_panel,
                    UniZx_panel, UniZy_panel, UniZz_panel);
            cudaThreadSynchronize();
            nactive_h = 0;
        }
        printf("kernel is ok\n");

        //GPU to CPU
        int counts=0;
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after adder is "<<counts<<endl;
        //outevents(&counts,events_d,"blur.dat");

//insert proper digitizer module in the following part

        energywindow<<<counts/512+1,512>>>(counts_d,events_d, counts, 50000,1000000);
        cudaThreadSynchronize();
        quicksort_d(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after thresholder is "<<counts<<endl;
        //outevents(&counts,events_d,"thresholder.dat");

        addnoise<<<10,100>>>(counts_d, events_d, 7.57e-6f, 450000, 30000, 1e-3f);
        cudaThreadSynchronize();
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after noise is "<<counts<<endl;
        //outevents(&counts,events_d,"noise.dat");//*/

        orderevents(&counts,events_d);//make events globally ordered by site number, and then ordered by flight time in each volume

        cout<<"order is ok"<<endl;//*/
        //outevents(&counts,events_d,"afterorder.dat");

        deadtime<<<counts/512+1,512>>>(counts_d,events_d, counts, 2.2e-6f, 0);
        cudaThreadSynchronize();
        cout<<"deadtime is ok\n";
        quicksort_d(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after deadtime is "<<counts<<endl;
        //outevents(&counts,events_d,"deadtime.dat");

        energywindow<<<counts/512+1,512>>>(counts_d,events_d, counts, 350000,700000);
        cudaThreadSynchronize();
        quicksort_d(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of singles is "<<counts<<endl;
        outevents(&counts,events_d,"singles.dat");

        coinsorter<<<counts/512+1,512>>>(counts_d,events_d,2.4e-8f,4,counts);
        cudaThreadSynchronize();
        quicksort_d(events_d,0,counts,3);
        cudaMemcpy(&counts,counts_d,sizeof(int),cudaMemcpyDeviceToHost);
        cout<<"counts of events after coincidence is "<<counts<<endl;
        outevents(&counts,events_d,"coincidence.dat");//*/
    }

    cudaFree(events_d);
    cudaFree(counts_d);

    int nsstk_h = 0;
    if( cudaMemcpyFromSymbol(&nsstk_h, nsstk, sizeof(int), 0, cudaMemcpyDeviceToHost) != cudaSuccess)
        cout << "error in getting nsstk" << endl;
    //cout<< "particle number    " << nsstk_h << endl;

    int size=sizeof(int)*(nsstk_h);
    void *tempData;
    cudaMalloc( (void **) &tempData, size);

    if( cudaMemcpyFromSymbol(tempData, sid, size, 0,cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "error in getting sf " << endl;

    outputData(tempData,size, fname1, "ab");
    cudaFree(tempData);

    void *tempData2;
    cudaMalloc( (void **) &tempData2, sizeof(float)*nsstk_h);

    if( cudaMemcpyFromSymbol(tempData2, sf, sizeof(float)*nsstk_h, 0,cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "error in getting sf " << endl;

    outputData(tempData2,sizeof(float)*(nsstk_h), fname0, "ab");
    cudaFree(tempData2);

    int temp = 0;
    if( cudaMemcpyToSymbol(nsstk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice)
            != cudaSuccess)
        cout << "error in setting nsstk" << endl;
}

void clrStk()
//      clear both particle stacks
{
    int temp = 0;
    if( cudaMemcpyToSymbol(nsstk, &temp, sizeof(int), 0, cudaMemcpyHostToDevice)
            != cudaSuccess)
        cout << "error in setting nsstk" << endl;

}

bool loadFromPSfile(Source source, int first, int last)
//  load particles from PS file, return TRUE if no particles available
{
    int n = last - first + 1;

//  load to GPU
    memcpy(&(xbuffer),&(source.xbuffer[first+1]),sizeof(float4)*n);
    memcpy(&(vxbuffer),&(source.vxbuffer[first+1]),sizeof(float4)*n);

    if(cudaMemcpyToSymbol(x_gPET, &(xbuffer[0]),
                          sizeof(float4)*n, 0, cudaMemcpyHostToDevice)!= cudaSuccess)

        cout << "error in setting x_gPET" << endl;

    if(cudaMemcpyToSymbol(vx_gPET, &(vxbuffer[0]),
                          sizeof(float4)*n, 0, cudaMemcpyHostToDevice)!= cudaSuccess)
        cout << "error in setting vx_gPET" << endl;



    //cout << "Loading " << n << " particles from source " << endl;

    return true;
}

#endif
