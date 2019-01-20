#ifndef __FINALIZE_CU__
#define __FINALIZE_CU__
#include "digitizer.h"
#include <fstream>
#include <iostream>

compare_parn compare1;
compare_siten compare2;
compare_t compare3;

void quicksort(Event*  hits,int start, int stop, int sorttype)
{   
    //CPU sort function for ordering events in cpu memory
    //sorttype  1 for ordering by particle #,2 for site number, 3 for flight time
    switch(sorttype)
    {
    case 1:
    {   sort(hits+start,hits+stop,compare1);
        break;
    }
    case 2:
    {   sort(hits+start,hits+stop,compare2);
        break;
    }
    case 3:
    {   sort(hits+start,hits+stop,compare3);
        break;
    }
    }
}
void quicksort_d(Event* events_d,int start, int stop, int sorttype)
{
    //GPU version for ordering the events in gpu memory,
    //more suitable for large scale sorting
    //sorttype  1 for ordering by particle #,2 for site number, 3 for flight time

    /*Event* events=(Event*) malloc(sizeof(Event)*(stop-start-1));
    cudaMemcpy(events,events_d,sizeof(Event)*(stop-start-1),cudaMemcpyDeviceToHost);
    quicksort(events,0,stop-start,sorttype);
    cudaMemcpy(events_d,events,sizeof(Event)*(stop-start-1),cudaMemcpyHostToDevice);
    free(events);//*/
    printf("gpu sort starts!!\n");
    thrust::device_ptr<Event> hits=thrust::device_pointer_cast(events_d);
    switch(sorttype)
    {
    case 1:
    {   thrust::sort(hits+start,hits+stop,compare1);
        break;
    }
    case 2:
    {   thrust::sort(hits+start,hits+stop,compare2);
        break;
    }
    case 3:
    {   thrust::sort(hits+start,hits+stop,compare3);
        break;
    }
    }
    printf("gpu sort finishs!!\n");//*/
}

void orderevents(int* counts,Event* events_d)
{
    /*quicksort_d(events_d,0,counts[0],2);
    Event* events=(Event*) malloc(sizeof(Event)*counts[0]);
    cudaMemcpy(events,events_d,sizeof(Event)*counts[0],cudaMemcpyDeviceToHost);
    cout<<"total events "<<counts[0]<<endl;
    int start=0;
    for(int i=1;i<counts[0];)
    {
       while(events[i].siten==events[start].siten&&(i<counts[0]))
           i++;
       if(i>start+1) quicksort_d(events_d,start,i,3);
       start=i;
       i++;
    }
    cudaMemcpy(events_d,events,sizeof(Event)*counts[0],cudaMemcpyHostToDevice);
    free(events);*/
    Event* events=(Event*) malloc(sizeof(Event)*counts[0]);
    cudaMemcpy(events,events_d,sizeof(Event)*counts[0],cudaMemcpyDeviceToHost);
    quicksort(events,0,counts[0],2);
    int start=0;
    for(int i=1; i<counts[0];)
    {
        while(events[i].siten==events[start].siten&&(i<counts[0]))
            i++;
        if(i>start+1) quicksort(events,start,i,3);
        start=i;
        i++;
    }
    cudaMemcpy(events_d,events,sizeof(Event)*counts[0],cudaMemcpyHostToDevice);
    free(events);
}
void outputData(const char *srcname, const int size, const char *outputfilename, const char *mode)
//      output data to file
{
    void *tempData;
    cudaMalloc( (void **) &tempData, size);
    if( cudaMemcpyFromSymbol(tempData, srcname, size, 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "error in getting " << srcname << endl;

    outputData(tempData,size, outputfilename, mode);
    cudaFree(tempData) ;
}
void outputData(void *src, const int size, const char *outputfilename, const char *mode)
//      output data to file
{
//      copy data from GPU to CPU
    void *tempData_h = malloc( size );
    cudaMemcpy( tempData_h, src, size, cudaMemcpyDeviceToHost) ;

//      write results to file
    FILE *fp;
    fp = fopen(outputfilename, mode);
    if( fp == NULL )
    {
        cout << "Can not open file to write results.";
        exit(1);
    }
    fwrite (tempData_h, size, 1 , fp );
    fclose(fp);

//      free space
    free(tempData_h);
}
int outevents(int* num_d, Event* totalevents_d, const char *outputfilename)
{
//copy data from device to host
//renewed at 1025, do not have to be the memory on GPU
    int num;
    if(cudaMemcpy(&num, num_d, sizeof(int), cudaMemcpyDeviceToHost)!=cudaSuccess)
        num=num_d[0];
    cout<<"num is "<<num<<endl;
    Event* tempData_h =(struct Event*) malloc( sizeof(Event)*num);
    if(cudaMemcpy(tempData_h, totalevents_d, sizeof(Event)*num, cudaMemcpyDeviceToHost)!=cudaSuccess)
        memcpy(tempData_h, totalevents_d, sizeof(Event)*num);
    cout<<"data has been copied"<<endl;
//  write results to file
    ofstream out(outputfilename,ios::app|ios::binary);
    out.write((char*) tempData_h,sizeof(Event)*num);
    out.close();
    cout<<"data has been written to "<<outputfilename<<"\n";
//  free space
    free(tempData_h);
    return 1;
}

void fina()
/*******************************************************************
c*    finalizes the gCTD system
c******************************************************************/
{
//      mark the start time
    time_t start_time, end_time;
    float time_diff;

    start_time = clock();


//      printf any results during computing
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();


//      free lamph info
    cudaUnbindTexture(lamph_tex) ;
    cudaFreeArray(lamph) ;

//      free cmpt info
    cudaUnbindTexture(compt_tex);
    cudaFreeArray(compt);
    cudaUnbindTexture(s_tex) ;
    cudaFreeArray(sArray) ;

//      free phote info
    cudaUnbindTexture(phote_tex);
    cudaFreeArray(phote);

//      free rayle info
    cudaUnbindTexture(rayle_tex) ;
    cudaFreeArray(rayle) ;
    cudaUnbindTexture(f_tex);
    cudaFreeArray(fArray);

//      free wck info
    cudaUnbindTexture(woock_tex) ;
    cudaFreeArray(woock) ;

    cudaFree(cuseed);
    cudaFree(iseed1);
    cudaFree(sf);
    cudaFree(sid);
    cudaFree(x_gPET);
    cudaFree(vx_gPET);

//  cudaFree(x_phap_gPET);
//  cudaFree(vx_phap_gPET);

//      mark the end timer
    printf("\n");
    printf("\n");
    printf("Finalize: Done.\n");

    end_time = clock();
    time_diff = ((float)end_time - (float)start_time)/CLOCKS_PER_SEC;
    printf("\n\n****************************************\n");
    printf("Finalization time: %f s.\n\n",time_diff);
    printf("****************************************\n\n\n");
}
#endif

