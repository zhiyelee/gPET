#ifndef __FINALIZE_CU__
#define __FINALIZE_CU__

void outputData(const char* srcname, const int size, const char* outputfilename, const char* mode)
//      output data to file
{
    void* tempData;
    cudaMalloc((void**)&tempData, size);
    if (cudaMemcpyFromSymbol(tempData, srcname, size, 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
        cout << "error in getting " << srcname << endl;

    outputData(tempData, size, outputfilename, mode);
    cudaFree(tempData);
}

void outputData(void* src, const int size, const char* outputfilename, const char* mode)
//      output data to file
{
    //      copy data from GPU to CPU
    void* tempData_h = malloc(size);
    cudaMemcpy(tempData_h, src, size, cudaMemcpyDeviceToHost);
    //  cout << "out put data ... mode " << mode <<" filename "<< outputfilename<< endl;
    //      write results to file
    FILE* fp;
    fp = fopen(outputfilename, mode);
    if (fp == NULL) {
        cout << "Can not open file to write results.";
        exit(1);
    }
    fwrite(tempData_h, size, 1, fp);
    fclose(fp);

    //      free space
    free(tempData_h);
}

void fina()
/*******************************************************************
c*    finalizes the gCTD system
c******************************************************************/
{
    //free memories
    //      mark the start time
    time_t start_time, end_time;
    float time_diff;

    start_time = clock();

    //      printf any results during computing
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();

    //      output dose results

    //      outputData("fEscore", sizeof(float)*NXYZ, outputAveName, "wb");
    //      outputData("fEscor2", sizeof(float)*NXYZ, outputStdName, "wb");

    //      free lamph info
    cudaUnbindTexture(lamph_tex);
    cudaFreeArray(lamph);

    //      free cmpt info
    cudaUnbindTexture(compt_tex);
    cudaFreeArray(compt);
    cudaUnbindTexture(s_tex);
    cudaFreeArray(sArray);

    //      free phote info
    cudaUnbindTexture(phote_tex);
    cudaFreeArray(phote);

    //      free rayle info
    cudaUnbindTexture(rayle_tex);
    cudaFreeArray(rayle);
    cudaUnbindTexture(f_tex);
    cudaFreeArray(fArray);

    //      free mat, struc, and dens info
    cudaUnbindTexture(mat_tex);
    cudaFreeArray(mat);

    cudaUnbindTexture(dens_tex);
    cudaFreeArray(dens);

    //      free texture file for electron transport
    cudaUnbindTexture(scssp_tex);
    cudaFreeArray(scssp);
    //      free st info
    cudaUnbindTexture(stsp_tex);
    cudaFreeArray(stsp);
    //      free scp info
    cudaUnbindTexture(scpsp_tex);
    cudaFreeArray(scpsp);
    //      free qsurf info
    cudaUnbindTexture(q_tex);
    cudaFreeArray(qArray);
    //      free bwsp info
    cudaUnbindTexture(bwsp_tex);
    cudaFreeArray(bwsp);

    //      free wck info
    cudaUnbindTexture(woock_tex);
    cudaFreeArray(woock);

    cudaUnbindTexture(icdf_ZDist_tex);
    cudaFreeArray(icdf_ZDist);
    cudaUnbindTexture(icdf_EDist_tex);
    cudaFreeArray(icdf_EDist);
    cudaUnbindTexture(icdf_PhiDist_tex);
    cudaFreeArray(icdf_PhiDist);

    cudaUnbindTexture(mear_tex);
    cudaFreeArray(mear);

    cudaFree(escore);
    /*  cudaFree(fEscore);
        cudaFree(fEscor2); */

    cudaFree(x_phap_gBrachy);
    cudaFree(vx_phap_gBrachy);

    //      mark the end timer
    printf("\n");
    printf("\n");
    printf("Finalize: Done.\n");

    end_time = clock();
    time_diff = ((float)end_time - (float)start_time) / 1000.0;
    printf("\n\n****************************************\n");
    printf("Finalization time: %f ms.\n\n", time_diff);
    printf("****************************************\n\n\n");
}

PatientDose getDose()
{

    PatientDose patientDose;
    patientDose.doseAve = new float[NXYZ];
    // patientDose.doseStd =  new float[NXYZ];

    //        void *tempData;
    //        cudaMalloc( (void **) &tempData, NXYZ*sizeof(float));
    //
    //        if( cudaMemcpyFromSymbol(tempData, "fEscore", NXYZ*sizeof(float), 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
    //        cout << "error in getting fEscore" << endl;
    //        cudaMemcpy(patientDose.doseAve, tempData, NXYZ*sizeof(float), cudaMemcpyDeviceToHost) ;
    //
    //        if( cudaMemcpyFromSymbol(tempData, "fEscor2", NXYZ*sizeof(float), 0, cudaMemcpyDeviceToDevice) != cudaSuccess)
    //        cout << "error in getting fEscore2" << endl;
    //        cudaMemcpy(patientDose.doseStd, tempData, NXYZ*sizeof(float), cudaMemcpyDeviceToHost) ;

    //cudaMemcpy(patientDose.doseAve, fEscore, NXYZ*sizeof(float), cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(patientDose.doseStd, fEscor2, NXYZ*sizeof(float), cudaMemcpyDeviceToHost) ;

    cudaMemcpy(patientDose.doseAve, escore, NXYZ * sizeof(float), cudaMemcpyDeviceToHost);

    patientDose.totalParticleWeight = totalWeight_gBrachy;

    //        cudaFree(tempData);

    return patientDose;
}

#endif
