#ifndef __LIBPENMATH_H__
#define __LIBPENMATH_H__




/**************************************************************
C                       SUBROUTINE SPLINE
C*************************************************************/
void spline(float *X, float *Y, float *A, float *B, float *C, float *D, float S1, float SN, int N)
//	possible error from FORTRAN to C

/*	CUBIC SPLINE INTERPOLATION BETWEEN TABULATED DATA.

C  INPUT:
C     X(I) (I=1, ...,N) ........ GRID POINTS.
C                     (THE X VALUES MUST BE IN INCREASING ORDER).
C     Y(I) (I=1, ...,N) ........ CORRESPONDING FUNCTION VALUES.
C     S1,SN ..... SECOND DERIVATIVES AT X(1) AND X(N).
C             (THE NATURAL SPLINE CORRESPONDS TO TAKING S1=SN=0).
C     N ........................ NUMBER OF GRID POINTS.
C
C     THE INTERPOLATING POLYNOMIAL IN THE I-TH INTERVAL, FROM
C  X(I) TO X(I+1), IS PI(X)=A(I)+X*(B(I)+X*(C(I)+X*D(I))).
C
C  OUTPUT:
C     A(I),B(I),C(I),D(I) ...... SPLINE COEFFICIENTS.
C
C     REF.: M.J. MARON, 'NUMERICAL ANALYSIS: A PRACTICAL
C           APPROACH', MACMILLAN PUBL. CO., NEW YORK 1982.
C*************************************************************/
{
	//    IMPLICIT DOUBLE PRECISION (A-H,O-Z)
    //  DIMENSION X(N),Y(N),A(N),B(N),C(N),D(N)
    
	if(N < 4)
	{
		printf("SPLINE INTERPOLATION CANNOT BE PERFORMED WITH %d POINTS. STOP.\n",N);
		exit(1);
	}

//	linear interpolation
/*	for(int i = 0; i< N-1; i++)
	{
		B[i] = (Y[i+1]-Y[i])/(X[i+1]-X[i]);
		A[i] = (Y[i]*X[i+1] - X[i]*Y[i+1])/(X[i+1]-X[i]);
		C[i] = 0.0;
		D[i] = 0.0;
	}*/
    
	int N1 = N-1;
    	int N2 = N-2;
//	AUXILIARY ARRAYS H(=A) AND DELTA(=D).
    	for(int i = 0; i < N1; i++)
	{
		if(X[i+1]-X[i] < 1.0e-10)
		{
			printf("SPLINE X VALUES NOT IN INCREASING ORDER. STOP.\n");
			exit(1);
		}
		A[i] = X[i+1] - X[i];
		D[i] = (Y[i+1] - Y[i])/A[i];
	}
		
//	SYMMETRIC COEFFICIENT MATRIX (AUGMENTED).
    	for(int i = 0; i < N2; i++)
	{
		B[i] = 2.0F * (A[i] + A[i+1]);
		int k = N1 - i - 1;
		D[k] = 6.0F * (D[k] - D[k-1]);
	}

	D[1] -= A[0] * S1;
	D[N1-1] -= A[N1-1] * SN;
//	GAUSS SOLUTION OF THE TRIDIAGONAL SYSTEM.
	for(int i = 1; i < N2; i++)
	{
		float R = A[i]/B[i-1];
		B[i] -= R * A[i];
		D[i+1] -= R * D[i];
	}
//	THE SIGMA COEFFICIENTS ARE STORED IN ARRAY D.
	D[N1-1] = D[N1-1]/B[N2-1];	
	for(int i = 1; i < N2; i++)
	{
		int k = N1 - i - 1;
		D[k] = (D[k] - A[k] * D[k+1])/B[k-1];
	}
	D[N-1] = SN;
//	SPLINE COEFFICIENTS.
	float SI1 = S1;
    for(int i = 0; i < N1; i++)
	{
		float SI = SI1;
		SI1 = D[i+1];
		float H = A[i];
		float HI = 1.0F/H;
		A[i] = (HI/6.0F)*(SI*X[i+1]*X[i+1]*X[i+1]-SI1*X[i]*X[i]*X[i])
			+HI*(Y[i]*X[i+1]-Y[i+1]*X[i])
			+(H/6.0F)*(SI1*X[i]-SI*X[i+1]);
		B[i] = (HI/2.0F)*(SI1*X[i]*X[i]-SI*X[i+1]*X[i+1])
				+HI*(Y[i+1]-Y[i])+(H/6.0F)*(SI-SI1);
		C[i] = (HI/2.0F)*(SI*X[i+1]-SI1*X[i]);
		D[i] = (HI/6.0F)*(SI1-SI);
	}
	return;

}


void inirngG()
/*******************************************************************
c*    Set iseed1 and iseed2 for all threads with random numbers    *
c*                                                                 *
c*    Input:                                                       *
c*    Output:                                                      *
c*      iseed1 -> random number                                    *
c*		iseed2 -> random number                            *
c******************************************************************/
{
//	initialize rand seeds at CPU
	srand ( (unsigned int)time(NULL) );

//	generate randseed at CPU
	for(int i = 0; i < NPART; i++)
	{
		iseed1_h[i] = rand();
	}

//	copy to GPU 
	cudaMemcpyToSymbol(iseed1, iseed1_h, sizeof(int)*NPART, 0, cudaMemcpyHostToDevice) ;

	int nblocks;
        nblocks = 1 + (NPART - 1)/NTHREAD_PER_BLOCK_GBRACHY ;
	setupcuseed<<<nblocks, NTHREAD_PER_BLOCK_GBRACHY>>>();
	cudaThreadSynchronize();

        printf("Rand seeds are initialized. The first 10 seeds are:\n");
        printf("iseed \n");
        for(int i = 0; i < 10; i++)
        {
                if(i < NPART)
                        printf("%d \n",iseed1_h[i]);
        }

}


__global__ void setupcuseed()
//	setup random seeds
{
	const int id = blockIdx.x*blockDim.x + threadIdx.x;
//      obtain current id on thread

        if( id < NPART)
        {
			curand_init(iseed1[id], id, 0, &cuseed[id]);
		}

                if(id==1)
                        printf("first cuseed %d \n",cuseed[id]);


}





#endif
