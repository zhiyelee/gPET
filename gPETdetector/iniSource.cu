#ifndef __INISOURCE_CU__
#define __INISOURCE_CU__
#include "gObject.h"
#include <fstream>
//using namespace std;

void readinput(char* ptr)
{
    char buffer[200];
    gets(buffer);
    printf("%s\n", buffer);
    scanf("%s\n",ptr);
    printf("%s\n",ptr);
}
void readinput(int* ptr)
{
    char buffer[200];
    gets(buffer);
    printf("%s\n", buffer);
    scanf("%d\n",ptr);
    printf("%d\n",(*ptr));
}
void readinput(float* ptr)
{
    char buffer[200];
    gets(buffer);
    printf("%s\n", buffer);
    scanf("%f\n",ptr);
    printf("%f\n",(*ptr));
}
/* reverse:  reverse string s in place */
void reverse(char s[])
{
    int i, j;
    char c;

    for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}
// itoa:  convert n to characters in s
void itoa(int n, char s[])
{
    int i, sign;

    if ((sign = n) < 0)  /* record sign */
        n = -n;          /* make n positive */
    i = 0;
    do {       /* generate digits in reverse order */
        s[i++] = n % 10 + '0';   /* get next digit */
    } while ((n /= 10) > 0);     /* delete it */
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    reverse(s);
}



/* strlen: return length of s */
int strlen(char s[])
{
    int i = 0;
    while (s[i] != '\0')
        ++i;
    return i;
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
        printf("ERROR, io from other unit has not been programmed...\n");
        exit(1);
    }
    strcpy(physics,phyname);
    *n = (int)strlen(phyname);

    return;
}

Source ReadSource(char fname[100],int NSource, int NRepeat)
{
    //read source from PSF file
    Source source;
    ifstream infile(fname,ios::binary);
    int start, stop;
    float data[8];
    if(!infile)
    {
        printf("open FILE error\n");
        exit(1);
    }
    else
    {
        start=infile.tellg();
        infile.seekg(0, ios::end);
        stop=infile.tellg();
        source.NSource = NSource<(stop-start)/32?NSource:(stop-start)/32;
        for(int j=0;j<NRepeat;j++)
        {
            infile.seekg(0, ios::beg);
            for(int i=0;i<source.NSource;i++)
            {
                infile.read(reinterpret_cast <char*> (&data), sizeof(data));
                source.xbuffer.push_back(make_float4(data[0],data[1],data[2],data[3]));
                source.vxbuffer.push_back(make_float4(data[4],data[5],data[6],data[7]));

                if(j==0 && i<5)
                {
                    printf("the first %d particle: %f %f %f %f %f %f %f %f\n",i,data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7] );
                }
            }
        }
        infile.close();
    }
    source.NSource = source.NSource * NRepeat;
    printf("finish read: source PSF;\n\n");
    return source;
}
void iniSource(Source source)
/*******************************************************************
c*    Initializes the gCTD system                                  *
c******************************************************************/
{//not used in this version. The gpu source memory initialization is done by reading cpu source in several times
    //because reading too many particles may cause the crush of gpu memory 

    printf(" \n");
    printf("init: GPU source PSF;\n");
    printf("      information from this stream follows:\n");

    int NSource=source.NSource;
    float4 *tempx = new float4[NSource];
    float4 *tempvx = new float4[NSource];

    cudaMalloc( (void **) &x_phap_gPET, sizeof(float4)*NSource);
    cudaMalloc( (void **) &vx_phap_gPET, sizeof(float4)*NSource);

    memcpy(&tempx[0], &source.xbuffer[0], sizeof(float4)*NSource);
    memcpy(&tempvx[0], &source.vxbuffer[0], sizeof(float4)*NSource);

    cudaMemcpy(x_phap_gPET, tempx, sizeof(float4)*NSource, cudaMemcpyHostToDevice);
    cudaMemcpy(vx_phap_gPET, tempvx, sizeof(float4)*NSource, cudaMemcpyHostToDevice);

    free(tempx);
    free(tempvx);

    printf("finish init: source PSF;\n\n");


}
#endif
