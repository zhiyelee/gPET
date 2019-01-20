#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <iostream>
#include <vector>
#include <ctime>
#include <omp.h>
#include<cstring>
using namespace std;
#include "gObject.h"
#include "main.h"
// create node
struct object_t InitializeObject()
{
    struct object_t q;

    q.panel = 0;

    q.lengthx=0.0f;
    q.lengthy=0.0f;
    q.lengthz=0.0f;

    q.MODx=0.0f;
    q.MODy=0.0f;
    q.MODz=0.0f;

    q.Mspacex=0.0f;
    q.Mspacey=0.0f;
    q.Mspacez=0.0f;

    q.LSOx=0.0f;
    q.LSOy=0.0f;
    q.LSOz=0.0f;

    q.spacex=0.0f;
    q.spacey=0.0f;
    q.spacez=0.0f;

    q.offsetx=0.0f;
    q.offsety=0.0f;
    q.offsetz=0.0f;

    q.directionx=0.0f;
    q.directiony=0.0f;
    q.directionz=0.0f;

    q.UniXx=0.0f;
    q.UniXy=0.0f;
    q.UniXz=0.0f;

    q.UniYx=0.0f;
    q.UniYy=0.0f;
    q.UniYz=0.0f;

    q.UniZx=0.0f;
    q.UniZy=0.0f;
    q.UniZz=0.0f;

    return q;
}

// read data from file, and build up the multiway tree

void read_file_ro(struct object_t** objectArray, struct object_v** objectMaterial, int* total_Panels, char fname[100])
/********************************************************************************
c* read geometry files using rotational definition                              *
c* Input:                                                                       *
c*  fname: input geometry file                                                  *
c* Output:                                                                      *
    objectArray: buildup geometry                                               *
c*  total_Panels: total panel numbers                                           *
/*******************************************************************************/
{
    printf("loading PET detector geometry parameters ... %s\n");

    FILE* fp=util_fopen(fname,"r");
    char buffer[256];
    int count = 0;
    fgets(buffer, 256, fp);
    fscanf(fp, "%d \n", &count);
    *total_Panels = count;
    cout << "total panels "<<*total_Panels << endl;

    float rot[3];
    fgets(buffer, 256, fp);
    fscanf(fp, "%f %f %f\n", &rot[0], &rot[1], &rot[2]);
    cout << "panel rotational axis "<<rot[0] <<" "<<rot[1] <<" "<<rot[2]<< endl;

    float rotAng;
    fgets(buffer, 256, fp);
    fscanf(fp, "%f\n", &rotAng);
    cout << "panel rotational angle "<<rotAng<< endl;


    // read the file for the second time, to load all the parameters
    struct object_t* temp;
    temp = (object_t*)util_malloc(*total_Panels*sizeof(object_t));
    struct object_v* temp1;
    temp1 = (object_v*)util_malloc(2*sizeof(object_v));
    for (int i = 0; i < *total_Panels; i++)
    {
        temp[i] = InitializeObject();
    }

    int mat = 0, pane = 0;
    float den = 0.0f, lenx=0.0f, leny=0.0f, lenz=0.0f, Mx=0.0f,My=0.0f,Mz=0.0f, Msx=0.0f,Msy=0.0f,Msz=0.0f;
    float Lx=0.0f,Ly=0.0f,Lz=0.0f,sx=0.0f, sy=0.0f, sz=0.0f, ox=0.0f, oy=0.0f, oz=0.0f, dx=0.0f,dy=0.0f,dz=0.0f;
    float UXx=0.0f, UXy=0.0f, UXz=0.0f, UYx=0.0f, UYy=0.0f, UYz=0.0f, UZx=0.0f, UZy=0.0f, UZz=0.0f;

// only two materials (0:LSO, 1: air)
    fgets(buffer, 256, fp);
    for (int i = 0; i < 2; i++)
    {
        fscanf(fp, "%d %f \n", &mat, &den);
        temp1[i].material=mat;
        temp1[i].density=den;
        printf("mat=%d, den=%f\n", mat, den);
    }
    fgets(buffer, 256, fp);

// read in parameter for each Panel
    for (int i = 0; i < 1; i++)
    {
        printf("i=%d\n", i);
        fgets(buffer, 256, fp);
        fscanf(fp, "%d \n", &pane);
        temp[i].panel = pane;
        printf("panel=%d\n", pane);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &lenx, &leny, &lenz);
        temp[i].lengthx = lenx;
        temp[i].lengthy = leny;
        temp[i].lengthz = lenz;
        printf("lengthx=%f, lengthy=%f, lengthz=%f\n", lenx, leny, lenz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &Mx, &My, &Mz);
        temp[i].MODx = Mx;
        temp[i].MODy = My;
        temp[i].MODz = Mz;
        printf("MODx=%f, MODy=%f, MODz=%f\n", Mx, My, Mz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &Msx, &Msy, &Msz);
        temp[i].Mspacex = Msx;
        temp[i].Mspacey = Msy;
        temp[i].Mspacez = Msz;
        printf("Mspacex=%f, Mspacey=%f, Mspacez=%f\n", Msx, Msy, Msz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &Lx, &Ly, &Lz);
        temp[i].LSOx = Lx;
        temp[i].LSOy = Ly;
        temp[i].LSOz = Lz;
        printf("LSOx=%f, LSOy=%f, LSOz=%f\n", Lx, Ly, Lz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &sx, &sy, &sz);
        temp[i].spacex = sx;
        temp[i].spacey = sy;
        temp[i].spacez = sz;
        printf("spacex=%f, spacey=%f, spacez=%f\n", sx, sy, sz);



        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &dx, &dy, &dz);
        temp[i].directionx = dx;
        temp[i].directiony = dy;
        temp[i].directionz = dz;
        printf("directionx=%f, directiony=%f, directionz=%f\n", dx, dy, dz);



        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &ox, &oy, &oz);
        temp[i].offsetx = ox;
        temp[i].offsety = oy;
        temp[i].offsetz = oz;
        printf("offsetx=%f, offsety=%f, offsetz=%f\n", ox, oy, oz);


        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &UXx, &UXy, &UXz);
        temp[i].UniXx = UXx;
        temp[i].UniXy = UXy;
        temp[i].UniXz = UXz;
        printf("UniXx=%f, UniXy=%f, UniXz=%f\n", UXx, UXy, UXz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &UYx, &UYy, &UYz);
        temp[i].UniYx = UYx;
        temp[i].UniYy = UYy;
        temp[i].UniYz = UYz;
        printf("UniYx=%f, UniYy=%f, UniYz=%f\n", UYx, UYy, UYz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &UZx, &UZy, &UZz);
        temp[i].UniZx = UZx;
        temp[i].UniZy = UZy;
        temp[i].UniZz = UZz;
        printf("UniZx=%f, UniZy=%f, UniZz=%f\n", UZx, UZy, UZz);

        fgets(buffer, 256, fp);
    }

    float vec[3];
    float ang;
    for (int i = 1; i < *total_Panels; i++)
    {
        if (*total_Panels<=1)
            break;
        printf("i=%d\n", i);
        temp[i].panel = i;
        printf("panel=%d\n", temp[i].panel);

        temp[i].lengthx = temp[0].lengthx;
        temp[i].lengthy = temp[0].lengthy;
        temp[i].lengthz = temp[0].lengthz;
        printf("lengthx=%f, lengthy=%f, lengthz=%f\n", temp[i].lengthx, temp[i].lengthy, temp[i].lengthz);

        temp[i].MODx = temp[0].MODx;
        temp[i].MODy = temp[0].MODy;
        temp[i].MODz = temp[0].MODz;

        temp[i].Mspacex = temp[0].Mspacex;
        temp[i].Mspacey = temp[0].Mspacey;
        temp[i].Mspacez = temp[0].Mspacez;

        temp[i].LSOx = temp[0].LSOx;
        temp[i].LSOy = temp[0].LSOy;
        temp[i].LSOz = temp[0].LSOz;

        temp[i].spacex = temp[0].spacex;
        temp[i].spacey = temp[0].spacey;
        temp[i].spacez = temp[0].spacez;

        temp[i].directionx = temp[0].directionx;
        temp[i].directiony = temp[0].directiony;
        temp[i].directionz = temp[0].directionz;


        ang=rotAng*PI/180.0f*i;

        vec[0] = temp[0].offsetx;
        vec[1] = temp[0].offsety;
        vec[2] = temp[0].offsetz;
        temp[i].offsetx=(1-cosf(ang))*(vec[0]*rot[0])*rot[0]+cosf(ang)*vec[0]+sinf(ang)*(rot[1]*vec[2]-rot[2]*vec[1]);
        temp[i].offsety=(1-cosf(ang))*(vec[1]*rot[1])*rot[1]+cosf(ang)*vec[1]+sinf(ang)*(rot[2]*vec[0]-rot[0]*vec[2]);
        temp[i].offsetz=(1-cosf(ang))*(vec[2]*rot[2])*rot[2]+cosf(ang)*vec[2]+sinf(ang)*(rot[0]*vec[1]-rot[1]*vec[0]);
        printf("offsetx=%f, offsety=%f, offsetz=%f\n", temp[i].offsetx, temp[i].offsety, temp[i].offsetz);

        vec[0] = temp[0].UniXx;
        vec[1] = temp[0].UniXy;
        vec[2] = temp[0].UniXz;
        temp[i].UniXx = (1-cosf(ang))*(vec[0]*rot[0])*rot[0]+cosf(ang)*vec[0]+sinf(ang)*(rot[1]*vec[2]-rot[2]*vec[1]);
        temp[i].UniXy = (1-cosf(ang))*(vec[1]*rot[1])*rot[1]+cosf(ang)*vec[1]+sinf(ang)*(rot[2]*vec[0]-rot[0]*vec[2]);
        temp[i].UniXz = (1-cosf(ang))*(vec[2]*rot[2])*rot[2]+cosf(ang)*vec[2]+sinf(ang)*(rot[0]*vec[1]-rot[1]*vec[0]);
        printf("UniXx=%f, UniXy=%f, UniXz=%f\n", temp[i].UniXx, temp[i].UniXy, temp[i].UniXz);

        vec[0] = temp[0].UniYx;
        vec[1] = temp[0].UniYy;
        vec[2] = temp[0].UniYz;
        temp[i].UniYx = (1-cosf(ang))*(vec[0]*rot[0])*rot[0]+cosf(ang)*vec[0]+sinf(ang)*(rot[1]*vec[2]-rot[2]*vec[1]);
        temp[i].UniYy = (1-cosf(ang))*(vec[1]*rot[1])*rot[1]+cosf(ang)*vec[1]+sinf(ang)*(rot[2]*vec[0]-rot[0]*vec[2]);
        temp[i].UniYz = (1-cosf(ang))*(vec[2]*rot[2])*rot[2]+cosf(ang)*vec[2]+sinf(ang)*(rot[0]*vec[1]-rot[1]*vec[0]);
        printf("UniYx=%f, UniYy=%f, UniYz=%f\n", temp[i].UniYx, temp[i].UniYy, temp[i].UniYz);

        vec[0] = temp[0].UniZx;
        vec[1] = temp[0].UniZy;
        vec[2] = temp[0].UniZz;
        temp[i].UniZx = (1-cosf(ang))*(vec[0]*rot[0])*rot[0]+cosf(ang)*vec[0]+sinf(ang)*(rot[1]*vec[2]-rot[2]*vec[1]);
        temp[i].UniZy = (1-cosf(ang))*(vec[1]*rot[1])*rot[1]+cosf(ang)*vec[1]+sinf(ang)*(rot[2]*vec[0]-rot[0]*vec[2]);
        temp[i].UniZz = (1-cosf(ang))*(vec[2]*rot[2])*rot[2]+cosf(ang)*vec[2]+sinf(ang)*(rot[0]*vec[1]-rot[1]*vec[0]);
        printf("UniZx=%f, UniZy=%f, UniZz=%f\n", temp[i].UniZx, temp[i].UniZy, temp[i].UniZz);
    }
    *objectArray = temp;
    *objectMaterial = temp1;
    // close file
    fclose(fp);
    printf("\n");
    printf("\n");
}

// two times reading the file
void read_file(struct object_t** objectArray, struct object_v** objectMaterial, int* total_Panels, char fname[100])
{
    // read the file for the first time, get the total Panel number

    printf("loading PET detector geometry parameters ... %s\n");

    FILE* fp0=util_fopen(fname,"r");
    char buffer[256];
    int count = 0;
    while(fgets(buffer, 256, fp0) != NULL)
    {
        if (strstr(buffer, "%%%%") != NULL)
            count += 1;
    }
    *total_Panels = count-1;
    cout << *total_Panels << endl;
    fclose(fp0);

    // read the file for the second time, to load all the parameters
    struct object_t* temp;
    temp = (object_t*)util_malloc(*total_Panels*sizeof(object_t));
    struct object_v* temp1;
    temp1 = (object_v*)util_malloc(2*sizeof(object_v));
    for (int i = 0; i < *total_Panels; i++)
    {
        temp[i] = InitializeObject();
    }

    FILE* fp;
//    fp = util_fopen("config.geo", "r"); // open file
    fp = util_fopen(fname, "r");

    int mat = 0, pane = 0;
    float den = 0.0f, lenx=0.0f, leny=0.0f, lenz=0.0f, Mx=0.0f,My=0.0f,Mz=0.0f, Msx=0.0f,Msy=0.0f,Msz=0.0f;
    float Lx=0.0f,Ly=0.0f,Lz=0.0f,sx=0.0f, sy=0.0f, sz=0.0f, ox=0.0f, oy=0.0f, oz=0.0f, dx=0.0f,dy=0.0f,dz=0.0f;
    float UXx=0.0f, UXy=0.0f, UXz=0.0f, UYx=0.0f, UYy=0.0f, UYz=0.0f, UZx=0.0f, UZy=0.0f, UZz=0.0f;

// only two materials (0:LSO, 1: air)
    fgets(buffer, 256, fp);
    for (int i = 0; i < 2; i++)
    {
        fscanf(fp, "%d %f \n", &mat, &den);
        temp1[i].material=mat;
        temp1[i].density=den;
        printf("mat=%d, den=%f\n", mat, den);
    }
    fgets(buffer, 256, fp);

// read in parameter for each Panel
    for (int i = 0; i < *total_Panels; i++)
    {

        printf("i=%d\n", i);
        fgets(buffer, 256, fp);
        fscanf(fp, "%d \n", &pane);
        temp[i].panel = pane;
        printf("panel=%d\n", pane);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &lenx, &leny, &lenz);
        temp[i].lengthx = lenx;
        temp[i].lengthy = leny;
        temp[i].lengthz = lenz;
        printf("lengthx=%f, lengthy=%f, lengthz=%f\n", lenx, leny, lenz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &Mx, &My, &Mz);
        temp[i].MODx = Mx;
        temp[i].MODy = My;
        temp[i].MODz = Mz;
        printf("MODx=%f, MODy=%f, MODz=%f\n", Mx, My, Mz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &Msx, &Msy, &Msz);
        temp[i].Mspacex = Msx;
        temp[i].Mspacey = Msy;
        temp[i].Mspacez = Msz;
        printf("Mspacex=%f, Mspacey=%f, Mspacez=%f\n", Msx, Msy, Msz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &Lx, &Ly, &Lz);
        temp[i].LSOx = Lx;
        temp[i].LSOy = Ly;
        temp[i].LSOz = Lz;
        printf("LSOx=%f, LSOy=%f, LSOz=%f\n", Lx, Ly, Lz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &sx, &sy, &sz);
        temp[i].spacex = sx;
        temp[i].spacey = sy;
        temp[i].spacez = sz;
        printf("spacex=%f, spacey=%f, spacez=%f\n", sx, sy, sz);


        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &dx, &dy, &dz);
        temp[i].directionx = dx;
        temp[i].directiony = dy;
        temp[i].directionz = dz;
        printf("directionx=%f, directiony=%f, directionz=%f\n", dx, dy, dz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &ox, &oy, &oz);
        temp[i].offsetx = ox;
        temp[i].offsety = oy;
        temp[i].offsetz = oz;
        printf("offsetx=%f, offsety=%f, offsetz=%f\n", ox, oy, oz);


        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &UXx, &UXy, &UXz);
        temp[i].UniXx = UXx;
        temp[i].UniXy = UXy;
        temp[i].UniXz = UXz;
        printf("UniXx=%f, UniXy=%f, UniXz=%f\n", UXx, UXy, UXz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &UYx, &UYy, &UYz);
        temp[i].UniYx = UYx;
        temp[i].UniYy = UYy;
        temp[i].UniYz = UYz;
        printf("UniYx=%f, UniYy=%f, UniYz=%f\n", UYx, UYy, UYz);

        fgets(buffer, 256, fp);
        fscanf(fp, "%f %f %f\n", &UZx, &UZy, &UZz);
        temp[i].UniZx = UZx;
        temp[i].UniZy = UZy;
        temp[i].UniZz = UZz;
        printf("UniZx=%f, UniZy=%f, UniZz=%f\n", UZx, UZy, UZz);

        fgets(buffer, 256, fp);
        if (strstr(buffer, "%")==NULL)
        {
            printf("number of panel error!\n");
            exit(EXIT_FAILURE);
        }
    }

    *objectArray = temp;
    *objectMaterial = temp1;
    // close file
    fclose(fp);
    printf("\n");
    printf("\n");
}

