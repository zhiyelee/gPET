#ifndef __LIBGEOM_CU__
#define __LIBGEOM_CU__


__device__ int4 getAbsVox(float4 xtemp)
//      return the absolute vox index according to the coordinate
{
    float4 xtemp2;
    xtemp2.z = xtemp.z+Offsetz_gBrachy;
    xtemp2.y = xtemp.y+Offsety_gBrachy;
    xtemp2.x = xtemp.x+Offsetx_gBrachy;

    int4 temp;
    temp.z = xtemp2.z*idz_gBrachy;
    temp.y = xtemp2.y*idy_gBrachy;
    temp.x = xtemp2.x*idx_gBrachy;
//the following give the boundry condition
    temp.w = (xtemp2.x <= 0.0f || xtemp2.x >= Unxvox*dx_gBrachy || xtemp2.y <= 0.0f || xtemp2.y >= Unyvox*dy_gBrachy || xtemp2.z <= 0.0f || xtemp2.z >= Unzvox*dz_gBrachy)?-1 : 1;
    return temp;
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

__host__ __device__ int getabs(int xvox, int yvox, int zvox, int nx, int ny, int nz)
/*******************************************************************
c*    Gets the absolute voxel # from the coordinate voxel #s       *
c*                                                                 *
c*    Input:                                                       *
c*      vox -> coordinate voxel #s                                 *
c*    Comments:                                                    *
c*      -> inigeo() must be called before 1st call                 *
c*      -> if the particle is out of universe, absvox is set to 0  *
c******************************************************************/
{
    //return zvox + yvox*Unzvox + xvox*Unzvox*Unyvox;
    //return zvox + yvox*nz + xvox*nz*ny;
    return xvox + yvox*nx + zvox*nx*ny;
}


__device__ float inters(float4 *vtemp, float4* xtemp, int4 *voxtemp, int *indexvox, int *dvox)
//	find intersection with vox boundary
{
    float smaybe;
    float returnValue;

    const int id = blockIdx.x*blockDim.x + threadIdx.x;

//	Checking out all the voxel walls for the smallest distance...
    float tempiv;

    tempiv = (vtemp->z != 0.0f)? 1.0f/vtemp->z : INF;
    if (tempiv > 0.0)
    {
        returnValue = ((voxtemp->z+1) * dz_gBrachy - xtemp->z) * tempiv;
        *indexvox = 3;
        *dvox = +1;
    }
    else
    {
        returnValue = (voxtemp->z * dz_gBrachy - xtemp->z) * tempiv;
        *indexvox = 3;
        *dvox = -1;
    }

    /* if(id == 10000)
    	{
    cuPrintf("xtemp->x= %f,xtemp->y= %f,xtemp->z= %f,vxtemp->x= %f,vxtemp->y= %f,vxtemp->z= %f\n", xtemp->x,xtemp->y,xtemp->z,vtemp->x,vtemp->y,vtemp->z);
    cuPrintf("voxtemp->x= %d,voxtemp->y= %d,voxtemp->z= %d,voxtemp->w= %d\n", voxtemp->x,voxtemp->y,voxtemp->z,voxtemp->w);
    cuPrintf("dx_gBrachy=%f,dy_gBrachy=%f,dz_gBrachy=%f\n ",dx_gBrachy,dy_gBrachy,dz_gBrachy);

    cuPrintf("tempiv = %f, returnValue = %f\n", tempiv,returnValue);
    } */

    tempiv = (vtemp->y != 0.0f)? 1.0f/vtemp->y : INF;
    if (tempiv > 0.0)
    {
        smaybe = ((voxtemp->y+1) * dy_gBrachy - xtemp->y) * tempiv;
        if (smaybe < returnValue)
        {
            returnValue = smaybe;
            *indexvox = 2;
            *dvox = +1;
        }
    }
    else
    {
        smaybe = ( voxtemp->y * dy_gBrachy - xtemp->y) * tempiv;
        if (smaybe < returnValue)
        {
            returnValue = smaybe;
            *indexvox = 2;
            *dvox = -1;
        }
    }

    /* if(id == 10000)
    cuPrintf("tempiv = %f, returnValue = %f\n", tempiv,smaybe); */

    tempiv = (vtemp->x != 0.0f)? 1.0f/vtemp->x : INF;
    if (tempiv > 0.0)
    {
        smaybe = ((voxtemp->x+1) * dx_gBrachy - xtemp->x) * tempiv;
        if (smaybe < returnValue)
        {
            returnValue = smaybe;
            *indexvox = 1;
            *dvox = +1;
        }
    }
    else
    {
        smaybe = (voxtemp->x * dx_gBrachy - xtemp->x) * tempiv;
        if (smaybe < returnValue)
        {
            returnValue = smaybe;
            *indexvox = 1;
            *dvox = -1;
        }
    }
    /* if(id == 10000){
    cuPrintf("tempiv = %f, returnValue = %f\n", tempiv,smaybe);
    cuPrintf("indexvox = %d, dvox = %d\n", *indexvox,*dvox);

    }
     */
//	Make sure we won't get neg value to avoid interpretation problems...
    return returnValue*(returnValue >= 0.0f);
}

__device__ void chvox(int4 *voxtemp, int indexvox, int dvox)
//    Changes voxel according to the information passed by inters()
{
    if (indexvox == 3)
    {
        voxtemp->z += dvox;
        if (voxtemp->z >= 0 && voxtemp->z < Unzvox)
        {
            //voxtemp->w += dvox;
            voxtemp->w =1;
        }
        else
        {
            voxtemp->w = -1;
        }
    }
    else if (indexvox == 2)
    {
        voxtemp->y += dvox;
        if (voxtemp->y >= 0 && voxtemp->y < Unyvox)
        {
            //voxtemp->w += dvox*Unzvox;
            voxtemp->w =1;
        }
        else
        {
            voxtemp->w = -1;
        }
    }
    else
    {
        voxtemp->x += dvox;
        if (voxtemp->x >= 0 && voxtemp->x < Unxvox)
        {
            //voxtemp->w += dvox*Unzvox*Unyvox;
            voxtemp->w =1;
        }
        else
        {
            voxtemp->w = -1;
        }
    }
}

#endif
