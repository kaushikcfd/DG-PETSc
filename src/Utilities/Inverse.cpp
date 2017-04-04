#include <lapacke.h>
#include <cstring>
#include "../../includes/Utilities/SolveAxb.h"

void inverse(double *A, double *Ainv,unsigned N)
{
    memcpy(Ainv,A,N*N*sizeof(double));
    int info;
    int ipiv[N];
    info =  LAPACKE_dgetrf(LAPACK_ROW_MAJOR,N,N,Ainv,N,ipiv);
    if(info==0)
        info =  LAPACKE_dgetri(LAPACK_ROW_MAJOR,N,Ainv,N,ipiv);
    if(info!=0)
        exit(0);
}
