#include <lapacke.h>
#include <cstring>
#include <iostream>
#include "../../includes/Utilities/SolveAxb.h"

using namespace std;


void solveAxb(double *A, double *x, double *b, unsigned N)
{
    double *B    = new double[N*N];
    double *c    = new double[N];
    memcpy(B,A,N*N*sizeof(double));
    memcpy(c,b,N*sizeof(double));
    int ipiv[N];
    int info;

    info    =   LAPACKE_dgesv(LAPACK_ROW_MAJOR,N,1,B,N,ipiv,c,1);
    if(info!=0){
        cerr << "The Linear solve `Ax=b` was not succesful. Error Code: " << info << endl;
    }
    memcpy(x,c,N*sizeof(double));

    delete[] B;
    delete[] c;
    return ;
}
