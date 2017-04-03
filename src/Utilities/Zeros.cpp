#include "../../includes/Utilities/Zeros.h"


void zeros(double *x, unsigned n)
{
    unsigned i;
    for(i=0; i<n; i++)
        x[i]    =   0.0;
}

void zeros(double *A, unsigned m, unsigned n)
{
    unsigned i;
    for(i=0;i<m*n;i++)
            A[i] = 0.0;
}

