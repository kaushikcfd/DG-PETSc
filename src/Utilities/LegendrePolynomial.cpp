#include "../../includes/Utilities/Zeros.h"
#include <cstring>

void legendrePolynomial(double *LegPolyn, unsigned int n)
{
    double *LegPolyn_1,*LegPolyn_2,*temp;

    LegPolyn_1  =    new double [n+1];
    LegPolyn_2  =    new double [n+1];
    temp        =    new double [n+1];
    zeros(LegPolyn,n+1);
    zeros(LegPolyn_1,n+1);
    zeros(LegPolyn_2,n+1);
    zeros(temp,n+1);
    


    unsigned i,j;

    LegPolyn_1[0]   =   0;
    LegPolyn_1[1]   =   1;
    LegPolyn_2[0]   =   1;

    if(n==0)
    {
        LegPolyn[0] =   1;
    }

    if(n==1)
    {
        LegPolyn[0] =   0.0;
        LegPolyn[1] =   1.0;
        return ;
    }

    for(i=2;i<=n;i++)
    {
        for(j=1;j<=i;j++)
        {
            LegPolyn[j] =   ((2.0*i-1.0)/(i))*LegPolyn_1[j-1] - ((i-1.0)/(i))*LegPolyn_2[j];
        }

        LegPolyn[0] =    - ((i-1.0)/(i))*LegPolyn_2[0];
        memcpy(temp,LegPolyn_1,i*(sizeof(double)));
        memcpy(LegPolyn_1,LegPolyn,(i+1)*sizeof(double));
        memcpy(LegPolyn_2,temp,(i)*sizeof(double));
    }

    delete [] LegPolyn_1;
    delete [] LegPolyn_2;
    delete [] temp;
    return ;
}
