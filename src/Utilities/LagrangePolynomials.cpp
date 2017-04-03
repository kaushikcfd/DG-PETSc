#include "../../includes/Utilities/LobattoNodes.h"
#include "../../includes/Utilities/Inverse.h"
#include "../../includes/Utilities/Transpose.h"

void lagrangePolynomials(double *Polynomials, unsigned N)
{
    double VanderMonde[N+1][N+1];
    double *Nodes;
    Nodes   =   new double[N+1];
    unsigned i,j;
    lobattoNodes(Nodes,N+1);

    for(i=0;i<=N;i++)
    {
        VanderMonde[i][0]   =   1.0;
        for(j=1;j<=N;j++)
            VanderMonde[i][j]   =   VanderMonde[i][j-1]*Nodes[i];
    }

    transpose(*VanderMonde,N+1);
    inverse(&VanderMonde[0][0],Polynomials,N+1);

    delete[] Nodes;
    return;
}
