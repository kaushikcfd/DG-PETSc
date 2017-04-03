#include <functional>
#include <cstring>
#include "../../includes/Utilities/LagrangePolynomials.h"
#include "../../includes/Utilities/PolyEval.h"
#include "../../includes/Utilities/PolyDeriv.h"
#include "../../includes/Utilities/LobattoIntegration.h"
#include "../../includes/Utilities/MassMatrix.h"

using namespace std;

void derivativeMatrix(double *DerivativeMatrix,unsigned N)
{
    double Poly[N+1][N+1];
    double **poly, **deriv;
    poly   =   new double*[N+1];
    deriv  =   new double*[N+1];
    lagrangePolynomials(*Poly,N);
    unsigned i,j;

    for(i=0;i<=N;i++)
    {
        poly[i] =   new double[N+1];
        deriv[i]=   new double[N];
        memcpy(poly[i],Poly[i],(N+1)*sizeof(double));
        polyDeriv(poly[i],deriv[i],N+1);
    }

    function<double(double)> eval;
    for(i=0;i<=N;i++)
    {
        for(j=0;j<=N;j++)
        {
            eval = [&poly,&deriv,&i,&j,&N](double x){return (((polyEval(poly[i],N,x))*(polyEval(deriv[j],N-1,x))));};
            DerivativeMatrix[i*(N+1)+j] = lobattoIntegration(-1.0,1.0,N+1,eval);
        }
    }

    for(i=0;i<=N;i++)
    {
        delete[] poly[i];
        delete[] deriv[i];
    }

    delete[] poly;
    delete[] deriv;
    return ;
}

void twoDDerivativeMatrixX(double *DerivativeMatrix, unsigned N)
{
    double m[N+1][N+1];
    double d[N+1][N+1];
    derivativeMatrix(*d,N);
    massMatrix(*m,N);

    unsigned i1,i2,j1,j2;
    for(i1=0;i1<=N;i1++)
        for(j1=0;j1<=N;j1++)
            for(i2=0;i2<=N;i2++)
                for(j2=0;j2<=N;j2++)
                    DerivativeMatrix[(i1*(N+1)+j1)*(N+1)*(N+1)+i2*(N+1)+j2] = m[i1][i2]*d[j1][j2];

    return ;
}

void twoDDerivativeMatrixY(double *DerivativeMatrix, unsigned N)
{
    double m[N+1][N+1];
    double d[N+1][N+1];
    derivativeMatrix(*d,N);
    massMatrix(*m,N);

    unsigned i1,i2,j1,j2;
    for(i1=0;i1<=N;i1++)
        for(j1=0;j1<=N;j1++)
            for(i2=0;i2<=N;i2++)
                for(j2=0;j2<=N;j2++)
                    DerivativeMatrix[(i1*(N+1)+j1)*(N+1)*(N+1)+i2*(N+1)+j2] = d[i1][i2]*m[j1][j2];

    return ;
}
