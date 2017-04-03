#include <cstring>
#include <functional>

#include "../../includes/Utilities/LagrangePolynomials.h"
#include "../../includes/Utilities/Zeros.h"
#include "../../includes/Utilities/MassMatrix.h"
#include "../../includes/Utilities/PolyEval.h"

using namespace std;

void fluxMatrix(double *FluxMatrix, unsigned N)
{
    double Poly[N+1][N+1];
    double **poly;
    poly   =   new double*[N+1];
    lagrangePolynomials(*Poly,N);
    unsigned i,j;

    for(i=0;i<=N;i++)
    {
        poly[i] =   new double[N+1];
        memcpy(poly[i],Poly[i],(N+1)*sizeof(double));
    }

    function<double(double)> eval;
    for(i=0;i<=N;i++)
    {
        for(j=0;j<=N;j++)
        {
            eval = [&poly,&i,&j,&N](double x){return (((polyEval(poly[i],N,x))*(polyEval(poly[j],N,x))));};
            FluxMatrix[i*(N+1)+j] = eval(1)-eval(-1);
        }
    }

    for(i=0;i<=N;i++)
        delete[] poly[i];

    delete[] poly;
    return ;
}

void twoDFluxMatrix1(double *Flux1, unsigned N)
{
    unsigned i1,i2,j1,j2;
    double M[N+1][N+1];
    massMatrix(*M,N);
    zeros(Flux1,(N+1)*(N+1),(N+1)*(N+1));

    i1=i2=0;

    for(j1=0;j1<=N;j1++)
            for(j2=0;j2<=N;j2++)
                Flux1[(i1*(N+1)+j1)*(N+1)*(N+1)+i2*(N+1)+j2] = M[j1][j2];
}

void twoDFluxMatrix3(double *Flux3, unsigned N)
{
    unsigned i1,i2,j1,j2;
    double M[N+1][N+1];
    massMatrix(*M,N);
    zeros(Flux3,(N+1)*(N+1),(N+1)*(N+1));

    i1=i2=N;

    for(j1=0;j1<=N;j1++)
            for(j2=0;j2<=N;j2++)
                Flux3[(i1*(N+1)+j1)*(N+1)*(N+1)+i2*(N+1)+j2] = M[j1][j2];
}

void twoDFluxMatrix2(double *Flux2, unsigned N)
{
    unsigned i1,i2,j1,j2;
    double M[N+1][N+1];
    massMatrix(*M,N);
    zeros(Flux2,(N+1)*(N+1),(N+1)*(N+1));

    j1=j2=N;
    for(i1=0;i1<=N;i1++)
        for(i2=0;i2<=N;i2++)
                Flux2[(i1*(N+1)+j1)*(N+1)*(N+1)+i2*(N+1)+j2] = M[i1][i2];
}

void twoDFluxMatrix4(double *Flux4, unsigned N)
{
    unsigned i1,i2,j1,j2;
    double M[N+1][N+1];
    massMatrix(*M,N);
    zeros(Flux4,(N+1)*(N+1),(N+1)*(N+1));

    j1=j2=0;
    for(i1=0;i1<=N;i1++)
        for(i2=0;i2<=N;i2++)
                Flux4[(i1*(N+1)+j1)*(N+1)*(N+1)+i2*(N+1)+j2] = M[i1][i2];
}
