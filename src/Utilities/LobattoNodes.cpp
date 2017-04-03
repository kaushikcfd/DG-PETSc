#include <algorithm>

#include "../../includes/Utilities/LobattoNodes.h"
#include "../../includes/Utilities/LegendrePolynomial.h"
#include "../../includes/Utilities/PolyEval.h"
#include "../../includes/Utilities/PolyDeriv.h"
#include "../../includes/Utilities/PolynomialRoots.h"
using namespace std;

void lobattoNodes(double *Nodes, unsigned N)
{
    double *Poly,*DerivedPoly;
    Poly        =   new double[N];
    DerivedPoly =   new double[N-1];
    legendrePolynomial(Poly,N-1);
    polyDeriv(Poly,DerivedPoly,N-1);
	Nodes[0]   =   -1;
    polynomialRoots(DerivedPoly, &(Nodes[1]), N-2);
	Nodes[N-1] =   1;
	sort(Nodes,Nodes+N);

    delete[] Poly;
    delete[] DerivedPoly;
}
