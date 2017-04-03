#include "../../includes/Utilities/PolyDeriv.h"

void polyDeriv(double *GivenPoly, double *DerivedPoly, unsigned deg)
{
	for(int i = 1;i<=deg;i++)
        DerivedPoly[i-1] = i*GivenPoly[i];
	return ;
}
