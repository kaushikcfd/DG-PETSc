#include "../../includes/Utilities/PolynomialRoots.h"
#include <gsl/gsl_poly.h>

void polynomialRoots(const double *Poly, double *Roots, int deg) {
    int i;
    double* z = new double[2*deg];
    gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc (deg+1);
    gsl_poly_complex_solve (Poly, deg+1, w, z);

    for(i=0; i<deg; i++)
        Roots[i] = z[2*i];

    gsl_poly_complex_workspace_free (w);
    delete[] z;
    return ;
}
