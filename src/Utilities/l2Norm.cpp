#include "../../includes/Utilities/l2Norm.h"
#include <cmath>

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis    This is the function to compute the L2-norm between 2 column vectors x and y
 *
 * @Param x     First column vector     
 * @Param y     Second column vector
 * @Param N     The size of the array
 *
 * @Returns     L2-norm between the vectors x and y.
 */
/* ----------------------------------------------------------------------------*/
double l2Norm(double* x, double* y, int N) {
    double result = 0.0;

    for (int i=0; i < N ; i++)
        result += ((x[i]-y[i])*(x[i] - y[i])) ;

    result = sqrt(result);

    return result;
}
