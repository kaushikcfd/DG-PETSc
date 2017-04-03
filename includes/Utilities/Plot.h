#ifndef PLOT_H
#define PLOT_H

#include <string>
using namespace std;

void plot(double *X, double *Y, double N, string GraphTitle, string LegendTitle, string fileName);

/**
  * A similar 1-D plot but with the y-axis ranges.
  */
void plot(double *X, double *Y, double N,double Ymin, double Ymax, string GraphTitle, string LegendTitle, string fileName );

/**
 * This is a plot function which takes three 2D arrays and makes the surface plots of them.
 * @param X    The X-Domain.
 * @param Y    The Y-Domain.
 * @param Z    The Z-Domain.
 */
void plot(double *X, double *Y, double *Z, unsigned Nx, unsigned Ny,double X1, double X2,double Y1, double Y2,double Z1, double Z2, string Name);

#endif
