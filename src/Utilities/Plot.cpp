#include <cstdio>
#include <iostream>
#include <string>
#include <cstdlib>
#include "../../includes/Utilities/Display.h"
#include "../../includes/Utilities/Plot.h"
using namespace std;

void plot(double *X, double *Y, double N, string GraphTitle = "", string LegendTitle = "", string fileName = "")
{
    unsigned i;
    FILE* pfile;
    /*Starting the dumping of data to the file `temp.dat`*/
    pfile=freopen("temp.dat","w",stdout);
        printf("# X\tY\n");
        for(i=0; i<N; i++)
            printf("%.6f\t%.6f\n",X[i],Y[i]);
    fclose(stdout);
    /*Writing the file for the GNUPLOT*/
    pfile=freopen("temp.gnu","w",stdout);
        printf("reset\n");
        printf("set terminal jpeg interlace enhanced size 1366,768\n");
        printf("set output \"%s.jpg\"\n", fileName.c_str());
        printf("set key right box\n");
        printf("set grid\n");
        printf("set title '%s'\n",GraphTitle.c_str());
        printf("set ylabel 'Y'\n");
        printf("set xlabel 'X'\n");
        printf("plot 'temp.dat' title '%s' w lines ls 1\n",LegendTitle.c_str());
    fclose(stdout);
    /*Writing commands to the terminal for making the plot.*/
    pfile=freopen ("/dev/tty", "a", stdout);
    i=system("gnuplot> load 'temp.gnu'");
    i=system("rm -rf temp* load");
    return ;
}

/**
  * A similar 1-D plot but with the y-axis ranges.
  */
void plot(double *X, double *Y, double N,double Ymin, double Ymax, string GraphTitle = "", string LegendTitle = "", string fileName = "")
{
    unsigned i;
    FILE* pfile;
    /*Starting the dumping of data to the file `temp.dat`*/
    pfile=freopen("temp.dat","w",stdout);
        printf("# X\tY\n");
        for(i=0; i<N; i++)
            printf("%.6f\t%.6f\n",X[i],Y[i]);
    fclose(stdout);
    /*Writing the file for the GNUPLOT*/
    pfile=freopen("temp.gnu","w",stdout);
        printf("reset\n");
        printf("set terminal jpeg interlace enhanced size 1366,768\n");
        printf("set output \"%s.jpg\"\n", fileName.c_str());
        printf("set key right box\n");
        printf("set grid\n");
        printf("set title '%s'\n",GraphTitle.c_str());
        printf("set yrange [%f:%f]\n",Ymin,Ymax);
        printf("set ylabel 'Y'\n");
        printf("set xlabel 'X'\n");
        printf("plot 'temp.dat' title '%s' w lines ls 1\n",LegendTitle.c_str());
    fclose(stdout);
    /*Writing commands to the terminal for making the plot.*/
    pfile=freopen ("/dev/tty", "a", stdout);
    i=system("gnuplot> load 'temp.gnu'");
    i=system("rm -rf temp* load");
    return ;
}

/**
 * This is a plot function which takes three 2D arrays and makes the surface plots of them.
 * @param X    The X-Domain.
 * @param Y    The Y-Domain.
 * @param Z    The Z-Domain.
 */
void plot(double *X, double *Y, double *Z, unsigned Nx, unsigned Ny,double X1, double X2,double Y1, double Y2,double Z1, double Z2, string Name)
{
    int i;
    FILE* pfile;
    pfile=freopen("X.dat","w",stdout);
    display(X,Nx,Ny);
    fclose(stdout);
    pfile=freopen("Y.dat","w",stdout);
    display(Y,Nx,Ny);
    fclose(stdout);
    pfile=freopen("Z.dat","w",stdout);
    display(Z,Nx,Ny);
    fclose(stdout);
    pfile=freopen("temp.m","w",stdout);
    printf("load X.dat;\n");
    printf("load Y.dat;\n");
    printf("load Z.dat;\n");
    printf("h=figure(1);\n");
    printf("surf(X,Y,Z);\n");
    printf("axis([%.2f %.2f %.2f %.2f %.2f %.2f]);\n",X1,X2,Y1,Y2,Z1,Z2);
    printf("H = 8; W = 8;\nset(h,\'PaperUnits\',\'inches\');\nset(h,\'PaperOrientation\',\'portrait\');\nset(h,\'PaperSize\',[H,W]);\n");
    printf("print(h,\'-djpg\',\'-color\',\'%s.jpg\');\n",Name.c_str());
    fclose(stdout);

    pfile=freopen ("/dev/tty", "a", stdout);
    i=system("octave temp.m");
    i=system("rm -rf temp* load");

    return ;
}
