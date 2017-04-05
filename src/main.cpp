/**
 * File: main.cpp
 * brief: The main purpose of this file is to create an Advection Solver using PETSc.
 * This would include creation of GLOBAL matrices.
 * These GLOBAL matrices are sparse matrices. Hence we would need to assemble a PETSc sparse matrix using the previously computed local matrices. 
 */
#include "petsc.h"
#include <string>
#include <fstream>
#include "../includes/Utilities/LobattoNodes.h"
#include "../includes/Utilities/MassMatrix.h"
#include "../includes/Utilities/DerivativeMatrix.h"
#include "../includes/Utilities/FluxMatrix.h"

using namespace std;

static char help[] = "Hello World Program\n\n";

PetscReal U(PetscReal x, PetscReal y) {
    return 1.0;
}

PetscReal V(PetscReal x, PetscReal y) {
    return 0.0;
}

double Q(double x, double y) {
    return (exp(-(x*x +  y*y)*16.0));
}

void initial_conditions(Vec x, Vec y, Vec u, Vec v, Vec q, Vec f_x, Vec f_y, PetscInt ne_x, PetscInt ne_y, PetscInt n, PetscReal x1, PetscReal y1, PetscReal x2, PetscReal y2) {
    PetscInt n_p    =   ne_x*ne_y*(n+1)*(n+1);
    PetscInt    *ix         = new PetscInt[n_p];
    double   *nodes         = new double[n_p];  
    PetscReal   *x_dummy    = new PetscReal[n_p];
    PetscReal   *y_dummy    = new PetscReal[n_p];
    PetscReal   *u_dummy    = new PetscReal[n_p];
    PetscReal   *v_dummy    = new PetscReal[n_p];
    PetscReal   *q_dummy    = new PetscReal[n_p];
    PetscReal   *fx_dummy    = new PetscReal[n_p];
    PetscReal   *fy_dummy    = new PetscReal[n_p];

    PetscInt i,j,k1,k2, node_index;
    PetscReal x_curr=x1, y_curr=y1;
    PetscReal dx, dy;
    dx = (x2-x1)/ne_x;
    dy = (y2-y1)/ne_y;

    lobattoNodes(nodes, n+1);

    for(j=0; j<ne_y; j++) {
        x_curr = x1;
        for(i=0; i<ne_x; i++) {
            for(k2=0; k2<=n ; k2++) {
                for(k1=0; k1 <=n; k1++) {
                    node_index = (j*ne_x + i)*(n+1)*(n+1) + (k2*(n+1)+k1);
                    x_dummy[node_index] = (PetscReal)(x_curr + 0.5*dx*(1.0 + nodes[k1]));
                    y_dummy[node_index] = (PetscReal)(y_curr + 0.5*dy*(1.0 + nodes[k2]));
                    u_dummy[node_index] = U(x_dummy[node_index], y_dummy[node_index]);
                    v_dummy[node_index] = V(x_dummy[node_index], y_dummy[node_index]);
                    q_dummy[node_index] = Q(x_dummy[node_index], y_dummy[node_index]);
                    fx_dummy[node_index] = u_dummy[node_index]*q_dummy[node_index];
                    fy_dummy[node_index] = v_dummy[node_index]*q_dummy[node_index];
                }
            }
            x_curr += dx;
        }
        y_curr += dy;
    }
    for(i=0;i<n_p;i++)
        ix[i] = i;

    VecSetValues(x, n_p, ix, x_dummy, INSERT_VALUES);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    VecSetValues(y, n_p, ix, y_dummy, INSERT_VALUES);
    VecAssemblyBegin(y);
    VecAssemblyEnd(y);
    VecSetValues(u, n_p, ix, u_dummy, INSERT_VALUES);
    VecAssemblyBegin(u);
    VecAssemblyEnd(u);
    VecSetValues(v, n_p, ix, v_dummy, INSERT_VALUES);
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
    VecSetValues(q, n_p, ix, q_dummy, INSERT_VALUES);
    VecAssemblyBegin(q);
    VecAssemblyEnd(q);
    VecSetValues(f_x, n_p, ix, fx_dummy, INSERT_VALUES);
    VecAssemblyBegin(f_x);
    VecAssemblyEnd(f_x);
    VecSetValues(f_y, n_p, ix, fy_dummy, INSERT_VALUES);
    VecAssemblyBegin(f_y);
    VecAssemblyEnd(f_y);

    delete[]   ix; 
    delete[]   nodes;    
    delete[]   x_dummy; 
    delete[]   y_dummy; 
    delete[]   u_dummy; 
    delete[]   v_dummy;
    delete[]   q_dummy; 
    return ;
}

void createGlobalMatrix(Mat global, PetscInt ne_x, PetscInt ne_y, PetscInt n, string matrixType) {
    // Declaring the iterators
    PetscInt i, j, k;
    PetscInt   *idxm = new PetscInt[(n+1)*(n+1)];
    PetscInt   *idxn = new PetscInt[(n+1)*(n+1)];

    // Allocating space for the local Matrix.
    PetscReal   *loc = new PetscReal[(n+1)*(n+1)*(n+1)*(n+1)];
    if(matrixType == "Mass") {
        twoDMassMatrix(loc, n);
    }
    else if(matrixType == "Derivative_x") {
        twoDDerivativeMatrixX(loc, n);
    }
    else if(matrixType == "Derivative_y") {
        twoDDerivativeMatrixY(loc, n);
    }
    else if(matrixType == "Flux_right") {
        twoDFluxMatrix2(loc, n);
    }
    else if(matrixType == "Flux_top") {
        twoDFluxMatrix3(loc, n);
    }    
    else if(matrixType == "Flux_left") {
        twoDFluxMatrix4(loc, n);
    }
    else if(matrixType == "Flux_bottom") {
        twoDFluxMatrix1(loc, n);
    }
    else {
         PetscPrintf(PETSC_COMM_SELF, "Did not find such local matrix.\n");
    }

    for(j=0; j<ne_y; j++) {
        for(i=0; i<ne_x; i++) {
            for(k=0; k<(n+1)*(n+1); k++) {
                idxm[k] = (j*ne_x + i)*(n+1)*(n+1) + k;
                idxn[k] = (j*ne_x + i)*(n+1)*(n+1) + k;
            }
            MatSetValues(global, (n+1)*(n+1), idxm, (n+1)*(n+1), idxn, loc, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(global, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(global, MAT_FINAL_ASSEMBLY);

    return ;
}

void writeVTK(Vec x, Vec y, Vec q, PetscInt ne_x, PetscInt ne_y, PetscInt n, string filename) {
    PetscReal   *x_array;
    PetscReal   *y_array;
    PetscReal   *q_array;

    VecGetArray(x, &x_array);
    VecGetArray(y, &y_array);
    VecGetArray(q, &q_array);

    ofstream pFile;
    pFile.open(filename);
    
    PetscInt i, j, k, k1, k2;

    // Printing the preamble for the .vtk file.
    pFile << "# vtk DataFile Version 3.0\nNavier Stokes DG\nASCII\nDATASET UNSTRUCTURED_GRID\n";
    // The information of the number of points.
    pFile << "POINTS\t" << (n+1)*(n+1)*ne_x*ne_y << "\tdouble\n";

    // Writing the point co-ordinates.
    for ( j = 0; j < ne_y; j++ )
        for ( i = 0; i < ne_x; i++ )
            for( k = 0; k < (n+1)*(n+1); k++ )
                pFile << x_array[(j*ne_x + i)*(n+1)*(n+1) + k] << "\t" << y_array[(j*ne_x + i)*(n+1)*(n+1) + k] <<"\t"<< 0.0 <<endl;

    pFile << "\n\n";

    // Specifying the information about the CELLS.
    pFile << "CELLS\t" << (n*n*ne_x*ne_y) <<"\t" << 5*(n*n*ne_x*ne_y) << endl;

    // Writing information about the structure of the cells.
    for ( j = 0; j < ne_y; j++ ) {
        for ( i = 0; i < ne_x; i++ ) {
            for( k1 = 0; k1 < n; k1++ ) {
                for ( k2 = 0; k2 < n; k2++ ) {
                    k   =   (j*ne_x+i)*(n+1)*(n+1) +   k1*(n+1)    +   k2;
                    pFile << 4 << "\t" << k << "\t" << k+1 << "\t" << k+n+2 << "\t" << k+n+1 << endl;
                }
            }
        }
    }
    pFile << "\n\n";

    // Specifying the information about the CELL TYPES.
    pFile << "CELL_TYPES " << (n*n*ne_x*ne_y) << endl;

    // `9` is the CELL TYPE CODE for specifying that it is a quad.
    for ( i = 0; i < (n*n*ne_x*ne_y); i++)
        pFile << "9\n";
    pFile << "\n\n";

    // Specifying the information about the values of the scalars.
    
    pFile << "POINT_DATA\t"<< (n+1)*(n+1)*ne_x*ne_y <<"\n";
        
    pFile << "SCALARS\tQ\tdouble\nLOOKUP_TABLE default\n";
        
    // Writing the value of the POINT_DATA, for the variable[variableNames[k1]] 
    for ( j = 0; j < ne_y; j++ ){
        for ( i = 0; i < ne_x; i++ ) {
            for( k = 0; k < (n+1)*(n+1); k++ ) {
                pFile << q_array[(j*ne_x + i)*(n+1)*(n+1) + k] << endl;
            }
        }
    }

    VecRestoreArray(x, &x_array);
    VecRestoreArray(y, &y_array);
    VecRestoreArray(q, &q_array);
    
    pFile.close(); // Closing the file.
    return ;
}

void updateNumericalFlux() {

}

int main(int argc, char *argv[])
{   
    /// Constants that define the problem.
    PetscInt ne_x = 10, ne_y = 10;   /// Number of elements in the x and y direction resp.
    PetscInt n = 4;                  /// The order of interpolation

    /// Setting other constants
    PetscInt n_p = (n+1)*(n+1)*ne_x*ne_y;

    /// Setting the domain.
    PetscReal x1, y1, x2, y2, dx, dy;
    x1 = y1 = -1.0;
    x2 = y2 =  1.0;
    dx = (x2-x1)/ne_x;
    dy = (y2-y1)/ne_y;


    /// Declaring the vectors
    Vec     x, y;
    Vec     u, v;
    Vec     q, f_x, f_y;
    Vec     q_star_x, q_star_y;

    // Declaring the matrices, small letter for local, Capital for global
    Mat M, D_x, D_y, F_right, F_top, F_left, F_bottom;

    PetscInitialize(&argc,&argv,(char*)0,help);

    // Creating vectors
    VecCreateSeq(PETSC_COMM_SELF, n_p, &x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &y);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &u);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &v);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &f_x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &f_y);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q_star_x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q_star_y);
    
    // Creating matrices
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &M);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &D_x);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &D_y);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_right);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_top);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_left);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_bottom);

    // Providing initial conditions.
    initial_conditions(x, y, u, v, q, f_x, f_y, ne_x, ne_y, n, x1, y1, x2, y2);
    
    // Creating Global matrices
    createGlobalMatrix(M, ne_x, ne_y, n, "Mass");
    createGlobalMatrix(D_x, ne_x, ne_y, n, "Derivative_x");
    createGlobalMatrix(D_y, ne_x, ne_y, n, "Derivative_y");
    createGlobalMatrix(F_right, ne_x, ne_y, n, "Flux_right");
    createGlobalMatrix(F_top, ne_x, ne_y, n, "Flux_top");
    createGlobalMatrix(F_left, ne_x, ne_y, n, "Flux_left");
    createGlobalMatrix(F_bottom, ne_x, ne_y, n, "Flux_bottom");

    // Multiplying with Jacobians
    MatScale(M, 0.25*dx*dy);
    MatScale(D_x, 0.5*dy);
    MatScale(D_y, 0.5*dx);
    MatScale(F_right, 0.5*dy);
    MatScale(F_left, 0.5*dy);
    MatScale(F_top, 0.5*dx);
    MatScale(F_bottom, 0.5*dx);

    // Transposing the derivative matrices
    MatTranspose(D_x, MAT_REUSE_MATRIX,&D_x);
    MatTranspose(D_y, MAT_REUSE_MATRIX,&D_y);

    // Writing the initial conditions
    writeVTK(x, y, q, ne_x, ne_y, n, "initial_conditions.vtk");

    PetscFinalize();
    return 0;
}