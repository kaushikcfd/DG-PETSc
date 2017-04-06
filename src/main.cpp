/**
 * File: main.cpp
 * brief: The main purpose of this file is to create an Advection Solver using PETSc.
 * This would include creation of GLOBAL matrices.
 * These GLOBAL matrices are sparse matrices. Hence we would need to assemble a PETSc sparse matrix using the previously computed local matrices. 
 */
#include "petsc.h"
#include <string>
#include <fstream>
#include <cmath>
#include "../includes/Utilities/LobattoNodes.h"
#include "../includes/Utilities/MassMatrix.h"
#include "../includes/Utilities/DerivativeMatrix.h"
#include "../includes/Utilities/FluxMatrix.h"
#include "../includes/Utilities/Inverse.h"

#define MAX_ABS(a, b)(fabs(a)>fabs(b)?fabs(a):fabs(b))

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
    else if(matrixType == "Mass_Inverse") {
        PetscReal   *massMatrix = new PetscReal[(n+1)*(n+1)*(n+1)*(n+1)];
        twoDMassMatrix(massMatrix, n);
        inverse(massMatrix,loc,(n+1)*(n+1));
        delete[] massMatrix;
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

    delete[] idxm;
    delete[] idxn;

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

void updateNumericalFlux(Vec f_x, Vec f_y, Vec f_star_x, Vec f_star_y, Vec u, Vec v, PetscInt ne_x, PetscInt ne_y, PetscInt n) {
    
    PetscInt i, j, k1, k2, node_index, neighbor_index;
    PetscReal lambda ;
    PetscReal   *f_x_array;
    PetscReal   *f_y_array;
    PetscReal   *f_star_x_array;
    PetscReal   *f_star_y_array;
    PetscReal   *u_array;
    PetscReal   *v_array;

    VecGetArray(f_x, &f_x_array);
    VecGetArray(f_y, &f_y_array);
    VecGetArray(f_star_x, &f_star_x_array);
    VecGetArray(f_star_y, &f_star_y_array);
    VecGetArray(u, &u_array);
    VecGetArray(v, &v_array);
    
    /// For the bottom face, internal elements. Ignoring the bottomost elements case as it will be boundary
    /// k2=0, j=[1:ne_y-1], i=[0:ne_x-1], k1 = [0, n]
    for(j=1; j<ne_y; j++) {
        for(i=0; i<ne_x; i++) {
            for(k1=0; k1<=n; k1++) {
                node_index      = (j*ne_x + i)*(n+1)*(n+1) + k1;
                neighbor_index  = ((j-1)*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
                lambda = MAX_ABS(v_array[neighbor_index],v_array[node_index]);
                f_star_y_array[node_index] = 0.5*(
                                                f_y_array[node_index]
                                                +f_y_array[neighbor_index]
                                                -lambda*(f_y_array[node_index]
                                                        -f_y_array[neighbor_index])
                                                );
            }
        }
    }

    /// For the top face, internal elements. Ignoring the topmost elements case as it will be boundary
    /// k2=n, j=[0:ne_y-2], i=[0:ne_x-1], k1 = [0, n]
    for(j=0; j<ne_y-1; j++) {
        for(i=0; i<ne_x; i++) {
            for(k1=0; k1<=n; k1++) {
                node_index      = (j*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
                neighbor_index  = ((j+1)*ne_x + i)*(n+1)*(n+1) + k1;
                lambda = MAX_ABS(v_array[neighbor_index],v_array[node_index]);
                f_star_y_array[node_index] = 0.5*(
                                                f_y_array[node_index]
                                                +f_y_array[neighbor_index]
                                                +lambda*(f_y_array[node_index]
                                                        -f_y_array[neighbor_index])
                                                );
            }
        }
    }

    /// For the left face, internal elements. Ignoring the leftmost elements case as it will be boundary
    /// j=[0:ne_y-1], i=[1:ne_x-1], k2 = [0, n], k1=0
    for(j=0; j<ne_y; j++) {
        for(i=1; i<ne_x; i++) {
            for(k2=0; k2<=n; k2++) {
                node_index      = (j*ne_x + i)*(n+1)*(n+1) + (k2*(n+1));
                neighbor_index  = (j*ne_x + i-1)*(n+1)*(n+1) + (k2*(n+1)+n);
                lambda = MAX_ABS(u_array[neighbor_index],u_array[node_index]);
                f_star_x_array[node_index] = 0.5*(
                                                f_x_array[node_index]
                                                +f_x_array[neighbor_index]
                                                -lambda*(f_x_array[node_index]
                                                        -f_x_array[neighbor_index])
                                                );
            }
        }
    }

    /// For the right face, internal elements. Ignoring the rightmost elements case as it will be boundary
    /// j=[0:ne_y-1], i=[0:ne_x-2], k2 = [0, n], k1=n
    for(j=0; j<ne_y; j++) {
        for(i=0; i<ne_x-1; i++) {
            for(k2=0; k2<=n; k2++) {
                node_index      = (j*ne_x + i)*(n+1)*(n+1) + (k2*(n+1)+n);
                neighbor_index  = (j*ne_x + i+1)*(n+1)*(n+1) + (k2*(n+1));
                lambda = MAX_ABS(u_array[neighbor_index],u_array[node_index]);
                f_star_x_array[node_index] = 0.5*(
                                                f_x_array[node_index]
                                                +f_x_array[neighbor_index]
                                                +lambda*(f_x_array[node_index]
                                                        -f_x_array[neighbor_index])
                                                );
            }
        }
    }

    /// Imposing periodic boundary conditions.
    /// Handling the bottom boundary
    for(i = 0; i<ne_x; i++){
        for(k1 = 0; k1<=n; k1++){
            node_index      = i*(n+1)*(n+1) + k1;
            neighbor_index  = ((ne_y-1)*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
            lambda = MAX_ABS(v_array[neighbor_index],v_array[node_index]);
            f_star_y_array[node_index] = 0.5*(
                                            f_y_array[node_index]
                                            +f_y_array[neighbor_index]
                                            -lambda*(f_y_array[node_index]
                                                    -f_y_array[neighbor_index])
                                            );
        }
    }

    /// Handling the top boundary
    for(i = 0; i<ne_x; i++){
        for(k1 = 0; k1<=n; k1++){
            node_index      = ((ne_y-1)*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
            neighbor_index  = i*(n+1)*(n+1) + k1;
            lambda = MAX_ABS(v_array[neighbor_index],v_array[node_index]);
            f_star_y_array[node_index] = 0.5*(
                                            f_y_array[node_index]
                                            +f_y_array[neighbor_index]
                                            +lambda*(f_y_array[node_index]
                                                    -f_y_array[neighbor_index])
                                            );
        }
    }

    /// Handling the left boundary
    for (j=0; j<ne_y; j++) {
        for(k2=0; k2<=n; k2++) {
            node_index      =  j*ne_x*(n+1)*(n+1) + k2*(n+1);
            neighbor_index  = (j*ne_x + ne_x-1)*(n+1)*(n+1) + (k2*(n+1)+n);
            lambda = MAX_ABS(u_array[neighbor_index],u_array[node_index]);
            f_star_x_array[node_index] = 0.5*(
                                            f_x_array[node_index]
                                            +f_x_array[neighbor_index]
                                            -lambda*(f_x_array[node_index]
                                                    -f_x_array[neighbor_index])
                                            );
        }
    }

    /// Handling the right boundary
    for (j=0; j<ne_y; j++) {
        for(k2=0; k2<=n; k2++) {
            node_index      = (j*ne_x + ne_x-1)*(n+1)*(n+1) + (k2*(n+1)+n);
            neighbor_index  =  j*ne_x*(n+1)*(n+1) + k2*(n+1);
            lambda = MAX_ABS(u_array[neighbor_index],u_array[node_index]);
            f_star_x_array[node_index] = 0.5*(
                                            f_x_array[node_index]
                                            +f_x_array[neighbor_index]
                                            +lambda*(f_x_array[node_index]
                                                    -f_x_array[neighbor_index])
                                            );
            
        }
    }

    VecRestoreArray(f_x, &f_x_array);
    VecRestoreArray(f_y, &f_y_array);
    VecRestoreArray(f_star_x, &f_star_x_array);
    VecRestoreArray(f_star_y, &f_star_y_array);
    VecRestoreArray(u, &u_array);
    VecRestoreArray(v, &v_array);

    return ;

}

int main(int argc, char *argv[])
{   
    /// Constants that define the problem.
    PetscInt ne_x = 10, ne_y = 10;   /// Number of elements in the x and y direction resp.
    PetscInt n = 4;                  /// The order of interpolation
    PetscInt n_time = 100;
    PetscReal dt = 1e-2;

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
    Vec     f_star_x, f_star_y;
    Vec     k1, k2, k3;
    Vec     dummy, rhs;

    // Declaring the matrices, small letter for local, Capital for global
    Mat M_inv, D_x, D_y, F_right, F_top, F_left, F_bottom, D_trans_x, D_trans_y;

    PetscInitialize(&argc,&argv,(char*)0,help);

    // Creating vectors
    VecCreateSeq(PETSC_COMM_SELF, n_p, &x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &y);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &u);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &v);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &f_x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &f_y);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &f_star_x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &f_star_y);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &k1);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &k2);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &k3);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &rhs);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &dummy);
    
    // Creating matrices
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &M_inv);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &D_trans_x);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &D_trans_y);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &D_x);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &D_y);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_right);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_top);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_left);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, (n+1)*(n+1), NULL, &F_bottom);

    // Providing initial conditions.
    initial_conditions(x, y, u, v, q, f_x, f_y, ne_x, ne_y, n, x1, y1, x2, y2);
    
    // Creating Global matrices
    createGlobalMatrix(M_inv, ne_x, ne_y, n, "Mass_Inverse");
    createGlobalMatrix(D_trans_x, ne_x, ne_y, n, "Derivative_x");
    createGlobalMatrix(D_trans_y, ne_x, ne_y, n, "Derivative_y");
    createGlobalMatrix(F_right, ne_x, ne_y, n, "Flux_right");
    createGlobalMatrix(F_top, ne_x, ne_y, n, "Flux_top");
    createGlobalMatrix(F_left, ne_x, ne_y, n, "Flux_left");
    createGlobalMatrix(F_bottom, ne_x, ne_y, n, "Flux_bottom");

    // Multiplying with Jacobians
    MatScale(M_inv, 4.0/(dx*dy));
    MatScale(D_trans_x, 0.5*dy);
    MatScale(D_trans_y, 0.5*dx);
    MatScale(F_right, 0.5*dy);
    MatScale(F_left, 0.5*dy);
    MatScale(F_top, 0.5*dx);
    MatScale(F_bottom, 0.5*dx);

    // Transposing the derivative matrices
    MatTranspose(D_trans_x, MAT_INITIAL_MATRIX, &D_x);
    MatTranspose(D_trans_y, MAT_INITIAL_MATRIX, &D_y);

    // Starting the time-stepping
    for(PetscInt i=0; i<n_time; i++) {

        /// First Step for RK3:
        VecSet(rhs, 0.0);

        VecPointwiseMult(f_x, u, q);
        VecPointwiseMult(f_y, v, q);

        updateNumericalFlux(f_x, f_y, f_star_x, f_star_y, u, v, ne_x, ne_y, n);

        MatMult(D_x, f_x, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(D_y, f_y, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(F_right, f_star_x, dummy);
        VecAXPY(rhs, -1.0, dummy);

        MatMult(F_left, f_star_x, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(F_top, f_star_y, dummy);
        VecAXPY(rhs, -1.0, dummy);

        MatMult(F_bottom, f_star_y, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(M_inv, rhs, k1);
        VecAXPY(q, 0.5*dt, k1);

        /// Second Step for RK3:
        VecSet(rhs, 0.0);


        VecPointwiseMult(f_x, u, q);
        VecPointwiseMult(f_y, v, q);

        updateNumericalFlux(f_x, f_y, f_star_x, f_star_y, u, v, ne_x, ne_y, n);

        MatMult(D_x, f_x, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(D_y, f_y, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(F_right, f_star_x, dummy);
        VecAXPY(rhs, -1.0, dummy);

        MatMult(F_left, f_star_x, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(F_top, f_star_y, dummy);
        VecAXPY(rhs, -1.0, dummy);

        MatMult(F_bottom, f_star_y, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(M_inv, rhs, k2);
        VecAXPY(q, -1.5*dt, k1);
        VecAXPY(q,  2.0*dt, k2);

        /// Third Step for RK3:
        VecSet(rhs, 0.0);

        VecPointwiseMult(f_x, u, q);
        VecPointwiseMult(f_y, v, q);

        updateNumericalFlux(f_x, f_y, f_star_x, f_star_y, u, v, ne_x, ne_y, n);

        MatMult(D_x, f_x, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(D_y, f_y, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(F_right, f_star_x, dummy);
        VecAXPY(rhs, -1.0, dummy);

        MatMult(F_left, f_star_x, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(F_top, f_star_y, dummy);
        VecAXPY(rhs, -1.0, dummy);

        MatMult(F_bottom, f_star_y, dummy);
        VecAXPY(rhs, 1.0, dummy);

        MatMult(M_inv, rhs, k3);
        VecAXPY(q,  (7.0/6.0)*dt, k1);
        VecAXPY(q, -(4.0/3.0)*dt, k2);
        VecAXPY(q,  (1.0/6.0)*dt, k3);
    }

    writeVTK(x, y, q, ne_x, ne_y, n, "output.vtk");

    /// Freeing the space by destroying the vectors
    VecDestroy(&x);
    VecDestroy(&y);
    VecDestroy(&u);
    VecDestroy(&v);
    VecDestroy(&q);
    VecDestroy(&f_x);
    VecDestroy(&f_y);
    VecDestroy(&f_star_x);
    VecDestroy(&f_star_y);
    VecDestroy(&k1);
    VecDestroy(&k2);
    VecDestroy(&k3);
    VecDestroy(&dummy);
    VecDestroy(&rhs);

    /// Freeing the space by destroying the matrices
    MatDestroy(&M_inv);
    MatDestroy(&D_x);
    MatDestroy(&D_y);
    MatDestroy(&F_right);
    MatDestroy(&F_top);
    MatDestroy(&F_left);
    MatDestroy(&F_bottom);
    MatDestroy(&D_trans_x);
    MatDestroy(&D_trans_y);

    PetscFinalize();
    return 0;
}