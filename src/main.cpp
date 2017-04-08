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

PetscReal Q(PetscReal x, PetscReal y) {

    return sin(3.1415926*x);

    /// Dirichlet boundary conditions
    if(x== -1.0) {
        return 3.0;
    }
    else if(x== 1.0) {
        return 1.0;
    }
    else if(y== -1.0) {
        return 0.0;
    }
    else if(y== 1.0) {
        return 2.0;
    }
    else{
        return 0.0;
    }
}

void initial_conditions(Vec x, Vec y, Vec q, PetscInt ne_x, PetscInt ne_y, PetscInt n, PetscReal x1, PetscReal y1, PetscReal x2, PetscReal y2) {
    PetscInt n_p    =   ne_x*ne_y*(n+1)*(n+1);
    PetscInt    *ix         = new PetscInt[n_p];
    double   *nodes         = new double[n_p];  
    PetscReal   *x_dummy    = new PetscReal[n_p];
    PetscReal   *y_dummy    = new PetscReal[n_p];
    PetscReal   *q_dummy    = new PetscReal[n_p];

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
                    q_dummy[node_index] = Q(x_dummy[node_index], y_dummy[node_index]);
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
    VecSetValues(q, n_p, ix, q_dummy, INSERT_VALUES);
    VecAssemblyBegin(q);
    VecAssemblyEnd(q);

    delete[]   ix; 
    delete[]   nodes;    
    delete[]   x_dummy; 
    delete[]   y_dummy;
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

void computeNumericalFluxMatrices(Mat NF_right, Mat NF_top, Mat NF_left, Mat NF_bottom, PetscInt ne_x, PetscInt ne_y, PetscInt n) {
    
    PetscInt i, j, k1, k2, node_index, neighbor_index;
    
    /// For the bottom face, internal elements. Ignoring the bottomost elements case as it will be boundary
    /// k2=0, j=[1:ne_y-1], i=[0:ne_x-1], k1 = [0, n]
    for(j=1; j<ne_y; j++) {
        for(i=0; i<ne_x; i++) {
            for(k1=0; k1<=n; k1++) {
                node_index      = (j*ne_x + i)*(n+1)*(n+1) + k1;
                neighbor_index  = ((j-1)*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
                MatSetValue(NF_bottom, node_index, neighbor_index, 0.5, INSERT_VALUES);
                MatSetValue(NF_bottom, node_index, node_index, 0.5, INSERT_VALUES);
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
                MatSetValue(NF_top, node_index, neighbor_index, 0.5, INSERT_VALUES);
                MatSetValue(NF_top, node_index, node_index, 0.5, INSERT_VALUES);
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
                MatSetValue(NF_left, node_index, neighbor_index, 0.5, INSERT_VALUES);
                MatSetValue(NF_left, node_index, node_index, 0.5, INSERT_VALUES);
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
                MatSetValue(NF_right, node_index, neighbor_index, 0.5, INSERT_VALUES);
                MatSetValue(NF_right, node_index, node_index, 0.5, INSERT_VALUES);
            }
        }
    }

    /// Imposing free boundary conditions.
    /// Handling the bottom boundary
    for(i = 0; i<ne_x; i++){
        for(k1 = 0; k1<=n; k1++){
            node_index      = i*(n+1)*(n+1) + k1;
            neighbor_index  = ((ne_y-1)*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
            MatSetValue(NF_bottom, node_index, node_index, 1.0, INSERT_VALUES);
        }
    }

    /// Handling the top boundary
    for(i = 0; i<ne_x; i++){
        for(k1 = 0; k1<=n; k1++){
            node_index      = ((ne_y-1)*ne_x + i)*(n+1)*(n+1) + (n*(n+1)+k1);
            neighbor_index  = i*(n+1)*(n+1) + k1;
            MatSetValue(NF_top, node_index, node_index, 1.0, INSERT_VALUES);
        }
    }

    /// Handling the left boundary
    for (j=0; j<ne_y; j++) {
        for(k2=0; k2<=n; k2++) {
            node_index      =  j*ne_x*(n+1)*(n+1) + k2*(n+1);
            neighbor_index  = (j*ne_x + ne_x-1)*(n+1)*(n+1) + (k2*(n+1)+n);
            MatSetValue(NF_left, node_index, node_index, 1.0, INSERT_VALUES);
        }
    }

    /// Handling the right boundary
    for (j=0; j<ne_y; j++) {
        for(k2=0; k2<=n; k2++) {
            node_index      = (j*ne_x + ne_x-1)*(n+1)*(n+1) + (k2*(n+1)+n);
            neighbor_index  =  j*ne_x*(n+1)*(n+1) + k2*(n+1);
            MatSetValue(NF_right, node_index, node_index, 1.0, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(NF_bottom, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(NF_bottom, MAT_FINAL_ASSEMBLY);
    
    MatAssemblyBegin(NF_top, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(NF_top, MAT_FINAL_ASSEMBLY);

    MatAssemblyBegin(NF_left, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(NF_left, MAT_FINAL_ASSEMBLY);

    MatAssemblyBegin(NF_right, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(NF_right, MAT_FINAL_ASSEMBLY);

    return ;
}

int main(int argc, char *argv[]) {   
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
    Vec     q, q_x, q_xx;

    // Declaring the matrices, small letter for local, Capital for global
    Mat M_inv, D_x, D_y, F_right, F_top, F_left, F_bottom, D_trans_x, D_trans_y, NF_right, NF_top, NF_left, NF_bottom;

    Mat A1, A2, A3;

    PetscInitialize(&argc,&argv,(char*)0,help);

    // Creating vectors
    VecCreateSeq(PETSC_COMM_SELF, n_p, &x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &y);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q_x);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &q_xx);
    
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
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, 2, NULL, &NF_right);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, 2, NULL, &NF_top);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, 2, NULL, &NF_left);
    MatCreateSeqAIJ(PETSC_COMM_SELF, n_p, n_p, 2, NULL, &NF_bottom);


    // Providing initial conditions.
    initial_conditions(x, y, q, ne_x, ne_y, n, x1, y1, x2, y2);
    
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

    // Obtaining the numerical flux matrices
    computeNumericalFluxMatrices(NF_right, NF_top, NF_left, NF_bottom, ne_x, ne_y, n);

    MatMatMult(F_left, NF_left, MAT_INITIAL_MATRIX, PETSC_DECIDE, &A1);
    MatMatMult(F_right, NF_right, MAT_INITIAL_MATRIX, PETSC_DECIDE, &A2);
    
    MatAXPY(A2, -1, A1, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(A2, -1, D_x, DIFFERENT_NONZERO_PATTERN);
    MatMatMult(M_inv, A2, MAT_INITIAL_MATRIX, PETSC_DECIDE, &A3);
    
    MatMult(A3, q, q_x);
    MatMult(A3, q_x, q_xx);

    writeVTK(x, y, q_xx, ne_x, ne_y, n, "danda.vtk");

    /// Freeing the space by destroying the vectors
    VecDestroy(&x);
    VecDestroy(&y);
    VecDestroy(&q);

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