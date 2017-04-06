#include "../../includes/Solvers/AdvectionSolver.h"
#include "../../includes/Utilities/LobattoNodes.h"
#include "../../includes/Utilities/MassMatrix.h"
#include "../../includes/Utilities/DerivativeMatrix.h"
#include "../../includes/Utilities/FluxMatrix.h"
#include "../../includes/Utilities/Inverse.h"
#include <string>
#include <fstream>
#include <cmath>

#define MAX_ABS(a, b)(fabs(a)>fabs(b)?fabs(a):fabs(b))

AdvectionSolver::AdvectionSolver(PetscInt _ne_x, PetscInt _ne_y, PetscInt _n) {
    ne_x = _ne_x;
    ne_y = _ne_y;
    n = _n;
    time = 0.0;
    n_p = (n+1)*(n+1)*ne_x*ne_y;

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

    // Creating Global matrices
    createGlobalMatrix(M_inv, "Mass_Inverse");
    createGlobalMatrix(D_trans_x, "Derivative_x");
    createGlobalMatrix(D_trans_y, "Derivative_y");
    createGlobalMatrix(F_right, "Flux_right");
    createGlobalMatrix(F_top, "Flux_top");
    createGlobalMatrix(F_left, "Flux_left");
    createGlobalMatrix(F_bottom, "Flux_bottom");

    // Transposing the derivative matrices
    MatTranspose(D_trans_x, MAT_INITIAL_MATRIX, &D_x);
    MatTranspose(D_trans_y, MAT_INITIAL_MATRIX, &D_y);

}

void AdvectionSolver::setDomain(PetscReal _x1, PetscReal _y1, PetscReal _x2, PetscReal _y2) {
    x1 = _x1;
    y1 = _y1;
    x2 = _x2;
    y2 = _y2;
    return ;
}

void AdvectionSolver::setInitialConditions(function<PetscReal (PetscReal, PetscReal)> U, function<PetscReal (PetscReal, PetscReal)> V, function<PetscReal (PetscReal, PetscReal)> I) {
    PetscInt    *ix         = new PetscInt[n_p];
    double   *nodes         = new double[n_p];  
    PetscReal   *x_dummy    = new PetscReal[n_p];
    PetscReal   *y_dummy    = new PetscReal[n_p];
    PetscReal   *u_dummy    = new PetscReal[n_p];
    PetscReal   *v_dummy    = new PetscReal[n_p];
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
                    u_dummy[node_index] = U(x_dummy[node_index], y_dummy[node_index]);
                    v_dummy[node_index] = V(x_dummy[node_index], y_dummy[node_index]);
                    q_dummy[node_index] = I(x_dummy[node_index], y_dummy[node_index]);
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

    delete[]   ix; 
    delete[]   nodes;    
    delete[]   x_dummy; 
    delete[]   y_dummy; 
    delete[]   u_dummy; 
    delete[]   v_dummy;
    delete[]   q_dummy; 
    return ;
}

void AdvectionSolver::setSolver(PetscReal _dt, PetscReal _no_of_time_steps) {
    dt = _dt;
    no_of_time_steps = _no_of_time_steps;
    return ;
}
void AdvectionSolver::setBoundaryCondtions(string type) {
    boundaryType = type;
    return ;
}

void AdvectionSolver::createGlobalMatrix(Mat global, string matrixType) {
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

void AdvectionSolver::updateNumericalFlux() {
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
    
    if(boundaryType == "periodic") {
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
    }

    VecRestoreArray(f_x, &f_x_array);
    VecRestoreArray(f_y, &f_y_array);
    VecRestoreArray(f_star_x, &f_star_x_array);
    VecRestoreArray(f_star_y, &f_star_y_array);
    VecRestoreArray(u, &u_array);
    VecRestoreArray(v, &v_array);

    return ;

}


void AdvectionSolver::solve() {

    PetscReal dx, dy;
    dx = (x2-x1)/ne_x;
    dy = (y2-y1)/ne_y;

    Vec     k1, k2, k3;
    Vec     dummy, rhs;

    VecCreateSeq(PETSC_COMM_SELF, n_p, &k1);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &k2);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &k3);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &rhs);
    VecCreateSeq(PETSC_COMM_SELF, n_p, &dummy);

    // Multiplying with Jacobians
    MatScale(M_inv, 4.0/(dx*dy));
    MatScale(D_x, 0.5*dy);
    MatScale(D_y, 0.5*dx);
    MatScale(F_right, 0.5*dy);
    MatScale(F_left, 0.5*dy);
    MatScale(F_top, 0.5*dx);
    MatScale(F_bottom, 0.5*dx);

    // Starting the time-stepping
    for(PetscInt i=0; i<no_of_time_steps; i++) {

        /// First Step for RK3:
        VecSet(rhs, 0.0);

        VecPointwiseMult(f_x, u, q);
        VecPointwiseMult(f_y, v, q);

        updateNumericalFlux();

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

        updateNumericalFlux();

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

        updateNumericalFlux();

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

    VecDestroy(&k1);
    VecDestroy(&k2);
    VecDestroy(&k3);
    VecDestroy(&dummy);
    VecDestroy(&rhs);
    return ;
}

void AdvectionSolver::plot(string filename) {
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
    pFile << "# vtk DataFile Version 3.0\nDG-PETSc\nASCII\nDATASET UNSTRUCTURED_GRID\n";
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

AdvectionSolver::~AdvectionSolver() {
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
}