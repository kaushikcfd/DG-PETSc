/**
 * File: main.cpp
 * brief: The main purpose of this file is to create an Advection Solver using PETSc.
 * This would include creation of GLOBAL matrices.
 * These GLOBAL matrices are sparse matrices. Hence we would need to assemble a PETSc sparse matrix using the previously computed local matrices. 
 */
#include "petsc.h"
#include "../includes/Solvers/AdvectionSolver.h"

using namespace std;

static char help[] = "Advection Solver\n\n";

PetscReal U(PetscReal x, PetscReal y) {
    return 1.0;
}

PetscReal V(PetscReal x, PetscReal y) {
    return 0.0;
}

PetscReal Q(PetscReal x, PetscReal y) {
    return (exp(-(x*x +  y*y)*16.0));
}

int main(int argc, char *argv[])
{   
    PetscInitialize(&argc,&argv,(char*)0,help);
    
    /// Constants that define the problem.
    PetscInt ne_x = 10, ne_y = 10;   /// Number of elements in the x and y direction resp.
    PetscInt n = 4;                  /// The order of interpolation
    PetscInt n_time = 100;
    PetscReal dt = 1e-2;

    /// Setting the domain.
    PetscReal x1, y1, x2, y2;
    x1 = y1 = -1.0;
    x2 = y2 =  1.0;

    AdvectionSolver q(ne_x, ne_y, n);
    
    q.setDomain(x1, y1, x2, y2);
    q.setBoundaryCondtions("periodic");
    q.setInitialConditions(U, V, Q);
    q.setSolver(dt, n_time);
    q.solve();
    q.plot("output.vtk");
    q.destroy();

    PetscFinalize();
    return 0;
}