#ifndef ADVECTIONSOLVER_H
#define ADVECTIONSOLVER_H

#include "petsc.h"
#include <functional>
using namespace std;

class AdvectionSolver {
private:
    PetscInt ne_x, ne_y, n, n_p;
    PetscReal x1, y1, x2, y2;
    PetscReal time;
    PetscReal dt;
    PetscInt no_of_time_steps;
    string boundaryType;

    /// Declaring the vectors
    Vec     x, y;
    Vec     u, v;
    Vec     q, f_x, f_y;
    Vec     f_star_x, f_star_y;

    // Declaring the matrices, small letter for local, Capital for global
    Mat M_inv, D_x, D_y, F_right, F_top, F_left, F_bottom, D_trans_x, D_trans_y;

    /// Private functions
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This function computes the numeical flux from the fluxes at the edges and updates the vectors f_star_x, f_star_y accordingly.
     */
    /* ----------------------------------------------------------------------------*/
    void updateNumericalFlux();
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis This function gives a global matrix for the given matrix type
     * 
     * @Param global    This is the matrix whose value is to be filled
     * @Param matrixType This string denotes what type of matrix is to be filled in the global matrix.  
     */
    /* ----------------------------------------------------------------------------*/
    void createGlobalMatrix(Mat global, string matrixType);

public:
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This is the class constructor. The main function is to initialize the clas number of elements and the
     * order of interpolation.
     *
     * @Param _ne_x The number of elements in the x-direction.
     * @Param _ne_y The number of elements in the y-direction.
     * @Param _n    The order of interpolation used for getting the results.
     */
    /* ----------------------------------------------------------------------------*/
    AdvectionSolver(PetscInt _ne_x, PetscInt _ne_y, PetscInt _n);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This is the function for setting the domain of the problem.
     *
     * @Param _x1 The x-coordinate of the lower left corner of the domain.
     * @Param _y1 The y-coorindate of the lower left corner of the domain.
     * @Param _x2 The x-coordinate of the upper right corner of the domain.
     * @Param _y2 The y-coordinate of the upper right corner of the domain.
     */
    /* ----------------------------------------------------------------------------*/
    void setDomain(PetscReal _x1, PetscReal _y1, PetscReal _x2, PetscReal _y2);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This is the function to set the type of the boundary condition.
     *
     * @Param type This will tell the type of boundary conditions:
     *              - "periodic"  = Periodic Boundary Condition
     *              - "dirichlet" = Dirichlet Boundary Condition
     *              - "neumann"   = Neumann Boundary Condition.
     */
    /* ----------------------------------------------------------------------------*/
    void setBoundaryCondtions(string type);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This is the function used to give the initial input waveform as a function and the velocity provided.
     *
     * @Param functionU This is the function used to initialize the `U` velocity as an input.
     * @Param functionV This is the function used to initialize the `V` velocity as an input.
     * @Param functionI The input function which is used to initialize the waveform. The function takes 2 inputs x and y in
     * order.
     */
    /* ----------------------------------------------------------------------------*/
    void setInitialConditions(function<PetscReal(PetscReal, PetscReal)>U, function<PetscReal(PetscReal, PetscReal)>V, function<PetscReal(PetscReal, PetscReal)> I);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis   This function is used to set important solver parameters like dt, and no. of time steps.
     *
     * @Param _dt The time step for each iteration.
     * @Param _no_of_time_steps The number of time steps that must be used which is also the number of time iterations that must
     * be performed
     */
    /* ----------------------------------------------------------------------------*/
    void setSolver(PetscReal _dt, PetscReal _no_of_time_steps);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This function does all the main functionalitites of the solver. This must be called in order to solve
     * the problem
     */
    /* ----------------------------------------------------------------------------*/
    void solve();
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This function plots the function in vtk fileformat which can be further read by software packages like
     * ParaView.
     *
     * @Param filename This is the filename with which the information is to be stored.
     */
    /* ----------------------------------------------------------------------------*/
    void plot(string filename);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  Frees the memory
     */
    /* ----------------------------------------------------------------------------*/
    void destroy();

};

#endif