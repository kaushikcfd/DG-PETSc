#ifndef ADVECTIONSOLVER_H
#define ADVECTIONSOLVER_H

# include "../DG_Field_2d/DG_Field_2d.h" 
#include <functional>

using namespace std;

class AdvectionSolver {
private:
    int ne_x, ne_y, N;
    double x1, y1, x2, y2;
    DG_Field_2d* field;
    double time;
    double dt;
    int no_of_time_steps;

public:
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This is the class constructor. The main function is to initialize the clas number of elements and the
     * order of interpolation.
     *
     * @Param _ne_x The number of elements in the x-direction.
     * @Param _ne_y The number of elements in the y-direction.
     * @Param _N    The order of interpolation used for getting the results.
     */
    /* ----------------------------------------------------------------------------*/
    AdvectionSolver(int _ne_x, int _ne_y, int _N);
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
    void setDomain(double _x1, double _y1, double _x2, double _y2);
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
     * @Synopsis This is the function used to initialize the the veclocity of the domain  
     *
     * @Param functionU This is the function used to initialize the `U` velocity as an input.
     * @Param functionV This is the function used to initialize the `V` velocity as an input.
     */
    /* ----------------------------------------------------------------------------*/
    void setVelocity(function<double(double, double)>U, function<double(double, double)>V);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis  This is the function used to give the initial input waveform as a function.
     *
     * @Param I The input function which is used to initialize the waveform. The function takes 2 inputs x and y in
     * order.
     */
    /* ----------------------------------------------------------------------------*/
    void setInitialConditions(function<double(double, double)> I);
    /* ----------------------------------------------------------------------------*/
    /**
     * @Synopsis   This function is used to set important solver parameters like dt, and no. of time steps.
     *
     * @Param _dt The time step for each iteration.
     * @Param _no_of_time_steps The number of time steps that must be used which is also the number of time iterations that must
     * be performed
     */
    /* ----------------------------------------------------------------------------*/
    void setSolver(double _dt, double _no_of_time_steps);
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
};

#endif
