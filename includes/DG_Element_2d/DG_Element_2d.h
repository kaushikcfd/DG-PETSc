/**
 * @file DG_Element_2d.h
 * @Synopsis  This is the header file for the class which defines the DG_Element_2d class. This class contains all the
 * declarations which are necessary for defining a general 2d element. This class defines all the necessities for a
 * structured 2d DG element.
 * @author Kaushik Kulkarni
 * @version 1.0
 * @date 2017-02-18
 */



/* Copyright (c) 2016, Shivasubramanian Gopalakrishnan and Kaushik Kulkarni.
  All rights reserved.

   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

     -Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
     -Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  **AS IS** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
     */

#ifndef DG_ELEMENT_2D_H_
#define DG_ELEMENT_2D_H_

#include <map>
#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <cblas.h>

using namespace std;

class DG_Element_2d {
private:
    
    int N; /// This represents the order of polynomial interpolation which is done while solving the DG equations.
    double x_start, x_end, y_start, y_end; /// Since, this is a structured rectangular element. These variables define the position and size of the element.
    double* massMatrix;
    double* derivativeMatrix_x; /// This is the matrix which is laid out in 1-D, this would help us to find t    he $\frac{d}{dx}$ of any term. 
    double* derivativeMatrix_y; /// This is the matrix which is laid out in 1-D, this would help us to find t    he $\frac{d}{dy}$ of any term.
    double* fluxMatrix_top; /// This is the flux matrix for the top edge.
    double* fluxMatrix_right; // The Flux Matrix for the right edge.
    double* fluxMatrix_bottom; /// This would be the flux term for the the bottom edge.
    double* fluxMatrix_left; /// The Flux matrix for the left edge.
    double* inverseMassMatrix; /// The inverse of the mass matrix is also created only once in the field function. And just passed to each element.
     
    DG_Element_2d* topNeighbor;
    DG_Element_2d* rightNeighbor;
    DG_Element_2d* leftNeighbor;
    DG_Element_2d* bottomNeighbor;

public:

    double *X, *Y;
    map<string, double*> variable; /// This is the map which would contain all the variable that are needed by the problem.

    map<string, double**> boundaryTop;
    map<string, double**> boundaryRight;
    map<string, double**> boundaryLeft;
    map<string, double**> boundaryBottom;
    
    map<string, double**> neighboringTop;
    map<string, double**> neighboringRight;
    map<string, double**> neighboringBottom;
    map<string, double**> neighboringLeft;

    vector<string> boundaryVariables; /// This is the variable which stores the name of all the variables whose boundary and neighboring points are stored. 

    DG_Element_2d(int _N, double x1, double y1, double x2, double y2);
    void addVariable_withBoundary(string v);
    void addVariable_withoutBoundary(string v);
    void initializeVariable(string v, function<double(double, double)> f);
    void setNeighboringElement(char type, DG_Element_2d* neighbor );
    void setVariableNeighbors(string v);

    // Functions for various operations on the variables.
    void delByDelX(string v, string vDash, string fluxType, string fluxVariable);

    void delByDelY(string v, string vDash, string fluxType, string fluxVariable);

    // Functions to set the various operator matrices.
    void setMassMatrix(double* m);
    void setInverseMassMatrix(double* im);
    void setderivateMatrix_x(double* d);
    void setderivateMatrix_y(double* d);
    void setTopFluxMatrix(double* f);
    void setRightFluxMatrix(double* f);
    void setLeftFluxMatrix(double* f);
    void setBottomFluxMatrix(double* f);

    // Functions to apply linear operations on the variables.
    void axpy(double a, string x, string y);
    void scal(double a, string x);
    void setFunctionsForVariables(string x, string y, function<double(double, double)>, string z);
    void setFunctionsForVariables(string w, string x, string y, function<double(double, double, double)>, string z);

    // Functions to do various other operations on the elements.
    double l2Norm(string v1, string v2);

};

#endif
