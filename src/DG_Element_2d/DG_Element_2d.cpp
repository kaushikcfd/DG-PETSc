#include "../../includes/DG_Element_2d/DG_Element_2d.h"

#include "../../includes/Utilities/Zeros.h"
#include "../../includes/Utilities/Inverse.h"

#include "../../includes/Utilities/FluxMatrix.h"
#include "../../includes/Utilities/MassMatrix.h"
#include "../../includes/Utilities/DerivativeMatrix.h"
#include "../../includes/Utilities/LobattoNodes.h"

#include <cmath>

#define MAX(a, b)(a>b?a:b)

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This is the constructor which initializes the member variables of the class. It also populates the arrays
 * X, Y as once the co-ordinates of the rectangle is known these arrays can be computed.
 *
 * @Param[in] _N The order of the polynomial which is to be used.
 * @Param[in] x1 The x-coord of the bottom left corner.
 * @Param[in] y1 The y-coord of the bottom left corner.
 * @Param[in] x2 The x-coord of the top right corner.
 * @Param[in] y2 The y-coord of the top right corner.
 */
/* ----------------------------------------------------------------------------*/
DG_Element_2d::DG_Element_2d(int _N, double x1, double y1, double x2, double y2) {
    N = _N;
    x_start = x1;
    y_start = y1;
    x_end   = x2;
    y_end   = y2;


    X = new double[(N+1)*(N+1)];
    Y = new double[(N+1)*(N+1)];

    double* nodes = new double[N+1]; // Alloting space for the lobatto nodes.
    
    lobattoNodes(nodes, N+1); // Found the lobatto nodes in the range -1 to +1.

    for(int k1=0;k1<=N;k1++)
    {
        for(int k2=0;k2<=N;k2++)
        {
            X[k1*(N+1)+k2] = 0.5*(x1 + x2) + 0.5*(x2 - x1)*nodes[k2];
            Y[k1*(N+1)+k2] = 0.5*(y1 + y2) + 0.5*(y2 - y1)*nodes[k1];
        }
    }

    delete[] nodes;

    massMatrix = NULL;

    derivativeMatrix_x  =   NULL;
    derivativeMatrix_y  =   NULL; 
    fluxMatrix_top      =   NULL;
    fluxMatrix_right    =   NULL;
    fluxMatrix_bottom   =   NULL;
    fluxMatrix_left     =   NULL;
    inverseMassMatrix   =   NULL;
    topNeighbor         =   NULL;
    rightNeighbor       =   NULL;
    leftNeighbor        =   NULL;
    bottomNeighbor      =   NULL;

}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This functions creates space in order to take in one more variable on which operators are needed to be
 * applied.
 *
 * @Param v  This is the name of the variable which is to be added.
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::addVariable_withoutBoundary(string v) {
    double * newVariable = new double[(N+1)*(N+1)]; /// Allocating the space for the new variable which is to be created.
    variable[v] = newVariable; /// Now assigning the same to the map.

    return ;
}


/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This functions creates space in order to take in one more variable on which operators are needed to be
 * applied.
 *
 * @Param v  This is the name of the variable which is to be added.
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::addVariable_withBoundary(string v) {
    double * newVariable = new double[(N+1)*(N+1)]; /// Allocating the space for the new variable which is to be created.
    variable[v] = newVariable; /// Now assigning the same to the map.
    
    // **b_top is used because it will store the address to the boundary element, So whenever the actual value in the double* of the variable is changed then this will also change automatically. The same holds for all the other following mentioned variables.
    
    double **b_top       = new double* [N+1]; 
    double **b_bottom    = new double* [N+1];
    double **b_left      = new double* [N+1];
    double **b_right     = new double* [N+1];

   
    double **n_top       = new double* [N+1];
    double **n_left      = new double* [N+1];
    double **n_right     = new double* [N+1];
    double **n_bottom    = new double* [N+1];

    for(int i=0; i<=N; i++){
        n_bottom[i] =   b_bottom[i] = &(variable[v][i]);    
        n_top[i]    =   b_top[i]    = &(variable[v][N*(N+1)+i]);
        n_left[i]   =   b_left[i]   = &(variable[v][i*(N+1)]);
        n_right[i]  =   b_right[i]  = &(variable[v][i*(N+1)+N]);
    }


    boundaryTop[v]      = b_top;
    boundaryRight[v]    = b_right;
    boundaryBottom[v]   = b_bottom;
    boundaryLeft[v]     = b_left;


    neighboringTop[v]   = n_top;
    neighboringRight[v] = n_right;
    neighboringBottom[v]= n_bottom;
    neighboringLeft[v]  = n_left;

    boundaryVariables.push_back(v);

    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  Function to set the value of a variable with the help of a function.
 *
 * @Param v This is the name of the variable whose value is to be  set.
 * @Param f This is the f(x, y) which is used to initialize the variable.
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::initializeVariable(string v, function<double(double, double)> f) {
    for(int i=0; i<((N+1)*(N+1)); i++)
        variable[v][i] = f(X[i], Y[i]);
    
    return ;
}


void DG_Element_2d::setVariableNeighbors(string v) {
    int j;
    if(topNeighbor!=NULL) {
        for( j = 0 ; j <= N; j++) 
            neighboringTop[v][j] = topNeighbor->boundaryBottom[v][j];
    }
    if(rightNeighbor!=NULL) {
        for( j = 0 ; j <= N; j++) 
            neighboringRight[v][j] = rightNeighbor->boundaryLeft[v][j];
    }
    if(leftNeighbor!=NULL) {
        for( j = 0 ; j <= N; j++) 
            neighboringLeft[v][j] = leftNeighbor->boundaryRight[v][j];
    }
    if(bottomNeighbor!=NULL) {
        for( j = 0 ; j <= N; j++) 
            neighboringBottom[v][j] = bottomNeighbor->boundaryTop[v][j];
    }
    return ;
}


/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This function sets the boundary variables for an element. This must be called whenever a new variable with
 * boundary is added. 
 *
 * @Param v The variable whose information about the boundaries is to be stored
 * @Param type The type of the neighbor whose information is to be looked upon.
 * @Param neighbor The pointer to the element 
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::setNeighboringElement(char type, DG_Element_2d* neighbor) {
    string currentVariable;
    switch(type) {
        case 't' : // `t` or `T` for top 
        case 'T' :
            topNeighbor = neighbor;
            break;
        case 'r' : // `r` or `R` for right
        case 'R' :
            rightNeighbor = neighbor;
            break;
        case 'b' : // `b` or `B` for bottom
        case 'B' :
            bottomNeighbor = neighbor;
            break;
        case 'l' : // `l` or `L` for left
        case 'L' :
            leftNeighbor = neighbor;
            break;
        default:
            cout << "WARNING!. No such neighbor type " << type << endl;
    }
    return ;
}


/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This is the function to get the variable `v` differentiated partially w.r.t. `x` and then store it in the
 * variable `vDash`. The function also takes `fluxType` as an input which would describe the numerical scheme that
 * should be used in order to obtain the derivative.
 *
 * @Param v         Variable which is to be differentiated.
 * @Param vDash     Variable in which the derivative is to be stored.
 * @Param fluxType  The type of flux that is to be used. eg "central"
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::delByDelX(string v, string vDash, string fluxType, string fluxVariable = "") {
    double dy = (y_end - y_start);
    double dx = (x_end - x_start);
    
    if(fluxType == "central") {
        double* numericalFlux        =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable.
        double* auxillaryVariable    =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable, auxiallary variable
        zeros(numericalFlux, (N+1)*(N+1));                                                       
        for(int i=0; i<=N; i++){
            numericalFlux[i*(N+1)+N]    = 0.5*( *boundaryRight[v][i]    + *neighboringRight[v][i] ) ;   
            numericalFlux[i*(N+1)]    = 0.5*( *boundaryLeft[v][i]     + *neighboringLeft[v][i] ) ;  
        }
        /// vDash = -0.5*dy*D*v
        cblas_dgemv(CblasRowMajor, CblasTrans,   (N+1)*(N+1), (N+1)*(N+1), -0.5*dy, derivativeMatrix_x, (N+1)*(N+1), variable[v],   1, 0, auxillaryVariable, 1);

        /// Adding the numeical Flux terms as necessary.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  0.5*dy, fluxMatrix_right,   (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  -0.5*dy, fluxMatrix_left,    (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);

        /// Multiplying my Mass Inverse, this is the final step in getting the derivative.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1), 4.0/(dx*dy), inverseMassMatrix,(N+1)*(N+1), auxillaryVariable,1,0,variable[vDash],1);

        delete[] numericalFlux;
        delete[] auxillaryVariable;
    }

    else if(fluxType == "rusanov") {
        double* numericalFlux        =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable.
        double* auxillaryVariable    =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable, auxiallary variable
        zeros(numericalFlux, (N+1)*(N+1));                                                       
        for(int i=0; i<=N; i++){
            numericalFlux[i*(N+1)+N] = 0.5*(*boundaryRight[v][i] + *neighboringRight[v][i] + MAX(fabs(*boundaryRight[fluxVariable][i]), fabs(*neighboringRight[fluxVariable][i]))*(*boundaryRight[v][i] - *neighboringRight[v][i])  ) ;   
            numericalFlux[i*(N+1)]   = 0.5*(*boundaryLeft[v][i]  + *neighboringLeft[v][i]  - MAX(fabs(*boundaryLeft[fluxVariable][i]), fabs(*neighboringLeft[fluxVariable][i]))*(*boundaryLeft[v][i] - *neighboringLeft[v][i])  ) ;   
        }
        /// vDash = -0.5*dy*D*v
        cblas_dgemv(CblasRowMajor, CblasTrans,   (N+1)*(N+1), (N+1)*(N+1), -0.5*dy, derivativeMatrix_x, (N+1)*(N+1), variable[v],   1, 0, auxillaryVariable, 1);

        /// Adding the numeical Flux terms as necessary.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  0.5*dy, fluxMatrix_right,   (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  -0.5*dy, fluxMatrix_left,    (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);

        /// Multiplying my Mass Inverse, this is the final step in getting the derivative.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1), 4.0/(dx*dy), inverseMassMatrix,(N+1)*(N+1), auxillaryVariable,1,0,variable[vDash],1);

        delete[] numericalFlux;
        delete[] auxillaryVariable;

    }
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This is the function to get the variable `v` differentiated partially w.r.t. `y` and then store it in the
 * variable `vDash`. The function also takes `fluxType` as an input which would describe the numerical scheme that
 * should be used in order to obtain the derivative.
 *
 * @Param v         Variable which is to be differentiated.
 * @Param vDash     Variable in which the derivative is to be stored.
 * @Param fluxType  The type of flux that is to be used. eg "central"
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::delByDelY(string v, string vDash, string fluxType, string fluxVariable = "") {
    double dy = (y_end - y_start);
    double dx = (x_end - x_start);

    if(fluxType == "central") {
        double* numericalFlux        =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable.
        double* auxillaryVariable    =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable, auxiallary variable
        zeros(numericalFlux, (N+1)*(N+1));                                                       
        for(int i=0; i<=N; i++){
            numericalFlux[N*(N+1)+i]    = 0.5*( *boundaryTop[v][i]    + *neighboringTop[v][i] ) ;   
            numericalFlux[i]            = 0.5*( *boundaryBottom[v][i]     + *neighboringBottom[v][i] ) ;  
        }
        /// vDash = -0.5*dy*D*v
        cblas_dgemv(CblasRowMajor, CblasTrans,   (N+1)*(N+1), (N+1)*(N+1), -0.5*dx, derivativeMatrix_y, (N+1)*(N+1), variable[v],   1, 0, auxillaryVariable, 1);

        /// Adding the numeical Flux terms as necessary.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  0.5*dx, fluxMatrix_top,   (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  -0.5*dx, fluxMatrix_bottom,    (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);

        /// Multiplying my Mass Inverse, this is the final step in getting the derivative.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1), 4.0/(dx*dy), inverseMassMatrix,(N+1)*(N+1), auxillaryVariable,1,0,variable[vDash],1);

        delete[] numericalFlux;
        delete[] auxillaryVariable;
    }
    
    else if(fluxType == "rusanov") {
        double* numericalFlux        =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable.
        double* auxillaryVariable    =   new double[(N+1)*(N+1)]; /// Creating a temporary new variable, auxiallary variable
        zeros(numericalFlux, (N+1)*(N+1));                                                       
        for(int i=0; i<=N; i++){
            numericalFlux[N*(N+1)+i]= 0.5*(*boundaryTop[v][i] + *neighboringTop[v][i] + MAX(fabs(*boundaryTop[fluxVariable][i]), fabs(*neighboringTop[fluxVariable][i]))*(*boundaryTop[v][i] - *neighboringTop[v][i]));
            numericalFlux[i]        = 0.5*(*boundaryBottom[v][i] + *neighboringBottom[v][i] - MAX(fabs(*boundaryBottom[fluxVariable][i]), fabs(*neighboringBottom[fluxVariable][i]))*(*boundaryBottom[v][i] - *neighboringBottom[v][i])); 
        }
        /// vDash = -0.5*dy*D*v
        cblas_dgemv(CblasRowMajor, CblasTrans,   (N+1)*(N+1), (N+1)*(N+1), -0.5*dx, derivativeMatrix_y, (N+1)*(N+1), variable[v],   1, 0, auxillaryVariable, 1);

        /// Adding the numeical Flux terms as necessary.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  0.5*dx, fluxMatrix_top,   (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1),  -0.5*dx, fluxMatrix_bottom,    (N+1)*(N+1), numericalFlux, 1, 1, auxillaryVariable, 1);

        /// Multiplying my Mass Inverse, this is the final step in getting the derivative.
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (N+1)*(N+1),(N+1)*(N+1), 4.0/(dx*dy), inverseMassMatrix,(N+1)*(N+1), auxillaryVariable,1,0,variable[vDash],1);

        delete[] numericalFlux;
        delete[] auxillaryVariable;
    }
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This function takes Mass Matrix as an input
 *
 * @Param m This is the massMatix array(actually a matrix, but implemented as a 1-d array).
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::setMassMatrix(double *m) {
    massMatrix = m;
    return ;
}

void DG_Element_2d::setInverseMassMatrix(double* im) {
    inverseMassMatrix = im;
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  Function to set the x-derivative matrix.
 *
 * @Param d The array of the x-derivative matrix which is given as an input.
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::setderivateMatrix_x(double *d) {
    derivativeMatrix_x = d;
    return ;
}


/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  Function to set the y-derivative matrix.
 *
 * @Param d The array of the y-derivative matrix which is given as an input.
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::setderivateMatrix_y(double *d) {
    derivativeMatrix_y = d;
    return ;
}

/***THE FOLLOWING 4 FUNCTIONS HAVE THE SIMILAR JOB TO SET THE FLUX MATRICES.****/

void DG_Element_2d::setTopFluxMatrix(double* f) {
    fluxMatrix_top = f;
    return ;
}
void DG_Element_2d::setRightFluxMatrix(double* f){
    fluxMatrix_right = f;
    return ;
}

void DG_Element_2d::setLeftFluxMatrix(double* f){
    fluxMatrix_left = f;
    return ;
}

void DG_Element_2d::setBottomFluxMatrix(double* f){
    fluxMatrix_bottom = f;
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @brief      This is used to extern the function `saxpy` to the class DG_Element_2d. $\mathbf{y} \mapsto a\mathbf{x} + \mathbf{y}$
 *
 * @param[in]  a     The coefficient `a` of `x`
 * @param[in]  x     The column vector `x`
 * @param[in]  y     The column vector `y`
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::axpy(double a, string x, string y) {
    cblas_daxpy((N+1)*(N+1), a, variable[x], 1, variable[y], 1);
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @brief      This is used to extern the function `sscal` to the class DG_Element_2d. $\mathbf{x} \mapsto a\mathbf{x}$
 *
 * @param[in]  a     The coefficient `a` of `x`
 * @param[in]  x     The column vector `x`
 * @param[in]  y     The column vector `y`
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::scal(double a, string x) {
    cblas_dscal((N+1)*(N+1), a, variable[x], 1);
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This is the function used to change the value of variable z to f(x, y).
 *
 * @Param x The first parameter of the function.
 * @Param y The second parameter of the function.
 * @Param functionf The function `f` which is required for the intended mapping.
 * @Param z The variable in which the value is to be stored
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::setFunctionsForVariables(string x, string y, function<double(double, double)> f, string z) {
    for(int i = 0 ; i < (N+1)*(N+1); i++)
        variable[z][i] = f(variable[x][i],variable[y][i]);
    return ;
}

/* ----------------------------------------------------------------------------*/
/**
 * @Synopsis  This is the function used to change the value of variable z to f(w, x, y).
 *
 * @Param w The first parameter of the function
 * @Param x The second parameter of the function.
 * @Param y The third parameter of the function.
 * @Param functionf The function `f` which is required for the intended mapping.
 * @Param z The variable in which the value is to be stored
 */
/* ----------------------------------------------------------------------------*/
void DG_Element_2d::setFunctionsForVariables(string w, string x, string y, function<double(double, double, double)> f, string z) {
    for(int i = 0 ; i < (N+1)*(N+1); i++)
        variable[z][i] = f(variable[w][i],variable[x][i],variable[y][i]);
    return ;
}


double DG_Element_2d::l2Norm(string v1, string v2) {
    double* diff = new double[(N+1)*(N+1)];
    cblas_dscal((N+1)*(N+1), 0.0, diff, 1);
    cblas_daxpy((N+1)*(N+1),  1.0, variable[v1], 1, diff, 1);
    cblas_daxpy((N+1)*(N+1), -1.0, variable[v2], 1, diff, 1);
    double norm2 =  (cblas_dnrm2((N+1)*(N+1),diff, 1));    
    delete[] diff;
    return norm2;
}
