#include <functional>
#include "../../includes/Utilities/LobattoIntegration.h"
#include "../../includes/Utilities/LobattoNodes.h"
#include "../../includes/Utilities/LobattoWeights.h"
using namespace std;


double lobattoIntegration(double start, double end, unsigned N, function<double(double)> f) {
    N++;
    double *Nodes,*Weights,*Values;
    Nodes   =   new double[N];
    Weights =   new double[N];
    Values  =   new double[N];
    unsigned i;
    double integral=0.0;

    if(start>=end)
	{
		printf("ERROR: Please look into it there is some error in the given inputs for lobattoIntegration()\n");
		return 0.0;
	}

	lobattoNodes(Nodes,N);
    lobattoWeights(Weights,N);

    for (i=0;i<N;i++)
	   Nodes[i] = 0.5*(start+end) +  (0.5*(start-end))*Nodes[i];///Made a shift from the computational space to the physical space.


	for(i=0;i<N;i++)
		Values[i] =   f(Nodes[i]);

	for(i=0;i<N;i++)
		integral += (Values[i]*Weights[i]);

    delete[]    Nodes;
    delete[]    Weights;
    delete[]    Values;

	return (0.5*(end-start)*(integral));
}
