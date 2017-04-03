#ifndef LOBATTOINTEGRATION_H
#define LOBATTOINTEGRATION_H

#include <functional>

using namespace std;

double lobattoIntegration(double start, double end, unsigned N, function<double(double)> f);

#endif
