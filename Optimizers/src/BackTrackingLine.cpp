//
// Created by fran on 09.07.17.
//

#include "BackTrackingLine.h"

double
AI::Optimizers::LinearOptimizers::Backtrackline(const double &alpha, const double &beta,
                                               std::function<double(const arma::vec &)> feval,
                                               const arma::vec &grad, const arma::vec &dx,
                                               const arma::vec &x0) {
    double t = 1;
    auto direction = arma::sum(grad.t() * dx);
    auto f0 = feval(x0);

    auto extrapol =[&f0,&alpha,&beta,&direction](double s){return f0+s*alpha*beta*direction;};
    double y = feval(x0+t*dx);
    double xtr = extrapol(t);
    while( y>xtr){

        t*=beta;
        y = feval(x0+t*dx);
        xtr = extrapol(t);
    }

    return t;
}
