//
// Created by fran on 08.07.17.
//

#include "NewtonMethod.h"
#include "BackTrackingLine.h"

AI::Optimizers::OptimizationResult
AI::Optimizers::NewtonMethod::Run(const arma::vec &x0, int niter, const double &tol) {



    arma::vec x = x0;
    arma::vec g = fgrad(x0);
    for (int i=0;i<niter;++i)
    {
        auto gf = fgrad(x);
        arma::mat h_n = this->fhess(x);

        arma::mat h(2,2);
        arma::inv(h,h_n);

        auto g1 = gf(0);
        auto g2 = gf(1);

        auto l = arma::sum(gf.t() * h * gf);
        if(l*l/2.0 < tol){
            break;
        }

        arma::mat dx = -h*gf;
        auto d1 = dx(0);
        auto d2 = dx(1);
        double t = LinearOptimizers::Backtrackline(0.1,0.7,eval,g,dx,x);

        x = x + t*dx;

    }

    OptimizationResult result = {x,eval(x)};
    return result;
}
