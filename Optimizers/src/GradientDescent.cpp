//
// Created by fran on 13.07.17.
//

#include <Optimizers/include/BackTrackingLine.h>
#include "GradientDescent.h"
#include "BackTrackingLine.h"

AI::Optimizers::GradientDescent::GradientDescent(const AI::Optimizers::TFunction &f, const double &lr_):IOptimizer(f),lr(lr_) {

}

AI::Optimizers::OptimizationResult
AI::Optimizers::GradientDescent::Run(const arma::vec &x0, int niter, const double &tol) {
    arma::vec x = x0;
    double fold = eval(x);
    double fnew = fold;
    auto x1 = x0(0);
    auto x2 = x0(1);

    for (int i=0;i<niter;++i)
    {
        arma::mat gf =lr>0.0? fgrad(x)*lr:fgrad(x);

        arma::mat gnorm2 = gf.t() * gf;

        double gnorm = std::sqrt(gnorm2(0,0));
        arma::mat gfnorm = gf/gnorm;

        auto g1 = gfnorm(0);
        auto g2 = gfnorm(1);


        arma::mat dx;
        dx = -gf;
        auto d1 = dx(0);
        auto d2 = dx(1);
        double t = LinearOptimizers::Backtrackline(0.1,0.7,eval,gf,dx,x);

        x = x + t*dx;
        fold=fnew;
        fnew = eval(x);
        if(std::abs(fold-fnew)<tol)
            break;

        x1 = x(0);
        x2 = x(1);

    }

    OptimizationResult result = {x,eval(x)};
    return result;
}
