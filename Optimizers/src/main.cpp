//
// Created by fran on 08.07.17.
//
#include <stdio.h>
#include <iostream>
#include <armadillo>
#include "OptStatistics.h"
#include <memory>
#include "IOptimizer.h"
#include "NewtonMethod.h"
#include "GradientDescent.h"
#include "BFGS.h"
namespace{
    const double xa = 121.155;
    const double xb = 125.001;
}

double feval1(const arma::vec& x){

    auto a = x(0)-xa;
    auto b = x(1)-xb;
    return a*a+b*b;
}

arma::vec fgrad_eval1(const arma::vec& x){


    auto a = x(0)-xa;
    auto b = x(1)-xb;
    return arma::vec({2*a,2*b});
}

arma::mat fhessian_eval1(const arma::vec& x){

    arma::mat h(2,2);
    h(0,0) = 2.0;
    h(0,1)=0.0;
    h(1,0)=0.0;
    h(1,1)=2.0;

    return std::move(h);
}
int main(int argv,char *args[])
{

    {
        std::cout<<"Running BFGS method on Quadratic form"<<std::endl;

        AI::Optimizers::BFGS method(feval1);
        method
                .withGradient(fgrad_eval1);

        arma::vec x0({0.0,0.0});
        auto r = method.Run(x0,10000,0.00001);
        std::cout<<r.x(0)<<","<<r.x(1)<<std::endl;

    }
    /*
   {
       std::cout<<"Running Newton method on Quadratic form"<<std::endl;

       AI::Optimizers::NewtonMethod method(feval1);
       method
               .withGradient(fgrad_eval1)
               ->withHessian(fhessian_eval1);

       arma::vec x0({0.0,0.0});
       auto r = method.Run(x0,10,0.0001);
       std::cout<<r.x(0)<<","<<r.x(1)<<std::endl;

   }

   {
       std::cout<<"Running Gradient Descend method on Quadratic form"<<std::endl;

       AI::Optimizers::GradientDescent method(feval1);
       method.withGradient(fgrad_eval1);
       std::cout<<method.lr<<std::endl;

       arma::vec x0({0.000631,-0.000003});
       std::cout<<method.lr<<std::endl;
       auto r = method.Run(x0,1000,0.00001);
       std::cout<<r.x(0)<<","<<r.x(1)<<std::endl;

   }
*/

    std::shared_ptr<AI::Optimizers::OptStatistics> stats(new AI::Optimizers::OptStatistics());

    arma::vec p({1.3,1.23,11.1});
    stats->addpoint(p,0.00145);
    auto stats2 = stats->getStats();
}