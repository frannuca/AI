//
// Created by fran on 08.07.17.
//

#include "Optimizers/include/BFGS.h"
#include "BackTrackingLine.h"
#include <boost/range/irange.hpp>
#include <ios>
#include <string>

AI::Optimizers::OptimizationResult AI::Optimizers::BFGS::Run (const arma::vec &x0 , int niter , const double &tol) {


    arma::mat H0= finiteDifferenceHessian(x0);
    arma::mat h_n(H0.n_rows,H0.n_cols);
    H0.print();
    arma::inv(h_n,H0);


    arma::vec x = x0;
    arma::vec xold = x0;
    arma::vec g = fgrad(x0);
    for (int i=0;i<niter;++i)
    {
        std::cout<<".----------------,"<<std::endl;
        h_n.print();
        auto gf = fgrad(x);




        auto g1 = gf(0);
        auto g2 = gf(1);

        auto l =std::sqrt( arma::sum(gf.t() * gf));
        if(l < tol){
            break;
        }

        arma::mat dx = -h_n*gf;
        auto d1 = dx(0);
        auto d2 = dx(1);
        double t = LinearOptimizers::Backtrackline(0.01,0.7,eval,g,dx,x);
        xold=x;
        x = x + t*dx;
        h_n = updateHessian(xold,x,h_n);
        h_n.print();

    }

    OptimizationResult result = {x,eval(x)};
    return result;
}


arma::mat AI::Optimizers::BFGS::finiteDifferenceHessian (const arma::vec &x0) {
    int n = x0.size();
    arma::mat B(n,n);

    auto fdeval=[&](int i,const arma::vec& xp,std::function<double(arma::vec)> funceval ){
        arma::vec x=xp;
        x(i)+=dt;
        double z = (funceval(xp)-funceval(x))/dt;
        return z;
    };

    for(int i=0;i<n;++i){
        auto feval_i = [&](const arma::vec& xp){ return fdeval(i,xp,this->eval);};
        auto tt = feval_i(x0);

        for(int j=i;j<n;++j){
            auto a=fdeval(j,x0,feval_i);
            B(i,j) = a;
            B(j,i)=B(i,j);
        }
    }

    return B;

}

arma::mat AI::Optimizers::BFGS::eyeScaledHessian (const arma::vec &x0) {
    return arma::mat();
}

arma::mat AI::Optimizers::BFGS::updateHessian (const arma::mat &xk , const arma::mat &xk_1 , const arma::mat &hk) {
    xk_1.t().print();
    std::cout<<"-----------"<<std::endl;
    xk.t().print();
    arma::mat yk = this->fgrad(xk_1)-this->fgrad(xk);
    arma::mat sk = xk_1-xk;
    arma::mat  I = arma::eye(xk.size(),xk.size());
    arma::mat rkm = yk.t()*sk;
    double rk = (std::abs(rkm(0,0))>1e5  || std::abs(rkm(0,0))<1e-5)  ?0.0:1.0/rkm(0,0);
    if(rk==0){
     return hk;
    }
    else{
        arma::mat a = I-rk*(sk*yk.t());
        arma::mat b = I-rk*(yk*sk.t());
        arma::mat c = rk*(sk*sk.t());
        arma::mat hk_1 = a*hk*b+c;
        return hk_1;

    }

}
