//
// Created by fran on 08.07.17.
//

#ifndef AI_QUASINEWTONRAPHSOM_H
#define AI_QUASINEWTONRAPHSOM_H

#include "IOptimizer.h"
namespace AI {
    namespace Optimizers {

        class BFGS : public IOptimizer {
        public:
            using IOptimizer::IOptimizer;
            OptimizationResult Run(const arma::vec &x0, int niter, const double &tol) override ;
        private:
            arma::mat finiteDifferenceHessian(const arma::vec& x0);
            arma::mat updateHessian (const arma::mat &xk , const arma::mat &xk_1 , const arma::mat &h);
            arma::mat eyeScaledHessian(const arma::vec& x0);

            static constexpr double dt=1e-3;
        };
    }
}

#endif //AI_QUASINEWTONRAPHSOM_H
