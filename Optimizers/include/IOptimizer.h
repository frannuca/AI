//
// Created by fran on 01.07.17.
//

#ifndef AI_IOPTIMIZER_H
#define AI_IOPTIMIZER_H

#include "armadillo"
#include <functional>
#include <memory>

namespace AI {
    namespace Optimizers {

        using TFunction = std::function<double (const arma::vec &)>;
        using TVFunction = std::function<arma::vec (const arma::vec &)>;
        using TMFunction = std::function<arma::mat (const arma::vec &)>;


        struct OptimizationResult {
            arma::vec x;
            double fx;
        };

        class IOptimizer {
        public:
            virtual ~IOptimizer () {}

            IOptimizer () = delete;

            IOptimizer (const TFunction &f);

            const AI::Optimizers::IOptimizer *withEqConstraint (const TFunction &g) const;

            const AI::Optimizers::IOptimizer *withIneqConstraint (const TFunction &g) const;

            const AI::Optimizers::IOptimizer *withGradient (const TVFunction &g) const;

            const AI::Optimizers::IOptimizer *withHessian (const TMFunction &g) const;


            virtual OptimizationResult Run (const arma::vec &x0 , int niter , const double &tol) = 0;

        protected:
            TFunction eval;
            mutable TFunction feq;
            mutable TFunction fineq;
            mutable TVFunction fgrad;
            mutable TMFunction fhess;

        };

    }
}


#endif //AI_GRAPH_H
