//
// Created by fran on 13.07.17.
//

#ifndef AI_GRADIENTDESCENT_H
#define AI_GRADIENTDESCENT_H


#include "armadillo"
#include <functional>
#include "IOptimizer.h"

namespace AI {
    namespace Optimizers {

        class GradientDescent: public IOptimizer {
        public:
            GradientDescent()= delete;

            GradientDescent(const TFunction &f,const double& lr=0.0);
            OptimizationResult Run(const arma::vec &x0, int niter, const double &tol) override ;

        public:
            double lr;
        };

    }
}


#endif //AI_GRADIENTDESCENT_H
