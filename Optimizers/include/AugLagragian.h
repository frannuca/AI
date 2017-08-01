//
// Created by fran on 08.07.17.
//

#ifndef AI_AUGLAGRAGIAN_H
#define AI_AUGLAGRAGIAN_H
#include "IOptimizer.h"
namespace AI {
    namespace Optimizers {

        class AugLagragian: public IOptimizer {
        public:
            using IOptimizer::IOptimizer;
            OptimizationResult Run(const arma::vec &x0, int niter, const double &tol) override ;
        };
    }
}

#endif //AI_AUGLAGRAGIAN_H
