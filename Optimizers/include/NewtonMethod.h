//
// Created by fran on 08.07.17.
//

#ifndef AI_NEWTONMETHOD_H
#define AI_NEWTONMETHOD_H
#include "armadillo"
#include <functional>
#include "IOptimizer.h"

namespace AI {
    namespace Optimizers {

        class NewtonMethod: public IOptimizer {
        public:
            using IOptimizer::IOptimizer;
            OptimizationResult Run(const arma::vec &x0, int niter, const double &tol) override ;
        };

    }
}
#endif //AI_NEWTONMETHOD_H
