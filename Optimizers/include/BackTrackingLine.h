//
// Created by fran on 09.07.17.
//

#ifndef AI_BACKTRACKINGLINE_H
#define AI_BACKTRACKINGLINE_H

#include <armadillo>

namespace AI {
    namespace Optimizers {

        class LinearOptimizers {
        public:
            static double Backtrackline(const double &alpha, const double &beta,
                                                   std::function<double(const arma::vec &)> feval,
                                                   const arma::vec &grad, const arma::vec &dx,
                                                   const arma::vec &x0);

        };
    }
}



#endif //AI_BACKTRACKINGLINE_H
