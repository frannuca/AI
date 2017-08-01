//
// Created by fran on 08.07.17.
//

#ifndef AI_OPTSTATISTICS_H
#define AI_OPTSTATISTICS_H
#include "armadillo"
#include <tuple>
#include <vector>

namespace AI {
    namespace Optimizers {

        class OptStatistics {
        public:
        OptStatistics& addpoint(arma::vec x, double feval);
            const std::vector<std::tuple<arma::vec,double>> getStats();

        private:
            std::vector<std::tuple<arma::vec,double>> points;
        };

    }
}


#endif //AI_OPTSTATISTICS_H
