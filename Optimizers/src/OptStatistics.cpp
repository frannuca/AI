//
// Created by fran on 08.07.17.
//

#include "OptStatistics.h"
namespace AI{
    namespace Optimizers{

        OptStatistics& OptStatistics::addpoint(arma::vec x, double feval) {
            points.push_back(std::make_tuple(x,feval));
            return *this;
        }

        const std::vector<std::tuple<arma::vec, double>> OptStatistics::getStats() {
            return this->points;
        }
    }
}