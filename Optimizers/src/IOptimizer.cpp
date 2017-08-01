//
// Created by fran on 08.07.17.
//

#include "IOptimizer.h"

AI::Optimizers::IOptimizer::IOptimizer(const AI::Optimizers::TFunction &f):eval(f) {

}

const AI::Optimizers::IOptimizer* AI::Optimizers::IOptimizer::withEqConstraint(const AI::Optimizers::TFunction &g) const {
    this->feq=g;
    return this;
}

const AI::Optimizers::IOptimizer*  AI::Optimizers::IOptimizer::withIneqConstraint(const AI::Optimizers::TFunction &g) const{
    this->fineq=g;
    return this;
}

const AI::Optimizers::IOptimizer*  AI::Optimizers::IOptimizer::withGradient(const AI::Optimizers::TVFunction &g) const {
    this->fgrad=g;
    return this;
}

const AI::Optimizers::IOptimizer*  AI::Optimizers::IOptimizer::withHessian(const AI::Optimizers::TMFunction &g) const {
    this->fhess=g;
    return this;
}
