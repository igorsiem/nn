/**
 * \file learn.h
 * The core 'learning' implementation
 */

#include <functional>
#include <Eigen/Core>
#include "sigmoid.h"

#ifndef _nn_learn_h_included
#define _nn_learn_h_included

namespace nn {

/**
 * \brief Perform a single learning iteration on a set of samples, weights,
 * predictions and outcomes
 * 
 * \tparam SamplesT The type of the samples matrix-like object
 * 
 * \tparam WeightsT The tye of the samples vector-like object; this must have
 * a `Scalar` sub-type that matches that of `SamplesT`
 * 
 * \tparam PredictionT The type of the prediction vector; this must have a
 * `Scalar` sub-type that matches that of `SamplesT`
 * 
 * \tparam OutcomesT The type of the 'real outcomes' vector; this must have
 * a `Scalar` subtype that matcheds that of `SamplesT`
 * 
 * \tparam ActivatorT The type of the Activator class; this must have static
 * methods called `activate` and `activate_d`, being the activation function
 * and its derivative
 * 
 * \param X The samples matrix, with samples arranged in rows
 * 
 * \param W The weights column vector; this must have the same number of
 * elements as the columns in the `X` matrix
 * 
 * \param pred The predictions column vector; this must have the same number
 * of elements as the columns in the `X' matrix
 * 
 * \param outcomes The 'true outcomes' vector; this must have the same number
 * of elements as the columns in the `X` matrix
 */
template <
    typename SamplesT
    , typename WeightsT
    , typename PredictionT
    , typename OutcomesT
    , typename ActivatorT>
void learn(
        const SamplesT& X
        , WeightsT& W
        , PredictionT& pred
        , const OutcomesT& outcomes)
{
    pred = ActivatorT::activate(X * W);
    auto pred_error = outcomes - pred;
    auto pred_delta =
        (pred_error.array() * ActivatorT::activate_d(pred).array()).matrix();
    auto W_delta = X.transpose() * pred_delta;
    W += W_delta;
}

/**
 * \brief Perform a single learning iteration on a set of samples, weights,
 * predictions and outcomes
 * 
 * This overload takes an instance of the Activator class so that invoking
 * the function can be done using type-deduction, rather than enumerating
 * each type argument.
 * 
 * \tparam SamplesT The type of the samples matrix-like object
 * 
 * \tparam WeightsT The tye of the samples vector-like object; this must have
 * a `Scalar` sub-type that matches that of `SamplesT`
 * 
 * \tparam PredictionT The type of the prediction vector; this must have a
 * `Scalar` sub-type that matches that of `SamplesT`
 * 
 * \tparam OutcomesT The type of the 'real outcomes' vector; this must have
 * a `Scalar` subtype that matcheds that of `SamplesT`
 * 
 * \tparam ActivatorT The type of the Activator class; this must have static
 * methods called `activate` and `activate_d`, being the activation function
 * and its derivative
 * 
 * \param X The samples matrix, with samples arranged in rows
 * 
 * \param W The weights column vector; this must have the same number of
 * elements as the columns in the `X` matrix
 * 
 * \param pred The predictions column vector; this must have the same number
 * of elements as the columns in the `X' matrix
 * 
 * \param outcomes The 'true outcomes' vector; this must have the same number
 * of elements as the columns in the `X` matrix
 * 
 * \param activator An instance of the `ActivatorT` class; note that the
 * object is a dummy, and is not used, because the activation function and
 * its derivative are static methods of the class
 */
template <
    typename SamplesT
    , typename WeightsT
    , typename PredictionT
    , typename OutcomesT
    , typename ActivatorT>
void learn(
        const SamplesT& X
        , WeightsT& W
        , PredictionT& pred
        , const OutcomesT& outcomes
        , ActivatorT)
{
    learn<SamplesT, WeightsT, PredictionT, OutcomesT, ActivatorT>(
        X, W, pred, outcomes);
}

}   // end nn namespace

#endif
