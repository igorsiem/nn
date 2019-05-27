/**
 * \file neural_net.h
 * The core 'learning' implementation
 */

#include <functional>
#include <Eigen/Core>
#include "sigmoid.h"

#ifndef _nn_learn_h_included
#define _nn_learn_h_included

namespace nn {

/**
 * \brief A class encapsulating the functionality of a neural net
 * 
 * This class has static methods for performing the learning functions on
 * datasets.
 * 
 * \tparam ActivatorT The type of the Activator class; this must have static
 * methods called `activate` and `activate_d`, being the activation function
 * and its derivative
 * 
 */
template <typename ActivatorT>
class neural_net
{

    public:

    /**
     * \brief A sub-type aliasing the Activator type
     */
    using activator_t = ActivatorT;

    /**
     * \brief Perform a single learning iteration on a set of samples,
     * weights, predictions and outcomes
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
        , typename OutcomesT>
    static void learn(
            const SamplesT& X
            , WeightsT& W
            , PredictionT& pred
            , const OutcomesT& outcomes)
    {
        pred = activator_t::activate(X * W);
        auto pred_error = outcomes - pred;
        auto pred_delta =
            (pred_error.array()
            * activator_t::activate_d(pred).array()).matrix();
        auto W_delta = X.transpose() * pred_delta;
        W += W_delta;
    }

    /**
     * \brief Perform a given number of iterations of the learning function
     * on a set of samples, weights, predictions and outcomes
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
     * \param n The number of times to iterate
     */
    template <
        typename SamplesT
        , typename WeightsT
        , typename PredictionT
        , typename OutcomesT>
    static void learn(
            const SamplesT& X
            , WeightsT& W
            , PredictionT& pred
            , const OutcomesT& outcomes
            , unsigned int n)
    {
        for (unsigned int i = 0; i < n; i++)
            learn(X, W, pred, outcomes);
    }

};  // end neural_net class

}   // end nn namespace

#endif
