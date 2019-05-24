/**
 * \file learn.h
 * The core 'learning' implementation
 */

#include <functional>
#include "sigmoid.h"

#ifndef _nn_learn_h_included
#define _nn_learn_h_included

namespace nn {

///template <typename inT, typename outT>
///using act_fn = std::function<outT(inT)>;

template <typename samplesT, typename weightsT, typename predT, typename yT>
void learn(
        const samplesT& X
        , weightsT& W
        , predT& pred
        , const yT& y)
{
    pred = sigmoid(X * W);
    auto pred_error = y - pred;
    auto pred_delta = (pred_error.array() * sigmoid_d(pred).array()).matrix();
    auto W_delta = X.transpose() * pred_delta;
    W += W_delta;
}

}   // end nn namespace


#endif
