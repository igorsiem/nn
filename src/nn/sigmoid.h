/**
 * \file sigmoid.h
 * The `sigmoid` function
 */

#include <cmath>
#include <functional>

#ifndef _nn_sigmoid_h_included
#define _nn_sigmoid_h_included

/**
 * \brief All neural-net-related declarations
 */
namespace nn {

/**
 * \brief Apply the sigmoid function to an Eigen array-like object
 * 
 * This serves as one candidate for an activation function in the neural
 * net.
 * 
 * \typename arrayT The array-like type (anything that has a `Scalar` subtype
 * and a `unaryExpr` method)
 * 
 * \param arr The input array
 * 
 * \return The result of applying the sigmoid function to the input array
 */
template <typename arrayT>
auto sigmoid(const arrayT& arr)
{
    return arr.unaryExpr(
        [](typename arrayT::Scalar n)
        {
            return static_cast<typename arrayT::Scalar>((1/(1+exp(-n))));
        });
}   // end sigmoid function

/**
 * \brief Calculate the first derivative of the sigmoid function on an
 * array-like Eigen object *of sigmoid function values*
 * 
 * \typename arrayT The array-like type (anything that has a `Scalar` subtype
 * and a `unaryExpr` method)
 * 
 * \param arr The input array; this should be the result of the `sigmoid`
 * function
 * 
 * \return The first derivative of the sigmoid function value
 */
template <typename arrayT>
auto sigmoid_d(const arrayT& arr)
{
    return arr.unaryExpr(
        [](typename arrayT::Scalar n)
        {
            return static_cast<typename arrayT::Scalar>((n * (1-n)));
        });
}

}   // end nn namespace

#endif
