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
 * \brief Activation framework using the sigmoid function and its first
 * derivative
 * 
 * The sigmoid function is one type of activator that may be used with neural
 * nets.
 */
class sigmoid_activator
{

    public:

    /**
     * \brief Apply the sigmoid function to an Eigen array-like object
     * 
     * This serves as one candidate for an activation function in the neural
     * net.
     * 
     * \tparam ArrayT The array-like type (anything that has a `Scalar`
     * subtype and a `unaryExpr` method)
     * 
     * \param arr The input array
     * 
     * \return The result of applying the sigmoid function to the input array
     */
    template <typename ArrayT>
    static auto activate(const ArrayT& arr)
    {
        return arr.unaryExpr(
            [](typename ArrayT::Scalar n)
            {
                return static_cast<typename ArrayT::Scalar>((1/(1+exp(-n))));
            });    
    }

    /**
     * \brief Calculate the first derivative of the sigmoid function on an
     * array-like Eigen object *of sigmoid function values*
     * 
     * \typename ArrayT The array-like type (anything that has a `Scalar`
     * subtype and a `unaryExpr` method)
     * 
     * \param arr The input array; this should be the result of the `sigmoid`
     * function
     * 
     * \return The first derivative of the sigmoid function value
     */
    template <typename ArrayT>
    static auto activate_d(const ArrayT& arr)
    {
        return arr.unaryExpr(
            [](typename ArrayT::Scalar n)
            {
                return static_cast<typename ArrayT::Scalar>((n * (1-n)));
            });
    }

};  // end sigmoid_activator class

}   // end nn namespace

#endif
