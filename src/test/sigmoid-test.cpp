/**
 * \file sigmoid-test.cpp
 * Tests for the sigmoid function
 */

#include <iostream>
#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <nn/nn.h>

// Test sigmoid and sigmoid_d activation functions
TEST_CASE("sigmoid", "[unit]")
{
    Eigen::Matrix<double, 1, Eigen::Dynamic>
        vec(1, 21)          // input data
        , sig               // result of sigmoid function
        , six_exp(1, 21)    // expected result of sigmoid function
        , der               // result of sigmoid derivative function
        , der_exp(1, 21);   // expected result of derivative
    vec << -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5
        , 2, 2.5, 3, 3.5, 4, 4.5, 5;

///    sig = nn::sigmoid(vec);
    sig = nn::sigmoid_activator::activate(vec);

    six_exp << 0.00669285, 0.0109869, 0.0179862, 0.0293122, 0.0474259,
        0.0758582, 0.119203, 0.182426, 0.268941, 0.377541, 0.5, 0.622459,
        0.731059, 0.817574, 0.880797, 0.924142, 0.952574, 0.970688, 0.982014,
        0.989013, 0.993307;

    // std::cout << std::endl << "[ ";
    for (int i = 0; i < sig.size(); i++)
    {
        // std::cout << sig(i) << ", ";
        REQUIRE(sig(i) == Approx(six_exp(i)));
    }
    // std::cout << "]" << std::endl;

    // Now the derivative - note that this uses the sigmoid value
///    der = nn::sigmoid_d(sig);
    der = nn::sigmoid_activator::activate_d(sig);

    der_exp << 0.00664806, 0.0108662, 0.0176627, 0.028453, 0.0451767,
        0.0701037, 0.104994, 0.149146, 0.196612, 0.235004, 0.25, 0.235004,
        0.196612, 0.149146, 0.104994, 0.0701037, 0.0451767, 0.028453,
        0.0176627, 0.0108662, 0.00664806;

    // std::cout << std::endl << "[ ";
    for (int i = 0; i < vec.size(); i++)
    {
        // std::cout << der(i) << ", ";
        REQUIRE(der(i) == Approx(der_exp(i)));
    }
    // std::cout << "]" << std::endl;

}   // end sigmoid tests
