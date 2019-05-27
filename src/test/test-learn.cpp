/**
 * \brief test-learn.cpp
 * Test the core `learn` function
 */

#include <iostream>
#include <chrono>
#include <catch2/catch.hpp>
#include <Eigen/Core>
#include <fmt/format.h>
#include <nn/nn.h>

using namespace fmt::literals;

// Basic test of the core `nn::learn` method. This test (and its expected
// values) is taken from "A Neural Network in 10 lines of C++ Code"
// (https://cognitivedemons.wordpress.com/2017/07/06/a-neural-network-in-10-lines-of-c-code/)
TEST_CASE("learn", "[unit]")
{
    using ScalarT = float;
    Eigen::Matrix<ScalarT, 4, 4> X;
    Eigen::Matrix<ScalarT, 4, 1> y, W, pred, pred_exp;

    X <<
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2,
        6.2, 3.4, 5.4, 2.3,
        5.9, 3.0, 5.1, 1.8;

    y << 0.0, 0.0, 1.0, 1.0;

    W << 0.5, 0.5, 0.5, 0.5;

    for (int i = 0; i < 50; i++)
    {

        nn::sigmoid_activator activator;
        nn::learn(X, W, pred, y, activator);

    }   //end learning loop

    pred_exp << 0.0511965, 0.0696981, 0.931842, 0.899579;

    for (int i = 0; i < pred.size(); i++)
        REQUIRE(pred(i) == Approx(pred_exp(i)));

}   // end learn unit test
