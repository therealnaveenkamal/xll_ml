// fms_perceptron.h
// A perceptron is a hyperplane separating two sets of points in R^n.
// Given sets S_0 and S_1, find a vector w and a scalar b such that
// w.x < 0 for x in S_0 and w.x > 0 for x in S_1.
#pragma once

#include <vector>
#include "fms_error.h"
#include "fms_linalg.h"
// https://cppreference.net/cpp/numeric/linalg.html
// #include <linalg>

namespace fms::perceptron {

    // Update weights w given point x and label y in {true, false}
    // Return if the update trained the data.
    template<class T = double>
    bool update(std::span<T> w, const std::span<T> x, bool y, T alpha = 1.0)
    {
        ensure (w.size() == x.size() || !"weight and point must have the same size");
		// changed int y to bool y so we don't need this check
        //ensure (y == 0 or y == 1 || !"label must be 0 or 1");

		bool y_ = fms::linalg::dot(w, x) > 0;
        // Check if misclassified
        if (y_ != y) {
            // Update: w = alpha dy x + w   ,
            fms::linalg::axpy(alpha * (y - y_), x, w, w);
        }

        return y == y_;
    }

    // loop updates until trained
    // return the number of iterations
    template<class T = double>
    std::size_t train(std::span<T> w, std::span<T> x, bool y, T alpha = 1.0, std::size_t n = 100)
    {
        size_t n0 = n;

        while (n && false == fms::perceptron::update(w, x, y, alpha)) {
            --n; // limit loops
        }

        return n0 - n;
    }

    template<class T = double>
    class neuron {
        std::vector<T> w;
    public:
        neuron(size_t n = 0)
            : w(n)
		{ }
        // RAII
        neuron(std::span<T> w)
            : w(w.begin(), w.end())
        { }
        neuron(const neuron&) = default;
        neuron& operator=(const neuron&) = default;
        neuron(neuron&&) = default;
        neuron& operator=(neuron&&) = default;
        ~neuron() = default;

        std::span<T> weights()
        {
            return std::span<T>(w);
        }

        bool update(const std::span<const T>& x, int y, double alpha = 1.0)
        {
            return perceptron::update(std::span(w), x, y, alpha);
		}
        bool train(std::span<T>& x, bool y, T alpha = 1.0, std::size_t n = 100)
        {
			return perceptron::train(w, x, y, alpha, n);
        }
		// Train collection of (x,y) pairs.
        using pair = std::pair<std::span<T>, int>;
        void train(std::span<pair> xy, double alpha = 1.0)
        {
            for (auto [x, y] : xy) {
                train(x, y, alpha);
            }
        }
    };
 
} // namespace fms::perceptron