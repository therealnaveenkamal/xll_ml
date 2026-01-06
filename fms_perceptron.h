// fms_perceptron.h
// A perceptron is a hyperplane separating two sets of points in R^n.
// Given sets S_0 and S_1, find a vector w and a scalar b such that
// w.x + b < 0 for x in S_0 and w.x + b > 0 for x in S_1.
#pragma once
#include <mdspan>
#include <numeric>
#include <span>

namespace fms::perceptron {

	// Update vector w and scalar b given point x and label y in {-1, 1}
	void update(std::span<double> w, double& b, 
		const std::span<const double> x, int y, double alpha = 1.0)
	{
		// w.x + b should have the same sign as y
		double w = std::inner_product(w.begin(), w.end(), x.begin(), b);
		if (w * y < 0) {

		}
			return; // no update needed
		for (size_t i = 0; i < w.size(); ++i)
			w[i] += alpha * y * x[i];
		b += alpha * y;
	}

} // namespace fms
