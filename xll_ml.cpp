// xll_ml.cpp
#include "fms_perceptron.h"
#undef ensure
#include "xll24/include/xll.h"

#define CATEGORY L"ML"

using namespace xll;
using namespace fms::perceptron;

AddIn xai_perceptron_update(
	Function(XLL_FP, L"xll_perceptron_update", L"PERCEPTRON.UPDATE")
	.Arguments({
		Arg(XLL_FP, L"w", L"is an array of weights."),
		Arg(XLL_FP, L"x", L"is an array representing the input vector."),
		Arg(XLL_BOOL, L"y", L"is the label"),
		Arg(XLL_DOUBLE, L"alpha", L"is the learning rate. (default=1.0)", 1.0)
		})
	.FunctionHelp(L"Update perceptron weights input vector and label.")
	.Category(CATEGORY)
);
_FP12* WINAPI xll_perceptron_update(_FP12* pw, _FP12*  px, BOOL y, double alpha)
{
#pragma XLLEXPORT
	try {
		alpha = alpha ? alpha : 1;
		auto w = span(*pw);
		auto x = span(*px);

		update(w, x, y, alpha);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCDNAME__ ": unknown exception");
	}
	return pw;
}

AddIn xai_perceptron_train(
	Function(XLL_FP, L"xll_perceptron_train", L"PERCEPTRON.TRAIN")
	.Arguments({
		Arg(XLL_FP, L"w", L"is an array of weights."),
		Arg(XLL_FP, L"x", L"is an array representing the input vector."),
		Arg(XLL_BOOL, L"y", L"is the label."),
		Arg(XLL_DOUBLE, L"alpha", L"is the learning rate. (default=1.0)", 1.0),
		Arg(XLL_UINT, L"n", L"is the maximum number of iterations. (default=100)", 100),
		})
	.Category(CATEGORY)
	.FunctionHelp(L"Train perceptron weights on single input vector and label.")

);
_FP12* WINAPI xll_perceptron_train(_FP12* pw, _FP12* px, BOOL y, double alpha, UINT n)
{
#pragma XLLEXPORT
	try {
		alpha = alpha ? alpha : 1;
		n = n ? n : 100;
		auto w = span(*pw);
		auto x = span(*px);

		train(w, x, y, alpha);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCDNAME__ ": unknown exception");
	}
	
	return pw;
}

AddIn xai_neuron_(
	Function(XLL_HANDLEX, L"xll_neuron_", L"\\NEURON")
	.Arguments({
		Arg(XLL_FP, L"w", L"is an array of initial weights."),
		})
	.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Return handle to a neuron with given weights.")
);
HANDLEX WINAPI xll_neuron_(_FP12* pw)
{	
#pragma XLLEXPORT
	HANDLEX h = INVALID_HANDLEX;

	try {
		handle<neuron<>> h_(new neuron<>(span(*pw)));
		ensure(h_);

		h = h_.get();
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCDNAME__ ": unknown exception");
	}

	return h;
}

AddIn xai_neuron(
	Function(XLL_FP, L"xll_neuron", L"NEURON")
	.Arguments({
		Arg(XLL_HANDLEX, L"h", L"is a handle returned by \\NEURON."),
		})
		.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Return array of weights.")
);
_FP12* WINAPI xll_neuron(HANDLEX h)
{
#pragma XLLEXPORT
	static FPX w;

	try {
		handle<neuron<>> h_(h);
		ensure(h_);

		std::span<double> weights = h_->weights();
		FPX w_((int)weights.size(), 1, weights.data());
		w.swap(w_);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCDNAME__ ": unknown exception");
	}

	return w.get();
}

AddIn xai_neuron_update(
	Function(XLL_HANDLEX, L"xll_neuron", L"NEURON.UPDATE")
	.Arguments({
		Arg(XLL_HANDLEX, L"h", L"is a handle to a neuron."),
		Arg(XLL_FP, L"x", L"is an array representing the input vector."),
		Arg(XLL_BOOL, L"y", L"is the label."),
		Arg(XLL_DOUBLE, L"alpha", L"is the learning rate. (defaul 1.0)", 1.0),
		})
		.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Return handle of updated neuron.")
);
HANDLEX WINAPI xll_neuron_update(HANDLEX h, _FP12* px, BOOL y, double alpha)
{
#pragma XLLEXPORT
	try {
		handle<neuron<>> h_(h);
		ensure(h_);

		alpha = alpha ? alpha : 1;
		auto x = span(*px);

		//ensure (h_->update(x, y, alpha);
	}
	catch (const std::exception& ex) {
		h = INVALID_HANDLEX;
		XLL_ERROR(ex.what());
	}
	catch (...) {
		h = INVALID_HANDLEX;
		XLL_ERROR(__FUNCDNAME__ ": unknown exception");
	}

	return h;
}
/*
AddIn xai_neuron_train(
	Function(XLL_FP, L"xll_neuron", L"NEURON.TRAIN")
	.Arguments({
		Arg(XLL_HANDLEX, L"h", L"is a handle to a neuron."),
		Arg(XLL_FP, L"x", L"is an array representing the input vector."),
		Arg(XLL_FP, L"y", L"is the label."),
		Arg(XLL_DOUBLE, L"alpha", L"is the learning rate. (default=1.0)", 1.0),
		})
		.Uncalced()
	.Category(CATEGORY)
	.FunctionHelp(L"Create a neuron with given weights.")
);
_FP12* WINAPI xll_neuron_train(HANDLEX h, _FP12* px, _FP12* py, double alpha)
{
#pragma XLLEXPORT
	static FPX w;

	try {
		ensure(columns(*px) == size(*py));

		alpha = alpha ? alpha : 1.0;

		handle<neuron<>> h_(h);
		ensure(h_);

		std::vector<std::pair<std::span<double>, int>> xy(size(*py));
		for (int i = 0; i < size(*py); ++i) {
			xy[i] = { std::span(xll::row(*px, i)), py->array[i] != 0};
		}
		auto data = std::span(xy.begin(), xy.end());
		h_->train(data, alpha);

		FPX w_((int)h_->w.size(), 1, h_->w.data());
		w.swap(w_);
	}
	catch (const std::exception& ex) {
		XLL_ERROR(ex.what());
	}
	catch (...) {
		XLL_ERROR(__FUNCDNAME__ ": unknown exception");
	}

	return w.get();
}
*/