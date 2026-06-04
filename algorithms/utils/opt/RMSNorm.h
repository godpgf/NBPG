#pragma once
#include <cassert>
#include "utils/Point/point.h"

namespace ant
{
    template <typename T>
    struct RMSNorm
    {
        static void forward(const Point<T> &x, Point<T> &y, float eps = 1e-6)
        {
            assert(x.params.dims == y.params.dims);
            auto s = 1.0f / x.rms();
            for (unsigned i = 0; i < x.params.dims; ++i)
            {
                y.data()[i] = x.data()[i] * s;
            }
        }

        static void forward(const PointRange<Point<T>> &x, PointRange<Point<T>> &y, float eps = 1e-6)
        {
#pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i)
            {
                const Point<T> xi = x[i];
                Point<T> yi = y[i];
                forward(xi, yi, eps);
            }
        }

        static void backward(const Point<T> &x, const Point<T> &y_grad, Point<T> &x_grad)
        {
            assert(x.params.dims == y_grad.params.dims);
            assert(x.params.dims == x_grad.params.dims);

            const unsigned d = x.params.dims;
            const float r = x.rms();
            const float inv_r = 1.0f / r;
            const float inv_r3_d = inv_r * inv_r * inv_r / d;
            
            float sum_x_grady = 0.0f;
            for (unsigned i = 0; i < d; ++i) {
                sum_x_grady += x.data()[i] * y_grad.data()[i];
            }
            
            for (unsigned i = 0; i < d; ++i) {
                x_grad.data()[i] = y_grad.data()[i] * inv_r - x.data()[i] * inv_r3_d * sum_x_grady;
            }
        }

        static void backward(const PointRange<Point<T>> &x, const PointRange<Point<T>> &y_grad, PointRange<Point<T>> &x_grad){
#pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i)
            {
                const Point<T> xi = x[i];
                const Point<T> y_gradi = y_grad[i];
                Point<T> x_gradi = x_grad[i];
                backward(xi, y_gradi, x_gradi);
            }
        }
    };
}