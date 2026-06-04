#pragma once
#include <algorithm>
#include "utils/Point/point.h"

namespace ant
{
    template <typename T>
    struct ReLU
    {
        // ========== 前馈：单个样本 ==========
        static void forward(const Point<T> x, Point<T> y)
        {
            assert(x.params.dims == y.params.dims);
            const unsigned d = x.params.dims;
            
            for (unsigned i = 0; i < d; ++i) {
                y.data()[i] = std::max(0.0f, x.data()[i]);
            }
        }

        // ========== 前馈：批量样本 ==========
        static void forward(const PointRange<Point<T>> &x, PointRange<Point<T>> &y)
        {
            assert(x.size() == y.size());
            assert(x.dimension() == y.dimension());
            
            #pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i)
            {
                forward(x[i], y[i]);
            }
        }

        // ========== 反向传播：单个样本 ==========
        // 需要前向时的输入x来判断哪些位置被置0
        static void backward(const Point<T> x, const Point<T> y_grad, Point<T> x_grad)
        {
            assert(x.params.dims == y_grad.params.dims);
            assert(x.params.dims == x_grad.params.dims);
            const unsigned d = x.params.dims;
            
            for (unsigned i = 0; i < d; ++i) {
                // ReLU导数：x>0时为1，否则为0
                x_grad.data()[i] = (x.data()[i] > 0.0f) ? y_grad.data()[i] : 0.0f;
            }
        }

        // ========== 反向传播：批量样本 ==========
        static void backward(const PointRange<Point<T>> &x, 
                            const PointRange<Point<T>> &y_grad, 
                            PointRange<Point<T>> &x_grad)
        {
            assert(x.size() == y_grad.size());
            assert(x.size() == x_grad.size());
            
            #pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i)
            {
                backward(x[i], y_grad[i], x_grad[i]);
            }
        }
    };
}