#pragma once
#include <cassert>
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include "utils/Point/sparse_point.h"
#include "optimizer.h"

namespace ant
{
    template <typename T>
    struct SparseLinear
    {
        SparseLinear(Optimizer<T>* optimizer, uint32_t in_features, uint32_t sparse_in_features, uint32_t out_features, bool use_bias = true)
            : use_bias(use_bias),
              mat(out_features, typename SparsePoint<T>::Parameters(in_features, sparse_in_features)),
              mat_grad(out_features, in_features),
              bias(use_bias ? 1 : 0, out_features),
              bias_grad(use_bias ? 1 : 0, out_features)
        {
            mat.kaiming_uniform_init(mat.dimension());
            bias.zeros_();
            if(optimizer != nullptr){
                optimizer->register_opt(mat, mat_grad);
                if(use_bias){
                    optimizer->register_opt(bias, bias_grad);
                }
            }

        }
        void forward(const PointRange<Point<T>> &x, PointRange<Point<T>> &y)
        {

#pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i)
            {
                if (use_bias)
                {
                    y[i].copy_(bias[0]);
                }
                else
                {
                    y[i].zeros_();
                }

                for (auto j = 0; j < y.dimension(); ++j)
                {
                    y[i].data()[j] += mat[j].dot(x[i]);
                }
            }
        }

        void backward(const PointRange<Point<T>> &x, const PointRange<Point<T>> &y_gard, PointRange<Point<T>> &x_grad)
        {
#pragma omp parallel for
            for (auto i = 0; i < y_gard.size(); ++i)
            {
                if (use_bias)
                {
                    bias_grad[0] += y_gard[i];
                }
                for (auto j = 0; j < y_gard.dimension(); ++j)
                {
                    mat_grad[j].add_feat(x[i], y_gard[i].data()[j]);
                }
            }

            x_grad.zeros_();
#pragma omp parallel for
            for (auto i = 0; i < y_gard.size(); ++i)
            {
                for (auto j = 0; j < y_gard.dimension(); ++j)
                {
                    x_grad[i].add_feat(mat[j].data(), mat[j].params.sparse_dims, y_gard[i].data()[j]);
                }
            }
        }


    protected:
        bool use_bias;
        PointRange<SparsePoint<T>> mat;
        PointRange<Point<T>> bias;
        PointRange<Point<T>> mat_grad;
        PointRange<Point<T>> bias_grad;
    };

}