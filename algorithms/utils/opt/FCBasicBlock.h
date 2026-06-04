#pragma once
#include "linear.h"
#include "RMSNorm.h"
#include "relu.h"

namespace ant
{
    template <typename T>
    struct FCBasicBlock
    {
        FCBasicBlock(Optimizer<T>* optimizer, uint32_t max_batch_size, uint32_t in_features, uint32_t out_features, uint32_t stride = 1, bool pre_norm=true)
            : fc1(optimizer, in_features, out_features, false),
              fc2(optimizer, out_features, out_features, false),
              shortcut(stride > 1 ? optimizer : nullptr, stride > 1 ? in_features : 0, stride > 1 ? out_features : 0, false),
              stride(stride),
              pre_norm(pre_norm),
              hidden1(max_batch_size, out_features),
              hidden2(max_batch_size, out_features),
              hidden_shortcut(max_batch_size, out_features),
              relu1(max_batch_size, out_features),
              norm_hidden1(max_batch_size, pre_norm ? in_features : out_features),
              norm_hidden2(max_batch_size, out_features),
              norm_hidden_shortcut(max_batch_size, pre_norm ? in_features : out_features),
              tmp_x_grad(max_batch_size, in_features)
        {
        }

        void forward(const PointRange<Point<T>> &x, PointRange<Point<T>> &y)
        {
            auto cur_batch_size = x.size();

            hidden1.resize(cur_batch_size);
            hidden2.resize(cur_batch_size);
            hidden_shortcut.resize(cur_batch_size);
            relu1.resize(cur_batch_size);
            // relu_out.resize(cur_batch_size);
            norm_hidden1.resize(cur_batch_size);
            norm_hidden2.resize(cur_batch_size);
            norm_hidden_shortcut.resize(cur_batch_size);    

            if(pre_norm){
                RMSNorm<float>::forward(x, norm_hidden1);
                fc1.forward(norm_hidden1, hidden1);
                ReLU<float>::forward(hidden1, relu1);

                RMSNorm<float>::forward(relu1, norm_hidden2);
                fc2.forward(norm_hidden2, hidden2);

                y.copy_(hidden2);
            } else {
                fc1.forward(x, hidden1);
                RMSNorm<float>::forward(hidden1, norm_hidden1);
                ReLU<float>::forward(norm_hidden1, relu1);

                fc2.forward(relu1, hidden2);
                RMSNorm<float>::forward(hidden2, norm_hidden2);

                y.copy_(norm_hidden2);
            }

            if(stride == 1){
                y += x;
            } else {
                if(pre_norm){
                    RMSNorm<float>::forward(x, norm_hidden_shortcut);
                    shortcut.forward(norm_hidden_shortcut, hidden_shortcut);
                    y += hidden_shortcut;
                } else {
                    shortcut.forward(x, hidden_shortcut);
                    RMSNorm<float>::forward(hidden_shortcut, norm_hidden_shortcut);
                    y += norm_hidden_shortcut;
                }   
            }
        }

        void backward(const PointRange<Point<T>> &x, const PointRange<Point<T>> &y_grad, PointRange<Point<T>> &x_grad)
        {
            auto cur_batch_size = x.size();
            uint32_t in_features = x.dimension();
            uint32_t out_features = y_grad.dimension();
            x_grad.zeros_();
            tmp_x_grad.resize(cur_batch_size);
            
            // ========== Shortcut 分支的反向传播 ==========
            if (stride == 1) {
                // 恒等映射：y = main + x，所以 x 的梯度直接传递
                x_grad += y_grad;
            } else {
                // norm_hidden_shortcut已经不需要再使用，用它来存:归一化前的残差
                PointRange<Point<T>>& grad_norm_shortcut = norm_hidden_shortcut;
                if(pre_norm){
                    shortcut.backward(x, y_grad, grad_norm_shortcut);
                    RMSNorm<T>::backward(x, grad_norm_shortcut, x_grad);
                }else{
                    RMSNorm<T>::backward(hidden_shortcut, y_grad, grad_norm_shortcut);
                    shortcut.backward(x, grad_norm_shortcut, x_grad);
                }

            }
            
            // ========== 主分支的反向传播 ==========
            if (pre_norm) {
                // 用norm_hidden2的空间存储grad_norm_hidden2
                auto& tmp_grad = norm_hidden2;
                fc2.backward(norm_hidden2, y_grad, tmp_grad);
                RMSNorm<T>::backward(relu1, tmp_grad, tmp_grad);
                ReLU<T>::backward(hidden1, tmp_grad, tmp_grad);
                
                fc1.backward(norm_hidden1, tmp_grad, tmp_x_grad);
                RMSNorm<T>::backward(x, tmp_x_grad, tmp_x_grad);
            } else {
                auto& tmp_grad = norm_hidden2;
                RMSNorm<T>::backward(hidden2, tmp_grad, tmp_grad);

                auto& tmp_grad2 = relu1;
                fc2.backward(relu1, tmp_grad, tmp_grad2);
                ReLU<T>::backward(norm_hidden1, tmp_grad2, tmp_grad2);
                RMSNorm<T>::backward(hidden1, tmp_grad2, tmp_grad2);
                fc1.backward(x, tmp_grad2, tmp_x_grad);
            }
            x_grad += tmp_x_grad;
        }


    protected:
        Linear<T> fc1;
        Linear<T> fc2;
        Linear<T> shortcut;
        uint32_t stride;
        bool pre_norm;
        // 零时参数
        PointRange<Point<T>> hidden1;
        PointRange<Point<T>> hidden2;
        PointRange<Point<T>> hidden_shortcut;
        PointRange<Point<T>> relu1;
        // PointRange<Point<T>> relu_out;
        PointRange<Point<T>> norm_hidden1;
        PointRange<Point<T>> norm_hidden2;
        PointRange<Point<T>> norm_hidden_shortcut; 
        PointRange<Point<T>> tmp_x_grad;      
    };
}