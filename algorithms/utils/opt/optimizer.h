#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"

namespace ant
{
    template <typename T>
    class Optimizer
    {
    public:
        virtual void step() = 0;
        virtual void register_opt(PointRange<Point<T>>& w, PointRange<Point<T>>& w_grad)
        {
            params.push_back(&w);
            params_grad.push_back(&w_grad);
        }


        virtual void zero_grads(){
            for(auto grad : params_grad){
                grad->zeros_();
            }
        }

    protected:
        std::vector<PointRange<Point<T>> *> params;
        std::vector<PointRange<Point<T>> *> params_grad;

    };

    template <typename T>
    class SGDOptimizer : public Optimizer<T>
    {
    public:
        SGDOptimizer(float lr) : lr(lr) {}

        virtual void step() {
            for(auto i = 0; i < this->params.size(); ++i){
                auto* w = this->params[i];
                auto* w_grad = this->params_grad[i];
                (*w).add_feat((*w_grad), -this->lr);
            }

            for(auto i = 0; i < this->sparse_params.size(); ++i){
                auto* w = this->sparse_params[i];
                auto* w_grad = this->sparse_params_grad[i];
                (*w).add_feat((*w_grad), -this->lr);
            }
        }

        float lr;
    };

    template <typename T>
    class AdamOptimizer : public Optimizer<T>
    {
    public:
        AdamOptimizer(float lr, float beta1=0.9f, float beta2=0.999f,float epsilon=1e-5f) : lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}
        virtual ~AdamOptimizer(){
            for(auto m : m_){
                delete m;
            }
            for(auto v : v_){
                delete v;
            }
            
        }

        virtual void register_opt(PointRange<Point<T>>& w, PointRange<Point<T>>& w_grad)
        {
            Optimizer<T>::register_opt(w, w_grad);
            PointRange<Point<T>>* m = new PointRange<Point<T>>(w_grad.size(), w_grad.dimension());
            PointRange<Point<T>>* v = new PointRange<Point<T>>(w_grad.size(), w_grad.dimension());
            m_.push_back(m);
            v_.push_back(v);
        }


        virtual void step() {
            ++t;
            step_params(this->params, this->params_grad, m_, v_);
        }

        template<typename paramsType>
        void step_params(std::vector<paramsType*> cur_params, std::vector<PointRange<Point<T>> *> cur_params_grad, std::vector<PointRange<Point<T>> *> cur_m_, std::vector<PointRange<Point<T>> *> cur_v_){
            float m_scale = 1.0f / (1.0f - std::pow(beta1, t));
            float v_scale = 1.0f / (1.0f - std::pow(beta2, t));

            for(auto i = 0; i < cur_params.size(); ++i){
                auto* w = cur_params[i];
                auto* w_grad = cur_params_grad[i];
                auto* m = cur_m_[i];
                auto* v = cur_v_[i];
                // 更新m和v
                (*m) *= beta1;
                m->add_feat((*w_grad), (1.0f - beta1));
                (*v) *= beta2;
                v->add_feat_square((*w_grad), (1.0f - beta2));
                // 更新w

                #pragma omp parallel for
                for(auto index = 0; index < w->size(); ++index){
                    // auto* w_ptr = (*w)[index].data();
                    auto* m_ptr = (*m)[index].data();
                    auto* v_ptr = (*v)[index].data();
                    std::vector<std::pair<unsigned, T>> cache((*w).dimension());
                    for(unsigned j = 0; j < (*w).dimension(); ++j){
                        cache[j].first = j;
                        cache[j].second = -lr * m_ptr[j] * m_scale / (std::sqrt(v_ptr[j] * v_scale) + epsilon);
                        // w_ptr[j] -= lr * m_ptr[j] * m_scale / (std::sqrt(v_ptr[j] * v_scale) + epsilon);
                    }
                    (*w)[index].add_feat(cache.data());
                }
            }
        }

        std::vector<PointRange<Point<T>> *> m_;
        std::vector<PointRange<Point<T>> *> v_;

        float lr;
        float beta1;
        float beta2;
        float epsilon;
        size_t t;
    };  
}