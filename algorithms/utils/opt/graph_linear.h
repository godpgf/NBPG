#pragma once
#include <set>
#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include "utils/Graph/graph.h"
#include "RMSNorm.h"
#include "optimizer.h"

namespace ant
{

    template <typename indexType, typename T>
    struct GraphLinear
    {
        using id_act = std::pair<indexType, T>;
        GraphLinear(Optimizer<T> *optimizer,
                    indexType nodeNum,
                    unsigned dims,
                    uint32_t max_batch_size,
                    uint32_t max_step,
                    uint32_t head_num,
                    uint32_t L,
                    bool pre_norm = true,
                    bool deep_norm = false) : nodeNum(nodeNum),
                                              max_batch_size(max_batch_size),
                                              max_step(max_step),
                                              head_num(head_num),
                                              L(L),
                                              pre_norm(pre_norm),
                                              deep_norm(deep_norm),
                                              w_act(nodeNum, dims / head_num),
                                              w_act_grad(nodeNum, dims / head_num),
                                              w_feat(nodeNum, dims),
                                              w_feat_grad(nodeNum, dims),
                                              hidden(max_step * max_batch_size, dims),
                                              norm_hidden(max_step * max_batch_size, dims),
                                              action(max_step * max_batch_size, L * head_num),
                                              norm_action(max_step * max_batch_size, L * head_num),
                                              nodeIds(max_step * max_batch_size, L * head_num)
        {
            // 初始化模型参数
            w_act.kaiming_uniform_init(dims);
            w_feat.kaiming_uniform_init(L * L);
            if (optimizer != nullptr)
            {
                optimizer->register_opt(w_act, w_act_grad);
                optimizer->register_opt(w_feat, w_feat_grad);
            }
        }

        void forward(const PointRange<Point<T>> &x,
                     PointRange<Point<T>> &y,
                     const Graph<indexType> &G)
        {
            // 多个向量相加方差会改变，用这个系数缩放方差
            const float h_scale = 1 / std::sqrt(2);
            auto cur_batch_size = x.size();
            action.zeros_();
            nodeIds.zeros_();
            auto sub_dim = w_act.dimension();
            size_t head_node_num = G.size() / head_num;

            // pre_norm
            // x->norm->w_act->norm->w_feat + x
            // post_norm
            // x->w_act->norm->w_feat->norm + x

#pragma omp parallel for
            for (size_t i = 0; i < cur_batch_size; ++i)
            {
                std::vector<id_act> candidates;
                candidates.reserve(L + (G.max_degree() + 1));
                std::set<indexType> visited;
                visited.insert(0);

                for (auto step = 0; step < max_step; ++step)
                {
                    for (auto head_id = 0; head_id < head_num; ++head_id)
                    {
                        // （1）利用输出变量y来暂存当前节点的输入
                        auto input_vec = y[i].getSubPoint(head_id, sub_dim);
                        if (step == 0)
                        {
                            input_vec.copy_(x[i].getSubPoint(head_id, sub_dim));
                        }
                        else
                        {
                            auto pre_h = get_step_layer(hidden, i, step - 1);
                            input_vec.copy_(pre_h.getSubPoint(head_id, sub_dim));
                        }

                        // （2）对于pre-norm，需要计算归一化后的数据
                        if (pre_norm)
                        {
                            auto after_norm = get_step_layer(norm_hidden, i, step).getSubPoint(head_id, sub_dim);
                            RMSNorm<T>::forward(input_vec, after_norm);
                            input_vec.copy_(after_norm);
                        }

                        // （3）搜索：遍历邻居的邻居，计算注意力值，并按照注意力值对这些邻居就行排序
                        if (step == 0)
                        {
                            fill_candidates(input_vec, head_id * head_node_num, G, visited, candidates);
                        }
                        else
                        {
                            auto pre_action = get_step_layer(action, i, step - 1).getSubPoint(head_id, L);
                            auto pre_nodeIds = get_step_layer(nodeIds, i, step - 1).getSubPoint(head_id, L);
                            for (auto j = 0; j < L; ++j)
                            {
                                auto index = pre_nodeIds[j];
                                auto act = pre_action[j];
                                if (act <= 1e-5)
                                    break;
                                fill_candidates(input_vec, index, G, visited, candidates);
                            }
                        }

                        // （4）记录当前层的最优节点，和这些节点的注意力值
                        fill_action(i, head_id, step, candidates);
                    }

                    // （5）得到当前检索出来的特征
                    auto h = get_step_layer(hidden, i, step);
                    {
                        h.zeros_();
                        auto cur_action = get_step_layer(action, i, step);
                        auto cur_nodeIds = get_step_layer(nodeIds, i, step);
                        for (auto j = 0; j < L * head_num; ++j)
                        {
                            auto index = cur_nodeIds[j];
                            auto act = cur_action[j];
                            if (act > 1e-5)
                                h.add_feat(w_feat[index], act);
                        }

                        // 如果是post-norm，需要对特征就行归一化，并记录归一化前的数据，以便计算梯度
                        if (!pre_norm)
                        {
                            auto before_norm = get_step_layer(norm_hidden, i, step);
                            before_norm.copy_(h);
                            RMSNorm<T>::forward(before_norm, h);
                        }
                    }

                    // （6）加上跨层连接
                    if (step > 0)
                    {
                        auto pre_h = get_step_layer(hidden, i, step - 1);
                        h += pre_h;
                    }
                    else
                    {
                        h += x[i];
                    }

                    // （7）对输出就行缩放，目的是保证方差不变
                    if (deep_norm)
                    {
                        h *= h_scale;
                    }
                }

                auto h = get_step_layer(hidden, i, max_step - 1);
                y[i].copy_(h);
            }
        }

        void fill_jump_nodes(int batch_id, std::vector<std::pair<indexType, indexType>>& res){
            for(int head_id = 0; head_id < head_num; ++head_id){
                int offset = head_id * L;
                for (auto step = 0; step < max_step; ++step){
                    auto cur_action = get_step_layer(action, batch_id, step);
                    auto cur_nodeIds = get_step_layer(nodeIds, batch_id, step);
                    
                    for(int j = 0; j < L; ++j){
                        auto act = cur_action[offset + j];
                        if(act <= 1e-5)
                            break;
                        res.push_back(std::pair<indexType, indexType>(step, cur_nodeIds[offset + j]));
                    }
                }
            }

        }

        void backward(const PointRange<Point<T>> &x,
                      PointRange<Point<T>> &y_grad,
                      PointRange<Point<T>> &x_grad)
        {
            auto cur_batch_size = x.size();
            // 多个向量相加方差会改变，用这个系数缩放方差
            const float h_scale = 1 / std::sqrt(2);
            auto sub_dim = w_act.dimension();

            // pre_norm
            // x->norm->w_act->norm->w_feat + x
            // post_norm
            // x->w_act->norm->w_feat->norm + x

#pragma omp parallel for
            for (size_t i = 0; i < cur_batch_size; ++i)
            {
                for (int step = max_step - 1; step >= 0; --step)
                {
                    // 当前层的输出数据已经不再有用，所以用它来暂存当前层的输出区
                    auto h_grad = get_step_layer(hidden, i, step);
                    if (step == max_step - 1)
                    {
                        h_grad.copy_(y_grad[i]);
                    }

                    // （1）对残差就行缩放（因为前馈时有缩放操作）
                    if (deep_norm)
                    {
                        h_grad *= h_scale;
                    }

                    // （2）将残差传递到上一层，用x_grad来暂存上一层的残差
                    auto pre_h_grad = x_grad[i];
                    pre_h_grad.copy_(h_grad);

                    // （3）如果使用post-norm，需要计算归一化前的梯度
                    if (!pre_norm)
                    {
                        auto before_norm = get_step_layer(norm_hidden, i, step);
                        RMSNorm<T>::backward(before_norm, h_grad, h_grad);
                    }

                    const auto pre_h = (step > 0) ? get_step_layer(hidden, i, step - 1) : x[i];

                    // （4）计算注意力值和特征层的梯度
                    auto cur_action = get_step_layer(action, i, step);
                    auto &cur_action_grad = cur_action;
                    auto cur_nodeIds = get_step_layer(nodeIds, i, step);

                    for (auto j = 0; j < L * head_num; ++j)
                    {
                        auto act = cur_action.data()[j];
                        if (act <= 1e-5)
                        {
                            cur_action.data()[j] = 0;
                            continue;
                        }

                        auto index = cur_nodeIds[j];
                        // 计算当前注意力值的梯度，以及w_feat的梯度
                        float act_grad = h_grad.dot(w_feat[index]);
#pragma omp critical
                        {
                            w_feat_grad[index].add_feat(h_grad, act);
                        }

                        // 缓存注意力值的梯度
                        cur_action_grad.data()[j] = act_grad;
                    }

                    for (auto head_id = 0; head_id < head_num; ++head_id)
                    {
                        auto before_norm = get_step_layer(norm_action, i, step).getSubPoint(head_id, L);
                        auto sub_action_grad = cur_action_grad.getSubPoint(head_id, L);
                        RMSNorm<T>::backward(before_norm, sub_action_grad, sub_action_grad);
                    }

                    h_grad.zeros_();

                    // （5）计算在注意力之前的梯度
                    for (auto head_id = 0; head_id < head_num; ++head_id)
                    {
                        auto after_norm = get_step_layer(norm_hidden, i, step).getSubPoint(head_id, sub_dim);
                        const auto cur_pre_h = pre_h.getSubPoint(head_id, sub_dim);
                        auto cur_h_grad = h_grad.getSubPoint(head_id, sub_dim);
                        for (auto j = head_id * L; j < (head_id + 1) * L; ++j)
                        {
                            auto act_grad = cur_action_grad.data()[j];
                            if (abs(act_grad) < 1e-5)
                                continue;
                            auto index = cur_nodeIds[j];

                            cur_h_grad.add_feat(w_act[index], act_grad);
#pragma omp critical
                            {

                                if (pre_norm)
                                {
                                    w_act_grad[index].add_feat(after_norm, act_grad);
                                }
                                else
                                {
                                    w_act_grad[index].add_feat(cur_pre_h, act_grad);
                                }
                            }
                        }
                        if (pre_norm)
                        {
                            RMSNorm<T>::backward(cur_pre_h, cur_h_grad, cur_h_grad);
                        }
                    }

                    pre_h_grad += h_grad;
                    if (step > 0)
                    {
                        // pre_h 计算完已经没有用了，可以用来保存梯度
                        auto _pre_h_grad = get_step_layer(hidden, i, step - 1);
                        _pre_h_grad.copy_(pre_h_grad);
                    }
                }
            }
        }

        void set_max_step(uint32_t max_step) { this->max_step = max_step; }

    protected:
        template <typename dataType>
        Point<dataType> get_step_layer(PointRange<Point<dataType>> &points, uint32_t batch_id, uint32_t step)
        {
            // 特殊的，由于第一层的输入数据是x，所以把第一个step的输入数据作为最后一层的输出！！！
            return points[step * max_batch_size + batch_id];
        }

        void fill_candidates(Point<T> p, indexType nodeId, const Graph<indexType> &G, std::set<indexType> visited, std::vector<id_act> &candidates)
        {
            auto firstEdges = G[nodeId];
            for (indexType fi = 0; fi < firstEdges.size(); fi++)
            {
                auto index = firstEdges[fi];
                if (visited.find(index) == visited.end())
                {
                    visited.insert(index);
                    T act = p.dot(w_act[index]);
                    if (act > 0)
                    {
                        candidates.push_back(id_act(index, act));
                    }
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](id_act a, id_act b)
                      { return a.second > b.second; });
            if (candidates.size() > L)
                candidates.resize(L);
        }

        void fill_action(uint32_t batch_id, uint32_t head_id, uint32_t step, std::vector<id_act> &candidates)
        {
            auto cur_action = get_step_layer(action, batch_id, step).getSubPoint(head_id, L);
            auto cur_nodeIds = get_step_layer(nodeIds, batch_id, step).getSubPoint(head_id, L);
            for (auto j = 0; j < candidates.size(); ++j)
            {
                auto edg = candidates[j];
                cur_action.data()[j] = edg.second;
                cur_nodeIds.data()[j] = edg.first;
            }
            candidates.resize(0);

            {
                auto before_norm = get_step_layer(norm_action, batch_id, step).getSubPoint(head_id, L);
                before_norm.copy_(cur_action);
                RMSNorm<T>::forward(before_norm, cur_action);
            }
        }

    protected:
        indexType nodeNum;
        uint32_t max_batch_size;
        uint32_t max_step;
        uint32_t head_num;
        uint32_t L;
        bool pre_norm;
        bool deep_norm;
        // 核心参数 ------------------------------------
        PointRange<Point<T>> w_act;
        PointRange<Point<T>> w_act_grad;
        PointRange<Point<T>> w_feat;
        PointRange<Point<T>> w_feat_grad;

        // 临时参数-------------------------------------
        // 记录每一个block的输出，计算梯度时要用
        PointRange<Point<T>> hidden;
        PointRange<Point<T>> norm_hidden;
        // 记录每一层节点的注意力值
        PointRange<Point<T>> action;
        PointRange<Point<T>> norm_action;
        // 记录每一层选择的节点id
        PointRange<Point<indexType>> nodeIds;
    };
}