#include "utils/Point/point_range.h"
#include "utils/Point/point.h"
#include <vector>
#include <random>
#include <algorithm>
#include <omp.h>

namespace ant
{

    template<typename indexType>
    void train_node_vectors(std::vector<std::pair<indexType, indexType>>& jump_nodes, PointRange<Point<float>>& p) {
        /**
         * 使用graph embedding算法训练node_vectors中的参数
         */
        // 初始化p向量 (Xavier/Kaiming)
        p.kaiming_uniform_init(p.size() + p.dimension());
        #pragma omp parallel for
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] /= p[i].norm();
        }


        /**
         * jump_nodes 是一个记录节点访问信息的序列，其中每个元素 v 由两部分组成：
         *   - v.first  ：step（步数）
         *   - v.second ：节点 ID
         * 例如：[(0,1), (0,5), (1,6), (1,9), ..., (10,89), (0,2), (0,7), ...]
         *
         * 该序列由多个子序列拼接而成。在同一子序列内，step 值单调递增；
         * 当 step 重新变为 0 时，即表示新一个子序列的开始。
         * （上例中，从 (0,1) 到 (10,89) 为第一个子序列，之后 step 归零开始第二个子序列。）
         *
         * 在任意一个子序列中，若两个元素 vi 和 vj 的 step 值满足 |vi.step - vj.step| ≤ 1，
         * 则认为它们对应的节点互为邻居，即 vi.second 与 vj.second 是邻居关系。
         *
         * 训练目标与图嵌入（Graph Embedding）算法相似：
         *   - 邻居节点对的向量点积应尽可能大；
         *   - 非邻居节点对（或随机采样的节点对）的点积应尽可能小。
         *
         * 与传统图嵌入方法的主要区别在于邻居的定义方式：
         *   - 传统方法（如 DeepWalk、Node2Vec）通过随机游走生成节点序列，
         *     序列中在一定窗口范围内的节点均视为邻居，并要求它们之间的相似度（点积）最大化；
         *   - 本方法则严格限定在同一子序列内、step 差值不超过 1 的节点对作为邻居进行优化。
         *
         * 每个节点的邻居信息存储在数据结构 p 中，例如 p[vi.second] 中保存了节点 vi.second 的邻居列表。
         */

        // ========== 构建邻居关系 ==========
        std::vector<std::vector<indexType>> neighbors(p.size());

        size_t seq_start = 0;
        for (size_t i = 0; i <= jump_nodes.size(); ++i) {
            if (i == jump_nodes.size() || jump_nodes[i].first == 0) {
                if (i > seq_start) {
                    for (size_t j = seq_start; j < i; ++j) {
                        indexType node_u = jump_nodes[j].second;
                        for (size_t k = seq_start; k < i; ++k) {
                            if (j == k) continue;
                            if (std::abs(static_cast<long long>(jump_nodes[j].first) -
                                         static_cast<long long>(jump_nodes[k].first)) <= 1) {
                                indexType node_v = jump_nodes[k].second;
                                if (node_u != node_v) {
                                    neighbors[node_u].push_back(node_v);
                                }
                            }
                        }
                    }
                }
                seq_start = i;
            }
        }

        // 去重邻居列表
        #pragma omp parallel for
        for (size_t i = 0; i < neighbors.size(); ++i) {
            auto& nbrs = neighbors[i];
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }

        // ========== 训练参数 ==========
        const int num_epochs = 100;
        const int num_negative_samples = 5;
        const float learning_rate = 0.01f;
        const float margin = 1.0f;

        std::random_device rd;

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            float total_loss = 0.0f;

            #pragma omp parallel
            {
                // 每个线程独立的随机生成器
                std::mt19937 gen(rd() + omp_get_thread_num());
                std::uniform_int_distribution<indexType> node_dist(0, static_cast<indexType>(p.size() - 1));

                #pragma omp for reduction(+:total_loss)
                for (indexType center_id = 0; center_id < static_cast<indexType>(p.size()); ++center_id) {
                    if (neighbors[center_id].empty()) continue;

                    Point<float> center_vec = p[center_id];
                    std::vector<float> grad_center(p.dimension(), 0.0f);

                    // 正样本：邻居节点
                    for (indexType neighbor_id : neighbors[center_id]) {
                        Point<float> neighbor_vec = p[neighbor_id];
                        float pos_dot = center_vec.dot(neighbor_vec);
                        total_loss += -pos_dot;

                        for (unsigned d = 0; d < p.dimension(); ++d) {
                            grad_center[d] -= neighbor_vec[d];
                        }
                    }

                    // 负样本：随机采样非邻居节点（修复 size_t 下溢）
                    int valid_neg = 0;
                    while (valid_neg < num_negative_samples) {
                        indexType neg_id = node_dist(gen);
                        if (neg_id == center_id) continue;

                        bool is_neighbor = false;
                        for (indexType nbr : neighbors[center_id]) {
                            if (nbr == neg_id) {
                                is_neighbor = true;
                                break;
                            }
                        }
                        if (is_neighbor) continue;

                        Point<float> neg_vec = p[neg_id];
                        float neg_dot = center_vec.dot(neg_vec);
                        float neg_loss = std::max(0.0f, margin + neg_dot);

                        if (neg_loss > 0.0f) {
                            total_loss += neg_loss;
                            for (unsigned d = 0; d < p.dimension(); ++d) {
                                grad_center[d] += neg_vec[d];
                            }
                        }

                        ++valid_neg;
                    }

                    // 更新中心节点向量
                    for (unsigned d = 0; d < p.dimension(); ++d) {
                        center_vec[d] -= learning_rate * grad_center[d];
                    }

                    // 归一化
                    float norm = center_vec.norm();
                    if (norm > 1e-6f) {
                        center_vec /= norm;
                    }

                    p[center_id].copy_(center_vec);
                }
            }

            // 全局归一化
            #pragma omp parallel for
            for (size_t i = 0; i < p.size(); ++i) {
                float norm = p[i].norm();
                if (norm > 1e-6f) {
                    p[i] /= norm;
                }
            }
        }
    }

}