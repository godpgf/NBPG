#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <vector>

#include "query_params.h"

namespace ant
{

    // Vamana / DiskANN 风格的 RobustPrune，支持可选的最大夹角增强版（PR > 0）。
    //
    // 约定：IdDist::dist 与 PointRange::distance 一致，欧氏度量下为平方距离。
    //
    // alpha     : 距离剪枝。若 alpha * d(p*, p') <= d(p, p')，则 p' 被 p* 遮挡而剪掉。
    // cos_angle : 角度剪枝。在顶点 p 处，若 cos∠(p*, p, p') > cos_angle（夹角过小），则剪掉 p'。
    // start_prune_id : 距离排序后，前 start_prune_id - 1 个最近邻视为强制保留（不参与被剪）。
    struct Prune
    {
        uint32_t R;  // 剪枝后最多保留的邻居数
        uint32_t PR; // 增强剪枝候选池容量；0 表示使用标准贪心剪枝

        Prune(uint32_t R, uint32_t num_workers = 1, uint32_t PR = 0)
            : R(R), PR(PR), num_workers(num_workers)
        {
            if (PR > 0)
            {
                tmp_score.resize(static_cast<size_t>(num_workers) * PR);
            }
        }

        bool usePruneEnhance() const { return PR > 0; }

        // 去重（按 index）并按 dist 升序排序。
        template <typename id_dist>
        void sort_candidates(id_dist *candidates, size_t &candidates_size)
        {
            dedupe_by_index(candidates, candidates_size);
            std::sort(candidates, candidates + candidates_size,
                      [](const id_dist &a, const id_dist &b) { return id_dist_less(a, b); });
        }

        // 同上；可选 dir_bias_scale 对高度数节点做距离偏置，并将中心点 p 排到末尾再剔除。
        template <typename indexType, typename id_dist, typename graphType>
        void sort_candidates(indexType p, graphType &G, id_dist *candidates, size_t &candidates_size,
                             float dir_bias_scale = 0)
        {
            dedupe_by_index(candidates, candidates_size);

            auto biased_less = [&](const id_dist &a, const id_dist &b)
            {
                id_dist lhs = a;
                id_dist rhs = b;
                if (dir_bias_scale > 0)
                {
                    lhs.dist *= dir_bias_scale * std::log(G[lhs.index].size() + 1) + 1;
                    rhs.dist *= dir_bias_scale * std::log(G[rhs.index].size() + 1) + 1;
                }
                if (lhs.index == p)
                    lhs.dist = std::numeric_limits<float>::max();
                if (rhs.index == p)
                    rhs.dist = std::numeric_limits<float>::max();
                return id_dist_less(lhs, rhs);
            };
            std::sort(candidates, candidates + candidates_size, biased_less);

            while (candidates_size > 0 && candidates[candidates_size - 1].index == p)
                --candidates_size;
        }

        // 入口：PR > 0 走最大夹角增强剪枝，否则走标准 RobustPrune。
        // 返回额外距离计算次数；结果写回 candidates[0:candidates_size)。
        template <typename indexType, typename id_dist, typename PointRange>
        size_t robustPrune(indexType p, id_dist *candidates, size_t &candidates_size, PointRange &Points,
                           double alpha, double cos_angle, size_t start_prune_id = 1)
        {
            if (usePruneEnhance())
            {
                return maxAnglePrune(p, candidates, candidates_size, Points, alpha, cos_angle, start_prune_id);
            }
            return limitedSimilarityPrune(p, candidates, candidates_size, Points, alpha, cos_angle, start_prune_id);
        }

        // 增强剪枝：每次从剩余候选中选与已选集合 C 夹角最大的点加入 C，再对剩余候选做 RobustPrune 过滤。
        // score[i] 记录候选 i 与 C 中已选点的最大余弦（对应最小夹角）。
        template <typename indexType, typename id_dist, typename PointRange>
        size_t maxAnglePrune(indexType p, id_dist *candidates, size_t &candidates_size, PointRange &Points,
                             double alpha, double cos_angle, size_t start_prune_id = 1)
        {
            assert(PR >= candidates_size);
            if (alpha <= kEpsilon && cos_angle <= kEpsilon)
            {
                std::cout << "RobustPrune requires alpha or cos_angle; both are unset." << std::endl;
                abort();
            }

            const auto worker_id = static_cast<size_t>(omp_get_thread_num() % num_workers);
            float *score = tmp_score.data() + worker_id * PR;
            std::fill(score, score + candidates_size, -2.f);

            size_t distance_comps = 0;
            size_t selected_size = start_prune_id > 0 ? start_prune_id - 1 : 0;

            while (selected_size < R && selected_size < candidates_size)
            {
                // 循环不变量：candidates[selected_size] 在剩余候选中与 C 的夹角最大（score 最小）。
                id_dist &cur = candidates[selected_size];
                const indexType p_star = cur.index;
                if (p_star == p || p_star == static_cast<indexType>(-1))
                {
                    std::cout << "RobustPrune encountered invalid candidate index." << std::endl;
                    abort();
                }
                ++selected_size;

                size_t survivors_end = selected_size;
                for (size_t i = selected_size; i < candidates_size; ++i)
                {
                    const indexType p_prime = candidates[i].index;
                    assert(p_prime != static_cast<indexType>(-1));

                    ++distance_comps;
                    const float dist_star_prime = Points[p_star].distance(Points[p_prime]);
                    const float dist_p_prime = candidates[i].dist;
                    const float cos_val = cosine_at_p(cur.dist, dist_p_prime, dist_star_prime);
                    score[i] = std::max(score[i], cos_val);

                    if (!survives_prune_enhanced(alpha, cos_angle, dist_star_prime, dist_p_prime, cos_val))
                        continue;

                    if (i > survivors_end)
                    {
                        std::swap(candidates[survivors_end], candidates[i]);
                        std::swap(score[survivors_end], score[i]);
                    }
                    // 将下一个要选的最大夹角候选交换到 survivors_end。
                    if (score[survivors_end] < score[selected_size - 1])
                    {
                        std::swap(candidates[survivors_end], candidates[selected_size - 1]);
                        std::swap(score[survivors_end], score[selected_size - 1]);
                    }
                    ++survivors_end;
                }
                candidates_size = survivors_end;
            }

            candidates_size = selected_size;
            return distance_comps;
        }

        // 标准 RobustPrune（Vamana）：按距离升序贪心选点，并用 alpha 或 cos_angle 剔除被遮挡候选。
        // 被剪候选标记 index = -1；最终结果紧凑写入 candidates 头部。
        template <typename indexType, typename id_dist, typename PointRange>
        size_t limitedSimilarityPrune(indexType p, id_dist *candidates, size_t &candidates_size, PointRange &Points,
                                      double alpha, double cos_angle, size_t start_prune_id = 1)
        {
            size_t distance_comps = 0;

            if (alpha <= kEpsilon && cos_angle <= kEpsilon)
            {
                if (candidates_size > R)
                    candidates_size = R;
                return 0;
            }

            size_t selected_size = 0;
            size_t scan_idx = 0;

            while (selected_size < R && scan_idx < candidates_size)
            {
                const indexType p_star = candidates[scan_idx].index;
                ++scan_idx;
                if (p_star == p || p_star == static_cast<indexType>(-1))
                    continue;

                candidates[selected_size++] = candidates[scan_idx - 1];

                for (size_t i = scan_idx; i < candidates_size; ++i)
                {
                    if (i < start_prune_id)
                        continue;

                    indexType p_prime = candidates[i].index;
                    if (p_prime == static_cast<indexType>(-1))
                        continue;

                    ++distance_comps;
                    const float dist_star_prime = Points[p_star].distance(Points[p_prime]);
                    const float dist_p_prime = candidates[i].dist;

                    if (is_pruned_by_distance(alpha, dist_star_prime, dist_p_prime))
                    {
                        candidates[i].index = -1;
                    }
                    else if (is_pruned_by_angle(cos_angle, candidates[scan_idx - 1].dist, dist_p_prime, dist_star_prime))
                    {
                        candidates[i].index = -1;
                    }
                }
            }

            candidates_size = selected_size;
            return distance_comps;
        }

    private:
        static constexpr double kEpsilon = 1e-5;

        std::vector<float> tmp_score;
        uint32_t num_workers;

        // 在顶点 p 处，由边 (p,p*) 与 (p,p') 张成的夹角余弦；输入距离均为平方距离。
        static float cosine_at_p(float dist_p_star_sq, float dist_p_prime_sq, float dist_star_prime_sq)
        {
            const float b = std::sqrt(dist_p_prime_sq);
            const float c = std::sqrt(dist_p_star_sq);
            if (b == 0.f || c == 0.f)
                return 1.f;
            return (dist_p_prime_sq + dist_p_star_sq - dist_star_prime_sq) / (2.f * b * c);
        }

        static bool is_pruned_by_distance(double alpha, float dist_star_prime_sq, float dist_p_prime_sq)
        {
            return alpha > kEpsilon && alpha * dist_star_prime_sq <= dist_p_prime_sq;
        }

        static bool is_pruned_by_angle(double cos_angle, float dist_p_star_sq, float dist_p_prime_sq,
                                       float dist_star_prime_sq)
        {
            if (cos_angle <= kEpsilon)
                return false;
            return cosine_at_p(dist_p_star_sq, dist_p_prime_sq, dist_star_prime_sq) > cos_angle;
        }

        // 增强版：alpha 与 cos_angle 同时启用时，满足任一保留条件即不被剪。
        static bool survives_prune_enhanced(double alpha, double cos_angle, float dist_star_prime_sq,
                                            float dist_p_prime_sq, float cos_val)
        {
            if (alpha > kEpsilon && alpha * dist_star_prime_sq > dist_p_prime_sq)
                return true;
            if (cos_angle > kEpsilon && cos_val < cos_angle)
                return true;
            return alpha <= kEpsilon && cos_angle <= kEpsilon;
        }

        template <typename id_dist>
        static void dedupe_by_index(id_dist *candidates, size_t &candidates_size)
        {
            std::sort(candidates, candidates + candidates_size,
                      [](const id_dist &a, const id_dist &b) { return a.index < b.index; });
            candidates_size = static_cast<size_t>(
                std::unique(candidates, candidates + candidates_size,
                            [](const id_dist &a, const id_dist &b) { return a.index == b.index; }) -
                candidates);
        }
    };

} // namespace ant
