#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "utils/utilities.h"
#include "utils/Graph/query_params.h"

namespace ant
{

    // 基于随机聚类树为 NN-Descent 生成初始近邻列表
    template <typename PointRange, typename indexType>
    struct ClusterPID
    {
        using PR = PointRange;
        using edge = std::pair<indexType, indexType>;
        using pid = IdDist<indexType>;

        static std::pair<indexType, indexType> select_two_random(const std::vector<indexType> &active_indices,
                                                                 Random &rnd)
        {
            const size_t n = active_indices.size();
            const size_t first_index = rnd.ith_rand(0) % n;
            const size_t second_index_unshifted = rnd.ith_rand(1) % (n - 1);
            const size_t second_index = (second_index_unshifted < first_index)
                                            ? second_index_unshifted
                                            : (second_index_unshifted + 1);

            return {active_indices[first_index], active_indices[second_index]};
        }

        // 对叶子簇做全对距离计算，保留每个点的 top-K 近邻
        static void naive_neighbors(PR &Points, const std::vector<indexType> &active_indices,
                                    size_t max_k, std::vector<std::vector<pid>> &intermediate_edges,
                                    QueryStatic &BuildStats)
        {
#pragma omp parallel for
            for (size_t i = 0; i < active_indices.size(); ++i)
            {
                const indexType index = active_indices[i];
                const size_t cluster_size = active_indices.size();

                auto less = [](const pid &a, const pid &b)
                { return id_dist_less<indexType>(a, b); };
                std::priority_queue<pid, std::vector<pid>, decltype(less)> top_k(less);

                BuildStats.increment_dist(index, cluster_size - 1);
                for (size_t j = 0; j < cluster_size; ++j)
                {
                    if (j == i)
                        continue;

                    const indexType neighbor = active_indices[j];
                    const float dist = Points[index].distance(Points[neighbor]);
                    const pid candidate(neighbor, dist);
                    if (top_k.size() >= max_k)
                    {
                        if (dist < top_k.top().dist)
                        {
                            top_k.pop();
                            top_k.push(candidate);
                        }
                    }
                    else
                    {
                        top_k.push(candidate);
                    }
                }

                std::vector<pid> cluster_best(top_k.size());
                for (auto it = cluster_best.rbegin(); it != cluster_best.rend(); ++it)
                {
                    *it = top_k.top();
                    top_k.pop();
                }

                auto &current = intermediate_edges[index];
                std::vector<pid> merged;
                merged.reserve(cluster_best.size() + current.size());
                std::merge(cluster_best.begin(), cluster_best.end(),
                           current.begin(), current.end(),
                           std::back_inserter(merged), less);
                merged.erase(std::unique(merged.begin(), merged.end(),
                                         [](const pid &a, const pid &b)
                                         { return a.index == b.index; }),
                             merged.end());
                if (merged.size() > max_k)
                    merged.resize(max_k);

                current = std::move(merged);
            }
        }

        static void random_clustering(PR &Points, std::vector<indexType> &active_indices,
                                      Random &rnd, size_t cluster_size, size_t max_k,
                                      std::vector<std::vector<pid>> &intermediate_edges,
                                      QueryStatic &BuildStats)
        {
            if (active_indices.size() <= cluster_size)
            {
                naive_neighbors(Points, active_indices, max_k, intermediate_edges, BuildStats);
                return;
            }

            const auto [pivot_first, pivot_second] = select_two_random(active_indices, rnd);
            auto left_rnd = rnd.fork(0);
            auto right_rnd = rnd.fork(1);

            std::vector<indexType> left_indices;
            std::vector<indexType> right_indices;
            left_indices.reserve(active_indices.size());
            right_indices.reserve(active_indices.size());

            BuildStats.increment_dist(pivot_first, 1);
            if (Points[pivot_first].distance(Points[pivot_second]) < 1e-5f)
            {
                // 两个 pivot 重合时按索引均分，避免无法切分
                const size_t half = active_indices.size() / 2;
                for (size_t i = 0; i < active_indices.size(); ++i)
                {
                    if (i < half)
                        left_indices.push_back(active_indices[i]);
                    else
                        right_indices.push_back(active_indices[i]);
                }
            }
            else
            {
                BuildStats.increment_dist(pivot_first, active_indices.size());
                BuildStats.increment_dist(pivot_second, active_indices.size());

                std::vector<bool> closer_to_first(active_indices.size());
#pragma omp parallel for
                for (size_t i = 0; i < active_indices.size(); ++i)
                {
                    const indexType ind = active_indices[i];
                    const float dist_first = Points[ind].distance(Points[pivot_first]);
                    const float dist_second = Points[ind].distance(Points[pivot_second]);
                    closer_to_first[i] = dist_first <= dist_second;
                }

                for (size_t i = 0; i < active_indices.size(); ++i)
                {
                    if (closer_to_first[i])
                        left_indices.push_back(active_indices[i]);
                    else
                        right_indices.push_back(active_indices[i]);
                }
            }

            random_clustering(Points, left_indices, left_rnd, cluster_size, max_k,
                              intermediate_edges, BuildStats);
            random_clustering(Points, right_indices, right_rnd, cluster_size, max_k,
                              intermediate_edges, BuildStats);
        }

        static void random_clustering_wrapper(PR &Points, size_t cluster_size, size_t max_k,
                                              std::vector<std::vector<pid>> &intermediate_edges,
                                              QueryStatic &BuildStats)
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<indexType> uni(0, static_cast<indexType>(Points.size() - 1));

            Random rnd(uni(rng));

            std::vector<indexType> active_indices(Points.size());
            std::iota(active_indices.begin(), active_indices.end(), static_cast<indexType>(0));

            random_clustering(Points, active_indices, rnd, cluster_size, max_k,
                              intermediate_edges, BuildStats);
        }

        // 重复构建多棵聚类树，合并为 NN-Descent 的初始近邻
        static void multiple_clustertrees(PR &Points, size_t cluster_size, size_t num_clusters, size_t max_k,
                                          std::vector<std::vector<pid>> &intermediate_edges,
                                          QueryStatic &BuildStats)
        {
            intermediate_edges.resize(Points.size());
            for (size_t i = 0; i < num_clusters; ++i)
            {
                random_clustering_wrapper(Points, cluster_size, max_k, intermediate_edges, BuildStats);
                std::cout << "Cluster " << i << std::endl;
            }
        }
    };

}
