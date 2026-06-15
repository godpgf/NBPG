#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "utils/utilities.h"
#include "utils/Graph/query_params.h"

namespace ant
{

    template <typename PointRange, typename indexType>
    struct NNDescent
    {
        using pid = IdDist<indexType>;
        using edge = std::pair<indexType, indexType>;
        using labelled_edge = std::pair<indexType, pid>;
        using inv_neighbor_list = std::pair<indexType, std::vector<indexType>>;

        double delta;
        uint32_t L;
        int max_rounds = 64;

        NNDescent(uint32_t L, double delta, int max_rounds)
            : L(L), delta(delta), max_rounds(max_rounds)
        {
        }

        template <typename GraphI, typename Prune>
        void undirect_and_prune(GraphI &G, Prune &prune, PointRange &Points, double alpha, double cos_angle,
                                std::vector<std::vector<pid>> &old_neighbors, QueryStatic &BuildStats)
        {
            const size_t num_vertices = old_neighbors.size();
            std::vector<std::vector<labelled_edge>> directed_edges(num_vertices);

#pragma omp parallel for
            for (indexType i = 0; i < num_vertices; ++i)
            {
                const size_t neighbor_count = old_neighbors[i].size();
                std::vector<labelled_edge> edges(neighbor_count);
                for (size_t j = 0; j < neighbor_count; ++j)
                {
                    edges[j] = {old_neighbors[i][j].index, pid(i, old_neighbors[i][j].dist)};
                }
                directed_edges[i] = std::move(edges);
            }

            std::vector<labelled_edge> flat_edges;
            flat(directed_edges, flat_edges);

            std::vector<std::pair<indexType, std::vector<pid>>> undirected_graph;
            groupby(flat_edges, undirected_graph);

            const size_t merge_chunk = std::max<size_t>(1, static_cast<size_t>(undirected_graph.size() * delta));
            for (size_t chunk_begin = 0; chunk_begin < undirected_graph.size(); chunk_begin += merge_chunk)
            {
                const size_t chunk_end = std::min(chunk_begin + merge_chunk, undirected_graph.size());

#pragma omp parallel for
                for (size_t i = chunk_begin; i < chunk_end; ++i)
                {
                    const indexType vertex = undirected_graph[i].first;
                    auto &merged = old_neighbors[vertex];
                    merge_and_dedupe(merged, undirected_graph[i].second);
                }
            }

#pragma omp parallel for
            for (size_t index = 0; index < num_vertices; ++index)
            {
                size_t candidate_count = old_neighbors[index].size();
                if (alpha > 1e-5 || cos_angle > 1e-5)
                {
                    const size_t dist_calls = prune.robustPrune(static_cast<indexType>(-1),
                                                                old_neighbors[index].data(),
                                                                candidate_count, Points, alpha, cos_angle);
                    BuildStats.increment_dist(index, dist_calls);
                }

                if (candidate_count > G.max_degree())
                    candidate_count = G.max_degree();
                G[index].update_neighbors(old_neighbors[index].data(), candidate_count);
            }
        }

        static void reverse_graph(const std::vector<std::vector<pid>> &old_neighbors,
                                  std::vector<inv_neighbor_list> &inv_neighbors)
        {
            const size_t num_vertices = old_neighbors.size();
            std::vector<std::vector<edge>> directed_edges(num_vertices);

#pragma omp parallel for
            for (indexType i = 0; i < num_vertices; ++i)
            {
                const size_t neighbor_count = old_neighbors[i].size();
                std::vector<edge> edges(neighbor_count);
                for (size_t j = 0; j < neighbor_count; ++j)
                {
                    edges[j] = {old_neighbors[i][j].index, i};
                }
                directed_edges[i] = std::move(edges);
            }

            std::vector<edge> flat_edges;
            flat(directed_edges, flat_edges);
            groupby(flat_edges, inv_neighbors);

#pragma omp parallel for
            for (indexType i = 0; i < inv_neighbors.size(); ++i)
            {
                auto &sources = inv_neighbors[i].second;
                std::sort(sources.begin(), sources.end());
                sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
            }
        }

        void nn_descent_chunk(PointRange &Points, const std::vector<int> &changed,
                              std::vector<int> &new_changed, inv_neighbor_list *begin,
                              inv_neighbor_list *end, std::vector<std::vector<pid>> &old_neighbors,
                              QueryStatic &BuildStats)
        {
            const size_t chunk_size = end - begin;
            std::vector<std::vector<labelled_edge>> candidate_edges(chunk_size);

            // 利用反向边发现新的候选近邻，并按 NN-Descent 规则插入
#pragma omp parallel for schedule(auto)
            for (size_t i = 0; i < chunk_size; ++i)
            {
                const indexType vertex = (begin + i)->first;
                const auto &sources = (begin + i)->second;
                const auto &current_neighbors = old_neighbors[vertex];

                std::unordered_set<indexType> excluded;
                excluded.reserve(current_neighbors.size() + 1);
                excluded.insert(vertex);
                for (const auto &neighbor : current_neighbors)
                    excluded.insert(neighbor.index);

                std::vector<indexType> filtered_candidates;
                filtered_candidates.reserve(sources.size());
                for (const indexType source : sources)
                {
                    if (excluded.find(source) == excluded.end())
                        filtered_candidates.push_back(source);
                }

                auto &edges = candidate_edges[i];
                edges.reserve(L * 2);

                size_t dist_calls = 0;
                for (size_t l = 0; l < filtered_candidates.size(); ++l)
                {
                    const indexType j = filtered_candidates[l];
                    const float j_max = old_neighbors[j].back().dist;
                    for (size_t m = l + 1; m < filtered_candidates.size(); ++m)
                    {
                        const indexType k = filtered_candidates[m];
                        if (!changed[j] && !changed[k])
                            continue;

                        const float dist = Points[j].distance(Points[k]);
                        ++dist_calls;
                        const float k_max = old_neighbors[k].back().dist;
                        if (dist < j_max)
                            edges.emplace_back(j, pid(k, dist));
                        if (dist < k_max)
                            edges.emplace_back(k, pid(j, dist));
                    }
                }
                BuildStats.increment_dist(vertex, dist_calls);
            }

            std::vector<labelled_edge> flat_edges;
            flat(candidate_edges, flat_edges);

            std::vector<std::pair<indexType, std::vector<pid>>> grouped_candidates;
            groupby(flat_edges, grouped_candidates);

#pragma omp parallel for schedule(auto)
            for (size_t i = 0; i < grouped_candidates.size(); ++i)
            {
                const indexType vertex = grouped_candidates[i].first;
                auto &candidates = grouped_candidates[i].second;
                std::sort(candidates.begin(), candidates.end(), pid_less);
                dedupe_by_index(candidates);

                auto merged = old_neighbors[vertex];
                merge_and_dedupe(merged, candidates);
                if (merged.size() > L)
                    merged.resize(L);

                const auto &previous = old_neighbors[vertex];
                if (merged.size() != previous.size() || merged.back().index != previous.back().index)
                {
                    old_neighbors[vertex] = std::move(merged);
                    new_changed[vertex] = 1;
                }
            }
        }

        size_t nn_descent(PointRange &Points, std::vector<int> &new_changed,
                          std::vector<std::vector<pid>> &old_neighbors, QueryStatic &BuildStats)
        {
            std::vector<int> changed = new_changed;
            std::fill(new_changed.begin(), new_changed.end(), 0);

            std::vector<inv_neighbor_list> inv_neighbors;
            reverse_graph(old_neighbors, inv_neighbors);

            constexpr int batch_size = 100000;
            inv_neighbor_list *chunk_begin = inv_neighbors.data();
            const inv_neighbor_list *const inv_end = inv_neighbors.data() + inv_neighbors.size();
            while (chunk_begin != inv_end)
            {
                const int remaining = static_cast<int>(inv_end - chunk_begin);
                inv_neighbor_list *chunk_end = chunk_begin + std::min(remaining, batch_size);
                nn_descent_chunk(Points, changed, new_changed, chunk_begin, chunk_end, old_neighbors, BuildStats);
                chunk_begin = chunk_end;
            }

            return std::accumulate(new_changed.begin(), new_changed.end(), static_cast<size_t>(0));
        }

        int nn_descent_wrapper(PointRange &Points, std::vector<std::vector<pid>> &old_neighbors,
                               QueryStatic &BuildStats)
        {
            const size_t num_vertices = Points.size();
            std::vector<int> changed(num_vertices, 1);
            int rounds = 0;

            size_t change_num = num_vertices;
            const size_t convergence_threshold = static_cast<size_t>(delta * num_vertices);

            while (change_num >= convergence_threshold && rounds < max_rounds)
            {
                change_num = nn_descent(Points, changed, old_neighbors, BuildStats);
                ++rounds;
                std::cout << change_num << " elements changed" << std::endl;
                std::cout << "Round " << rounds << " of " << max_rounds << " completed" << std::endl;
            }

            std::cout << "descent converged in " << rounds << " rounds";
            if (rounds < max_rounds)
                std::cout << " (Early termination)";
            std::cout << std::endl;
            return rounds;
        }

    private:
        static bool pid_less(const pid &a, const pid &b)
        {
            return id_dist_less<indexType>(a, b);
        }

        static void dedupe_by_index(std::vector<pid> &neighbors)
        {
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end(),
                                        [](const pid &a, const pid &b)
                                        { return a.index == b.index; }),
                            neighbors.end());
        }

        static void merge_and_dedupe(std::vector<pid> &dst, const std::vector<pid> &src)
        {
            std::vector<pid> merged;
            merged.reserve(dst.size() + src.size());
            std::merge(dst.begin(), dst.end(), src.begin(), src.end(),
                       std::back_inserter(merged), pid_less);
            dedupe_by_index(merged);
            dst = std::move(merged);
        }
    };

}
